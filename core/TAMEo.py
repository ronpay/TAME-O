import re
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from colorama import Fore, Style
from loguru import logger
from PIL import Image

from core.dataset_utils import build_concept_dir_map, get_history_path, get_image_path
from core.utils.grounding_utils import detect_with_mllm_grounding
from core.utils.memory_utils import ConceptManager, MemoryManager
from core.utils.mllm_factory import MLLMFactory
from core.utils.openset_identifier import OpenSetIdentifier
from core.utils.retrieval_utils import Retriever

MODEL_SHORTCUTS = {"qwen": "Qwen/Qwen3.5-9B"}


def get_model_id(model_arg: str) -> str:
    """Convert model shortcut to full model ID"""
    return MODEL_SHORTCUTS.get(model_arg, model_arg)


def get_model_short_name(model_id: str) -> str:
    """Get short name for model ID"""
    for short_name, full_id in MODEL_SHORTCUTS.items():
        if full_id == model_id:
            return short_name
    return model_id.split("/")[-1].replace("-", "_").lower()


class TAMEo:
    """TAME-O: Open-Set TAME with two-stage concept identification and CRA tracking"""

    def __init__(
        self,
        model_id: str,
        dataset_name: str = "LCMP-H",
        ablation: str = None,
        top_k: int = 3,
        num_recent: int = 3,
        decay_rate: float = 1.0,
        threshold: float = 0.6,
        mitigation_interval: int = 10,
        fusion_top_l: int = 3,
        tao: float = 0.6,
    ):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.ablation = ablation
        self.top_k = top_k
        self.num_recent = num_recent
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.mitigation_interval = mitigation_interval
        self.fusion_top_l = fusion_top_l
        self.tao = tao
        self.model_short_name = get_model_short_name(model_id)

        # Determine memory directory name based on ablation setting
        if self.ablation == "TAME":
            self.run_name = f"tame_{self.model_short_name}"
        else:
            self.run_name = f"tameo_{self.model_short_name}"

        self.storage_name = str(Path(self.dataset_name) / self.run_name)

        self.model = MLLMFactory(model_id)
        self.concept_dirs = build_concept_dir_map(dataset_name)
        self._retriever = None
        self._openset_identifier = None
        self._detector = None

        self.concept_manager = ConceptManager(self.storage_name)
        self.concept_manager.read_memories()

        # TAME-O extension: CRA tracking, plus fork-aware CRA (FCRA)
        self.identification_stats = {
            "correct": 0,
            "incorrect": 0,
            "new_concept_correct": 0,
            "new_concept_incorrect": 0,
        }
        self.fork_aware_correct_sum = 0.0
        self.fork_aware_total = 0

        # Eq. (11) CFR tracking: cumulative history turn t and GT concepts C_t seen up to turn t
        self.history_turn_count = 0
        self.seen_gt_concepts = set()

        # Fission mitigation configuration (disabled for TAME)
        self.enable_fission_mitigation = self.ablation != "TAME"
        self.last_mitigation_turn = 0

    def get_concept_dir(self, concept_id: str) -> Path:
        return self.concept_dirs[concept_id]

    def get_history_path(self, concept_id: str) -> Path:
        return get_history_path(self.dataset_name, concept_id)

    def get_image_path(self, concept_id: str, image_id: str) -> Path:
        return get_image_path(self.dataset_name, concept_id, image_id)

    @property
    def retriever(self) -> Retriever:
        """Lazy-load retriever on first access"""
        if self._retriever is None:
            logger.info("Loading retriever with Qwen3-VL-Embedding-2B (lazy-loaded).")
            self._retriever = Retriever(decay_rate=self.decay_rate)
        return self._retriever

    @property
    def openset_identifier(self) -> OpenSetIdentifier:
        """Lazy-load open-set identifier on first access"""
        if self._openset_identifier is None:
            # Pre-load detector to share with OpenSetIdentifier
            detector_instance = self.detector

            if self.ablation == "TAME":
                logger.info(f"Loading ORIGINAL TAME identifier (Ablation: {self.ablation})")
                from core.utils.openset_identifier import OpenSetIdentifier_original

                self._openset_identifier = OpenSetIdentifier_original(
                    self.retriever,
                    self.concept_manager,
                    self.model,
                    top_k=self.top_k,
                    threshold=self.threshold,
                )
                # Share detector instance to avoid loading duplicate models
                self._openset_identifier._detector = detector_instance
            else:
                logger.info("Loading TAME-O open-set identifier (lazy-loaded).")
                self._openset_identifier = OpenSetIdentifier(
                    self.retriever,
                    self.concept_manager,
                    self.model,
                    ablation=self.ablation,
                    top_k=self.top_k,
                    num_recent=self.num_recent,
                    threshold=self.threshold,
                )
                # Share detector instance to avoid loading duplicate models
                self._openset_identifier._detector = detector_instance
        return self._openset_identifier

    @property
    def detector(self):
        """Lazy-load detector on first access"""
        if self._detector is None:
            logger.info("Loading detector for portrait object detection (lazy-loaded)")
            from core.utils.retrieval_utils import Detector

            self._detector = Detector()
        return self._detector

    def _detect_main_object(self, img: Image.Image) -> Optional[Image.Image]:
        """Detect and crop the main object from portrait image

        Args:
            img: Input portrait image

        Returns:
            Cropped main object image, or original image if detection fails
        """
        try:
            result = detect_with_mllm_grounding(self.detector, self.model, img)
            crops = result.crops
            if crops:
                return crops[0]
            return img
        except Exception as e:
            logger.warning(f"Portrait: Object detection failed: {e}, using original image")
            return img

    def dump_numbered_list(self, list: list) -> str:
        return "\n".join([f"{i + 1}. {item}" for i, item in enumerate(list)])

    def _extract_yaml_payload(self, text: str) -> str:
        """Extract YAML content from a fenced code block (```yaml or ```yml).
        Also handles cases without code blocks where YAML starts with '-' or is '[]'."""
        match = re.search(r"```(?:yaml|yml)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: Look for start of YAML list or empty list
        # Check for empty list at the end of the text
        if text.strip().endswith("[]"):
            return "[]"

        # Look for the first line starting with "- " which indicates a YAML list item
        yaml_start_match = re.search(r"(^|\n)- ", text)
        if yaml_start_match:
            return text[yaml_start_match.start() :].strip()

        return text

    def memory_exists(self, concept_id: str) -> bool:
        """Check if memory files already exist for a concept"""
        memory_manager = MemoryManager(concept_id, self.storage_name)
        return memory_manager.static_memory_path.exists() or memory_manager.dynamic_memory_path.exists()

    def _remove_empty_concepts(self):
        """Check all memory concept folders and remove empty ones"""
        if not self.concept_manager.memory_path.exists():
            return

        for concept_dir in list(self.concept_manager.memory_path.iterdir()):
            if concept_dir.is_dir() and not any(concept_dir.iterdir()):
                logger.info(f"Removing empty concept folder: {concept_dir.name}")
                try:
                    concept_dir.rmdir()
                    self.concept_manager.memories = [
                        m for m in self.concept_manager.memories if m.concept_id != concept_dir.name
                    ]
                except Exception as e:
                    logger.warning(f"Failed to remove empty folder {concept_dir}: {e}")

    #! read_history_all will not be used, ignore it.
    def read_history_all(self):
        skipped_count = 0
        processed_count = 0

        for concept_id in self.concept_dirs:
            if self.memory_exists(concept_id):
                skipped_count += 1
                logger.info(f"Skipping {concept_id} - memory already exists")
                continue

            processed_count += 1
            self.read_history(concept_id)

        self.concept_manager.reload_all()
        logger.info(
            f"Memory building completed: {processed_count} processed, {skipped_count} skipped. ConceptManager reloaded."
        )

    def read_history(self, concept_id: str, metrics_tracker=None):
        """Read history and build memory for a concept.

        During history reading, if the open-set identifier mistakenly flags an existing concept as
        NEW_CONCEPT, fork subsequent memory updates into a new suffixed concept_id (e.g.,
        `<ground_truth>_1`) to align with the QA workflow behavior.
        """
        logger.info(f"Reading history for concept {concept_id}")
        history_path = self.get_history_path(concept_id)

        if not history_path.exists():
            logger.warning(f"History file not found: {history_path}")
            return

        with open(history_path, "r", encoding="utf-8") as f:
            history = yaml.safe_load(f)

        if not history:
            logger.warning(f"Empty history file: {history_path}")
            return

        active_concept_id = concept_id

        for i, turn in enumerate(history, 1):
            logger.info(f"{Fore.GREEN}Turn {i} (ground truth: {concept_id}) start{Style.RESET_ALL}")
            self.history_turn_count += 1
            self.seen_gt_concepts.add(concept_id)

            if not isinstance(turn, dict):
                logger.warning(f"Invalid turn format: {turn}")
                continue

            image_id = turn.get("image_id")
            if image_id:
                image_path = self.get_image_path(concept_id, image_id)
                image = Image.open(image_path).convert("RGB")
            else:
                image = None

            updated_concept_id = self.read_history_single_turn(
                concept_id=active_concept_id,
                img=image,
                question=turn.get("user_input", ""),
                ground_truth_concept_id=concept_id,
                metrics_tracker=metrics_tracker,
            )
            if updated_concept_id is None:
                self._remove_empty_concepts()
                logger.info("Skipped ConceptManager refresh: no image for new concept.")
            else:
                active_concept_id = updated_concept_id
                self.concept_manager.refresh(active_concept_id)
                self._remove_empty_concepts()
                logger.info(f"Refreshed ConceptManager for {active_concept_id}")

            merge_results = self._maybe_mitigate_fission()
            active_concept_id = self._resolve_concept_id_after_mitigation(active_concept_id, merge_results)

            # CFR after this turn's memory write and PCF
            self.report_cfr(metrics_tracker)

    def read_history_single_turn(
        self,
        concept_id: str,
        img: Optional[Image.Image],
        question: str,
        ground_truth_concept_id: Optional[str] = None,
        metrics_tracker=None,
    ) -> Optional[str]:
        """Process a single history turn and update memory (with CRA tracking)"""
        if not concept_id:
            logger.error("concept_id cannot be empty")
            return concept_id

        expected_concept_id = ground_truth_concept_id or concept_id
        concept_id_for_write = concept_id

        def get_cra_str():
            total = sum(self.identification_stats.values())
            correct = self.identification_stats["correct"] + self.identification_stats["new_concept_correct"]
            return f"CRA: {self.get_cra():.1%} ({correct}/{total})"

        def get_fcra_str():
            if self.fork_aware_total == 0:
                return "FCRA: 0.0% (0.000/0)"
            return (
                f"FCRA: {self.get_fork_aware_cra():.1%} "
                f"({self.fork_aware_correct_sum:.3f}/{self.fork_aware_total})"
            )

        try:
            # TAME-O extension: Track concept identification accuracy during history reading.
            # Run for both image and text-only turns (OpenSetIdentifier supports img=None).
            if self.concept_manager.has_concepts():
                identified_id, is_existing = self.identify_concept(
                    img, question, gt_concept_id=expected_concept_id
                )

                is_correct = self.openset_identifier.check_correctness(identified_id, expected_concept_id)
                fork_count = self.get_concept_fork_count(expected_concept_id)
                fork_weight = self.get_fork_memory_write_weight(identified_id, expected_concept_id)
                self.fork_aware_total += 1

                if is_existing and is_correct:
                    self.identification_stats["correct"] += 1
                    self.fork_aware_correct_sum += fork_weight
                    logger.info(
                        f"✅ CRA: Correctly identified {expected_concept_id} (Matched: {identified_id}) | "
                        f"{get_cra_str()} | {get_fcra_str()} | forks={fork_count} weight={fork_weight:.3f}"
                    )
                    concept_id_for_write = identified_id
                elif is_existing:
                    self.identification_stats["incorrect"] += 1
                    logger.warning(
                        f"❌ CRA: Misidentified as {identified_id}, expected {expected_concept_id} | "
                        f"{get_cra_str()} | {get_fcra_str()} | forks={fork_count} weight={fork_weight:.3f}"
                    )
                    concept_id_for_write = identified_id
                    logger.info(f"Trusting model: Switching memory target to {concept_id_for_write}")
                else:
                    concept_already_exists = any(
                        memory.concept_id == expected_concept_id for memory in self.concept_manager.memories
                    )
                    if concept_already_exists:
                        self.identification_stats["new_concept_incorrect"] += 1
                        logger.warning(
                            f"❌ CRA: Identified as new concept (ground truth: {expected_concept_id}) | "
                            f"{get_cra_str()} | {get_fcra_str()} | forks={fork_count} weight={fork_weight:.3f}"
                        )

                        # Fork memory updates to a new suffix because the model believes this is new.
                        if ground_truth_concept_id and self.memory_exists(expected_concept_id):
                            i = 1
                            while self.memory_exists(f"{expected_concept_id}_{i}"):
                                i += 1
                            concept_id_for_write = f"{expected_concept_id}_{i}"
                            logger.info(
                                f"Forking history memory updates: {expected_concept_id} -> {concept_id_for_write}"
                            )
                    else:
                        self.identification_stats["new_concept_correct"] += 1
                        self.fork_aware_correct_sum += fork_weight
                        logger.info(
                            f"✅ CRA: Identified as new concept (ground truth: {expected_concept_id}) | "
                            f"{get_cra_str()} | {get_fcra_str()} | forks={fork_count} weight={fork_weight:.3f}"
                        )
                        concept_id_for_write = expected_concept_id

                if metrics_tracker:
                    try:
                        cra_correct = (
                            self.identification_stats["correct"] + self.identification_stats["new_concept_correct"]
                        )
                        metrics_tracker.log_cra(self.get_cra(), cra_correct, sum(self.identification_stats.values()))
                        metrics_tracker.log_fcra(
                            self.get_fork_aware_cra(), self.fork_aware_correct_sum, self.fork_aware_total
                        )
                    except Exception as e:
                        logger.warning(f"CRA logging error: {e}")
        except Exception as e:
            logger.warning(f"CRA tracking error: {e}")

        memory_manager = MemoryManager(concept_id_for_write, self.storage_name)

        if not (memory_manager.static_memory_path.exists() or memory_manager.dynamic_memory_path.exists()):
            if img is None:
                logger.info(f"Skipping new concept {concept_id_for_write}: No image to save.")
                memory_manager.remove_concept_directory()
                return None

        memory_manager.increment_memory_write_count()

        dynamic_memory = memory_manager.read_dynamic_memory()

        memory_type = "dynamic"
        memory_prompt_type = "dynamic"

        memory_prompt = f"""You are a strict Data Entry Assistant.
Your task is to update the **Dynamic Memory** (temporary buffer) based on the **Current Conversation**.

## GOAL
Extract **new information** provided in the conversation.
- If the User provides new facts, preferences, or current status -> **ADD** to memory.
- If the User updates/corrects existing info -> **MODIFY** the existing memory.
- If the conversation is just chit-chat (greetings, thanks) -> **DO NOTHING** (Output []).

## INPUT DATA
**Existing Dynamic Memory:**
```yaml
{self.dump_numbered_list(dynamic_memory if memory_type == "dynamic" else memory_manager.read_static_memory())}

```

**Current Conversation:**

* User Input: "{question}"
* Attached Image: {"Yes" if img else "No"}

## CRITICAL RULES FOR SMALL MODELS

1. **Be Specific:** Do not write "User asked a question." Write "User asked about the weather."
2. **No Duplicates:** Check the "Existing Dynamic Memory" list. If the info is already there, do not add it again.
3. **Concise:** Keep memory strings under 15 words.
4. **Names Matter:** If the name of a concept is mentioned, must add that name to memory.
5. **Context:** If the name is known, use it instead of "it".

## EXAMPLES

**Conversation:** "My cat Luna is limping today."
**Action:** ADD -> "Luna is limping today."

**Conversation:** "Hello, how are you?"
**Action:** IGNORE (No useful info).

**Conversation:** (User corrects a mistake) "Actually, Luna is 5 years old, not 4."
**Action:** MODIFY (Find index of old age memory) -> "Luna is 5 years old."

## OUTPUT FORMAT

Provide a `# Analysis` comment first, then the YAML code box.

# Analysis:
# - The user mentioned [X]. This is new information. I will ADD it.
# - The user said "Thanks". This is noise. I will IGNORE it.

```yaml
- op: "add" # or "modify" or "remove"
  memory: "The specific content extracted"
  target_id: 1 # Only required for 'modify' or 'remove'
```

If no updates are needed, output: []

Generate the output: (Start with # Analysis then YAML code box)
"""

        if img is not None:
            memory_response = self.model.chat_img(memory_prompt, img, max_tokens=2048)
        else:
            memory_response = self.model.chat_text(memory_prompt, max_tokens=2048)
        memory_one_line = memory_response.replace("\n", " ")
        logger.info(
            f"{memory_prompt_type.title()} memory update: len={len(memory_response)} content={memory_one_line[:500]!r}"
        )

        memory_yaml_payload = self._extract_yaml_payload(memory_response)
        applied_ops = memory_manager.parse_update_ops(memory_yaml_payload, memory_type)
        if isinstance(applied_ops, list):
            logger.debug(f"Applied {len(applied_ops)} {memory_type} ops")
        else:
            logger.debug(f"Applied 0 {memory_type} ops (unparsed response)")

        try:
            parsed_ops = yaml.safe_load(memory_yaml_payload)
            if isinstance(parsed_ops, list):
                logger.info(f"Processed {len(parsed_ops)} dynamic memory operations")
            elif parsed_ops is None or (isinstance(parsed_ops, list) and len(parsed_ops) == 0):
                logger.info("Processed 0 dynamic memory operations")
        except Exception as e:
            logger.debug(f"Could not count dynamic operations: {e}")

        static_memory = memory_manager.read_static_memory()
        dynamic_memory = memory_manager.read_dynamic_memory()

        if img is not None and not memory_manager.portrait_path.exists():
            portrait_img = self._detect_main_object(img)
            memory_manager.save_portrait(portrait_img)

        updated_dynamic_memory = memory_manager.read_dynamic_memory()

        if updated_dynamic_memory:
            static_analysis_prompt = f"""You are a strict Memory Manager. Your goal is to move PERMANENT FACTS from Dynamic Memory to Static Memory, while leaving TEMPORARY EVENTS alone.

## INPUT DATA
Current Dynamic Memory (Recent observations):
```yaml
{self.dump_numbered_list(updated_dynamic_memory)}

```

Current Static Memory (Long-term facts):

```yaml
{self.dump_numbered_list(static_memory)}

```

## CLASSIFICATION RULES (CRITICAL)

1. **PERMANENT FACTS (MOVE these):**
* Characteristics that rarely change.
* Examples: Names, species, breeds, personality traits (e.g., "timid"), physical features (e.g., "white fur"), favorite foods, owner's name.
* Action: Create a `static_ops` to ADD it and a `dynamic_ops` to REMOVE it.


2. **TEMPORARY EVENTS (DO NOT MOVE):**
* Things happening right now or recently.
* Examples: Eating, sleeping, walking, feeling hungry, distinct conversations, mood swings.
* Action: **IGNORE**. Do not create any operations for these.


## OUTPUT FORMAT

Provide a `# Analysis` comment first, then the YAML code box.

Example Output Structure:

# Analysis:
# - Dynamic Memory 1 (Luna is sleeping): This is a temporary EVENT. Ignore.
# - Dynamic Memory 2 (Luna is a cat): This is a FACT. Move to Static.

```yaml
# **Dynamic Memory Operations** (to remove transferred items):
dynamic_ops:
- op: "remove"
  target_id: 2 # Removing Item 2 because it was moved

# **Static Memory Operations** (to add/modify/remove persistent information) (duplicate items should be removed):
static_ops:
- op: "add"
  memory: "Luna is a cat" # The content of the fact
```

If no Permanent Facts are found, return empty lists:

```yaml
dynamic_ops: []
static_ops: []
```

Generate the output: (Start with # Analysis then YAML code box)
"""

            if img is not None:
                memory_transform_response = self.model.chat_img(static_analysis_prompt, img, max_tokens=2048)
            else:
                memory_transform_response = self.model.chat_text(static_analysis_prompt, max_tokens=2048)

            logger.info(f"Static memory update: {memory_transform_response.replace('\n', ' ')}")

            try:
                transform_yaml_payload = self._extract_yaml_payload(memory_transform_response)
                parsed_response = yaml.safe_load(transform_yaml_payload)

                if isinstance(parsed_response, dict):
                    dynamic_ops = parsed_response.get("dynamic_ops", [])
                    if dynamic_ops:
                        dynamic_yaml = yaml.dump(dynamic_ops, allow_unicode=True)
                        memory_manager.parse_update_ops(f"```yaml\n{dynamic_yaml}```", "dynamic")

                    static_ops = parsed_response.get("static_ops", [])
                    if static_ops:
                        static_yaml = yaml.dump(static_ops, allow_unicode=True)
                        memory_manager.parse_update_ops(f"```yaml\n{static_yaml}```", "static")

                    logger.info(
                        f"Processed {len(dynamic_ops)} dynamic and {len(static_ops)} static memory operations"
                    )

            except Exception as e:
                logger.error(f"Error processing static memory from dynamic memory: {e}")
                logger.debug(f"Raw response: {memory_transform_response}")

        memory_manager.apply_fifo_to_dynamic_memory(max_size=10)
        return concept_id_for_write

    def _extract_base_concept_id(self, concept_id: str, gt_concepts: set = None) -> str:
        """Extract base concept ID, utilizing ground truth knowledge if available.

        Args:
            concept_id: Concept identifier
            gt_concepts: Set of known ground truth concept IDs

        Returns:
            Base concept ID (longest prefix found in gt_concepts, or concept_id if no parent found)
        """
        if gt_concepts:
            if concept_id in gt_concepts:
                return concept_id

            # Try stripping _\d+ suffixes to find a GT parent
            curr = concept_id
            while True:
                match = re.search(r"^(.*)_\d+$", curr)
                if not match:
                    break
                candidate = match.group(1)
                if candidate in gt_concepts:
                    return candidate
                curr = candidate
            return concept_id

        # Fallback if no GT provided: return original to avoid incorrect aggressive stripping
        return concept_id

    def _group_concept_identifiers(self) -> dict:
        """Group current memory identifiers by base ground-truth concept."""
        concept_groups = {}
        gt_concepts = set(self.concept_dirs)

        for memory in self.concept_manager.memories:
            base_id = self._extract_base_concept_id(memory.concept_id, gt_concepts)
            concept_groups.setdefault(base_id, []).append(memory.concept_id)

        return concept_groups

    def get_concept_fork_count(self, concept_id: str) -> int:
        """Return the number of extra identifiers currently associated with a concept."""
        if not self.concept_manager.has_concepts():
            return 0

        gt_concepts = set(self.concept_dirs)
        base_id = self._extract_base_concept_id(concept_id, gt_concepts)
        identifiers = self._group_concept_identifiers().get(base_id, [])
        return max(0, len(identifiers) - 1)

    def get_fork_memory_write_weight(self, predicted_id: Optional[str], gt_concept_id: str) -> float:
        """Return the predicted fork's share of memory-write counts for the GT concept."""
        if not self.concept_manager.has_concepts():
            return 1.0

        gt_concepts = set(self.concept_dirs)
        base_id = self._extract_base_concept_id(gt_concept_id, gt_concepts)
        identifiers = self._group_concept_identifiers().get(base_id, [])
        if not identifiers:
            return 1.0
        if not predicted_id:
            return 0.0

        write_counts = {
            memory.concept_id: self._get_memory_write_count(memory)
            for memory in self.concept_manager.memories
            if memory.concept_id in identifiers
        }
        total_writes = sum(write_counts.values())
        if total_writes <= 0:
            return 1.0 / len(identifiers)
        return write_counts.get(predicted_id, 0) / total_writes

    @staticmethod
    def _get_memory_write_count(memory) -> int:
        if hasattr(memory, "get_memory_write_count"):
            return memory.get_memory_write_count()
        return 1

    def calculate_cfr(self) -> Tuple[float, int, int, int]:
        """Calculate Concept Fission Rate (CFR) as defined in Eq. (11):

        CFR(t) = F(t) / max(t - |C_t|, 1), with F(t) = Σ_c max(0, |Î_c(t)| - 1)

        t is the cumulative history turn and |C_t| the number of GT concepts seen up to turn t,
        so t - |C_t| counts revisit turns, the maximum possible number of forks, and CFR ∈ [0, 1].

        Returns:
            Tuple of (cfr, total_fissions, denominator, total_identifiers)
            - total_fissions: surviving forked identifiers F(t) (numerator)
            - denominator: max(t - |C_t|, 1)
            - total_identifiers: Σ_c |Î_c(t)|, kept for the F/I robustness check
        """
        denominator = max(self.history_turn_count - len(self.seen_gt_concepts), 1)
        if not self.concept_manager.has_concepts():
            return 0.0, 0, denominator, 0

        concept_groups = self._group_concept_identifiers()

        total_fissions = 0
        total_identifiers = 0

        for identifiers in concept_groups.values():
            num_identifiers = len(identifiers)
            total_identifiers += num_identifiers
            total_fissions += max(0, num_identifiers - 1)

        cfr = total_fissions / denominator

        return cfr, total_fissions, denominator, total_identifiers

    def report_cfr(self, metrics_tracker=None):
        """Log Eq. (11) CFR for the current history turn, after its memory write (and PCF)."""
        try:
            cfr, total_fissions, denominator, total_identifiers = self.calculate_cfr()

            # Log CFR to the metrics tracker
            if metrics_tracker and total_identifiers > 0:
                metrics_tracker.log_cfr(cfr, total_fissions, denominator, self.history_turn_count)

            # Display CFR status
            if cfr == 0.0:
                symbol = "✓"
                message = "CFR: No fission detected"
            elif cfr < 0.3:
                symbol = "✓"
                message = "CFR: Low fission rate"
            else:
                symbol = "⚠"
                message = "CFR: Elevated fission rate"

            logger.info(
                f"{symbol} {message} | CFR: {cfr:.3f} ({total_fissions}/{denominator}) | "
                f"step={self.history_turn_count} gt_concepts={len(self.seen_gt_concepts)} "
                f"identifiers={total_identifiers}"
            )
        except Exception as e:
            logger.warning(f"CFR tracking error: {e}")

    def _maybe_mitigate_fission(self):
        """Run PCF exactly on T-step intervals of the global history turn counter."""
        if not self.enable_fission_mitigation:
            return None

        if self.history_turn_count - self.last_mitigation_turn >= self.mitigation_interval:
            logger.info(f"Running fission mitigation at interaction step {self.history_turn_count}")
            results = self._run_fission_mitigation()
            self.last_mitigation_turn = self.history_turn_count
            return results

        return None

    def _resolve_concept_id_after_mitigation(self, concept_id: str, merge_results: Optional[dict]) -> str:
        """Return the surviving concept id if PCF merged the active concept."""
        if not merge_results:
            return concept_id

        merge_map = {merge["source"]: merge["target"] for merge in merge_results.get("merges", [])}
        resolved_id = concept_id
        seen = set()
        while resolved_id in merge_map and resolved_id not in seen:
            seen.add(resolved_id)
            resolved_id = merge_map[resolved_id]

        if resolved_id != concept_id:
            logger.info(f"Active concept updated after PCF merge: {concept_id} -> {resolved_id}")
        return resolved_id

    def _run_fission_mitigation(self):
        """Run concept fission mitigation"""
        if not self.enable_fission_mitigation:
            logger.debug("Fission mitigation disabled")
            return None

        import time

        from core.utils.fission_mitigator import ConceptFissionMitigator

        logger.info(f"{Fore.CYAN}===== FISSION MITIGATION START ====={Style.RESET_ALL}")

        start_time = time.time()

        mitigator = ConceptFissionMitigator(
            concept_manager=self.concept_manager,
            retriever=self.retriever,
            mllm=self.model,
            similarity_top_k=self.fusion_top_l,
            tao=self.tao,
        )

        results = mitigator.detect_and_merge_fissions()

        elapsed = time.time() - start_time

        logger.info(f"{Fore.CYAN}===== FISSION MITIGATION COMPLETE ====={Style.RESET_ALL}")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Merged: {results['merged_count']} pairs")
        logger.info(f"Failed: {results['failed_count']} pairs")

        for merge in results["merges"]:
            logger.info(f"  • {merge['source']} → {merge['target']} (sim={merge['similarity']:.3f})")

        return results

    def identify_concept(
        self, img: Optional[Image.Image], question: str, gt_concept_id: Optional[str] = None
    ) -> Tuple[Optional[str], bool]:
        """TAME-O: Open-set concept identification (supports image or text-only turns)"""
        concept_id, is_existing = self.openset_identifier.identify(img, question, gt_concept_id=gt_concept_id)
        return concept_id, is_existing

    def get_context_prompt(
        self, concept_id: str, question: str | None = None, img: Optional[Image.Image] = None
    ) -> str:
        """Get context prompt with optional memory alignment"""
        if not concept_id:
            logger.error("concept_id cannot be empty")
            return ""

        memory_manager = MemoryManager(concept_id, self.storage_name)

        static_memory = memory_manager.read_static_memory()
        dynamic_memory = memory_manager.read_dynamic_memory()

        if question:
            original_context = self._build_memory_context(concept_id, static_memory, dynamic_memory)

            alignment_prompt = f"""You are an AI assistant. Your task is to extract relevant facts from the provided memory to help answer a question.

# Available Memory
{original_context}

# User Request
- Question: "{question}"
- Image Provided: {"Yes" if img else "No"}

# Instructions
1.  **Extract Relevant Facts:** From the memory, pull out details that are direct answers, useful context, or visual descriptions (especially if an image is provided).
2.  **List Facts as Bullets:** Present the extracted information as a simple bulleted list.
3.  **Do Not Answer the Question:** This is critical. Only output the list of facts.

# Output Format
- [A relevant fact 1 from memory]
- [A relevant fact 2 from memory]
- ...

Generate the aligned memory:
"""

            if img is not None:
                aligned_response = self.model.chat_img(alignment_prompt, img, max_tokens=2048)
            else:
                aligned_response = self.model.chat_text(alignment_prompt, max_tokens=2048)

            logger.info(f"Memory aligned for question: {question[:50]}...")
            return aligned_response
        else:
            return self._build_memory_context(concept_id, static_memory, dynamic_memory)

    def answer_question(
        self, concept_id: str, question: str, context_prompt: str, img: Optional[Image.Image] = None
    ) -> str:
        """Answer a free-text question using memory context"""
        if not concept_id or not question:
            logger.error("concept_id and question cannot be empty")
            return "Error: Invalid parameters provided."

        answer_prompt = f"""You are a personalized AI assistant with detailed knowledge about specific concepts and their current states. You have access to both static stable information and dynamic contextual information about the concept.

# CONCEPT INFORMATION
{context_prompt}

# CURRENT QUESTION
User asks: "{question}"
{"Image provided: Yes" if img else "Image provided: No"}

# YOUR TASK
Answer the user's question comprehensively using the available memory information. Consider both:
1. **Static characteristics**: Permanent traits, preferences, and features
2. **Dynamic context**: Recent events, temporary changes, current states

# RESPONSE GUIDELINES
- Provide accurate information based on the memory context
- If the question involves preferences or characteristics, consider both static patterns and recent changes
- If asking about identification from an image, use both stable visual features and recent contextual changes
- If there's a conflict between static and dynamic information, acknowledge both and explain the current situation
- Be conversational and helpful while being precise about the information you have
- Always provide a clear and concise response in a single paragraph.

Generate your response:"""

        if img is not None:
            response = self.model.chat_img(answer_prompt, img, max_tokens=2048)
        else:
            response = self.model.chat_text(answer_prompt, max_tokens=2048)

        return response

    def answer_choice_question(
        self,
        concept_id: str,
        question: str,
        options: List[str],
        context_prompt: str,
        img: Optional[Image.Image] = None,
    ) -> str:
        """Answer a multiple choice question"""
        if not concept_id or not question or not options:
            logger.error("concept_id, question, and options cannot be empty")
            return "Error: Invalid parameters provided."

        if len(options) != 4:
            logger.error("Exactly 4 options must be provided")
            return "Error: Exactly 4 options required."

        formatted_options = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])

        choice_prompt = f"""You are a personalized AI assistant with detailed knowledge about specific concepts and their current states. You have access to both static stable information and dynamic contextual information about the concept.

# CONCEPT INFORMATION
{context_prompt}

# MULTIPLE CHOICE QUESTION
Question: "{question}"
{"Image provided: Yes" if img else "Image provided: No"}

Options:
{formatted_options}

# YOUR TASK
Based on the available memory information about the concept, select the most appropriate answer from the given options. Consider both:
1. **Static characteristics**: Permanent traits, preferences, and features
2. **Dynamic context**: Recent events, temporary changes, current states

# RESPONSE GUIDELINES
- Analyze each option carefully against the memory information
- Choose the option that best matches the concept's characteristics and current context
- If there's a conflict between static and dynamic information, prioritize the most relevant information for the question
- Respond with ONLY the letter (A, B, C, or D) of your chosen answer

Generate your response (only the letter):"""

        if img is not None:
            response = self.model.chat_img(choice_prompt, img, max_tokens=2048)
        else:
            response = self.model.chat_text(choice_prompt, max_tokens=2048)

        choice_match = re.search(r"[ABCD]", response.strip().upper())
        if choice_match:
            return choice_match.group()
        else:
            logger.warning(f"Could not extract valid choice from response: {response}")
            return "A"

    def complete_qa_workflow(
        self,
        img_path: str,
        question: str,
        options: Optional[List[str]] = None,
        options_answer: Optional[str] = None,
        ground_truth_concept_id: Optional[str] = None,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """TAME-O: Complete QA workflow with open-set identification"""
        if not img_path or not question:
            logger.error("img_path and question cannot be None/empty")
            return None, "Error: Invalid parameters provided.", None

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            return None, "Error: Could not open image.", None

        concept_id, is_existing = self.identify_concept(img, question, gt_concept_id=ground_truth_concept_id)

        if not is_existing:
            logger.warning(f"NEW_CONCEPT detected but expected {ground_truth_concept_id}")
            if ground_truth_concept_id:
                if self.memory_exists(ground_truth_concept_id):
                    i = 1
                    while self.memory_exists(f"{ground_truth_concept_id}_{i}"):
                        i += 1
                    concept_id = f"{ground_truth_concept_id}_{i}"
                    logger.info(
                        f"Ground truth {ground_truth_concept_id} exists. Assigning new ID: {concept_id}"
                    )
                else:
                    concept_id = ground_truth_concept_id
            else:
                return None, "Error: New concept detected with no ground truth.", None

        if concept_id is None:
            return None, "Error: Could not identify any concept in the image.", None

        if ground_truth_concept_id:
            if not is_existing:
                is_correct = False
            else:
                is_correct = self.openset_identifier.check_correctness(concept_id, ground_truth_concept_id)

            status_symbol = " ✅" if is_correct else " ❌"
            logger.info(
                f"{Fore.CYAN}Concept Identification{Style.RESET_ALL}: Predicted={concept_id}, Ground Truth={ground_truth_concept_id}{status_symbol}"
            )
        else:
            logger.info(f"{Fore.CYAN}Concept Identification{Style.RESET_ALL}: Predicted={concept_id}")

        context_prompt = self.get_context_prompt(concept_id, question, img)

        free_text_answer = self.answer_question(concept_id, question, context_prompt, img)

        choice_answer = None
        if options and len(options) == 4:
            choice_answer = self.answer_choice_question(concept_id, question, options, context_prompt, img)

        return concept_id, free_text_answer, choice_answer

    def get_cra(self) -> float:
        """TAME-O: Calculate Concept Recognition Accuracy"""
        total = sum(self.identification_stats.values())
        if total == 0:
            return 0.0
        correct_total = (
            self.identification_stats["correct"] + self.identification_stats["new_concept_correct"]
        )
        return correct_total / total

    def get_fork_aware_cra(self) -> float:
        """Calculate fork-aware Concept Recognition Accuracy."""
        if self.fork_aware_total == 0:
            return 0.0
        return self.fork_aware_correct_sum / self.fork_aware_total

    def _build_memory_context(self, concept_id: str, static_memory: list, dynamic_memory: list) -> str:
        """Build memory context string"""
        context_parts = []

        if static_memory:
            context_parts.append("## STATIC MEMORY")

            for memory_item in static_memory:
                if isinstance(memory_item, str):
                    context_parts.append(f"- {memory_item}")
                else:
                    context_parts.append(f"- {str(memory_item)}")

        context_parts.append("\n## DYNAMIC MEMORY")
        if dynamic_memory:
            for memory_item in dynamic_memory:
                if isinstance(memory_item, str):
                    context_parts.append(f"- {memory_item}")
                else:
                    context_parts.append(f"- {str(memory_item)}")
        else:
            context_parts.append("No recent contextual information available.")

        return "\n".join(context_parts)
