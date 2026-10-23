from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image

from core.utils.grounding_utils import detect_with_mllm_grounding


@dataclass
class Stage0Result:
    """Stage 0 preprocessing output"""

    original_img: Optional[Image.Image]
    original_text: str
    segmented_objects: List[Image.Image]  # Top-3 confidence objects
    semantic_query: str
    concept_name_keyword: Optional[str]
    direct_keywords: List[str]
    has_objects: bool
    is_text_only: bool
    grounding_labels: List[str]
    grounding_label_source: Optional[str]


class OpenSetIdentifier:
    """TAME-O two-stage open-set concept identification"""

    def __init__(
        self,
        retriever,
        concept_manager,
        mllm,
        ablation=None,
        top_k=3,
        num_recent=3,
        dynamic_weight=0.5,
        static_weight=0.5,
        threshold=0.6,
    ):
        self.retriever = retriever
        self.concept_manager = concept_manager
        self.mllm = mllm
        self.ablation = ablation
        self.top_k = top_k
        self.num_recent = num_recent
        self.dynamic_weight = dynamic_weight
        self.static_weight = static_weight
        self.threshold = threshold
        self._detector = None  # Lazy-loaded detector
        self.conversation_history = deque(maxlen=3)  # Store last 3 conversations

    @property
    def detector(self):
        """Lazy-load detector on first access"""
        if self._detector is None:
            logger.info("Loading detector for Stage 0 (lazy-loaded)")
            from core.utils.retrieval_utils import Detector

            self._detector = Detector()
        return self._detector

    def _segment_question_objects(
        self, question_img: Image.Image, question_text: str
    ) -> Tuple[List[Image.Image], List[str], str]:
        """Segment top-3 objects from question image using Grounding DINO"""
        try:
            detection_result = detect_with_mllm_grounding(
                self.detector,
                self.mllm,
                question_img,
                question_text=question_text,
            )
            all_objects = detection_result.crops

            if not all_objects:
                logger.info("Stage 0: No objects detected")
                return [], detection_result.labels, detection_result.label_source

            # Return top-3 objects (already sorted by confidence in detector)
            top_k_objects = all_objects[:3]
            logger.info(
                "Stage 0: Detected {} objects with {} labels {}, using top-{}",
                len(all_objects),
                detection_result.label_source,
                detection_result.labels,
                len(top_k_objects),
            )
            return top_k_objects, detection_result.labels, detection_result.label_source

        except Exception as e:
            logger.warning(f"Stage 0: Object detection failed: {e}")
            return [], [], "generic_fallback"

    def stage0_preprocess(self, question_img: Optional[Image.Image], question_text: str) -> Stage0Result:
        """Stage 0: Preprocess question for enhanced retrieval"""

        # Text preprocessing (always run)
        semantic_query, concept_name_keyword, direct_keywords = self._preprocess_question_text(question_text)

        # Image segmentation (only if image provided)
        segmented_objects = []
        has_objects = False
        grounding_labels: List[str] = []
        grounding_label_source: Optional[str] = None

        if question_img is not None:
            segmented_objects, grounding_labels, grounding_label_source = self._segment_question_objects(
                question_img, question_text
            )
            has_objects = len(segmented_objects) > 0

        return Stage0Result(
            original_img=question_img,
            original_text=question_text,
            segmented_objects=segmented_objects,
            semantic_query=semantic_query,
            concept_name_keyword=concept_name_keyword,
            direct_keywords=direct_keywords,
            has_objects=has_objects,
            is_text_only=(question_img is None),
            grounding_labels=grounding_labels,
            grounding_label_source=grounding_label_source,
        )

    def _calculate_scores(self, concept, stage0_result) -> dict:
        """Calculate dynamic and static scores for a concept"""
        # Dynamic score: Use semantic query instead of original text
        dynamic_memory_with_timestamps = concept.get_dynamic_memory_with_timestamps()

        try:
            if len(dynamic_memory_with_timestamps) > 0:
                query_text = stage0_result.semantic_query

                # Multi-object query: try each segmented object
                if stage0_result.has_objects:
                    object_scores = []
                    for obj_img in stage0_result.segmented_objects:
                        temporal_score = self.retriever.temporal_weighted_retrieval(
                            obj_img, query_text, dynamic_memory_with_timestamps
                        )
                        object_scores.append(temporal_score)

                    dynamic_score = max(object_scores) if object_scores else 0.0
                else:
                    # Fallback to original image
                    dynamic_score = self.retriever.temporal_weighted_retrieval(
                        stage0_result.original_img, query_text, dynamic_memory_with_timestamps
                    )
            else:
                dynamic_score = 0.0
        except Exception as e:
            logger.warning(
                "Stage 1: dynamic scoring failed for concept_id={} (dyn_mem_count={}): {}",
                concept.concept_id,
                len(dynamic_memory_with_timestamps),
                e,
            )
            logger.debug(
                "Stage 1: dynamic scoring context: has_objects={}, obj_count={}, query_snippet={!r}",
                stage0_result.has_objects,
                len(stage0_result.segmented_objects),
                stage0_result.semantic_query[:200],
            )
            dynamic_score = 0.0

        # Static score: Enhanced with keyword matching
        portrait = concept.read_portrait()
        static_memory = concept.read_static_memory()

        img_similarity = 0.0
        text_sim = 0.0

        try:
            # Portrait similarity (unchanged - use original image)
            if portrait is not None and stage0_result.original_img is not None:
                portrait_emb = self.retriever.encode_passage_images([portrait])
                query_emb = self.retriever.encode_query_images([stage0_result.original_img])
                img_similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(query_emb), torch.tensor(portrait_emb), dim=1
                ).item()
            else:
                img_similarity = 0.0

            # Text similarity with semantic query
            if len(static_memory) > 0:
                static_embs = self.retriever.encode_passage_text(static_memory)

                # Use semantic query instead of original text
                query_emb = self.retriever.encode_passage_text([stage0_result.semantic_query])

                text_similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(query_emb), torch.tensor(static_embs), dim=1
                )
                text_sim = torch.max(text_similarity).item()
            else:
                text_sim = 0.0

            static_score: int | float = 0.7 * img_similarity + 0.3 * text_sim

        except Exception as e:
            logger.warning(
                "Stage 1: static scoring failed for concept_id={} (static_mem_count={}): {}",
                concept.concept_id,
                len(static_memory),
                e,
            )
            logger.debug(
                "Stage 1: static scoring context: has_portrait={}, has_query_img={}, query_snippet={!r}",
                portrait is not None,
                stage0_result.original_img is not None,
                stage0_result.semantic_query[:200],
            )
            static_score = 0.0

        return {
            "dynamic_score": dynamic_score,
            "static_score": static_score,
            "img_similarity": img_similarity,
            "text_sim": text_sim,
        }

    def _log_concept_scores(self, concept_id, scores, prefix=""):
        logger.debug(
            f"{prefix}Concept {concept_id}: Dynamic={scores['dynamic_score']:.4f}, Static={scores['static_score']:.4f} "
            f"(ImgSim={scores['img_similarity']:.4f}, TextSim={scores['text_sim']:.4f})"
        )

    def stage1_retrieve(
        self, question_img: Optional[Image.Image], question_text: str
    ) -> Tuple[list, Stage0Result, dict]:
        """Stage 1: MLLM-free retrieval with two-way weighting (enhanced with Stage 0)"""
        # Stage 0: Preprocess
        stage0_result = self.stage0_preprocess(question_img, question_text)
        logger.debug(
            "Stage 1: preprocess summary: is_text_only={}, semantic_query_len={}, keywords={}",
            stage0_result.is_text_only,
            len(stage0_result.semantic_query),
            stage0_result.direct_keywords,
        )

        if not self.concept_manager.has_concepts():
            logger.info("No existing concepts, treating as new concept")
            return [], stage0_result, {}

        concepts = self.concept_manager.memories
        all_scores_dicts = []
        dynamic_scores = []
        img_similarities = []
        text_sims = []

        for concept in concepts:
            scores = self._calculate_scores(concept, stage0_result)
            all_scores_dicts.append(scores)

            dynamic_scores.append(scores["dynamic_score"])
            img_similarities.append(scores["img_similarity"])
            text_sims.append(scores["text_sim"])

        # Fallback: if dynamic_score is 0.0, set to min(non-zero dynamic scores)
        non_zero_dynamics = [s for s in dynamic_scores if s > 0.0]
        if non_zero_dynamics:
            min_dynamic = min(non_zero_dynamics)
            dynamic_scores = [s if s > 0.0 else min_dynamic for s in dynamic_scores]

        # Fallback for img_similarity
        non_zero_imgs = [s for s in img_similarities if s > 0.0]
        if non_zero_imgs:
            min_img = min(non_zero_imgs)
            img_similarities = [s if s > 0.0 else min_img for s in img_similarities]

        # Fallback for text_sim
        non_zero_texts = [s for s in text_sims if s > 0.0]
        if non_zero_texts:
            min_text = min(non_zero_texts)
            text_sims = [s if s > 0.0 else min_text for s in text_sims]

        # Recompute static scores based on adjusted components
        static_scores = [
            0.7 * img_sim + 0.3 * txt_sim
            for i, (img_sim, txt_sim) in enumerate(zip(img_similarities, text_sims))
        ]

        # Update scores dicts and log them (using the adjusted values)
        all_scores_map = {}
        for i, concept in enumerate(concepts):
            # Update the specific components that might have changed due to fallback
            scores = all_scores_dicts[i]
            scores["dynamic_score"] = dynamic_scores[i]
            scores["img_similarity"] = img_similarities[i]
            scores["text_sim"] = text_sims[i]
            scores["static_score"] = static_scores[i]

            self._log_concept_scores(concept.concept_id, scores)
            all_scores_map[concept.concept_id] = scores

        dynamic_scores_tensor = torch.tensor(dynamic_scores)
        static_scores_tensor = torch.tensor(static_scores)

        combined_scores = (
            self.dynamic_weight * dynamic_scores_tensor + self.static_weight * static_scores_tensor
        )

        top_k = min(self.top_k, len(combined_scores))
        if top_k == 0:
            return [], stage0_result, all_scores_map

        top_k_values, top_k_indices = torch.topk(combined_scores, top_k)

        top_k_concepts = [
            {"idx": idx.item(), "concept_id": concepts[idx].concept_id, "score": top_k_values[i].item()}
            for i, idx in enumerate(top_k_indices)
        ]

        logger.info(
            f"Stage 1 retrieved top-{top_k} concepts: {[f'{c["concept_id"]} ({c["score"]:.3f})' for c in top_k_concepts]}"
        )
        return top_k_concepts, stage0_result, all_scores_map

    def stage2_verify(
        self, question_img: Optional[Image.Image], question_text: str, top_k_concepts: list
    ) -> Tuple[Optional[str], bool]:
        """Stage 2: MLLM verification for EXISTING vs NEW_CONCEPT"""
        if not top_k_concepts:
            return None, False

        prompt = self._build_verification_prompt(question_text, question_img, top_k_concepts)

        try:
            if question_img is not None:
                portraits = []
                for concept_info in top_k_concepts:
                    concept = self.concept_manager.memories[concept_info["idx"]]
                    portrait = concept.read_portrait()
                    if portrait is not None:
                        portraits.append(portrait)

                images = [question_img] + portraits
                response = self.mllm.chat_multi_img(prompt, images, max_tokens=2048)
            else:
                response = self.mllm.chat_text(prompt, max_tokens=2048)

            logger.debug(f"Stage 2 MLLM response: {response}")

            if "EXISTING:" in response:
                raw_label = response.split("EXISTING:")[1].strip().split()[0]
                candidate_label = raw_label.rstrip(".,;!?").upper()
                # The MLLM only sees neutral labels; map them back to concept identifiers here.
                label_to_concept = {
                    f"CANDIDATE_{i + 1}": concept_info["concept_id"]
                    for i, concept_info in enumerate(top_k_concepts)
                }

                if candidate_label.isdigit():
                    candidate_label = f"CANDIDATE_{candidate_label}"

                if candidate_label in label_to_concept:
                    concept_id = label_to_concept[candidate_label]
                    logger.info(f"Stage 2 identified EXISTING concept: {concept_id}")
                    return concept_id, True
                else:
                    logger.warning(
                        "Stage 2 hallucinated candidate label '{}' (cleaned='{}'). Valid candidates: {}. Treating as NEW_CONCEPT.",
                        raw_label,
                        candidate_label,
                        list(label_to_concept),
                    )
                    return None, False
            elif "NEW_CONCEPT" in response:
                logger.info("Stage 2 identified NEW_CONCEPT")
                return None, False
            else:
                logger.warning("Stage 2 failed to parse response, using stage 1 top-1")
                return top_k_concepts[0]["concept_id"], True

        except Exception as e:
            logger.error(f"Stage 2 MLLM error: {e}, falling back to stage 1 top-1")
            return top_k_concepts[0]["concept_id"], True

    def _build_verification_prompt(
        self, question_text: str, question_img: Optional[Image.Image], top_k_concepts: list
    ) -> str:
        """Build prompt for stage 2 MLLM verification"""
        has_image = "present" if question_img is not None else "absent"
        image_note = ""
        if question_img is not None:
            image_note = (
                "\nThe first attached image is the user's question image. "
                "Candidate portrait images, when available, follow in candidate order."
            )

        prompt = f"""You are a concept identification assistant. Your goal is to determine if the user's question refers to one of the EXISTING candidates or to a completely NEW concept.

# Question
{question_text}
Image: {has_image}{image_note}

"""

        # Add historical conversations if available
        if self.conversation_history:
            prompt += "# Historical Conversations (Recent Context):\n\n"
            for idx, conv_text in enumerate(self.conversation_history, 1):
                prompt += f"{idx}. {conv_text}\n"

        prompt += "# Candidates (Existing Memories):\n\n"

        concepts = self.concept_manager.memories
        for i, concept_info in enumerate(top_k_concepts):
            concept_idx = concept_info["idx"]
            concept = concepts[concept_idx]
            candidate_label = f"CANDIDATE_{i + 1}"

            static_memory = concept.read_static_memory()
            dynamic_memory_with_timestamps = concept.get_dynamic_memory_with_timestamps()

            prompt += f"\n## Candidate: {candidate_label}\n"
            if question_img is not None:
                prompt += f"- Portrait: candidate portrait image {i + 1} after the question image\n"

            prompt += "- Static Memory (latest 5):\n"
            if static_memory:
                for mem in static_memory[:5]:
                    prompt += f"  - {mem}\n"
            else:
                prompt += "  - (No static memory)\n"

            prompt += f"- Recent Dynamic Memory (latest {self.num_recent}):\n"
            if dynamic_memory_with_timestamps:
                recent_memories = dynamic_memory_with_timestamps[-self.num_recent :]
                for mem, _ in recent_memories:
                    prompt += f"  - {mem}\n"
            else:
                prompt += "  - (No dynamic memory)\n"

        prompt += """
# Instructions
Compare the entity in the Question against each Candidate.
- **Use Historical Conversations** for context: Previous identifications can help disambiguate similar entities.
- Check for mismatches in **Name**, **Species**, **Physical Appearance**, or **Core Attributes**.
- If the name mentioned in Question is different from name in candidates' memory, they definitely are not the same concept.
- If the Question describes an entity (e.g., "Hamster", "Person") that is fundamentally different from a Candidate (e.g., "Parrot", "Shark Doll"), they are NOT the same.
- If the Question explicitly mentions a candidate's proper name, select that candidate unless species or stable identity details contradict it.
- If the Question is an update about a known subject, accessories, locations, moods, poses, food, damage, weather, lighting, or temporary activities may differ from memory. These are not enough by themselves to make a new concept.
- For pets, compare stable identity details such as name, species, breed, coat color, markings, eye color, and recurring accessories. Do not reject a candidate only because the current event or object is new.
- For pets within the same species, select "NEW_CONCEPT" only when stable identity details clearly conflict or no candidate has enough stable identity support.
- If the tone of the question is to introduce a new concept (e.g., person, pet, or belongings), then select "NEW_CONCEPT".
- Context (e.g., "pet", "friend") might be similar, but specific details must match.
- Use only the neutral candidate labels shown above. Do not infer identity from labels.
- If NO candidate is a strong match, select "NEW_CONCEPT".

# Output Format
Output ONLY one of the following:
1. "EXISTING:CANDIDATE_N" (if it matches a candidate, e.g. EXISTING:CANDIDATE_1)
2. "NEW_CONCEPT" (if it is a new or different entity)

Generate your response:
"""

        return prompt

    def _preprocess_question_text(self, question_text: str) -> Tuple[str, Optional[str], List[str]]:
        """Preprocess question text using MLLM"""
        prompt = self._build_preprocessing_prompt(question_text)

        try:
            response = self.mllm.chat_text(prompt, max_tokens=2048)
            if response is None or not str(response).strip():
                logger.warning(
                    "Stage 0: Empty MLLM response for text preprocessing; falling back to original question text"
                )
                return question_text, None, self._extract_simple_keywords(question_text)

            semantic_query, concept_name_keyword, keywords = self._parse_preprocessing_response(response)
            if not semantic_query:
                logger.warning(
                    "Stage 0: Empty semantic query after parsing, falling back to original question text"
                )
                semantic_query = question_text
            logger.debug(
                f"Stage 0: Query='{semantic_query}', Name='{concept_name_keyword}', Keywords={keywords}"
            )
            return semantic_query, concept_name_keyword, keywords

        except Exception as e:
            logger.warning(f"Stage 0: Text preprocessing failed: {e}, using original")
            return question_text, None, self._extract_simple_keywords(question_text)

    def _build_preprocessing_prompt(self, question_text: str) -> str:
        """Build prompt for question preprocessing"""
        return f"""You are a query analyzer. Extract key information from the question.

# Question
{question_text}

# Task
1. Create a semantic query (1 sentence) (the form of declarative sentence) that captures the question's intent
2. Extract the name of the concept if explicitly mentioned (e.g., "Luna", "Buddy").
3. Extract the category keywords related to the concept (e.g., "dog", "cat", "bowl", "toy", "bag").

# Rules
- Semantic query: Rephrase to be searchable (e.g., "What is Luna eating?" -> "food Luna is currently eating")
- Concept Name: specific proper noun referring to the subject. Return null if none.
- Keywords: Only concrete terms (e.g., "Luna", "cat", "bowl")
- Keep it simple and direct

# Output Format
```yaml
semantic_query: "your semantic query here"
concept_name_keyword: "extracted name or null"
category_keywords:
  - keyword1
  - keyword2
```

Generate output:"""

    def _parse_preprocessing_response(self, response: str) -> Tuple[str, Optional[str], List[str]]:
        """Parse YAML response from MLLM"""
        import re

        import yaml

        yaml_match = re.search(r"```yaml\s*(.*?)```", response, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
        else:
            yaml_content = response

        try:
            parsed = yaml.safe_load(yaml_content)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected YAML mapping, got {type(parsed).__name__}")
            semantic_query = parsed.get("semantic_query", "")
            concept_name_keyword = parsed.get("concept_name_keyword")
            category_keywords = parsed.get("category_keywords", [])

            if concept_name_keyword == "null" or concept_name_keyword is None:
                concept_name_keyword = None
            else:
                concept_name_keyword = str(concept_name_keyword).strip()

            if not semantic_query:
                raise ValueError("Empty semantic query")

            return semantic_query, concept_name_keyword, category_keywords
        except Exception as e:
            logger.warning(f"Stage 0: YAML parsing failed: {e}; raw={response[:200]!r}")
            logger.debug(f"Stage 0: YAML candidate={yaml_content.strip()[:400]!r}")
            return response.strip(), None, []

    def _extract_simple_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction fallback (no MLLM)"""
        words = text.split()
        keywords = [w.strip("?.!,") for w in words if w and (w[0].isupper() or len(w) > 5)]
        return keywords[:5]

    @staticmethod
    def check_correctness(predicted_id: Optional[str], gt_id: Optional[str]) -> bool:
        """Check if predicted concept ID matches ground truth (allowing for suffixes)"""
        if not predicted_id or not gt_id:
            return predicted_id == gt_id
        return predicted_id == gt_id or predicted_id.startswith(f"{gt_id}_")

    def identify(
        self, question_img: Optional[Image.Image], question_text: str, gt_concept_id: Optional[str] = None
    ) -> Tuple[Optional[str], bool]:
        """Main identification method combining stage 1 and stage 2"""
        if not self.concept_manager.has_concepts():
            logger.info("No existing concepts, returning NEW_CONCEPT")
            return None, False

        top_k_concepts, stage0_result, _ = self.stage1_retrieve(question_img, question_text)

        if not top_k_concepts:
            logger.info("Stage 1 returned no candidates, returning NEW_CONCEPT")
            if gt_concept_id:
                logger.info(f"Missed Ground Truth: {gt_concept_id} (No candidates found)")
            return None, False

        concept_id, is_existing = self.stage2_verify(question_img, question_text, top_k_concepts)

        # Store conversation in history (text only)
        self.conversation_history.append(question_text)

        if gt_concept_id:
            identified_concept = concept_id if is_existing else None

            if not self.check_correctness(identified_concept, gt_concept_id):
                logger.info(f"Misidentification detected: Predicted={identified_concept}, GT={gt_concept_id}")

                # Log GT scores
                gt_concept = next(
                    (c for c in self.concept_manager.memories if c.concept_id == gt_concept_id), None
                )
                if gt_concept:
                    scores = self._calculate_scores(gt_concept, stage0_result)
                    self._log_concept_scores(gt_concept_id, scores, prefix="[GT Misidentified] ")
                else:
                    logger.warning(f"GT Concept {gt_concept_id} not found in memory manager")

                # Log Top-1 scores (from retrieval)
                if top_k_concepts:
                    top1_info = top_k_concepts[0]
                    top1_id = top1_info["concept_id"]
                    top1_concept = next(
                        (c for c in self.concept_manager.memories if c.concept_id == top1_id), None
                    )
                    if top1_concept:
                        scores = self._calculate_scores(top1_concept, stage0_result)
                        self._log_concept_scores(top1_id, scores, prefix="[Top-1 Misidentified] ")

        return concept_id, is_existing


class OpenSetIdentifier_original:
    """Original TAME concept identification (no open-set capability)"""

    def __init__(
        self,
        retriever,
        concept_manager,
        mllm,
        top_k=3,
        dynamic_weight=0.4,
        static_weight=0.6,
        threshold=0.6,
    ):
        self.retriever = retriever
        self.concept_manager = concept_manager
        # mllm and weights are unused in original TAME identification
        self.threshold = threshold
        self._detector = None

    @property
    def detector(self):
        """Lazy-load detector on first access"""
        if self._detector is None:
            logger.info("Loading detector for Original TAME (lazy-loaded)")
            from core.utils.retrieval_utils import Detector

            self._detector = Detector()
        return self._detector

    @staticmethod
    def check_correctness(predicted_id: Optional[str], gt_id: Optional[str]) -> bool:
        """Check if predicted concept ID matches ground truth (allowing for suffixes)"""
        if not predicted_id or not gt_id:
            return predicted_id == gt_id
        return predicted_id == gt_id or predicted_id.startswith(f"{gt_id}_")

    def identify(
        self, question_img: Optional[Image.Image], question_text: str, gt_concept_id: Optional[str] = None
    ) -> Tuple[Optional[str], bool]:
        """Identify concept using original TAME method (Detector + Retriever) with improvements"""
        threshold = self.threshold

        concept_profiles = self.concept_manager.get_concept_retrieval_target()
        if not concept_profiles:
            logger.warning("No concept profiles available")
            return None, False

        # Prepare concept embeddings
        target_texts = [p[1] for p in concept_profiles]

        # Encode Concept Texts
        concept_text_embs = self.retriever.encode_passage_text(target_texts)
        concept_text_tensor = torch.tensor(concept_text_embs)

        # Encode Concept Images (handle None)
        embedding_dim = concept_text_embs.shape[1]
        concept_image_tensor = torch.zeros((len(concept_profiles), embedding_dim))
        concept_has_img = torch.zeros(len(concept_profiles))

        images_to_encode = []
        indices_to_fill = []
        for i, (img, _) in enumerate(concept_profiles):
            if img is not None:
                images_to_encode.append(img)
                indices_to_fill.append(i)
                concept_has_img[i] = 1.0

        if images_to_encode:
            img_embs = self.retriever.encode_passage_images(images_to_encode)
            concept_image_tensor[indices_to_fill] = torch.tensor(img_embs)

        final_scores = torch.zeros(len(concept_profiles))

        # 1. Image Query
        if question_img is not None:
            labels = ["animal", "person", "household item", "personal belonging"]
            detected_objects = self.detector.detect_and_crop(question_img, labels)
            if not detected_objects:
                detected_objects = [question_img]

            query_img_embs = self.retriever.encode_query_images(detected_objects)
            query_img_tensor = torch.tensor(query_img_embs)

            sim_img_img = torch.mm(query_img_tensor, concept_image_tensor.T)
            sim_img_text = torch.mm(query_img_tensor, concept_text_tensor.T)

            # Weighted sum (per object)
            score_with_img = 0.7 * sim_img_img + 0.3 * sim_img_text
            score_without_img = sim_img_text

            obj_scores = concept_has_img * score_with_img + (1.0 - concept_has_img) * score_without_img

            if obj_scores.shape[0] > 0:
                image_scores, _ = torch.max(obj_scores, dim=0)
                final_scores = torch.max(final_scores, image_scores)

        # 2. Text Query
        if question_text:
            query_txt_emb = self.retriever.encode_passage_text([question_text])[0]
            query_txt_tensor = torch.tensor(query_txt_emb).unsqueeze(0)

            sim_txt_img = torch.mm(query_txt_tensor, concept_image_tensor.T)
            sim_txt_text = torch.mm(query_txt_tensor, concept_text_tensor.T)

            score_with_img = 0.7 * sim_txt_img + 0.3 * sim_txt_text
            score_without_img = sim_txt_text

            txt_scores = concept_has_img * score_with_img + (1.0 - concept_has_img) * score_without_img
            txt_scores = txt_scores.flatten()

            final_scores = torch.max(final_scores, txt_scores)

        max_score, best_idx = torch.max(final_scores, dim=0)
        best_idx = best_idx.item()
        max_score = max_score.item()

        concept_id = self.concept_manager.get_concept_id(best_idx)

        if max_score < threshold:
            logger.info(
                f"Original TAME (improved): Score {max_score:.4f} < {threshold}. Identified as NEW_CONCEPT."
            )
            if gt_concept_id:
                logger.info(f"Missed Ground Truth: {gt_concept_id} (Score too low)")
            return None, False

        if gt_concept_id:
            if not self.check_correctness(concept_id, gt_concept_id):
                logger.info(f"Misidentification detected: Predicted={concept_id}, GT={gt_concept_id}")

        logger.info(f"Original TAME (improved): Identified {concept_id} with score {max_score:.4f}")
        return concept_id, True
