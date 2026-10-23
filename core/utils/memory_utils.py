import re
import shutil
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from PIL import Image


class MemoryManager:
    def __init__(self, concept_id: str, model_name: str = "default"):
        self.concept_id = concept_id
        self.model_name = model_name
        base_path = Path("memory") / model_name / concept_id
        self.static_memory_path = base_path / "static.yaml"
        self.dynamic_memory_path = base_path / "dynamic.yaml"
        self.metadata_path = base_path / "metadata.yaml"
        self.portrait_path = base_path / "portrait.png"
        # Lazy creation: Do not create directories here. 
        # They will be created when data is actually written.
        self.static_memory = self.read_static_memory()
        self.dynamic_memory = self.read_dynamic_memory()

        # TAME-O extension: Track timestamps for dynamic memory
        self.dynamic_memory_with_timestamps = []
        self._init_timestamps()

    def remove_concept_directory(self):
        """Remove the entire concept directory"""
        concept_dir = self.static_memory_path.parent
        if concept_dir.exists() and concept_dir.is_dir():
            try:
                shutil.rmtree(concept_dir)
                logger.info(f"Removed concept directory: {concept_dir}")
            except Exception as e:
                logger.error(f"Failed to remove concept directory {concept_dir}: {e}")

    def _init_timestamps(self):
        """Initialize timestamps for existing dynamic memory items"""
        self.dynamic_memory_with_timestamps = [
            (memory, idx) for idx, memory in enumerate(self.dynamic_memory)
        ]

    def read_static_memory(self) -> list:
        if not self.static_memory_path.exists():
            return []
        with open(self.static_memory_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    def read_dynamic_memory(self) -> list:
        if not self.dynamic_memory_path.exists():
            return []
        with open(self.dynamic_memory_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    def read_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def write_metadata(self, metadata: dict):
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, allow_unicode=True)

    def get_memory_write_count(self) -> int:
        metadata = self.read_metadata()
        if "memory_write_count" in metadata:
            return int(metadata["memory_write_count"])

        concept_dir = self.static_memory_path.parent
        if concept_dir.exists():
            return 1
        return 0

    def increment_memory_write_count(self, amount: int = 1):
        metadata = self.read_metadata()
        metadata["memory_write_count"] = self.get_memory_write_count() + amount
        self.write_metadata(metadata)

    def read_portrait(self) -> Optional[Image.Image]:
        if not self.portrait_path.exists():
            return None
        return Image.open(self.portrait_path)

    def update_static_memory(
        self, memory: str, op: str, target_id: int = 0
    ):
        if not memory and op == "add":
            logger.warning("Cannot add empty memory")
            return

        static_memory = self.static_memory.copy()
        to_added = []

        if op == "add":
            cleaned_memory = memory.replace('\n', ' ') if memory else memory
            to_added.append(cleaned_memory)
            logger.debug(f"Adding to static memory: '{cleaned_memory}'")
        elif op == "remove":
            if 0 <= target_id < len(static_memory):
                item_to_remove = static_memory[target_id]
                static_memory[target_id] = "DELETE"
                logger.info(f"Removing from static memory at position {target_id + 1} (1-indexed): '{item_to_remove}'")
            else:
                logger.error(
                    f"Cannot remove from static memory: index {target_id + 1} (1-indexed) out of range. "
                    f"Valid range: [1, {len(static_memory)}]. Current static memory has {len(static_memory)} items."
                )
                return
        elif op == "modify":
            if 0 <= target_id < len(static_memory):
                old_value = static_memory[target_id]
                cleaned_memory = memory.replace('\n', ' ') if memory else memory
                static_memory[target_id] = cleaned_memory
                logger.info(f"Modifying static memory at position {target_id + 1} (1-indexed): '{old_value}' -> '{cleaned_memory}'")
            else:
                logger.error(
                    f"Cannot modify static memory: index {target_id + 1} (1-indexed) out of range. "
                    f"Valid range: [1, {len(static_memory)}]. Current static memory has {len(static_memory)} items."
                )
                return

        static_memory = [m for m in static_memory if m != "DELETE"]
        static_memory.extend(to_added)

        seen = set()
        static_memory = [x for x in static_memory if not (x in seen or seen.add(x))]

        self.static_memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.static_memory_path, "w", encoding="utf-8") as f:
            yaml.dump(static_memory, f, allow_unicode=True)
        self.static_memory = static_memory

    def update_dynamic_memory(
        self, memory: str, op: str, target_id: int = 0
    ):
        if not memory and op == "add":
            logger.warning("Cannot add empty memory")
            return

        dynamic_memory = self.dynamic_memory.copy()
        to_added = []

        if op == "add":
            cleaned_memory = memory.replace('\n', ' ') if memory else memory
            to_added.append(cleaned_memory)

            # TAME-O extension: Track timestamp for new memory
            new_timestamp = len(self.dynamic_memory_with_timestamps)
            self.dynamic_memory_with_timestamps.append((cleaned_memory, new_timestamp))

            logger.debug(f"Adding to dynamic memory: '{cleaned_memory}' with timestamp {new_timestamp}")
        elif op == "remove":
            if 0 <= target_id < len(dynamic_memory):
                item_to_remove = dynamic_memory[target_id]
                dynamic_memory[target_id] = "DELETE"

                # TAME-O extension: Remove from timestamped list
                if target_id < len(self.dynamic_memory_with_timestamps):
                    removed_item = self.dynamic_memory_with_timestamps[target_id]
                    self.dynamic_memory_with_timestamps[target_id] = ("DELETE", -1)

                logger.info(f"Removing from dynamic memory at position {target_id + 1} (1-indexed): '{item_to_remove}'")
            else:
                logger.error(
                    f"Cannot remove from dynamic memory: index {target_id + 1} (1-indexed) out of range. "
                    f"Valid range: [1, {len(dynamic_memory)}]. Current dynamic memory has {len(dynamic_memory)} items."
                )
                return
        elif op == "modify":
            if 0 <= target_id < len(dynamic_memory):
                old_value = dynamic_memory[target_id]
                cleaned_memory = memory.replace('\n', ' ') if memory else memory
                dynamic_memory[target_id] = cleaned_memory

                # TAME-O extension: Update in timestamped list (keep original timestamp)
                if target_id < len(self.dynamic_memory_with_timestamps):
                    original_timestamp = self.dynamic_memory_with_timestamps[target_id][1]
                    self.dynamic_memory_with_timestamps[target_id] = (cleaned_memory, original_timestamp)

                logger.info(f"Modifying dynamic memory at position {target_id + 1} (1-indexed): '{old_value}' -> '{cleaned_memory}'")
            else:
                logger.error(
                    f"Cannot modify dynamic memory: index {target_id + 1} (1-indexed) out of range. "
                    f"Valid range: [1, {len(dynamic_memory)}]. Current dynamic memory has {len(dynamic_memory)} items."
                )
                return

        dynamic_memory = [m for m in dynamic_memory if m != "DELETE"]
        dynamic_memory.extend(to_added)

        # TAME-O extension: Clean up timestamped list
        self.dynamic_memory_with_timestamps = [
            (m, t) for m, t in self.dynamic_memory_with_timestamps if m != "DELETE"
        ]

        seen = set()
        dynamic_memory = [x for x in dynamic_memory if not (x in seen or seen.add(x))]

        self.dynamic_memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dynamic_memory_path, "w", encoding="utf-8") as f:
            yaml.dump(dynamic_memory, f, allow_unicode=True)
        self.dynamic_memory = dynamic_memory

    def get_dynamic_memory_with_timestamps(self) -> list:
        """TAME-O extension: Get dynamic memory with timestamps for temporal weighting"""
        return self.dynamic_memory_with_timestamps

    def clean_dynamic_memory(self):
        if self.dynamic_memory_path.exists():
            self.dynamic_memory_path.unlink()
            self.dynamic_memory = []
            self.dynamic_memory_with_timestamps = []

    def save_portrait(self, portrait: Image.Image):
        if portrait is None:
            logger.warning("Cannot save None portrait")
            return

        self.portrait_path.parent.mkdir(parents=True, exist_ok=True)
        portrait.save(self.portrait_path)

    def apply_fifo_to_dynamic_memory(self, max_size: int = 10):
        """Apply FIFO rule to dynamic memory to limit its size"""
        dynamic_memory = self.read_dynamic_memory()

        if len(dynamic_memory) > max_size:
            dynamic_memory = dynamic_memory[-max_size:]
            logger.info(f"Applied FIFO rule: dynamic memory trimmed to {max_size} items")

            # TAME-O extension: Also trim timestamped list
            if len(self.dynamic_memory_with_timestamps) > max_size:
                self.dynamic_memory_with_timestamps = self.dynamic_memory_with_timestamps[-max_size:]

            self.dynamic_memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dynamic_memory_path, "w", encoding="utf-8") as f:
                yaml.dump(dynamic_memory, f, allow_unicode=True)
            self.dynamic_memory = dynamic_memory

    def parse_update_ops(self, response: str, memory_type: str) -> dict:
        """
        Parse the response (yaml format), and return a dict of update ops
        Then call update_static_memory or update_dynamic_memory to perform the update

        Expected format:
        op: add/remove/modify
        target_id: index for remove/modify operations
        memory: memory content
        Note: Both static and dynamic memory now use the same simple string format
        """
        def _clean_memory_value(value):
            if isinstance(value, str):
                return value.replace("\n", " ")
            return value

        def _ensure_dynamic_timestamps_aligned():
            if len(self.dynamic_memory_with_timestamps) != len(self.dynamic_memory):
                self.dynamic_memory_with_timestamps = [(m, idx) for idx, m in enumerate(self.dynamic_memory)]

        def _apply_ops_batch(ops: list[dict]):
            """
            Apply ops against a single pre-op snapshot to prevent target_id index shifting.

            The LLM generates `target_id` values based on the numbered memory shown in the prompt.
            If we apply ops sequentially and compact the list after each remove, later `target_id`s can
            become invalid (e.g., removing 1 then 2 in a 2-item list). Batch application marks removals
            first and compacts once at the end, so all target_id values refer to the same snapshot.
            """
            if memory_type == "static":
                working_memory = self.static_memory.copy()
                working_timestamps = None
                memory_path = self.static_memory_path
            else:
                _ensure_dynamic_timestamps_aligned()
                working_memory = self.dynamic_memory.copy()
                working_timestamps = self.dynamic_memory_with_timestamps.copy()
                memory_path = self.dynamic_memory_path

            snapshot_len = len(working_memory)
            normalized_ops: list[dict] = []

            for update_op in ops:
                if not update_op or not isinstance(update_op, dict):
                    continue

                op = (update_op.get("op", "add") or "add").strip().lower()
                memory = _clean_memory_value(update_op.get("memory", None))
                target_id = update_op.get("target_id", None)

                if isinstance(target_id, list):
                    target_ids = target_id
                else:
                    target_ids = [target_id]

                for one_target_id in target_ids:
                    normalized_ops.append({"op": op, "memory": memory, "target_id": one_target_id})

            modify_ops: list[tuple[int, object]] = []
            remove_target_idxs: list[int] = []
            add_memories: list[object] = []

            for op_item in normalized_ops:
                op = op_item["op"]
                memory = op_item["memory"]
                target_id = op_item["target_id"]

                if op in ["remove", "modify"] and target_id is None:
                    logger.error(f"Operation '{op}' requires target_id but none was provided. Skipping operation.")
                    continue

                if op == "add":
                    if not memory:
                        logger.warning("Cannot add empty memory")
                        continue
                    add_memories.append(memory)
                    continue

                try:
                    target_id_int = int(target_id)
                except (ValueError, TypeError):
                    logger.error(f"Invalid target_id type: {type(target_id)}, value: {target_id}. Skipping operation.")
                    continue

                if target_id_int <= 0:
                    logger.error(f"Invalid target_id: {target_id_int}. Must be >= 1 (1-indexed). Skipping operation.")
                    continue

                target_idx = target_id_int - 1
                if not (0 <= target_idx < snapshot_len):
                    logger.error(
                        f"target_id {target_id_int} (1-indexed) out of range. "
                        f"Valid range: [1, {snapshot_len}]. Current {memory_type} memory has {snapshot_len} items. Skipping operation."
                    )
                    continue

                if op == "modify":
                    modify_ops.append((target_idx, memory))
                elif op == "remove":
                    remove_target_idxs.append(target_idx)

            for target_idx, memory in modify_ops:
                old_value = working_memory[target_idx]
                working_memory[target_idx] = memory
                if working_timestamps is not None and target_idx < len(working_timestamps):
                    original_timestamp = working_timestamps[target_idx][1]
                    working_timestamps[target_idx] = (memory, original_timestamp)
                logger.info(
                    f"Modifying {memory_type} memory at position {target_idx + 1} (1-indexed): '{old_value}' -> '{memory}'"
                )

            for target_idx in sorted(set(remove_target_idxs), reverse=True):
                item_to_remove = working_memory[target_idx]
                working_memory[target_idx] = "DELETE"
                if working_timestamps is not None and target_idx < len(working_timestamps):
                    working_timestamps[target_idx] = ("DELETE", -1)
                logger.info(
                    f"Removing from {memory_type} memory at position {target_idx + 1} (1-indexed): '{item_to_remove}'"
                )

            filtered_memory = []
            filtered_timestamps = [] if working_timestamps is not None else None
            for idx, mem_item in enumerate(working_memory):
                if mem_item == "DELETE":
                    continue
                filtered_memory.append(mem_item)
                if filtered_timestamps is not None:
                    filtered_timestamps.append(working_timestamps[idx])

            for memory in add_memories:
                filtered_memory.append(memory)
                if filtered_timestamps is not None:
                    new_timestamp = len(filtered_timestamps)
                    filtered_timestamps.append((memory, new_timestamp))
                    logger.debug(f"Adding to dynamic memory: '{memory}' with timestamp {new_timestamp}")

            seen = set()
            deduped_memory = []
            deduped_timestamps = [] if filtered_timestamps is not None else None
            for idx, mem_item in enumerate(filtered_memory):
                if mem_item in seen:
                    continue
                seen.add(mem_item)
                deduped_memory.append(mem_item)
                if deduped_timestamps is not None:
                    deduped_timestamps.append(filtered_timestamps[idx])

            memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(memory_path, "w", encoding="utf-8") as f:
                yaml.dump(deduped_memory, f, allow_unicode=True)

            if memory_type == "static":
                self.static_memory = deduped_memory
            else:
                self.dynamic_memory = deduped_memory
                self.dynamic_memory_with_timestamps = deduped_timestamps or []

        m = re.search(r"```yaml\s*(.*?)```", response, re.S)
        response = m.group(1) if m else response
        try:
            parsed_response = yaml.safe_load(response)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}

        if isinstance(parsed_response, dict):
            parsed_response = [parsed_response]
        elif not isinstance(parsed_response, list):
            return {}

        _apply_ops_batch(parsed_response)

        return parsed_response


class ConceptManager:
    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.memory_path = Path("memory") / model_name
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.memories = []
        self.retrieval_target = []

    def read_memories(self):
        """Initialize memories from existing concept directories"""
        self.memories.clear()
        for memory_path in self.memory_path.iterdir():
            if memory_path.is_dir():
                concept_id = memory_path.name
                memory_manager = MemoryManager(concept_id, self.model_name)
                self.memories.append(memory_manager)

    def has_concepts(self) -> bool:
        """TAME-O extension: Check if any concepts exist"""
        return len(self.memories) > 0

    def refresh(self, concept_id: str):
        """Refresh a single concept's memory from disk"""
        for i, memory in enumerate(self.memories):
            if memory.concept_id == concept_id:
                logger.debug(f"Refreshing concept {concept_id} from disk")
                self.memories[i] = MemoryManager(concept_id, self.model_name)
                return
        logger.debug(f"Adding new concept {concept_id} to ConceptManager")
        self.memories.append(MemoryManager(concept_id, self.model_name))

    def reload_all(self):
        """Reload all concepts from disk"""
        logger.debug("Reloading all concepts from disk")
        self.read_memories()

    def get_concept_retrieval_target(self) -> list:
        """Return the retrieval targets"""
        self.retrieval_target.clear()

        for memory in self.memories:
            portrait = memory.read_portrait()
            static_memory = memory.read_static_memory()
            visual_prompt = ""

            for m in static_memory:
                if isinstance(m, str) and "visual" in m.lower():
                    visual_prompt += f"{m}\n"

            if not visual_prompt:
                visual_prompt = "NONE"

            self.retrieval_target.append((portrait, visual_prompt))

        return self.retrieval_target

    def get_concept_id(self, concept_idx: int) -> Optional[str]:
        """Get concept ID by index"""
        if not (0 <= concept_idx < len(self.memories)):
            logger.error(f"Invalid concept index: {concept_idx}")
            return None
        return self.memories[concept_idx].concept_id

    def get_concept_memory(self, concept_id: str) -> Optional[MemoryManager]:
        target_memory = None
        for memory in self.memories:
            if memory.concept_id == concept_id:
                print(memory.concept_id)
                target_memory = memory
                break
        target_static_memory = target_memory.read_static_memory()
        target_dynamic_memory = target_memory.read_dynamic_memory()
        return target_static_memory, target_dynamic_memory
