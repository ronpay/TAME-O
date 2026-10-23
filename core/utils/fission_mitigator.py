import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from colorama import Fore, Style
from loguru import logger
from PIL import Image


class ConceptFissionMitigator:
    """
    Detects and merges fissioned concepts that were incorrectly split during history reading.

    Uses a 3-step approach:
    1. PairwiseSimilarityRanking - Calculate similarity scores for all concept pairs
    2. MLLMFissionVerification - Use MLLM to verify if pairs represent the same entity
    3. ConceptMerging - Merge newer concepts into older ones
    """

    def __init__(self, concept_manager, retriever, mllm, similarity_top_k=3, tao=0.6):
        """
        Initialize the concept fission mitigator.

        Args:
            concept_manager: ConceptManager instance managing all concepts
            retriever: Retriever instance for encoding text and images
            mllm: MLLM instance for verification
            similarity_top_k: Number of top similar pairs L to verify (default: 3)
            tao: Threshold tau for filtering pairs by similarity score (default: 0.6)
        """
        self.concept_manager = concept_manager
        self.retriever = retriever
        self.mllm = mllm
        self.similarity_top_k = similarity_top_k
        self.tao = tao
        self.max_dynamic_size = 10

    def detect_and_merge_fissions(self) -> dict:
        """
        Main entry point: detect and merge fissioned concepts.

        Returns:
            Dictionary with merge statistics:
            {
                'merged_count': int,
                'failed_count': int,
                'merges': [{'target': str, 'source': str, 'similarity': float}, ...]
            }
        """
        logger.info("Starting concept fission detection and merging")

        # Step 1: Calculate similarities
        candidate_pairs = self.calculate_similarity_scores()
        logger.info(f"Step 1 complete: Found {len(candidate_pairs)} candidate pairs")

        if not candidate_pairs:
            logger.info("No candidate pairs found, skipping verification and merging")
            return {"merged_count": 0, "failed_count": 0, "merges": []}

        # Step 2: MLLM verification
        verified_pairs = self.verify_with_mllm(candidate_pairs)
        logger.info(f"Step 2 complete: Verified {len(verified_pairs)} fissioned pairs")

        if not verified_pairs:
            logger.info("No verified pairs, skipping merging")
            return {"merged_count": 0, "failed_count": 0, "merges": []}

        # Step 3: Merge concepts
        merge_results = self.merge_concepts(verified_pairs)
        logger.info(
            f"Step 3 complete: {merge_results['merged_count']} pairs merged, {merge_results['failed_count']} failed"
        )

        return merge_results

    def calculate_similarity_scores(self) -> List[Dict]:
        """
        Step 1: PairwiseSimilarityRanking
        Calculate similarity for all concept pairs and return top-k.

        Uses batch processing to encode all concepts once and compute similarities efficiently.

        Returns:
            List of top-k candidate pairs, each containing:
            {
                'concept_m': concept_id_m,
                'concept_n': concept_id_n,
                'similarity': combined_score,
                'text_sim': text_component,
                'visual_sim': visual_component
            }
        """
        concepts = self.concept_manager.memories
        n_concepts = len(concepts)

        if n_concepts < 2:
            logger.info(f"Only {n_concepts} concept(s) available, skipping similarity calculation")
            return []

        n_pairs = n_concepts * (n_concepts - 1) // 2
        logger.info(f"Calculating pairwise similarities for {n_concepts} concepts ({n_pairs} pairs)")

        # Batch encode all concepts in a single pass
        logger.info(f"Batch encoding {n_concepts} concepts...")
        encoded_data = self._batch_encode_all_concepts(concepts)

        # Generate all pair indices
        indices_m = []
        indices_n = []
        for i in range(n_concepts):
            for j in range(i + 1, n_concepts):
                indices_m.append(i)
                indices_n.append(j)

        # Batch similarity calculations
        logger.info(f"Computing {n_pairs} pair similarities...")
        text_sims = self._calculate_text_similarity_batch(
            encoded_data["text_embeddings"], indices_m, indices_n
        )

        visual_sims = self._calculate_visual_similarity_batch(
            encoded_data["portrait_embeddings"], encoded_data["portrait_masks"], indices_m, indices_n
        )

        # Combine scores as in the paper: S(c_m, c_n) = cos(text) + cos(visual)
        total_sims = text_sims + visual_sims

        # Build candidate pairs list
        candidate_pairs = []
        for idx in range(len(indices_m)):
            i, j = indices_m[idx], indices_n[idx]
            candidate_pairs.append(
                {
                    "concept_m": encoded_data["concept_ids"][i],
                    "concept_n": encoded_data["concept_ids"][j],
                    "similarity": total_sims[idx].item(),
                    "text_sim": text_sims[idx].item(),
                    "visual_sim": visual_sims[idx].item(),
                }
            )

        # Sort by similarity score (descending)
        candidate_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        # Filter by threshold (tao)
        if self.tao > 0:
            filtered_pairs = [pair for pair in candidate_pairs if pair["similarity"] >= self.tao]
            logger.info(
                f"Filtered {len(candidate_pairs)} pairs to {len(filtered_pairs)} pairs with threshold tao={self.tao}"
            )
            candidate_pairs = filtered_pairs

        # Return top-k
        top_pairs = candidate_pairs[: self.similarity_top_k]

        logger.debug(f"Top {len(top_pairs)} candidate pairs:")
        for pair in top_pairs:
            logger.debug(
                f"  {pair['concept_m']} <-> {pair['concept_n']}: {pair['similarity']:.3f} "
                f"(text={pair['text_sim']:.3f}, visual={pair['visual_sim']:.3f})"
            )

        return top_pairs

    def _calculate_pair_similarity(self, concept_m, concept_n) -> dict:
        """
        Calculate similarity between two concepts.

        NOTE: This method is kept for backward compatibility and validation.
        The main similarity calculation now uses batch methods in calculate_similarity_scores().

        Returns:
            Dictionary with similarity components:
            {'total': float, 'text': float, 'visual': float}
        """
        # Get memories
        static_m = concept_m.read_static_memory()
        static_n = concept_n.read_static_memory()
        dynamic_m = concept_m.get_dynamic_memory_with_timestamps()
        dynamic_n = concept_n.get_dynamic_memory_with_timestamps()

        # Combine all memories (text only)
        memories_m = static_m + [mem[0] for mem in dynamic_m]
        memories_n = static_n + [mem[0] for mem in dynamic_n]

        # Get portraits
        portrait_m = concept_m.read_portrait()
        portrait_n = concept_n.read_portrait()

        # Calculate text similarity component
        try:
            text_sim = self._calculate_text_similarity(memories_m, memories_n)
        except Exception as e:
            logger.warning(
                f"Text similarity failed for {concept_m.concept_id} vs {concept_n.concept_id}: {e}"
            )
            text_sim = 0.0

        # Calculate visual similarity component
        try:
            visual_sim = self._calculate_visual_similarity(portrait_m, portrait_n)
        except Exception as e:
            logger.warning(
                f"Visual similarity failed for {concept_m.concept_id} vs {concept_n.concept_id}: {e}"
            )
            visual_sim = 0.0

        # Combined score as in the paper: cos(text) + cos(visual)
        total_sim = text_sim + visual_sim

        return {"total": total_sim, "text": text_sim, "visual": visual_sim}

    def _calculate_text_similarity(self, memories_m: List[str], memories_n: List[str]) -> float:
        """
        Calculate text similarity using the formula:
        S_text = (1 / |M_m| * |M_n|) * sum(cos(φ(T_m^i), φ(T_n^j)))

        NOTE: This method is kept for backward compatibility and validation.
        The main similarity calculation now uses _calculate_text_similarity_batch().

        Args:
            memories_m: List of memory strings for concept m
            memories_n: List of memory strings for concept n

        Returns:
            Average pairwise cosine similarity
        """
        if not memories_m or not memories_n:
            logger.debug("One or both concepts have no memories, returning 0.0 similarity")
            return 0.0

        # Encode all memories
        emb_m = self.retriever.encode_passage_text(memories_m)
        emb_n = self.retriever.encode_passage_text(memories_n)

        # Convert to tensors
        emb_m_tensor = torch.tensor(emb_m)
        emb_n_tensor = torch.tensor(emb_n)

        # Normalize embeddings for cosine similarity
        emb_m_tensor = torch.nn.functional.normalize(emb_m_tensor, p=2, dim=1)
        emb_n_tensor = torch.nn.functional.normalize(emb_n_tensor, p=2, dim=1)

        # Compute all pairwise similarities: (|M_m| x |M_n|)
        # Using matrix multiplication: emb_m @ emb_n.T gives cosine similarities
        similarities = torch.mm(emb_m_tensor, emb_n_tensor.T)

        # Average over all pairs: (1 / |M_m| * |M_n|) * sum
        avg_similarity = similarities.mean().item()

        return avg_similarity

    def _calculate_visual_similarity(
        self, portrait_m: Optional[Image.Image], portrait_n: Optional[Image.Image]
    ) -> float:
        """
        Calculate visual similarity between portraits:
        S_visual = cos(φ(V_m), φ(V_n))

        NOTE: This method is kept for backward compatibility and validation.
        The main similarity calculation now uses _calculate_visual_similarity_batch().

        Args:
            portrait_m: Portrait image for concept m (or None)
            portrait_n: Portrait image for concept n (or None)

        Returns:
            Cosine similarity between portrait embeddings
        """
        if portrait_m is None or portrait_n is None:
            logger.debug("One or both portraits missing, skipping visual similarity")
            return 0.0

        # Encode portraits
        emb_m = self.retriever.encode_passage_images([portrait_m])[0]
        emb_n = self.retriever.encode_passage_images([portrait_n])[0]

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(emb_m).unsqueeze(0), torch.tensor(emb_n).unsqueeze(0), dim=1
        ).item()

        return similarity

    def _batch_encode_all_concepts(self, concepts) -> dict:
        """
        Batch encode all concepts' memories and portraits in a single pass.

        This method encodes all concept data upfront to avoid redundant encoding
        when computing pairwise similarities.

        Args:
            concepts: List of MemoryManager instances

        Returns:
            Dictionary containing:
            {
                'concept_ids': List[str] - concept IDs in order,
                'text_embeddings': List[torch.Tensor] - normalized embeddings per concept,
                'memory_texts': List[List[str]] - raw memory texts for Jaccard,
                'portrait_embeddings': torch.Tensor - shape (n_concepts, emb_dim),
                'portrait_masks': torch.BoolTensor - shape (n_concepts,) indicating valid portraits
            }
        """
        n_concepts = len(concepts)
        concept_ids = []
        text_embeddings = []
        memory_texts = []
        portraits_to_encode = []
        portrait_masks = []

        # Extract and encode text memories for each concept
        for concept in concepts:
            concept_ids.append(concept.concept_id)

            # Get memories (same as _calculate_pair_similarity)
            static_mem = concept.read_static_memory()
            dynamic_mem = concept.get_dynamic_memory_with_timestamps()
            memories = static_mem + [mem[0] for mem in dynamic_mem]

            memory_texts.append(memories)

            # Encode text memories
            if memories:
                emb = self.retriever.encode_passage_text(memories)
                emb_tensor = torch.tensor(emb)
                # Normalize for cosine similarity
                emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
                text_embeddings.append(emb_tensor)
            else:
                # Empty embeddings for concepts with no memories
                text_embeddings.append(torch.empty(0, 0))

            # Collect portraits
            portrait = concept.read_portrait()
            portraits_to_encode.append(portrait)
            portrait_masks.append(portrait is not None)

        # Batch encode all portraits at once
        valid_portraits = [p for p in portraits_to_encode if p is not None]
        if valid_portraits:
            portrait_embs = self.retriever.encode_passage_images(valid_portraits)
            portrait_embs_tensor = torch.tensor(portrait_embs)
            # Normalize for cosine similarity
            portrait_embs_tensor = torch.nn.functional.normalize(portrait_embs_tensor, p=2, dim=1)
        else:
            portrait_embs_tensor = torch.empty(0, 0)

        # Build full portrait embeddings array with zeros for missing portraits
        emb_dim = portrait_embs_tensor.shape[1] if portrait_embs_tensor.numel() > 0 else 512
        full_portrait_embs = torch.zeros(n_concepts, emb_dim)
        portrait_idx = 0
        for i, has_portrait in enumerate(portrait_masks):
            if has_portrait:
                full_portrait_embs[i] = portrait_embs_tensor[portrait_idx]
                portrait_idx += 1

        return {
            "concept_ids": concept_ids,
            "text_embeddings": text_embeddings,
            "memory_texts": memory_texts,
            "portrait_embeddings": full_portrait_embs,
            "portrait_masks": torch.tensor(portrait_masks),
        }

    def _calculate_visual_similarity_batch(
        self,
        portrait_embeddings: torch.Tensor,
        portrait_masks: torch.Tensor,
        indices_m: List[int],
        indices_n: List[int],
    ) -> torch.Tensor:
        """
        Calculate visual similarity for multiple concept pairs using pre-encoded portraits.

        Args:
            portrait_embeddings: Tensor of shape (n_concepts, emb_dim) with portrait embeddings
            portrait_masks: Boolean tensor of shape (n_concepts,) indicating valid portraits
            indices_m: List of indices for first concept in each pair
            indices_n: List of indices for second concept in each pair

        Returns:
            Tensor of shape (n_pairs,) containing cosine similarities (0.0 if portrait missing)
        """
        # Extract portraits for all pairs using advanced indexing
        portraits_m = portrait_embeddings[indices_m]  # Shape: (n_pairs, emb_dim)
        portraits_n = portrait_embeddings[indices_n]  # Shape: (n_pairs, emb_dim)

        # Batch cosine similarity (embeddings already normalized in _batch_encode_all_concepts)
        similarities = torch.nn.functional.cosine_similarity(portraits_m, portraits_n, dim=1)

        # Apply mask: set similarity to 0.0 if either portrait is missing
        mask = portrait_masks[indices_m] & portrait_masks[indices_n]
        similarities = torch.where(mask, similarities, torch.zeros_like(similarities))

        return similarities

    def _calculate_text_similarity_batch(
        self, text_embeddings: List[torch.Tensor], indices_m: List[int], indices_n: List[int]
    ) -> torch.Tensor:
        """
        Calculate text similarity for multiple concept pairs using pre-encoded embeddings.

        This method performs the same computation as _calculate_text_similarity() but
        operates on multiple pairs simultaneously by reusing pre-encoded embeddings.

        Args:
            text_embeddings: List of embedding tensors, one per concept.
                           Each tensor has shape (n_memories, embedding_dim).
            indices_m: List of indices for first concept in each pair
            indices_n: List of indices for second concept in each pair

        Returns:
            Tensor of shape (n_pairs,) containing average cosine similarities
        """
        similarities = []

        for i, j in zip(indices_m, indices_n):
            emb_m = text_embeddings[i]  # Already normalized
            emb_n = text_embeddings[j]  # Already normalized

            # Handle empty embeddings
            if emb_m.numel() == 0 or emb_n.numel() == 0:
                similarities.append(0.0)
                continue

            # Compute all pairwise similarities using matrix multiplication
            # Same formula as original: S_text = (1 / |M_m| * |M_n|) * sum(cos(φ(T_m^i), φ(T_n^j)))
            sim_matrix = torch.mm(emb_m, emb_n.T)
            avg_sim = sim_matrix.mean().item()
            similarities.append(avg_sim)

        return torch.tensor(similarities)

    def verify_with_mllm(self, candidate_pairs: List[Dict]) -> List[Dict]:
        """
        Step 2: MLLMFissionVerification
        Verify each candidate pair using MLLM.

        Args:
            candidate_pairs: List of candidate pairs from step 1

        Returns:
            List of verified fissioned pairs
        """
        verified_pairs = []

        logger.info(f"Verifying {len(candidate_pairs)} candidate pairs with MLLM")

        for idx, pair in enumerate(candidate_pairs, 1):
            logger.info(
                f"Verifying pair {idx}/{len(candidate_pairs)}: {pair['concept_m']} vs {pair['concept_n']}"
            )
            logger.debug(
                f"  Similarity: {pair['similarity']:.3f} "
                f"(Text: {pair['text_sim']:.3f}, Visual: {pair['visual_sim']:.3f})"
            )

            try:
                is_same = self._verify_pair_with_mllm(pair["concept_m"], pair["concept_n"])

                if is_same:
                    verified_pairs.append(pair)
                    logger.info(f"✓ Verified fission: {pair['concept_m']} ↔ {pair['concept_n']}")
                else:
                    logger.debug(f"✗ Not fissioned: {pair['concept_m']} ≠ {pair['concept_n']}")

            except Exception as e:
                logger.warning(f"MLLM verification failed for {pair}: {e}")

        return verified_pairs

    def _verify_pair_with_mllm(self, concept_m_id: str, concept_n_id: str) -> bool:
        """
        Verify if two concepts are fissioned using MLLM.

        Args:
            concept_m_id: ID of first concept
            concept_n_id: ID of second concept

        Returns:
            True if concepts represent the same entity (fissioned), False otherwise
        """
        concept_m = self._get_concept_by_id(concept_m_id)
        concept_n = self._get_concept_by_id(concept_n_id)

        if not concept_m or not concept_n:
            logger.warning(f"Could not find concepts {concept_m_id} or {concept_n_id}")
            return False

        portrait_m = concept_m.read_portrait()
        portrait_n = concept_n.read_portrait()

        prompt = self._build_verification_prompt(
            concept_m, concept_n, portrait_m is not None, portrait_n is not None
        )

        # Use multi-image chat (similar to Stage 2 in OpenSetIdentifier)
        try:
            if portrait_m and portrait_n:
                images = [portrait_m, portrait_n]
                response = self.mllm.chat_multi_img(prompt, images, max_tokens=512)
            elif portrait_m or portrait_n:
                # Only one portrait available
                portrait = portrait_m if portrait_m else portrait_n
                response = self.mllm.chat_img(prompt, portrait, max_tokens=512)
            else:
                # No portraits, text-only
                response = self.mllm.chat_text(prompt, max_tokens=512)

            logger.debug(f"MLLM verification for {concept_m_id} vs {concept_n_id}: {response}")

            # Parse response
            response_upper = response.strip().upper()

            # Check for verdict
            if "VERDICT: SAME" in response_upper:
                return True
            elif "VERDICT: DIFFERENT" in response_upper:
                return False

            # Fallback for old format or short answers. Check negative forms first.
            if "NOT SAME" in response_upper or "NOT THE SAME" in response_upper:
                return False
            if "DIFFERENT" in response_upper:
                return False
            if "SAME" in response_upper:
                return True
            else:
                # Ambiguous response - log and default to False (conservative)
                logger.warning(f"Ambiguous MLLM response: {response}")
                return False

        except Exception as e:
            logger.error(f"MLLM verification error for {concept_m_id} vs {concept_n_id}: {e}")
            return False

    def _build_verification_prompt(
        self, concept_m, concept_n, has_portrait_m: bool, has_portrait_n: bool
    ) -> str:
        """
        Build the verification prompt for MLLM.

        Args:
            concept_m: First MemoryManager instance
            concept_n: Second MemoryManager instance
            has_portrait_m: Whether concept m has a portrait
            has_portrait_n: Whether concept n has a portrait

        Returns:
            Formatted prompt string
        """
        static_m = concept_m.read_static_memory()
        static_n = concept_n.read_static_memory()
        dynamic_m = concept_m.get_dynamic_memory_with_timestamps()
        dynamic_n = concept_n.get_dynamic_memory_with_timestamps()

        # Determine image placeholders
        img_idx = 1
        if has_portrait_m:
            portrait_m_str = f"[IMAGE {img_idx}]"
            img_idx += 1
        else:
            portrait_m_str = "(No portrait available)"

        if has_portrait_n:
            portrait_n_str = f"[IMAGE {img_idx}]"
            img_idx += 1
        else:
            portrait_n_str = "(No portrait available)"

        prompt = f"""You are a concept similarity expert. Your task is to determine if two concepts represent the SAME entity that was incorrectly split (fissioned).

# CONCEPT A

## Portrait
{portrait_m_str}

## Static Memory (Permanent Facts):
{self._format_memory_list(static_m[:5])}

## Dynamic Memory (Recent Events):
{self._format_timestamped_memory(dynamic_m[-5:])}

---

# CONCEPT B

## Portrait
{portrait_n_str}

## Static Memory (Permanent Facts):
{self._format_memory_list(static_n[:5])}

## Dynamic Memory (Recent Events):
{self._format_timestamped_memory(dynamic_n[-5:])}

---

# INSTRUCTIONS

Compare the two concepts carefully to determine if they are the SAME entity.

**CRITICAL RULES FOR "SAME" VERDICT**:
1. Conclude "SAME" only when there is clear identity evidence: the portraits show the same individual/object, or the memories share a distinctive name/model/owner-specific identifier.
2. Generic category overlap is not enough. Two concepts can both be cats, headphones, siblings, classmates, or similar-looking people and still be different entities.
3. If both concepts have specific names, brands, models, colors, breeds, or personal details and those details differ, conclude "DIFFERENT".
4. If one concept uses a generic reference ("he", "the cat", "the headset") and the other is specific, conclude "SAME" only when the visual evidence or distinctive memories strongly connect them.
5. If one concept is an unnamed fragment with only a few memories, merge it into the richer concept when the fragment adds compatible later details and there is no contradictory identity evidence.

**GUIDELINES**:
- Assume these are memories accumulated over time. Small variations or new details are expected.
- Do not merge based on broad class, scene, pose, or activity alone.
- Do not require both concepts to repeat the same name. Later memory fragments often omit the name.
- If the evidence is weak, ambiguous, or only loosely compatible, favor "DIFFERENT".

# OUTPUT FORMAT

1. First, provide a brief reasoning (1-2 sentences) explaining why they are compatible or contradictory.
2. Then, provide the final verdict on a new line: "VERDICT: SAME" or "VERDICT: DIFFERENT".

Generate your response:"""

        return prompt

    def _format_memory_list(self, memories: List[str]) -> str:
        """Format a list of memories for the prompt."""
        if not memories:
            return "  (No memories)"
        return "\n".join([f"  - {mem}" for mem in memories])

    def _format_timestamped_memory(self, memory_list: List[Tuple]) -> str:
        """Format timestamped memory for the prompt."""
        if not memory_list:
            return "  (No memories)"
        return "\n".join([f"  - {mem[0]}" for mem in memory_list])

    def merge_concepts(self, verified_pairs: List[Dict]) -> dict:
        """
        Step 3: ConceptMerging
        Merge verified fissioned concept pairs.

        Strategy:
        - Keep the richest concept memory as the target.
        - Merge the other concepts into that target.
        - Delete merged source directories.

        Args:
            verified_pairs: List of verified fissioned pairs

        Returns:
            Dictionary with merge statistics
        """
        merge_results = {"merged_count": 0, "failed_count": 0, "merges": []}

        logger.info(f"Starting merge of {len(verified_pairs)} verified pairs")

        for component_ids, similarity_by_source in self._merge_components(verified_pairs):
            target_id = self._select_component_target(component_ids)
            ordered_source_ids = [
                memory.concept_id
                for memory in self.concept_manager.memories
                if memory.concept_id in component_ids and memory.concept_id != target_id
            ]
            for source_id in ordered_source_ids:
                if source_id == target_id:
                    continue
                if self._get_concept_by_id(source_id) is None:
                    continue

                logger.info(f"Merging {source_id} → {target_id}")
                success = self._merge_single_pair(target_id, source_id)

                if success:
                    merge_results["merged_count"] += 1
                    merge_results["merges"].append(
                        {
                            "target": target_id,
                            "source": source_id,
                            "similarity": similarity_by_source.get(source_id, 0.0),
                        }
                    )
                else:
                    merge_results["failed_count"] += 1

        # Reload ConceptManager to reflect changes
        if merge_results["merged_count"] > 0:
            self.concept_manager.reload_all()
            logger.info("ConceptManager reloaded after merging")

        return merge_results

    def _merge_components(self, verified_pairs: List[Dict]) -> List[Tuple[set[str], Dict[str, float]]]:
        """Convert pair decisions into connected components to avoid chain-merge failures."""
        if not verified_pairs:
            return []

        parent = {}

        def find(item):
            parent.setdefault(item, item)
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]

        def union(left, right):
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        similarity_by_source = {}
        for pair in verified_pairs:
            left = pair["concept_m"]
            right = pair["concept_n"]
            union(left, right)
            similarity = pair.get("similarity", 0.0)
            similarity_by_source[left] = max(similarity_by_source.get(left, 0.0), similarity)
            similarity_by_source[right] = max(similarity_by_source.get(right, 0.0), similarity)

        components = {}
        for concept_id in parent:
            components.setdefault(find(concept_id), set()).add(concept_id)

        return [(component, similarity_by_source) for component in components.values()]

    def _select_component_target(self, component_ids: set[str]) -> str:
        """Choose a merge target from memory content, not from concept identifier text."""
        ordered_ids = [
            memory.concept_id
            for memory in self.concept_manager.memories
            if memory.concept_id in component_ids
        ]
        if not ordered_ids:
            return next(iter(component_ids))
        return max(ordered_ids, key=self._concept_content_key)

    def _concept_content_key(self, concept_id: str) -> Tuple[int, int, int]:
        """Rank concepts by available evidence without reading identifier semantics."""
        concept = self._get_concept_by_id(concept_id)
        if concept is None:
            return (0, 0, 0)
        has_portrait = int(concept.portrait_path.exists())
        static_count = len(concept.read_static_memory())
        dynamic_count = len(concept.read_dynamic_memory())
        return (has_portrait, static_count + dynamic_count, static_count)

    def _merge_single_pair(self, target_id: str, source_id: str) -> bool:
        """
        Merge source concept into target concept.

        Steps:
        1. Load both concepts' memories
        2. Merge static memory (deduplicate)
        3. Merge dynamic memory (keep timestamps, sort)
        4. Keep target's portrait
        5. Write merged memories to target
        6. Delete source directory
        7. Update ConceptManager

        Args:
            target_id: Target concept ID (older, to keep)
            source_id: Source concept ID (newer, to merge and delete)

        Returns:
            True if merge successful, False otherwise
        """
        try:
            target = self._get_concept_by_id(target_id)
            source = self._get_concept_by_id(source_id)

            if not target or not source:
                logger.error(f"Could not find concepts {target_id} or {source_id}")
                return False

            # 1. Load memories
            static_target = target.read_static_memory()
            static_source = source.read_static_memory()
            dynamic_target = target.get_dynamic_memory_with_timestamps()
            dynamic_source = source.get_dynamic_memory_with_timestamps()

            # 2. Merge static memory (deduplicate)
            merged_static = list(set(static_target + static_source))

            # 3. Merge dynamic memory (combine and sort by timestamp)
            # Offset source timestamps to place them after target memories
            max_target_ts = -1
            if dynamic_target:
                max_target_ts = max((t for _, t in dynamic_target), default=-1)

            # Re-timestamp source memories
            offset_dynamic_source = []
            for i, (mem, _) in enumerate(dynamic_source):
                # Use i to maintain relative order within source, offset by target's end
                new_ts = max_target_ts + 1 + i
                offset_dynamic_source.append((mem, new_ts))

            merged_dynamic = dynamic_target + offset_dynamic_source
            merged_dynamic.sort(key=lambda x: x[1])  # Sort by timestamp

            # Apply FIFO if needed
            if len(merged_dynamic) > self.max_dynamic_size:
                merged_dynamic = merged_dynamic[-self.max_dynamic_size :]

            # Extract just the memory strings (without timestamps for saving)
            merged_dynamic_list = [mem[0] for mem in merged_dynamic]

            # 4. Write merged memories to target
            with open(target.static_memory_path, "w", encoding="utf-8") as f:
                yaml.dump(merged_static, f, allow_unicode=True)

            with open(target.dynamic_memory_path, "w", encoding="utf-8") as f:
                yaml.dump(merged_dynamic_list, f, allow_unicode=True)

            # Update target's in-memory state
            target.static_memory = merged_static
            target.dynamic_memory = merged_dynamic_list
            target.dynamic_memory_with_timestamps = merged_dynamic
            target_metadata = target.read_metadata()
            target_metadata["memory_write_count"] = (
                target.get_memory_write_count() + source.get_memory_write_count()
            )
            target.write_metadata(target_metadata)

            # 5. Delete source directory
            source_dir = source.static_memory_path.parent
            if source_dir.exists():
                try:
                    shutil.rmtree(source_dir)
                    logger.info(f"Deleted source concept directory: {source_dir}")
                except OSError as e:
                    logger.error(f"Failed to delete source directory {source_dir}: {e}")
                    # Attempt to force delete or handle specific errors if needed
                    # For now, just logging the specific error is helpful
            else:
                logger.warning(f"Source directory {source_dir} already gone")

            # 6. Update ConceptManager (remove source from memories list)
            self.concept_manager.memories = [
                m for m in self.concept_manager.memories if m.concept_id != source_id
            ]

            logger.info(f"Successfully merged {source_id} → {target_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to merge {source_id} → {target_id}: {e}")
            return False

    def _get_concept_by_id(self, concept_id: str):
        """Get MemoryManager instance by concept ID."""
        for concept in self.concept_manager.memories:
            if concept.concept_id == concept_id:
                return concept
        return None
