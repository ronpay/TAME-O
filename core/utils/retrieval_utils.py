import hashlib
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch import Tensor
from torchvision import ops
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class EmbeddingCache:
    """TAME-O extension: Cache embeddings for faster retrieval"""

    def __init__(self, cache_dir="cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def _compute_hash(self, content):
        """Compute hash for memory content"""
        if isinstance(content, Image.Image):
            buf = BytesIO()
            content.save(buf, format="PNG")
            return hashlib.md5(buf.getvalue()).hexdigest()
        else:
            return hashlib.md5(str(content).encode()).hexdigest()

    def _get_cache_key(self, concept_id, memory_type, content_hash):
        """Generate cache key based on content"""
        return f"{concept_id}_{memory_type}_{content_hash}"

    def get_embedding(self, concept_id, memory_type, content):
        """Get cached embedding or return None"""
        content_hash = self._compute_hash(content)
        cache_key = self._get_cache_key(concept_id, memory_type, content_hash)

        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                self.memory_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, removing corrupted cache")
                cache_file.unlink(missing_ok=True)
                return None

        return None

    def save_embedding(self, concept_id, memory_type, content, embedding):
        """Save embedding to cache"""
        content_hash = self._compute_hash(content)
        cache_key = self._get_cache_key(concept_id, memory_type, content_hash)

        self.memory_cache[cache_key] = embedding

        cache_file = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_file, embedding)

    def invalidate_concept(self, concept_id):
        """Invalidate all cache entries for a concept"""
        keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{concept_id}_")]
        for key in keys_to_remove:
            del self.memory_cache[key]
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                cache_file.unlink()


class Detector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def detect_and_crop(self, image: Image.Image, text_labels: list[str]) -> list[Image.Image]:
        text_prompt = " . ".join(text_labels) + " ."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        if len(boxes) == 0:
            return []

        priority_map = {label: 1 for label in labels}
        if "object" in priority_map:
            priority_map["object"] = 3
        if "pet" in priority_map:
            priority_map["pet"] = 2
        if "person" in priority_map:
            priority_map["person"] = 2

        priorities = torch.tensor([priority_map.get(l, 1) for l in labels], device=self.device)

        keep_indices = ops.nms(boxes, scores, iou_threshold=0.5)

        final_boxes = []
        final_indices = []

        sorted_indices = sorted(keep_indices.tolist(), key=lambda i: (priorities[i], -scores[i]))

        accepted_boxes = []

        for idx in sorted_indices:
            current_box = boxes[idx]
            is_redundant = False

            if len(accepted_boxes) > 0:
                accepted_tensor = torch.stack(accepted_boxes)
                iou = ops.box_iou(current_box.unsqueeze(0), accepted_tensor)[0]

                if (iou > 0.6).any():
                    is_redundant = True

            if not is_redundant:
                accepted_boxes.append(current_box)
                final_indices.append(idx)

        cropped_images = []
        for idx in final_indices:
            box = boxes[idx].tolist()
            box = [max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3])]
            cropped_image = image.crop(box)
            cropped_images.append(cropped_image)

        return cropped_images


class Retriever:
    QWEN_DEFAULT_BATCH_SIZE = 16
    # Hugging Face revision of Qwen/Qwen3-VL-Embedding-2B used in the paper experiments
    QWEN_EMBEDDING_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"

    def __init__(self, model_id: str = None, use_cache: bool = True, decay_rate: float = 1.0):
        import sys

        from huggingface_hub import snapshot_download

        self.device = "cuda"
        self.use_cache = use_cache
        self.decay_rate = decay_rate
        self.cache = EmbeddingCache() if use_cache else None

        # The model repository ships its embedder implementation under scripts/
        model_path = Path(
            snapshot_download(model_id or "Qwen/Qwen3-VL-Embedding-2B", revision=self.QWEN_EMBEDDING_REVISION)
        )
        qwen_utils_path = model_path / "scripts"
        if str(qwen_utils_path) not in sys.path:
            sys.path.insert(0, str(qwen_utils_path))

        from qwen3_vl_embedding import Qwen3VLEmbedder

        self.model = Qwen3VLEmbedder(str(model_path))

    def _encode_with_qwen(self, inputs, batch_size=QWEN_DEFAULT_BATCH_SIZE):
        """Encode using Qwen3-VL

        Args:
            inputs: List of dicts with keys: text, image, video, instruction
            batch_size: Number of inputs to process per forward pass

        Returns:
            Embeddings as numpy array
        """
        batched_embeddings = []
        for start_idx in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start_idx : start_idx + batch_size]
            embeddings = self.model.process(batch_inputs)
            # Convert to numpy if it's a tensor
            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.float().cpu().numpy()
            batched_embeddings.append(embeddings)

        if len(batched_embeddings) == 1:
            return batched_embeddings[0]
        return np.concatenate(batched_embeddings, axis=0)

    def _resolve_batch_size(self, batch_size):
        """Resolve the default batch size."""
        if batch_size is not None:
            return batch_size
        return self.QWEN_DEFAULT_BATCH_SIZE

    def _prepare_qwen_input(self, content, content_type="text"):
        """Prepare input format for Qwen3-VL

        Args:
            content: Either a PIL Image, list of PIL Images, or text string/list
            content_type: "text" or "image"

        Returns:
            List of input dicts for Qwen3-VL
        """
        from PIL import Image

        inputs = []

        # Handle lists
        if isinstance(content, list):
            for item in content:
                if content_type == "image":
                    if isinstance(item, Image.Image):
                        inputs.append({"image": item})
                    else:
                        raise TypeError(f"Expected PIL Image, got {type(item)}")
                else:  # text
                    inputs.append({"text": str(item)})
        else:
            # Single item
            if content_type == "image":
                if isinstance(content, Image.Image):
                    inputs.append({"image": content})
                else:
                    raise TypeError(f"Expected PIL Image, got {type(content)}")
            else:  # text
                inputs.append({"text": str(content)})

        return inputs

    def encode_query_images(self, images: list[Image.Image], batch_size: int | None = None):
        """Encode query images for retrieval"""
        batch_size = self._resolve_batch_size(batch_size)
        inputs = self._prepare_qwen_input(images, content_type="image")
        return self._encode_with_qwen(inputs, batch_size=batch_size)

    def encode_passage_text(self, texts: list[str], batch_size: int | None = None):
        """Encode passage text for retrieval"""
        batch_size = self._resolve_batch_size(batch_size)
        inputs = self._prepare_qwen_input(texts, content_type="text")
        return self._encode_with_qwen(inputs, batch_size=batch_size)

    def encode_passage_images(self, images: list[Image.Image], batch_size: int | None = None):
        """Encode passage images for retrieval"""
        batch_size = self._resolve_batch_size(batch_size)
        inputs = self._prepare_qwen_input(images, content_type="image")
        return self._encode_with_qwen(inputs, batch_size=batch_size)

    def encode_with_cache(self, content, concept_id, memory_type):
        """TAME-O extension: Encode with caching"""
        if self.cache:
            cached = self.cache.get_embedding(concept_id, memory_type, content)
            if cached is not None:
                return cached

        if isinstance(content, Image.Image):
            embedding = self.encode_passage_images([content])[0]
        else:
            embedding = self.encode_passage_text([content])[0]

        if self.cache:
            self.cache.save_embedding(concept_id, memory_type, content, embedding)

        return embedding

    def temporal_weighted_retrieval(self, query_img, query_text, dynamic_memory_with_timestamps, top_k=10):
        """TAME-O extension: Retrieval with temporal weighting for dynamic memory

        Returns normalized temporal weighted score following:
        S(c_k | q_i) = (1/Z_k) * Σ exp(-λ * (t_max - t_j) / N_k) * s_j

        If both query_img and query_text are available, returns mean of both scores.
        """
        if not dynamic_memory_with_timestamps:
            return 0.0

        memories = [m[0] for m in dynamic_memory_with_timestamps]
        timestamps = [m[1] for m in dynamic_memory_with_timestamps]
        N_k = len(timestamps)

        memory_embs = self.encode_passage_text(memories)

        max_timestamp = max(timestamps)
        temporal_weights = torch.tensor(
            [math.exp(-self.decay_rate * (max_timestamp - t) / N_k) for t in timestamps]
        )

        # Z_k: normalization constant (sum of temporal weights)
        Z_k = torch.sum(temporal_weights)

        scores = []

        # Compute image-based score if available
        if query_img:
            query_img_emb = self.encode_query_images([query_img])
            similarity_img = torch.nn.functional.cosine_similarity(
                torch.tensor(query_img_emb), torch.tensor(memory_embs), dim=1
            )
            weighted_sum_img = torch.sum(similarity_img * temporal_weights) / Z_k
            scores.append(weighted_sum_img.item())

        # Compute text-based score if available
        if query_text:
            query_text_emb = self.encode_passage_text([query_text])
            similarity_text = torch.nn.functional.cosine_similarity(
                torch.tensor(query_text_emb), torch.tensor(memory_embs), dim=1
            )
            weighted_sum_text = torch.sum(similarity_text * temporal_weights) / Z_k
            scores.append(weighted_sum_text.item())

        # Return mean if both available, otherwise return whichever is available
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0

    def retrieve_concept(
        self,
        query_images: list[Image.Image],
        target_concepts: list[tuple[Image.Image, str]],
        batch_size: int | None = None,
    ) -> int:
        query_embeddings = self.encode_query_images(query_images, batch_size=batch_size)

        concept_images = [concept[0] for concept in target_concepts]
        concept_texts = [concept[1] for concept in target_concepts]

        concept_image_embeddings = self.encode_passage_images(concept_images, batch_size=batch_size)
        concept_text_embeddings = self.encode_passage_text(concept_texts, batch_size=batch_size)

        query_tensor = torch.tensor(query_embeddings)
        concept_image_tensor = torch.tensor(concept_image_embeddings)
        concept_text_tensor = torch.tensor(concept_text_embeddings)

        image_similarities = torch.mm(query_tensor, concept_image_tensor.T)
        text_similarities = torch.mm(query_tensor, concept_text_tensor.T)

        weighted_similarities = 0.7 * image_similarities + 0.3 * text_similarities

        max_similarity, best_idx = torch.max(weighted_similarities.flatten(), dim=0)

        best_concept_idx = best_idx % len(target_concepts)

        return best_concept_idx.item()

    def retrieve(self, query: Tensor, passage: Tensor, top_k: int = 10):
        similarity = torch.nn.functional.cosine_similarity(query, passage, dim=1)
        top_k_indices = torch.topk(similarity, top_k).indices
        return top_k_indices
