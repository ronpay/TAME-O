import re
from dataclasses import dataclass
from typing import Optional

import yaml
from loguru import logger
from PIL import Image

GENERIC_GROUNDING_LABELS = ["animal", "person", "household item", "personal belonging"]


@dataclass
class GroundingDetectionResult:
    crops: list[Image.Image]
    labels: list[str]
    label_source: str


def _extract_yaml_payload(text: str) -> str:
    match = re.search(r"```(?:yaml|yml)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return text


def _normalize_grounding_label(label: object) -> str:
    normalized = str(label).strip().lower()
    normalized = re.sub(r"^[\s'\"`\-.,;:]+|[\s'\"`\-.,;:]+$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_grounding_labels(response: str) -> list[str]:
    yaml_payload = _extract_yaml_payload(response)

    try:
        parsed = yaml.safe_load(yaml_payload)
    except Exception as exc:
        logger.warning(f"Grounding label YAML parsing failed: {exc}")
        return []

    if not isinstance(parsed, dict):
        logger.warning(f"Grounding label response is not a YAML mapping: {type(parsed).__name__}")
        return []

    raw_labels = parsed.get("grounding_labels", [])
    if not isinstance(raw_labels, list):
        logger.warning(f"Grounding labels should be a list, got {type(raw_labels).__name__}")
        return []

    normalized_labels = []
    seen_labels = set()
    for raw_label in raw_labels:
        normalized_label = _normalize_grounding_label(raw_label)
        if not normalized_label or normalized_label in seen_labels:
            continue
        normalized_labels.append(normalized_label)
        seen_labels.add(normalized_label)
        if len(normalized_labels) == 3:
            break

    return normalized_labels


def build_grounding_label_prompt(question_text: Optional[str] = None) -> str:
    context_block = ""
    if question_text:
        context_block = f"""
# Extra Context
Question text: {question_text}

Use the question text only to disambiguate the main concept in the image.
"""

    return f"""You are preparing labels for Grounding-DINO.

# Task
Look at the provided image and identify the single main concept that should be grounded.
Return up to 3 short labels that are suitable for object grounding.
{context_block}
# Label Rules
- Focus on the main visible concept only.
- Prefer short common-noun phrases, not proper names.
- Make the labels specific enough to locate the concept in the image.
- Return:
  1. one primary noun phrase,
  2. one simpler alias,
  3. one broader category.
- Use lowercase.
- Do not include explanations.

# Output Format
```yaml
grounding_labels:
  - "primary label"
  - "simple alias"
  - "broader category"
```

Generate the output:
"""


def detect_with_mllm_grounding(
    detector,
    mllm,
    image: Image.Image,
    question_text: Optional[str] = None,
) -> GroundingDetectionResult:
    fallback_result = GroundingDetectionResult(
        crops=[],
        labels=list(GENERIC_GROUNDING_LABELS),
        label_source="generic_fallback",
    )

    try:
        prompt = build_grounding_label_prompt(question_text)
        response = mllm.chat_img(prompt, image, max_tokens=256)
        candidate_labels = parse_grounding_labels(response)
        if candidate_labels:
            logger.info(f"Grounding labels from MLLM: {candidate_labels}")
            detected_crops = detector.detect_and_crop(image, candidate_labels)
            if detected_crops:
                logger.info(f"Grounding succeeded with MLLM labels: {candidate_labels}")
                return GroundingDetectionResult(
                    crops=detected_crops,
                    labels=candidate_labels,
                    label_source="mllm",
                )
            logger.info("Grounding with MLLM labels returned no detections; retrying with generic fallback labels")
        else:
            logger.info("Grounding labels from MLLM were empty or unusable; using generic fallback labels")
    except Exception as exc:
        logger.warning(f"Grounding label generation failed: {exc}; using generic fallback labels")

    try:
        detected_crops = detector.detect_and_crop(image, GENERIC_GROUNDING_LABELS)
        if detected_crops:
            logger.info(f"Grounding succeeded with generic fallback labels: {GENERIC_GROUNDING_LABELS}")
        else:
            logger.info("Grounding with generic fallback labels returned no detections")
        fallback_result.crops = detected_crops
        return fallback_result
    except Exception as exc:
        logger.warning(f"Grounding detection failed with generic fallback labels: {exc}")
        return fallback_result
