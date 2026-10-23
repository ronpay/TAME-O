import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    concept_groups: tuple[str, ...]
    difficulties: tuple[str, ...]


DATASET_CONFIGS = {
    "LCMP-H": DatasetConfig(
        name="LCMP-H",
        root=Path("data/LCMP"),
        concept_groups=("concept_person", "concept_pet", "concept_obj"),
        difficulties=("hard",),
    ),
    "LCMP-O": DatasetConfig(
        name="LCMP-O",
        root=Path("data/LCMP-O"),
        concept_groups=("concept_person", "concept_pet", "concept_object"),
        difficulties=("hard",),
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    return DATASET_CONFIGS[dataset_name]


def get_dataset_difficulties(dataset_name: str) -> tuple[str, ...]:
    return get_dataset_config(dataset_name).difficulties


@lru_cache(maxsize=None)
def build_concept_dir_map(dataset_name: str) -> dict[str, Path]:
    config = get_dataset_config(dataset_name)
    concept_dirs = {}

    for group_name in config.concept_groups:
        group_dir = config.root / group_name
        for concept_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            concept_dirs[concept_dir.name] = concept_dir

    return concept_dirs


def get_concept_ids(dataset_name: str, shuffle: bool = False, seed: int = 42) -> list[str]:
    concept_ids = sorted(build_concept_dir_map(dataset_name))
    if shuffle:
        random.seed(seed)
        random.shuffle(concept_ids)
    return concept_ids


def get_concept_dir(dataset_name: str, concept_id: str) -> Path:
    return build_concept_dir_map(dataset_name)[concept_id]


def get_history_path(dataset_name: str, concept_id: str) -> Path:
    return get_concept_dir(dataset_name, concept_id) / "history.yaml"


def get_question_path(dataset_name: str, concept_id: str, difficulty: str) -> Path:
    if difficulty == "hard":
        return get_concept_dir(dataset_name, concept_id) / "hard_question.yaml"
    return get_concept_dir(dataset_name, concept_id) / "easy_question.yaml"


def get_image_path(dataset_name: str, concept_id: str, image_id: str) -> Path:
    concept_dir = get_concept_dir(dataset_name, concept_id)
    image_dir = concept_dir / "img"

    for suffix in (".png", ".jpg", ".jpeg"):
        image_path = image_dir / f"{image_id}{suffix}"
        if image_path.exists():
            return image_path

    return image_dir / f"{image_id}.png"
