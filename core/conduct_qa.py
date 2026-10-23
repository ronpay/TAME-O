import yaml
from loguru import logger

from core.dataset_utils import build_concept_dir_map


class QASystem:
    def __init__(self, model_id: str, dataset_name: str = "LCMP-H"):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.concept_dirs = build_concept_dir_map(dataset_name)
        self.questions = []
        self.cnt = 0

    def load_question_single(self, concept_id: str, q: dict, difficulty: str = "unknown"):
        """Load a single question with metadata"""
        qid = q["id"]
        question = q["question"]
        ideal_answer = q["evaluation_criteria"]["ideal_answer"]
        key_points = q["evaluation_criteria"]["key_points"]
        options = q["options"]
        options_answer = q["answer"]
        img_path = self.concept_dirs[concept_id] / "img" / f"{q['image_id']}.png"
        return {
            "qid": qid,
            "concept_id": concept_id,
            "question": question,
            "ideal_answer": ideal_answer,
            "key_points": key_points,
            "img_path": str(img_path),
            "options": options,
            "options_answer": options_answer,
            "difficulty": difficulty,
        }

    def load_question_by_concept(self, concept_id: str):
        """Load hard questions for a specific concept."""
        concept_dir = self.concept_dirs[concept_id]
        all_questions = []

        hard_question_path = concept_dir / "hard_question.yaml"
        try:
            with open(hard_question_path, "r", encoding="utf-8") as f:
                hard_questions = yaml.safe_load(f)
            if hard_questions:
                all_questions.extend(
                    [self.load_question_single(concept_id, q, "hard") for q in hard_questions]
                )
        except (FileNotFoundError, yaml.YAMLError, IOError) as e:
            logger.error(f"Error loading hard questions for concept {concept_id}: {e}")

        return all_questions

    def load_question_for_all_concepts(self):
        """Load all questions for all concepts"""
        if not self.concept_dirs:
            logger.error(f"No concepts found for dataset: {self.dataset_name}")
            return []

        questions = []
        for concept_id in self.concept_dirs:
            questions.extend(self.load_question_by_concept(concept_id))
        return questions

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt >= len(self.questions):
            raise StopIteration
        question = self.questions[self.cnt]
        self.cnt += 1
        return question

    def reset(self):
        """Reset the iterator to allow re-iteration"""
        self.cnt = 0
