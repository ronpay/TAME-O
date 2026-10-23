import argparse
import multiprocessing as mp
import random
import signal
import ssl
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from loguru import logger

from core.dataset_utils import build_concept_dir_map, get_dataset_difficulties, get_question_path
from core.utils.mllm_factory import MLLMFactory
from core.utils.models.api_model import APIModel

METRIC_KEYS = {
    "easy": ("choice_acc", "freetext_acc", "scoring_point"),
    "hard": ("choice_acc", "freetext_acc", "scoring_point_long", "scoring_point_short"),
}

_WORKER_EVALUATOR = None


def read_jsonl_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    frame = pd.read_json(path, lines=True)
    return frame.where(pd.notnull(frame), None).to_dict(orient="records")


def write_jsonl_records(records: List[Dict[str, Any]], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        path.write_text("", encoding="utf-8")
        return

    frame = pd.DataFrame(records)
    frame.to_json(path, orient="records", lines=True, force_ascii=False)


def load_single_concept_data(concept_id: str, dataset_name: str, difficulties: tuple[str, ...]) -> Dict[str, Any]:
    """Static function to load concept data for multiprocessing."""
    data = {difficulty: {} for difficulty in difficulties}

    for difficulty in difficulties:
        question_path = get_question_path(dataset_name, concept_id, difficulty)
        if not question_path.exists():
            continue

        with open(question_path, "r", encoding="utf-8") as handle:
            questions = yaml.safe_load(handle) or []

        for question in questions:
            data[difficulty][question["id"]] = question

    return data


def load_all_concepts(dataset_name: str) -> Dict[str, Any]:
    """Static function to load all concept data for multiprocessing."""
    concept_data = {}
    concept_dirs = build_concept_dir_map(dataset_name)
    difficulties = get_dataset_difficulties(dataset_name)

    for concept_id in concept_dirs:
        concept_data[concept_id] = load_single_concept_data(concept_id, dataset_name, difficulties)

    return concept_data


def use_multiprocessing(model_id: str) -> bool:
    """Only use multiprocessing for API-backed evaluators."""
    return model_id in APIModel.model_list()


def get_eval_process_count(model_id: str) -> int:
    if use_multiprocessing(model_id):
        return 2
    return 0


def initialize_eval_worker(model_id: str) -> None:
    """Load one evaluator per worker process and reuse it for all assigned rows."""
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = MLLMFactory(model_id)


def get_worker_evaluator(model_id: str):
    global _WORKER_EVALUATOR
    if _WORKER_EVALUATOR is None:
        _WORKER_EVALUATOR = MLLMFactory(model_id)
    return _WORKER_EVALUATOR


def evaluate_result_record(result: Dict[str, Any], concept_data: Dict[str, Any], evaluator):
    concept_id = result["concept_id"]
    question_id = result["question_id"]
    difficulty = result["difficulty"]

    if concept_id not in concept_data:
        return None, f"Concept {concept_id} not found"

    if difficulty not in concept_data[concept_id]:
        return None, f"Difficulty {difficulty} not found for concept {concept_id}"

    if question_id not in concept_data[concept_id][difficulty]:
        return None, f"Question {question_id} not found for {concept_id}/{difficulty}"

    ground_truth = concept_data[concept_id][difficulty][question_id]
    evaluation = evaluate_single_result_with_retry(evaluator, result, ground_truth)
    return evaluation, None


def evaluate_single_result_with_retry(evaluator, result, ground_truth, max_retries=3):
    """Helper function to evaluate with retry logic for network errors."""
    for attempt in range(max_retries):
        try:
            if result["difficulty"] == "easy":
                return evaluate_easy_question_static(result, ground_truth, evaluator)
            if result["difficulty"] == "hard":
                return evaluate_hard_question_static(result, ground_truth, evaluator)
            raise ValueError(f"Unknown difficulty: {result['difficulty']}")
        except (requests.exceptions.RequestException, ssl.SSLError, ConnectionError) as error:
            if attempt < max_retries - 1:
                wait_time = (2**attempt) * 5
                logger.warning(f"Network error on attempt {attempt + 1}, retrying in {wait_time}s: {error}")
                time.sleep(wait_time)
                continue
            raise error


def evaluate_single_result(args_tuple):
    """Worker function to evaluate a single result in multiprocessing."""
    result, concept_data, model_id = args_tuple

    try:
        evaluator = get_worker_evaluator(model_id)
        return evaluate_result_record(result, concept_data, evaluator)
    except Exception as error:
        return (
            None,
            f"Error evaluating {result.get('concept_id', 'unknown')}/{result.get('question_id', 'unknown')}: {error}",
        )


def evaluate_choice_accuracy_static(predicted_choice: str, ground_truth: Dict[str, Any]) -> int:
    """Static function for choice accuracy evaluation."""
    if "options" not in ground_truth or "answer" not in ground_truth:
        return 0

    correct_answer = ground_truth["answer"]
    options = ground_truth["options"]

    correct_choice = None
    for index, option in enumerate(options):
        if option == correct_answer:
            correct_choice = chr(ord("A") + index)
            break

    if correct_choice is None:
        return 0
    if predicted_choice is None:
        predicted_choice = random.choice(["A", "B", "C", "D"])

    return 1 if predicted_choice.upper() == correct_choice else 0


def evaluate_freetext_accuracy_static(predicted_answer: str, ground_truth: Dict[str, Any], evaluator) -> int:
    """Static function for freetext accuracy evaluation."""
    if "evaluation_criteria" not in ground_truth:
        return 0

    ideal_answer = ground_truth["evaluation_criteria"].get("ideal_answer", "")
    if not ideal_answer:
        return 0

    prompt = f"""Please evaluate whether the following predicted answer is an acceptable substitute for the ideal answer.

Ideal Answer: {ideal_answer}

Predicted Answer: {predicted_answer}

Consider:
- Does the predicted answer support the same general conclusion?
- Is the predicted answer factually compatible with the ideal answer (i.e., not contradictory)?
- You may overlook missing nuances or slightly less precise phrasing if the main intent is still preserved.

Respond with only "YES" if the predicted answer is acceptable, or "NO" if it is not."""

    try:
        response = evaluator.chat_text(prompt)
        return 1 if "YES" in response.strip().upper() else 0
    except Exception:
        return 0


def evaluate_scoring_point_static(predicted_answer: str, key_point: str, evaluator) -> int:
    """Static function for scoring point evaluation."""
    prompt = f"""Please evaluate whether the following answer addresses the given key point.

Key Point: {key_point}

Answer: {predicted_answer}

Respond with only "YES" if the answer clearly addresses this key point, or "NO" if it doesn't."""

    try:
        response = evaluator.chat_text(prompt)
        return 1 if "YES" in response.strip().upper() else 0
    except Exception:
        return 0


def evaluate_easy_question_static(result: Dict[str, Any], ground_truth: Dict[str, Any], evaluator) -> Dict[str, Any]:
    """Static function to evaluate an easy question result."""
    evaluation = {
        "concept_id": result["concept_id"],
        "question_id": result["question_id"],
        "difficulty": result["difficulty"],
    }

    predicted_choice = result.get("choice", "")
    evaluation["choice_acc"] = evaluate_choice_accuracy_static(predicted_choice, ground_truth)

    predicted_answer = result.get("answer", "")
    evaluation["freetext_acc"] = evaluate_freetext_accuracy_static(predicted_answer, ground_truth, evaluator)

    key_points = ground_truth.get("evaluation_criteria", {}).get("key_points", [])
    evaluation["scoring_point"] = 0
    for key_point in key_points:
        if evaluate_scoring_point_static(predicted_answer, key_point, evaluator):
            evaluation["scoring_point"] = 1
            break

    return evaluation


def evaluate_hard_question_static(result: Dict[str, Any], ground_truth: Dict[str, Any], evaluator) -> Dict[str, Any]:
    """Static function to evaluate a hard question result."""
    evaluation = {
        "concept_id": result["concept_id"],
        "question_id": result["question_id"],
        "difficulty": result["difficulty"],
    }

    predicted_choice = result.get("choice", "")
    evaluation["choice_acc"] = evaluate_choice_accuracy_static(predicted_choice, ground_truth)

    predicted_answer = result.get("answer", "")
    evaluation["freetext_acc"] = evaluate_freetext_accuracy_static(predicted_answer, ground_truth, evaluator)

    key_points = ground_truth.get("evaluation_criteria", {}).get("key_points", [])
    if len(key_points) >= 2:
        evaluation["scoring_point_long"] = evaluate_scoring_point_static(predicted_answer, key_points[0], evaluator)
        evaluation["scoring_point_short"] = evaluate_scoring_point_static(predicted_answer, key_points[1], evaluator)
    else:
        evaluation["scoring_point_long"] = 0
        evaluation["scoring_point_short"] = 0

    return evaluation


class ConceptEvaluator:
    def __init__(
        self,
        dataset_name: str = "LCMP-H",
        model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    ):
        """Initialize the evaluator with an LLM for freetext evaluation."""
        self.dataset_name = dataset_name
        self.model_id = model_id
        self.difficulties = get_dataset_difficulties(dataset_name)
        self.concept_data = {}
        self.load_all_concepts()

    def load_all_concepts(self):
        """Load all concept data for the selected dataset."""
        self.concept_data = load_all_concepts(self.dataset_name)

    def get_ground_truth(
        self, concept_id: str, question_id: int, difficulty: str
    ) -> Optional[Dict[str, Any]]:
        """Get ground truth data for a specific question."""
        if concept_id not in self.concept_data:
            logger.warning(f"Concept {concept_id} not found")
            return None

        if difficulty not in self.concept_data[concept_id]:
            logger.warning(f"Difficulty {difficulty} not found for concept {concept_id}")
            return None

        if question_id not in self.concept_data[concept_id][difficulty]:
            logger.warning(f"Question {question_id} not found for {concept_id}/{difficulty}")
            return None

        return self.concept_data[concept_id][difficulty][question_id]

    def filter_latest_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results to keep only the latest for each (concept_id, question_id, difficulty) combination."""
        grouped = defaultdict(list)
        for result in results:
            key = (result["concept_id"], result["question_id"], result["difficulty"])
            grouped[key].append(result)
        return [group[-1] for group in grouped.values()]

    def load_existing_evaluations(self, output_file: str) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
        """Load existing evaluation results from output file if it exists."""
        evaluations = {}
        for evaluation in read_jsonl_records(output_file):
            key = (evaluation["concept_id"], evaluation["question_id"], evaluation["difficulty"])
            evaluations[key] = evaluation
        if evaluations:
            logger.info(f"Loaded {len(evaluations)} existing evaluations from {output_file}")
        else:
            logger.info("No existing evaluation file found, starting fresh")
        return evaluations

    def initialize_stats(self) -> Dict[str, Dict[str, List[int]]]:
        return {
            difficulty: {metric: [] for metric in METRIC_KEYS[difficulty]}
            for difficulty in self.difficulties
        }

    def update_stats(self, stats: Dict[str, Dict[str, List[int]]], evaluation: Dict[str, Any]):
        difficulty = evaluation["difficulty"]
        if difficulty not in stats:
            return

        for metric in METRIC_KEYS[difficulty]:
            stats[difficulty][metric].append(evaluation[metric])

    def format_progress_stats(self, stats: Dict[str, Dict[str, List[int]]]) -> str:
        parts = []

        if "easy" in stats and stats["easy"]["choice_acc"]:
            easy_count = len(stats["easy"]["choice_acc"])
            easy_choice_avg = sum(stats["easy"]["choice_acc"]) / easy_count
            easy_freetext_avg = sum(stats["easy"]["freetext_acc"]) / easy_count
            easy_scoring_avg = sum(stats["easy"]["scoring_point"]) / easy_count
            parts.append(
                f"Easy({easy_count}): CA={easy_choice_avg:.3f} FA={easy_freetext_avg:.3f} SP={easy_scoring_avg:.3f}"
            )

        if "hard" in stats and stats["hard"]["choice_acc"]:
            hard_count = len(stats["hard"]["choice_acc"])
            hard_choice_avg = sum(stats["hard"]["choice_acc"]) / hard_count
            hard_freetext_avg = sum(stats["hard"]["freetext_acc"]) / hard_count
            hard_long_avg = sum(stats["hard"]["scoring_point_long"]) / hard_count
            hard_short_avg = sum(stats["hard"]["scoring_point_short"]) / hard_count
            parts.append(
                f"Hard({hard_count}): CA={hard_choice_avg:.3f} FA={hard_freetext_avg:.3f} SPL={hard_long_avg:.3f} SPS={hard_short_avg:.3f}"
            )

        return " | ".join(parts)

    def evaluate_results(self, input_file: str, output_file: str):
        """Main evaluation function with multiprocessing support."""
        logger.info(f"Loading results from {input_file}")

        results = read_jsonl_records(input_file)
        logger.info(f"Loaded {len(results)} results")

        filtered_results = self.filter_latest_results(results)
        logger.info(f"After filtering, have {len(filtered_results)} unique results")

        existing_evaluations = self.load_existing_evaluations(output_file)

        remaining_results = []
        for result in filtered_results:
            key = (result["concept_id"], result["question_id"], result["difficulty"])
            if key not in existing_evaluations:
                remaining_results.append(result)

        logger.info(f"Found {len(existing_evaluations)} existing evaluations")
        logger.info(f"Remaining to evaluate: {len(remaining_results)} results")

        stats = self.initialize_stats()
        for evaluation in existing_evaluations.values():
            self.update_stats(stats, evaluation)

        if not remaining_results:
            logger.info("All results have already been evaluated!")
            self.calculate_statistics(stats)
            return

        args_list = [(result, self.concept_data, self.model_id) for result in remaining_results]
        total_count = len(remaining_results)
        completed_count = 0
        new_evaluations = []

        def signal_handler(signum, frame):
            logger.info("Received interrupt signal, shutting down gracefully...")

        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            if use_multiprocessing(self.model_id):
                process_count = get_eval_process_count(self.model_id)
                logger.info(f"Starting multiprocessing evaluation with {process_count} worker(s)...")
                with mp.Pool(
                    processes=process_count,
                    initializer=initialize_eval_worker,
                    initargs=(self.model_id,),
                ) as pool:
                    iterator = pool.imap_unordered(evaluate_single_result, args_list)
                    for evaluation, error in iterator:
                        if error:
                            logger.warning(f"Evaluation error: {error}")
                            continue

                        if evaluation is None:
                            continue

                        new_evaluations.append(evaluation)
                        self.update_stats(stats, evaluation)
                        completed_count += 1

                        if completed_count % 10 == 0 or completed_count == total_count:
                            progress_pct = (completed_count / total_count) * 100
                            stats_message = self.format_progress_stats(stats)
                            base_message = f"Progress: {completed_count}/{total_count} ({progress_pct:.1f}%)"
                            logger.info(f"{base_message} | {stats_message}" if stats_message else base_message)
            else:
                logger.info("Starting single-process evaluation...")
                evaluator = MLLMFactory(self.model_id)
                for result in remaining_results:
                    evaluation, error = evaluate_result_record(result, self.concept_data, evaluator)
                    if error:
                        logger.warning(f"Evaluation error: {error}")
                        continue

                    if evaluation is None:
                        continue

                    new_evaluations.append(evaluation)
                    self.update_stats(stats, evaluation)
                    completed_count += 1

                    if completed_count % 10 == 0 or completed_count == total_count:
                        progress_pct = (completed_count / total_count) * 100
                        stats_message = self.format_progress_stats(stats)
                        base_message = f"Progress: {completed_count}/{total_count} ({progress_pct:.1f}%)"
                        logger.info(f"{base_message} | {stats_message}" if stats_message else base_message)
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
            return
        except Exception as error:
            logger.error(f"Multiprocessing error: {error}")
            return
        finally:
            signal.signal(signal.SIGINT, original_handler)

        all_evaluations = list(existing_evaluations.values()) + new_evaluations
        write_jsonl_records(all_evaluations, output_file)

        logger.info(f"Evaluation completed! Results saved to {output_file}")
        self.calculate_statistics(stats)

    def calculate_statistics(self, stats: Dict[str, Dict[str, List[int]]]):
        """Calculate and display aggregated statistics."""
        logger.info("=== EVALUATION STATISTICS ===")

        if "easy" in stats:
            if stats["easy"]["choice_acc"]:
                easy_count = len(stats["easy"]["choice_acc"])
                easy_choice_avg = sum(stats["easy"]["choice_acc"]) / easy_count
                easy_freetext_avg = sum(stats["easy"]["freetext_acc"]) / easy_count
                easy_scoring_avg = sum(stats["easy"]["scoring_point"]) / easy_count
                logger.info(
                    f"Easy (n={easy_count}): CA={easy_choice_avg:.3f} FA={easy_freetext_avg:.3f} SP={easy_scoring_avg:.3f}"
                )
            else:
                logger.info("Easy: No data")

        if "hard" in stats:
            if stats["hard"]["choice_acc"]:
                hard_count = len(stats["hard"]["choice_acc"])
                hard_choice_avg = sum(stats["hard"]["choice_acc"]) / hard_count
                hard_freetext_avg = sum(stats["hard"]["freetext_acc"]) / hard_count
                hard_long_avg = sum(stats["hard"]["scoring_point_long"]) / hard_count
                hard_short_avg = sum(stats["hard"]["scoring_point_short"]) / hard_count
                logger.info(
                    f"Hard (n={hard_count}): CA={hard_choice_avg:.3f} FA={hard_freetext_avg:.3f} SPL={hard_long_avg:.3f} SPS={hard_short_avg:.3f}"
                )
            else:
                logger.info("Hard: No data")


def main():
    parser = argparse.ArgumentParser(description="Evaluate concept-based Q&A results")
    parser.add_argument("input_file", help="Input JSONL file with results")
    parser.add_argument("output_file", help="Output JSONL file for evaluations")
    parser.add_argument(
        "--dataset",
        default="LCMP-H",
        choices=["LCMP-H", "LCMP-O"],
        help="Dataset used to resolve concept ground truth",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="Judge model ID, served by the OpenAI-compatible endpoint at API_URL",
    )

    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("log/eval")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"log/eval/{timestamp}.log")

    evaluator = ConceptEvaluator(dataset_name=args.dataset, model_id=args.model)
    evaluator.evaluate_results(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
