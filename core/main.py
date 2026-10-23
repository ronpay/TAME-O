import argparse
import sys
from pathlib import Path

from loguru import logger

from core.build_memory import build_memory_interleaved
from core.dataset_utils import get_concept_ids
from core.TAMEo import TAMEo, get_model_id
from core.utils.metrics_tracker import MetricsTracker


def setup_logger(debug: bool = False):
    """Configure logger for console output"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG" if debug else "INFO",
    )
    logger.add("log/{time}.log", level="DEBUG", encoding="utf8", colorize=False)


def get_concept_list(dataset_name: str):
    """Get a reproducibly shuffled list of concept directories (seed=42)."""
    concept_ids = get_concept_ids(dataset_name, shuffle=True, seed=42)
    if not concept_ids:
        logger.error(f"No concepts found for dataset: {dataset_name}")
    return concept_ids


def main():
    parser = argparse.ArgumentParser(description="TAME-O: Open-Set TAME")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen"],
        help="Model shortcut: qwen (Qwen3.5-9B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LCMP-H",
        choices=["LCMP-H", "LCMP-O"],
        help="Dataset to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="build",
        choices=["build"],
        help="Mode: build (interleaved learning-QA)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path (default: results/<dataset>/<run_name>.jsonl)"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["TAME"],
        help="Ablation mode: TAME",
    )
    parser.add_argument(
        "--debug",
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y", "on"},
        nargs="?",
        const=True,
        default=False,
        help='Enable DEBUG logging (e.g., "--debug=true")',
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top candidates K to retrieve in Stage 1 and verify in Stage 2 (default: 3)",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="Number of latest dynamic memory items R per candidate shown in Stage 2 (default: 3)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=1.0,
        help="Temporal decay coefficient lambda for memory retrieval (default: 1.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Novel-concept similarity threshold for the TAME ablation only, without LMM invocation (default: 0.6)",
    )
    parser.add_argument(
        "--mitigation-interval",
        type=int,
        default=10,
        help="Interval T for triggering concept fission mitigation (default: 10)",
    )
    parser.add_argument(
        "--l",
        type=int,
        default=3,
        help="Number of top similar concept pairs L verified per fusion round (default: 3)",
    )
    parser.add_argument(
        "--tao",
        type=float,
        default=0.6,
        help="Threshold tau for filtering concept pairs by similarity score (default: 0.6)",
    )

    args = parser.parse_args()
    setup_logger(args.debug)

    metrics_tracker = MetricsTracker()

    model_id = get_model_id(args.model)

    logger.info(f"Initializing TAME-O with model: {model_id}")

    agent = TAMEo(
        model_id,
        dataset_name=args.dataset,
        ablation=args.ablation,
        top_k=args.k,
        num_recent=args.r,
        decay_rate=args.lambda_,
        threshold=args.threshold,
        mitigation_interval=args.mitigation_interval,
        fusion_top_l=args.l,
        tao=args.tao,
    )

    concept_ids = get_concept_list(args.dataset)

    if not concept_ids:
        logger.error("No concepts found to process")
        return

    logger.info(f"Concepts to process ({len(concept_ids)}): {', '.join(concept_ids)}")

    if args.output:
        output_path = args.output
    else:
        output_path = str(Path("results") / args.dataset / f"{agent.run_name}.jsonl")

    logger.info("Mode: Interleaved learning-QA")
    build_memory_interleaved(agent, concept_ids, output_path, metrics_tracker)

    logger.info("TAME-O completed successfully!")


if __name__ == "__main__":
    main()
