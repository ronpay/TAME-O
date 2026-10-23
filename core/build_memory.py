from pathlib import Path
import json

import pandas as pd
from colorama import Fore, Style
from loguru import logger


def build_memory_interleaved(agent, concept_ids, output_path=None, metrics_tracker=None):
    """
    TAME-O interleaved learning-QA workflow:
    For each concept: learn history -> answer questions -> next concept
    """
    from core.conduct_qa import QASystem

    results = []
    qa_system = QASystem(agent.model_id, dataset_name=agent.dataset_name)

    total_questions = 0
    correct_questions = 0

    # Determine per-concept output directory
    per_concept_output_dir = Path("results") / agent.dataset_name / agent.run_name
    per_concept_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Per-concept results will be saved to: {per_concept_output_dir}")

    if output_path is None:
        output_file = Path("results") / agent.dataset_name / f"{agent.run_name}.jsonl"
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for concept_id in concept_ids:
        logger.info(f"{'=' * 80}")
        logger.info(f"Processing concept: {concept_id}")
        logger.info(f"{'=' * 80}")

        # Concept-specific output file
        concept_output_file = per_concept_output_dir / f"{concept_id}.jsonl"
        processed_qids = set()

        # Load existing results for resume capability
        if concept_output_file.exists():
            try:
                with open(concept_output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                if "question_id" in record:
                                    processed_qids.add(record["question_id"])
                            except json.JSONDecodeError:
                                continue
                logger.info(f"Resuming {concept_id}: Found {len(processed_qids)} answered questions.")
            except Exception as e:
                logger.error(f"Error reading existing results for {concept_id}: {e}")

        logger.info(f"Step 1/2: Building memory for {concept_id}")
        agent.read_history(concept_id, metrics_tracker=metrics_tracker)
        agent._remove_empty_concepts()

        logger.info(f"Step 2/2: Conducting QA for {concept_id}")
        questions = qa_system.load_question_by_concept(concept_id)

        for q in questions:
            if q["qid"] in processed_qids:
                logger.info(f"Skipping question {q['qid']} (already answered)")
                continue

            logger.info(
                f"{Fore.GREEN}{q['difficulty']} question of concept {concept_id} starting{Style.RESET_ALL}"
            )

            # --- DEBUG INFO ---
            try:
                cra = agent.get_cra()
                cra_correct = agent.identification_stats["correct"] + agent.identification_stats["new_concept_correct"]
                cra_total = sum(agent.identification_stats.values())
                fcra = agent.get_fork_aware_cra()
                cfr = agent.calculate_cfr()[0]
                logger.info(
                    f"DEBUG: CRA={cra:.1%} ({cra_correct}/{cra_total}) | "
                    f"FCRA={fcra:.1%} ({agent.fork_aware_correct_sum:.3f}/{agent.fork_aware_total}) | CFR={cfr:.3f}"
                )
            except Exception as e:
                logger.warning(f"DEBUG info gathering failed: {e}")
            # ------------------

            concept_id_pred, answer, choice = agent.complete_qa_workflow(
                img_path=q["img_path"],
                question=q["question"],
                options=q.get("options"),
                options_answer=q.get("options_answer"),
                ground_truth_concept_id=concept_id,
            )

            is_correct_choice = False
            options_answer = q.get("options_answer")
            options = q.get("options", [])
            correct_choice_letter = "?"

            if choice and options_answer:
                if options:
                    try:
                        correct_index = options.index(options_answer)
                        correct_choice_letter = chr(65 + correct_index)
                        is_correct_choice = choice == correct_choice_letter
                    except ValueError:
                        logger.warning(f"Option answer '{options_answer}' not found in options list.")
                        is_correct_choice = choice == options_answer
                else:
                    is_correct_choice = choice == options_answer

                if is_correct_choice:
                    correct_questions += 1
                total_questions += 1

            accuracy = correct_questions / total_questions if total_questions > 0 else 0.0

            if metrics_tracker:
                metrics_tracker.log_acc(accuracy, correct_questions, total_questions)

            result_symbol = "✔️" if is_correct_choice else "✖️"
            logger.info(
                f"Question Result: {choice} vs {correct_choice_letter} {result_symbol} | Accuracy: {accuracy:.2%} ({correct_questions}/{total_questions})"
            )

            result = {
                "concept_id": concept_id,
                "concept_id_predicted": concept_id_pred,
                "question_id": q["qid"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "answer": answer,
                "choice": choice,
            }
            results.append(result)

            # Immediately save the result
            try:
                with open(concept_output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Error saving result for question {q['qid']}: {e}")

            if agent.openset_identifier.check_correctness(concept_id_pred, concept_id):
                logger.info(f"✓ Correctly identified {concept_id}")
            else:
                logger.warning(f"✗ Misidentified: {concept_id_pred} != {concept_id}")

        logger.info(f"Completed concept: {concept_id} ({len(questions)} questions)")

    # Optional: still save the aggregated file, but mostly relying on per-concept files now
    if results:
        df = pd.DataFrame(results)
        df.to_json(output_file, orient="records", lines=True, force_ascii=False)
        logger.info(f"Aggregated results saved to: {output_file}")
    else:
        logger.info("No new results to save in aggregated file.")

    logger.info(f"{'=' * 80}")
    logger.info(f"Interleaved learning-QA completed!")
    logger.info(f"Total concepts processed: {len(concept_ids)}")
    logger.info(f"Total questions answered: {len(results)}")
    logger.info(f"Results saved to: {output_file}")

    cra = agent.get_cra()
    cra_correct = agent.identification_stats["correct"] + agent.identification_stats["new_concept_correct"]
    cra_total = sum(agent.identification_stats.values())
    fcra = agent.get_fork_aware_cra()
    logger.info(f"CRA (Concept Recognition Accuracy): {cra:.3f} ({cra_correct}/{cra_total})")
    logger.info(
        f"FCRA (Fork-Aware CRA): {fcra:.3f} ({agent.fork_aware_correct_sum:.3f}/{agent.fork_aware_total})"
    )
    logger.info(f"{'=' * 80}")

    return results
