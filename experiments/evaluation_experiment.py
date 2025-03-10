"""
experiments/evaluation_experiment.py

Evaluates completed debates from a given postprocessing folder by running
selected metrics (e.g., AI judgement) in an incremental fashion.

Key Features:
1) Incremental / partial saving:
   - After each AI judgment is computed, the result is stored to disk in
     'eval_results_current.json' (and 'analysis_current.json') so progress is not lost.
2) Resume-like behavior with '--complete_from':
   - If --complete_from is given, it points to an existing evaluation folder
     (e.g. results/evaluation_20250310_135147), and we attempt to merge any partial or final
     results found there with the fresh debates from --postprocess_folder, then resume
     only for unjudged debates.
3) Logging of progress and stats updates after each new debate is evaluated.

Usage Example:
  python experiments/evaluation_experiment.py \
    --postprocess_folder records/postprocess_records_20250310_135147 \
    --metrics judgement \
    --judge_model gpt-4o \
    --judge_max_tokens 100

Or to resume from an existing evaluation folder:
  python experiments/evaluation_experiment.py \
    --postprocess_folder records/postprocess_records_20250310_135147 \
    --complete_from results/evaluation_20250310_135147 \
    --metrics judgement \
    --judge_model gpt-4o \
    --judge_max_tokens 100

Output:
  If starting fresh:
    results/evaluation_YYYYMMDD_HHMMSS/
      - eval_results_current.json
      - analysis_current.json
      - postprocess_info.json
      When done, also:
      - eval_results_final.json
      - analysis_final.json

  If using --complete_from <eval_folder>, partial outputs are updated in that folder,
  culminating in eval_results_final.json and analysis_final.json when complete.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.logger import GlobalLogger
from src.metrics.ai_judgement import ai_judgement


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation Experiment CLI (Incremental)")

    parser.add_argument("--postprocess_folder", type=str, required=True,
                        help="Path to the postprocess folder, e.g. 'records/postprocess_records_20250310_135147'. "
                             "Must contain a 'completed_debates_<TIMESTAMP>.json' file.")
    parser.add_argument("--complete_from", type=str, default=None,
                        help="If provided, points to an existing evaluation folder under 'results/' to resume from, "
                             "e.g. 'results/evaluation_20250310_135147'. We'll merge any partial/final results found "
                             "there with the debates from --postprocess_folder, and continue incrementally.")
    parser.add_argument("--metrics", nargs='+', default=["judgement"],
                        help="Which metrics to apply, e.g. 'judgement'.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o",
                        help="Model name for AI judgement. E.g. 'gpt-4o', 'claude-3.5-haiku', etc.")
    parser.add_argument("--judge_max_tokens", type=int, default=100,
                        help="Max tokens for the AI judge's output.")
    return parser


def parse_winner_from_judgement(judgement_text: str) -> str:
    """
    Look for 'Winner: Debater 1', 'Winner: Debater 2', or 'Winner: Tie' (case-insensitive).
    Returns 'Debater 1', 'Debater 2', or 'Tie' if found, else 'Unknown'.
    """
    text_lower = judgement_text.lower()
    if "winner: debater 1" in text_lower:
        return "Debater 1"
    elif "winner: debater 2" in text_lower:
        return "Debater 2"
    elif "winner: tie" in text_lower:
        return "Tie"
    return "Unknown"


def recalc_stats(evaluated_debates):
    """
    Recalculate all stats from the current set of evaluated debates.
    Returns a dict with model_stats, pairwise_stats, topic_analysis, etc.

    'evaluated_debates' is a list of debate records, each possibly containing
    "evaluation": { "judge_verdict": ... } if it's been evaluated.
    """

    model_wins = defaultdict(int)
    model_ties = defaultdict(int)
    matchups = defaultdict(lambda: defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0, "total": 0}))
    topic_model_wins = defaultdict(lambda: defaultdict(int))

    all_models = set()
    total_evaluated = 0

    for debate_record in evaluated_debates:
        eval_obj = debate_record.get("evaluation", {})
        verdict = eval_obj.get("judge_verdict", None)
        if verdict not in ("Debater 1", "Debater 2", "Tie"):
            continue  # Not yet evaluated or unknown

        pro_model = debate_record.get("pro_model", "UnknownPro")
        con_model = debate_record.get("con_model", "UnknownCon")
        topic_idx = debate_record.get("topic_index", -1)
        all_models.add(pro_model)
        all_models.add(con_model)

        # update matchups
        matchups[pro_model][con_model]["total"] += 1
        matchups[con_model][pro_model]["total"] += 1

        if verdict == "Debater 1":
            model_wins[pro_model] += 1
            matchups[pro_model][con_model]["win"] += 1
            matchups[con_model][pro_model]["loss"] += 1
            topic_model_wins[topic_idx][pro_model] += 1
        elif verdict == "Debater 2":
            model_wins[con_model] += 1
            matchups[con_model][pro_model]["win"] += 1
            matchups[pro_model][con_model]["loss"] += 1
            topic_model_wins[topic_idx][con_model] += 1
        elif verdict == "Tie":
            matchups[pro_model][con_model]["tie"] += 1
            matchups[con_model][pro_model]["tie"] += 1
            model_ties[pro_model] += 1
            model_ties[con_model] += 1

        total_evaluated += 1

    all_models_list = sorted(all_models)
    model_stats = {}
    for m in all_models_list:
        # total debates for this model
        total_for_m = 0
        for x in all_models_list:
            if x == m:
                continue
            total_for_m += matchups[m][x]["total"]

        mwins = model_wins[m]
        mties = model_ties[m]
        model_stats[m] = {
            "wins": mwins,
            "ties": mties,
            "debates_participated": total_for_m,
            "win_rate": float(mwins) / total_for_m if total_for_m > 0 else 0.0
        }

    pairwise_stats = {}
    for mA in all_models_list:
        pairwise_stats[mA] = {}
        for mB in all_models_list:
            if mA == mB:
                continue
            pairwise_stats[mA][mB] = dict(matchups[mA][mB])

    topic_analysis = {}
    for t_idx, model_win_map in topic_model_wins.items():
        max_wins = 0
        winners = []
        for model, wcount in model_win_map.items():
            if wcount > max_wins:
                max_wins = wcount
                winners = [model]
            elif wcount == max_wins and wcount > 0:
                winners.append(model)
        topic_analysis[t_idx] = {
            "model_wins": dict(model_win_map),
            "best_models": winners
        }

    return {
        "model_stats": model_stats,
        "pairwise_stats": pairwise_stats,
        "topic_analysis": topic_analysis,
        "all_models_list": all_models_list,
        "total_evaluated_debates": total_evaluated
    }


def load_json_if_exists(path):
    """Utility function: load JSON from path if it exists, else return empty list or dict."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def build_debate_key(record):
    """
    Use (timestamp, topic_index, pro_model, con_model) as a unique key
    to identify a debate record (they are unique).
    """
    return (
        record.get("timestamp"),
        record.get("topic_index"),
        record.get("pro_model"),
        record.get("con_model"),
    )


def main():
    parser = get_parser()
    args = parser.parse_args()

    logger = GlobalLogger.get_logger("Evaluation")
    logger.info("=== Starting Incremental Evaluation of Completed Debates ===")

    # 1. Verify postprocess folder
    postproc_folder = args.postprocess_folder
    if not os.path.isdir(postproc_folder):
        logger.error(f"Postprocess folder does not exist: {postproc_folder}")
        return

    # Identify timestamp from postprocess folder
    folder_name = os.path.basename(postproc_folder)
    if folder_name.startswith("postprocess_records_"):
        timestamp_str = folder_name.replace("postprocess_records_", "")
    else:
        raise ValueError(f"Unexpected postprocess folder name: {folder_name}, it should be like 'postprocess_records_20250310_135147'")

    # The 'completed_debates_{timestamp}.json' file
    completed_debates_file = os.path.join(postproc_folder, f"completed_debates_{timestamp_str}.json")
    if not os.path.exists(completed_debates_file):
        logger.error(f"Cannot find completed debates file: {completed_debates_file}")
        return

    # Also read the postprocess_summarization for reference
    postproc_summary_file = os.path.join(postproc_folder, f"postprocess_summarization_{timestamp_str}.json")
    postproc_info = {}
    if os.path.exists(postproc_summary_file):
        try:
            with open(postproc_summary_file, "r", encoding="utf-8") as f:
                postproc_info = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load postprocess_summarization: {e}")

    # 2. Determine whether creating a new evaluation folder or continuing from `--complete_from`
    if args.complete_from:
        # Store partial results in the existing folder
        eval_folder_path = args.complete_from
        if not os.path.isdir(eval_folder_path):
            logger.error(f"--complete_from folder does not exist: {eval_folder_path}")
            return
        logger.info(f"Resuming incremental evaluation in existing folder: {eval_folder_path}")
    else:
        # Create new folder
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        eval_folder_name = f"evaluation_{timestamp_str}"
        eval_folder_path = os.path.join(results_dir, eval_folder_name)
        os.makedirs(eval_folder_path, exist_ok=True)
        logger.info(f"Creating new evaluation folder: {eval_folder_path}")

    # Paths for partial/final results
    eval_results_current_path = os.path.join(eval_folder_path, "eval_results_current.json")
    analysis_current_path = os.path.join(eval_folder_path, "analysis_current.json")
    postproc_info_path = os.path.join(eval_folder_path, "postprocess_info.json")

    # 3. Save a copy of the postproc_info to eval folder if not exists
    if postproc_info and not os.path.exists(postproc_info_path):
        with open(postproc_info_path, "w", encoding="utf-8") as f:
            json.dump(postproc_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote a copy of postprocess_summarization to {postproc_info_path}")

    # 4. Load the entire set of completed debates from the postprocess folder
    with open(completed_debates_file, "r", encoding="utf-8") as f:
        postproc_debates = json.load(f)

    # Convert to a key->record map
    postproc_map = {}
    for rec in postproc_debates:
        key = build_debate_key(rec)
        postproc_map[key] = rec

    logger.info(f"Loaded {len(postproc_map)} completed debates from postprocess folder.")

    # 5. If continuing from an existing evaluation folder, load partial results
    existing_evals = load_json_if_exists(eval_results_current_path)
    if existing_evals is None:
        existing_evals = load_json_if_exists(os.path.join(eval_folder_path, "eval_results_final.json"))
    if not existing_evals:
        existing_evals = []

    # Merge existing partial/final with the postproc debates
    existing_map = {}
    for rec in existing_evals:
        key = build_debate_key(rec)
        existing_map[key] = rec

    # Overwrite postproc_map with existing_map if present
    for k, v in existing_map.items():
        postproc_map[k] = v

    # Get a unified map
    all_debates = list(postproc_map.values())

    logger.info(f"After merging existing partial results, we have {len(all_debates)} total debates to evaluate.")

    # 6. Figure out which are not yet evaluated
    def is_evaluated(rec):
        ev = rec.get("evaluation", {})
        verdict = ev.get("judge_verdict", None)
        return (verdict in ("Debater 1", "Debater 2", "Tie", "Unknown"))

    to_process = [r for r in all_debates if not is_evaluated(r)]
    total_to_process = len(to_process)
    logger.info(f"{total_to_process} debates remain unjudged.")

    do_judgement = ("judgement" in args.metrics)

    # 7. Evaluate each unjudged debate, saving partial results after each
    for i, debate_record in enumerate(to_process, start=1):
        t_idx = debate_record.get("topic_index")
        pro_model = debate_record.get("pro_model", "UnknownPro")
        con_model = debate_record.get("con_model", "UnknownCon")
        logger.info(f"[Progress: {i}/{total_to_process}] Evaluating topic_idx={t_idx}, pro={pro_model}, con={con_model}")

        if do_judgement:
            turns = debate_record.get("debate", {}).get("turns", [])
            topic_text = debate_record.get("debate", {}).get("topic", "")
            judgement_text = ai_judgement(
                turns=turns,
                judge_model_name=args.judge_model,
                judge_max_tokens=args.judge_max_tokens,
                topic=topic_text
            )
            winner_str = parse_winner_from_judgement(judgement_text)

            if "evaluation" not in debate_record:
                debate_record["evaluation"] = {}
            debate_record["evaluation"].update({
                "judge_model": args.judge_model,
                "judge_max_tokens": args.judge_max_tokens,
                "judge_verdict": winner_str,
                "judge_explanation": judgement_text
            })

        # Update the map
        key = build_debate_key(debate_record)
        postproc_map[key] = debate_record

        # Recompute stats incrementally
        current_list = list(postproc_map.values())
        stats_data = recalc_stats(current_list)

        # Write partial results
        with open(eval_results_current_path, "w", encoding="utf-8") as f:
            json.dump(current_list, f, indent=2, ensure_ascii=False)

        analysis_obj = {
            "evaluation_time": datetime.utcnow().isoformat(),
            "judge_model": args.judge_model if do_judgement else None,
            "judge_max_tokens": args.judge_max_tokens if do_judgement else None,
            "metrics_used": args.metrics,
            "total_completed_debates": len(current_list),
            "total_evaluated_so_far": stats_data["total_evaluated_debates"],
            "model_stats": stats_data["model_stats"],
            "pairwise_stats": stats_data["pairwise_stats"],
            "topic_analysis": stats_data["topic_analysis"],
            "all_models_list": stats_data["all_models_list"],
        }
        with open(analysis_current_path, "w", encoding="utf-8") as f:
            json.dump(analysis_obj, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved partial results. Evaluated so far: {stats_data['total_evaluated_debates']}")

    # 8. Finalize
    logger.info("All debates have been processed or were already evaluated. Generating final files...")

    final_list = list(postproc_map.values())
    final_stats = recalc_stats(final_list)

    eval_results_final_path = os.path.join(eval_folder_path, "eval_results_final.json")
    with open(eval_results_final_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)

    analysis_final_path = os.path.join(eval_folder_path, "analysis_final.json")
    final_analysis_obj = {
        "evaluation_time": datetime.utcnow().isoformat(),
        "judge_model": args.judge_model if do_judgement else None,
        "judge_max_tokens": args.judge_max_tokens if do_judgement else None,
        "metrics_used": args.metrics,
        "total_completed_debates": len(final_list),
        "total_evaluated_debates": final_stats["total_evaluated_debates"],
        "model_stats": final_stats["model_stats"],
        "pairwise_stats": final_stats["pairwise_stats"],
        "topic_analysis": final_stats["topic_analysis"],
        "all_models_list": final_stats["all_models_list"],
        "postprocess_info": postproc_info if postproc_info else {}
    }
    with open(analysis_final_path, "w", encoding="utf-8") as f:
        json.dump(final_analysis_obj, f, indent=2, ensure_ascii=False)

    logger.info(f"Final results written to: {eval_results_final_path}")
    logger.info(f"Final analysis written to: {analysis_final_path}")
    logger.info("=== Incremental Evaluation Completed Successfully. ===")


if __name__ == "__main__":
    main()
