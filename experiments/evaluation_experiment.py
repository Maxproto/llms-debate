"""
experiments/evaluation_experiment.py

Evaluate each debate in `records/debates_*.json` with coverage, coherence,
and consistency computed separately for the pro and con agents, and output a single
CSV row per debate. The row contains:

  time_stamp, topic_index, topic,
  pro_model, pro_coverage, pro_coherence, pro_consistency,
  con_model, con_coverage, con_coherence, con_consistency,
  ai_judge_model, ai_judge_winner, ai_judgement
"""

import sys
import os
import json
import csv
import glob
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logger import GlobalLogger
from src.metrics.ai_judgement import ai_judgement
from src.metrics.consistency import compute_agent_consistency
from src.metrics.coverage import compute_agent_coverage
from src.metrics.coherence import compute_agent_coherence

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate debates: compute coverage, coherence, consistency for both pro and con in one CSV row + optional AI judge."
    )
    parser.add_argument("--record_dir", default="records", help="Directory containing debates_*.json.")
    parser.add_argument("--result_dir", default="results", help="Where to output the CSV.")
    parser.add_argument("--metrics", nargs='+', default=["coherence","coverage","consistency","judgement"],
                        help="Which metrics to apply, e.g. coherence coverage consistency judgement.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o",
                        help="Model name for AI judgement. E.g. gpt-4o, claude-3.5-haiku")
    parser.add_argument("--judge_max_tokens", type=int, default=400, help="Max tokens for AI judge's output.")
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="Sentence-Transformer for coherence.")
    return parser

def load_records(record_dir: str):
    pattern = os.path.join(record_dir, "debates_*.json")
    recs = []
    for fp in glob.glob(pattern):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                recs.extend(data)
            else:
                recs.append(data)
    return recs

def main():
    args = get_parser().parse_args()
    logger = GlobalLogger.get_logger("Evaluation")
    os.makedirs(args.result_dir, exist_ok=True)

    # Load debates
    records = load_records(args.record_dir)
    if not records:
        logger.info("No debate records found. Exiting.")
        return
    logger.info(f"Loaded {len(records)} debate records from {args.record_dir}")

    # Optional embedder for coherence
    embedder = None
    if "coherence" in args.metrics:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embed model: {args.embed_model}")
        embedder = SentenceTransformer(args.embed_model)

    # Define CSV columns (one row per debate)
    columns = [
        "time_stamp", "topic_index", "topic",
        "pro_model", "pro_coverage", "pro_coherence", "pro_consistency",
        "con_model", "con_coverage", "con_coherence", "con_consistency",
        "ai_judge_model", "ai_judge_winner", "ai_judgement"
    ]

    results = []
    for rec in records:
        debate = rec.get("debate", {})
        turns = debate.get("turns", [])
        pro_model = debate.get("pro_model", "")
        con_model = debate.get("con_model", "")
        topic = debate.get("topic", "")

        # Run AI judge once per debate if requested
        judge_text = ""
        judge_winner = ""
        judge_mdl = ""
        if "judgement" in args.metrics and args.judge_model:
            judge_text = ai_judgement(
                turns=turns,
                judge_model_name=args.judge_model,
                judge_max_tokens=args.judge_max_tokens,
                topic=topic
            )
            judge_mdl = args.judge_model
            # Naive parse of judge output
            w = "tie"
            if "Winner: Debater 1" in judge_text:
                w = pro_model
            elif "Winner: Debater 2" in judge_text:
                w = con_model
            judge_winner = w

        # Compute metrics for pro agent
        pro_cov = pro_coh = pro_cons = "N/A"
        if "coverage" in args.metrics:
            pro_cov = f"{compute_agent_coverage(turns, pro_model):.3f}"
        if "coherence" in args.metrics:
            c = compute_agent_coherence(turns, pro_model, embedder)
            pro_coh = f"{c:.3f}" if c >= 0 else "N/A"
        if "consistency" in args.metrics:
            cc = compute_agent_consistency(turns, pro_model)
            pro_cons = f"{cc:.3f}"

        # Compute metrics for con agent
        con_cov = con_coh = con_cons = "N/A"
        if "coverage" in args.metrics:
            con_cov = f"{compute_agent_coverage(turns, con_model):.3f}"
        if "coherence" in args.metrics:
            c = compute_agent_coherence(turns, con_model, embedder)
            con_coh = f"{c:.3f}" if c >= 0 else "N/A"
        if "consistency" in args.metrics:
            cc = compute_agent_consistency(turns, con_model)
            con_cons = f"{cc:.3f}"

        row = {
            "time_stamp": rec.get("timestamp", ""),
            "topic_index": rec.get("topic_index", -1),
            "topic": topic,
            "pro_model": pro_model,
            "pro_coverage": pro_cov,
            "pro_coherence": pro_coh,
            "pro_consistency": pro_cons,
            "con_model": con_model,
            "con_coverage": con_cov,
            "con_coherence": con_coh,
            "con_consistency": con_cons,
            "ai_judge_model": judge_mdl,
            "ai_judge_winner": judge_winner,
            "ai_judgement": judge_text
        }
        results.append(row)

    time_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.result_dir, f"debate_metrics_{time_str}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info(f"Wrote debate metrics to {out_csv}")
    logger.info("Done with per-records analysis. Next, run aggregate_results.py to summarize per-model performance.")

if __name__=="__main__":
    main()
