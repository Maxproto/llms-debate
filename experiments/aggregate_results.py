"""
experiments/aggregate_results.py

Reads the 'debate_metrics_*.csv' produced by evaluation_experiment.py,
and aggregates metrics per model. For each debate row, both the pro and con agent metrics
are extracted and tallied. For each unique model, the script computes:
  - Average coverage, coherence, consistency (from all appearances)
  - Total count of appearances (num_debates)
  - Number of wins (if ai_judge_winner equals the model)

The output CSV has columns:
  model, num_debates, avg_coverage, avg_coherence, avg_consistency, wins
"""

import sys
import os
import csv
import glob
import logging
import argparse
from datetime import datetime

def setup_logger():
    logger = logging.getLogger("aggregate")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger

def get_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate debate metrics (one row per debate with pro and con) into model-level summary."
    )
    parser.add_argument("--input_csv_pattern", default="results/debate_metrics_*.csv",
                        help="Glob pattern to find input CSVs from step 1.")
    parser.add_argument("--out_dir", default="results", help="Folder to store the final summary CSV.")
    return parser

def safe_float(val):
    if val in ["N/A", ""]:
        return None
    try:
        return float(val)
    except:
        return None

def main():
    logger = setup_logger()
    args = get_parser().parse_args()

    all_files = glob.glob(args.input_csv_pattern)
    if not all_files:
        logger.info(f"No CSV files found matching {args.input_csv_pattern}. Exiting.")
        return

    # Data storage: model -> aggregated stats
    model_data = {}
    # Process each debate row and update stats for both pro and con agents.
    for fp in all_files:
        logger.info(f"Reading {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Process pro agent data
                pro_model = row.get("pro_model", "")
                if pro_model:
                    if pro_model not in model_data:
                        model_data[pro_model] = {
                            "coverages": [],
                            "coherences": [],
                            "consistencies": [],
                            "num_debates": 0,
                            "wins": 0
                        }
                    cov = safe_float(row.get("pro_coverage", ""))
                    coh = safe_float(row.get("pro_coherence", ""))
                    cons = safe_float(row.get("pro_consistency", ""))
                    if cov is not None:
                        model_data[pro_model]["coverages"].append(cov)
                    if coh is not None:
                        model_data[pro_model]["coherences"].append(coh)
                    if cons is not None:
                        model_data[pro_model]["consistencies"].append(cons)
                    model_data[pro_model]["num_debates"] += 1
                    if row.get("ai_judge_winner", "") == pro_model:
                        model_data[pro_model]["wins"] += 1

                # Process con agent data
                con_model = row.get("con_model", "")
                if con_model:
                    if con_model not in model_data:
                        model_data[con_model] = {
                            "coverages": [],
                            "coherences": [],
                            "consistencies": [],
                            "num_debates": 0,
                            "wins": 0
                        }
                    cov = safe_float(row.get("con_coverage", ""))
                    coh = safe_float(row.get("con_coherence", ""))
                    cons = safe_float(row.get("con_consistency", ""))
                    if cov is not None:
                        model_data[con_model]["coverages"].append(cov)
                    if coh is not None:
                        model_data[con_model]["coherences"].append(coh)
                    if cons is not None:
                        model_data[con_model]["consistencies"].append(cons)
                    model_data[con_model]["num_debates"] += 1
                    if row.get("ai_judge_winner", "") == con_model:
                        model_data[con_model]["wins"] += 1

    # Create summary rows
    out_rows = []
    for model_name, stats in model_data.items():
        covs = stats["coverages"]
        coherences = stats["coherences"]
        consistencies = stats["consistencies"]
        avg_cov = sum(covs) / len(covs) if covs else 0.0
        avg_coh = sum(coherences) / len(coherences) if coherences else 0.0
        avg_cons = sum(consistencies) / len(consistencies) if consistencies else 0.0
        row = {
            "model": model_name,
            "num_debates": stats["num_debates"],
            "avg_coverage": f"{avg_cov:.3f}",
            "avg_coherence": f"{avg_coh:.3f}",
            "avg_consistency": f"{avg_cons:.3f}",
            "wins": stats["wins"]
        }
        out_rows.append(row)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.out_dir, f"model_performance_{timestamp}.csv")
    columns = ["model", "num_debates", "avg_coverage", "avg_coherence", "avg_consistency", "wins"]
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    logger.info(f"Wrote summary to {out_file}")

if __name__=="__main__":
    main()
