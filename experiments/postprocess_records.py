"""
experiments/postprocess_records.py

Postprocess debate records to:
1) Identify completed vs. incomplete vs. repeated debates.
2) A 'complete' set of debates for a single topic requires n_models*(n_models-1) unique (pro, con) pairs.
3) Collect repeated debates (where the same (topic_index, pro_model, con_model) appears multiple times).
4) Collect all distinct model names from the records.
5) For each topic in the min..max topic index range, record which pairs are missing
  so we can complete them later (in "topics_to_complete").
6) Output results into a new folder under "records" named `postprocess_records_{currenttime}`:
   - completed_debates_{currenttime}.json
   - incomplete_debates_{currenttime}.json
   - repeated_debates_{currenttime}.json
   - postprocess_summarization_{currenttime}.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.logger import GlobalLogger


def get_parser() -> argparse.ArgumentParser:
    """
    Returns the ArgumentParser for this postprocessing script.
    """
    parser = argparse.ArgumentParser(
        description="Postprocess debate record files to identify completed/incomplete/repeated debates."
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=5,
        help="Number of distinct models used in the debate experiments. "
             "A complete set for one topic is n_models*(n_models - 1) unique debates."
    )
    return parser


def main():
    # 1. Parse CLI
    parser = get_parser()
    args = parser.parse_args()

    # 2. Prepare a logger
    logger = GlobalLogger.get_logger("Postprocess")
    logger.info("=== Starting Postprocessing of Debate Records ===")

    # 3. Identify how many debates we expect per topic
    n_models = args.n_models
    expected_debates_per_topic = n_models * (n_models - 1)
    logger.info(f"Expecting {expected_debates_per_topic} unique debates per topic "
                f"for n_models={n_models}.")

    # 4. Read all debate record files in the records/ folder
    records_dir = "records"
    if not os.path.exists(records_dir):
        logger.error(f"Records folder not found: {records_dir}")
        return

    record_files = [
        f for f in os.listdir(records_dir)
        if f.endswith(".json") and f.startswith("debates")
    ]
    logger.info(f"Found {len(record_files)} JSON record files to process.")

    # Data structure to hold all debates:
    # records_by_topic[topic_index][(pro_model, con_model)] = list of raw record items
    records_by_topic = defaultdict(lambda: defaultdict(list))

    # collect all models encountered
    all_models = set()

    min_topic_index = None
    max_topic_index = None

    # 5. Aggregate data
    for file_name in record_files:
        file_path = os.path.join(records_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue

        # Each 'data' is expected to be a list of debate records
        for entry in data:
            # Example structure of entry:
            # {
            #   "timestamp": "2025-03-01T19:05:15.078239",
            #   "topic_index": 51,
            #   "pro_model": "claude-3.5-haiku",
            #   "con_model": "gpt-4o",
            #   "debate": {...}
            # }
            t_idx = entry.get("topic_index")
            p_model = entry.get("pro_model")
            c_model = entry.get("con_model")

            if t_idx is None or p_model is None or c_model is None:
                logger.debug(f"Skipping malformed record in {file_path}: {entry}")
                continue

            records_by_topic[t_idx][(p_model, c_model)].append(entry)

            # Track model names
            all_models.add(p_model)
            all_models.add(c_model)

            if min_topic_index is None or t_idx < min_topic_index:
                min_topic_index = t_idx
            if max_topic_index is None or t_idx > max_topic_index:
                max_topic_index = t_idx

    if min_topic_index is None or max_topic_index is None:
        logger.error("No valid debate records found. Nothing to process.")
        return

    logger.info(f"Collected debate data for topics in the range: "
                f"{min_topic_index}..{max_topic_index}")

    # Sort model names for consistent output
    sorted_all_models = sorted(all_models)
    model_count_detected = len(sorted_all_models)
    logger.info(f"Detected {model_count_detected} unique models in the records: {sorted_all_models}")

    # 6. Classify debates into completed, incomplete, repeated
    completed_debates = []
    incomplete_debates = []
    repeated_debates = []

    # track topic status (for summarization)
    # possible statuses: "complete", "incomplete", "missing"
    topic_status = {}

    # For enumerating all possible pairs among the detected models:
    # If the user strictly wants the n_models from CLI to match the actual set, raise a warning.:
    if model_count_detected != n_models:
        logger.warning(f"WARNING: Found {model_count_detected} models in records but --n_models={n_models} was set.")

    # create a set of all possible (pro, con) pairs from the discovered models
    # (excluding pro==con).
    all_possible_pairs = set()
    for m1 in sorted_all_models:
        for m2 in sorted_all_models:
            if m1 != m2:
                all_possible_pairs.add((m1, m2))

    # hold "topics_to_complete" -> topic_index => list of missing pairs
    topics_to_complete = defaultdict(list)

    # Range can be the entire min..max
    all_topic_indices = range(min_topic_index, max_topic_index + 1)

    for t_idx in all_topic_indices:
        pair_dict = records_by_topic.get(t_idx, {})
        if not pair_dict:
            # Means no debates for t_idx => missing
            topic_status[t_idx] = "missing"

            missing_pairs = all_possible_pairs
            for (pm, cm) in missing_pairs:
                topics_to_complete[t_idx].append({
                    "topic_index": t_idx,
                    "pro_model": pm,
                    "con_model": cm
                })
            continue

        # gather the canonical (first) debate for each pair
        # and store duplicates (2nd, 3rd, ...) in repeated
        canonical_debates_for_topic = []
        repeated_for_topic = []

        for pair, debate_list in pair_dict.items():
            if len(debate_list) > 1:
                # store all but the first as repeated
                repeated_for_topic.extend(debate_list[1:])
            # the first is the canonical
            canonical_debates_for_topic.append(debate_list[0])

        unique_pairs_count = len(pair_dict.keys())

        # Compare with the set of all_possible_pairs
        existing_pairs = set(pair_dict.keys())
        # Pairs that do NOT appear
        missing_pairs = all_possible_pairs - existing_pairs

        # If unique_pairs_count == expected_debates_per_topic => complete
        if unique_pairs_count == expected_debates_per_topic:
            topic_status[t_idx] = "complete"
            completed_debates.extend(canonical_debates_for_topic)
        else:
            topic_status[t_idx] = "incomplete"
            incomplete_debates.extend(canonical_debates_for_topic)

        # Mark repeated
        repeated_debates.extend(repeated_for_topic)

        # If topic is incomplete or missing, track missing pairs
        if topic_status[t_idx] in ("incomplete", "missing"):
            for (pm, cm) in missing_pairs:
                topics_to_complete[t_idx].append({
                    "topic_index": t_idx,
                    "pro_model": pm,
                    "con_model": cm
                })

    # Summaries
    num_complete_topics = sum(1 for s in topic_status.values() if s == "complete")
    num_incomplete_topics = sum(1 for s in topic_status.values() if s in ("incomplete", "missing"))

    # For "missing" specifically
    missing_topic_indices = [t for t, status in topic_status.items() if status == "missing"]
    incomplete_topic_indices = [t for t, status in topic_status.items() if status == "incomplete"]

    # Repeated topics summary: gather a mapping from topic -> list of (pro, con) from repeated
    repeated_topic_pairs = defaultdict(list)
    for rdeb in repeated_debates:
        t_idx = rdeb.get("topic_index")
        pair = (rdeb.get("pro_model"), rdeb.get("con_model"))
        repeated_topic_pairs[t_idx].append(pair)

    # 7. Create output folder
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    postproc_folder_name = f"postprocess_records_{now_str}"
    postproc_folder_path = os.path.join("records", postproc_folder_name)
    os.makedirs(postproc_folder_path, exist_ok=True)

    # 8. Write out the JSON files
    # 8a. completed_debates
    completed_path = os.path.join(postproc_folder_path,
                                  f"completed_debates_{now_str}.json")
    with open(completed_path, "w", encoding="utf-8") as f:
        json.dump(completed_debates, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(completed_debates)} records to {completed_path}")

    # 8b. incomplete_debates
    incomplete_path = os.path.join(postproc_folder_path,
                                   f"incomplete_debates_{now_str}.json")
    with open(incomplete_path, "w", encoding="utf-8") as f:
        json.dump(incomplete_debates, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(incomplete_debates)} records to {incomplete_path}")

    # 8c. repeated_debates
    repeated_path = os.path.join(postproc_folder_path,
                                 f"repeated_debates_{now_str}.json")
    with open(repeated_path, "w", encoding="utf-8") as f:
        json.dump(repeated_debates, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(repeated_debates)} records to {repeated_path}")

    # 8d. postprocess_summarization
    repeated_topic_indices = sorted(repeated_topic_pairs.keys())

    # Prepare "topics_to_complete" as a dict of str => list of dict
    # (stringify topic_index as a key)
    topics_to_complete_out = {}
    for t_idx, missing_list in topics_to_complete.items():
        # Sort by pro_model, then con_model for consistency
        sorted_missing = sorted(missing_list, key=lambda x: (x["pro_model"], x["con_model"]))
        topics_to_complete_out[str(t_idx)] = sorted_missing

    summarization = {
        "number_of_completed_topics": num_complete_topics,
        "number_of_incomplete_topics": num_incomplete_topics,
        "topic_index_range": [min_topic_index, max_topic_index],
        "missing_experiments": {
            "missing_topic_indices": missing_topic_indices,
            "incomplete_topic_indices": incomplete_topic_indices
        },
        "repeated_topics": {
            str(t): repeated_topic_pairs[t] for t in repeated_topic_indices
        },
        # Provide all models discovered in the entire data set
        "all_models_detected": sorted_all_models,
        # Provide the missing debates that are needed to complete each topic
        "topics_to_complete": topics_to_complete_out
    }

    summary_path = os.path.join(postproc_folder_path, f"postprocess_summarization_{now_str}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summarization, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote postprocessing summary to {summary_path}")

    logger.info("=== Postprocessing Completed. ===")


if __name__ == "__main__":
    main()
