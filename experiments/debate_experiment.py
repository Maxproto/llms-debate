"""
experiments/debate_experiment.py

Debate experiment script:
- Parse CLI for topic indices, model list, rounds, max_tokens
- Load topics, slice them, generate pairings
- For each pairing, run debate, log partial results
- Rename final file after done
- If --complete_from is provided, the script reads the "topics_to_complete" from
  a postprocess_summarization_{...}.json file and only runs those missing debates
  rather than generating pairings.
"""

import sys
import os
import argparse
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_debate_topics, generate_debate_pairings
from src.logger import DebateLogger
from src.runner import run_debate


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debate Batch Experiment CLI")
    parser.add_argument("--start", type=int, default=None,
                        help="Start topic index (inclusive). Default=0.")
    parser.add_argument("--end", type=int, default=None,
                        help="End topic index (inclusive). Default=last topic.")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of rebuttal rounds. Default=3.")
    parser.add_argument("--models", nargs='+', default=None,
                        help="List of model names (space-separated). "
                             "Default=all supported models: gpt-4o, claude-3.5-haiku, "
                             "mistral-small-latest, llama-3.2-3b, gemini-2.0-flash.")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Max tokens per model response. Default=200.")
    parser.add_argument("--complete_from", type=str, default=None,
                        help="Path to a postprocess_summarization_{...}.json file. "
                             "If provided, this script will run only the missing debates "
                             "specified under 'topics_to_complete' in that file, ignoring "
                             "--start, --end, and --models.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Prepare logger
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger = DebateLogger(record_dir="records")
    logger.info("=== Starting Debate Batch Experiment ===")

    # Load the full list of topics from the data file
    topic_file = os.path.join("data", "debate_topics.txt")
    all_topics = load_debate_topics(topic_file)
    num_topics = len(all_topics)
    logger.info(f"Loaded {num_topics} topics from {topic_file}")

    rounds = args.rounds
    max_tokens = args.max_tokens
    logger.info(f"Rounds = {rounds}, max_tokens = {max_tokens}")

    # If --complete_from is provided, run only the missing debates from that JSON
    if args.complete_from is not None:
        comp_file_path = args.complete_from
        if not os.path.exists(comp_file_path):
            logger.error(f"Cannot find --complete_from file: {comp_file_path}")
            return

        logger.info(f"Running completion mode from file: {comp_file_path}")

        with open(comp_file_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # "topics_to_complete": { "51": [ {"topic_index": 51, "pro_model": "...", "con_model":"..."} ], ...}
        topics_to_complete = summary_data.get("topics_to_complete", {})
        # create a list of all required pairs
        all_missing_debates = []
        for t_idx_str, pair_list in topics_to_complete.items():
            t_idx = int(t_idx_str)
            if not pair_list:
                continue
            for pair_entry in pair_list:
                # expected keys: topic_index, pro_model, con_model
                all_missing_debates.append(pair_entry)

        total_debates = len(all_missing_debates)
        logger.info(f"Total missing debates to run: {total_debates}")

        # store partial results in "debates_incomplete_inprogress.json"
        partial_file = f"debates_incomplete_inprogress_{now_str}.json"
        logger.info(f"Storing partial results in {partial_file}")

        debate_count = 0
        for debate_req in all_missing_debates:
            debate_count += 1
            t_idx = debate_req["topic_index"]
            pro = debate_req["pro_model"]
            con = debate_req["con_model"]
            if t_idx < 0 or t_idx >= num_topics:
                logger.error(f"Skipping invalid topic_index={t_idx}")
                continue

            topic_text = all_topics[t_idx]
            logger.info(f"Debate {debate_count}/{total_debates}: topic_idx={t_idx}, pro={pro}, con={con}")

            try:
                # run the debate
                debate_data = run_debate(topic_text, pro, con, rounds=rounds)
                # log the result
                logger.log_debate(debate_data, pro, con, topic_index=t_idx)
                # save partial JSON
                logger.save_partial_json(filename=partial_file)
            except Exception as e:
                logger.error(f"Debate failed: topic_idx={t_idx}, pro={pro}, con={con}, error={str(e)}")

        logger.info(f"All missing debates completed. Partial results in records/{partial_file}")

        final_filename = f"debates_incomplete_{now_str}.json"
        inprog_path = os.path.join("records", partial_file)
        final_path = os.path.join("records", final_filename)
        if os.path.exists(inprog_path):
            os.rename(inprog_path, final_path)
            logger.info(f"Renamed partial log to final: {final_path}")
        else:
            logger.error("No partial log found at the end of run (incomplete run?).")

        logger.info("=== Debate Batch Experiment FINISHED (Completion Mode) ===")
        return

    # Otherwise, run the normal approach

    # 1. Determine topic range
    start_idx = 0 if args.start is None else args.start
    end_idx = (num_topics - 1) if args.end is None else args.end
    if start_idx < 0 or end_idx >= num_topics or start_idx > end_idx:
        logger.error(f"Invalid index range: start={start_idx}, end={end_idx}, total={num_topics}")
        return
    logger.info(f"Using topic indices {start_idx}..{end_idx}")

    # 2. Models
    if args.models is None:
        model_list = ["gpt-4o", "claude-3.5-haiku", "mistral-small-latest",
                      "llama-3.2-3b", "gemini-2.0-flash"]
    else:
        model_list = args.models
    logger.info(f"Models: {model_list}")

    # 3. Generate pairings
    pairings = generate_debate_pairings(model_list, all_topics, start_idx, end_idx)
    total_debates = len(pairings)
    logger.info(f"Total debates to run: {total_debates}")

    # 4. Store partial results in "debates_inprogress.json"
    partial_file = f'debates_inprogress_{now_str}.json'
    logger.info(f"Storing partial results in {partial_file}")

    # 5. For each pairing, run the debate
    debate_count = 0
    for pairing in pairings:
        debate_count += 1
        t_idx = pairing["topic_index"]
        topic_text = pairing["topic"]
        pro = pairing["pro"]
        con = pairing["con"]

        logger.info(f"Debate {debate_count}/{total_debates}: topic_idx={t_idx}, pro={pro}, con={con}")
        try:
            # run the debate
            debate_data = run_debate(topic_text, pro, con, rounds=rounds)
            # log the result
            logger.log_debate(debate_data, pro, con, topic_index=t_idx)
            # save partial JSON
            logger.save_partial_json(filename=partial_file)
        except Exception as e:
            logger.error(f"Debate failed: topic_idx={t_idx}, pro={pro}, con={con}, error={str(e)}")

    logger.info(f"All debates completed. Partial results in records/{partial_file}")

    # 6. Final rename with model & range info
    short_models = "-".join([m.split("-")[0] for m in model_list])
    final_filename = f"debates_{short_models}_{start_idx}to{end_idx}_{now_str}.json"
    inprog_path = os.path.join("records", partial_file)
    final_path = os.path.join("records", final_filename)
    if os.path.exists(inprog_path):
        os.rename(inprog_path, final_path)
        logger.info(f"Renamed partial log to final: {final_path}")
    else:
        logger.error("No partial log found at the end of run")

    logger.info("=== Debate Batch Experiment FINISHED (Normal Mode) ===")


if __name__ == "__main__":
    main()
