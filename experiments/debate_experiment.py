"""
experiments/debate_experiment.py

Debate experiment script:
- Parse CLI for topic indices, model list, rounds, max_tokens
- Load topics, slice them, generate pairings
- For each pairing, run debate, log partial results
- Rename final file after done
"""

import sys
import os
import argparse
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
                        help="List of model names (space-separated). Default=all supported models: gpt-4o,"
                         " claude-3.5-haiku, mistral-small-latest, llama-3.2-3b, gemini-2.0-flash.")
    parser.add_argument("--max_tokens", type=int, default=400,
                        help="Max tokens per model response. Default=400.")
    return parser

def main():
    # 1. Parse CLI
    parser = get_parser()
    args = parser.parse_args()

    # 2. Prepare a logger
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger = DebateLogger(log_dir="logs", record_dir="records", log_filename=f'batch_experiment_{now_str}.log')
    logger.info("=== Starting Debate Batch Experiment ===")
    logger.info(f"Storing progress log in logs/batch_experiment_{now_str}.log")

    # 3. Load topics
    topic_file = os.path.join("data", "debate_topics.txt")
    all_topics = load_debate_topics(topic_file)
    num_topics = len(all_topics)
    logger.info(f"Loaded {num_topics} total topics from {topic_file}")

    # 4. Determine topic range
    start_idx = 0 if args.start is None else args.start
    end_idx = (num_topics - 1) if args.end is None else args.end
    if start_idx < 0 or end_idx >= num_topics or start_idx > end_idx:
        logger.error(f"Invalid index range: start={start_idx}, end={end_idx}, total={num_topics}")
        return
    logger.info(f"Using topic indices {start_idx}..{end_idx}")

    # 5. Models
    if args.models is None:
        model_list = ["gpt-4o", "claude-3.5-haiku", "mistral-small-latest", "llama-3.2-3b", "gemini-2.0-flash"]
    else:
        model_list = args.models
    logger.info(f"Models: {model_list}")

    rounds = args.rounds
    max_tokens = args.max_tokens
    logger.info(f"Rounds = {rounds}, max_tokens = {max_tokens}")

    # 6. Generate pairings
    pairings = generate_debate_pairings(model_list, all_topics, start_idx, end_idx)
    total_debates = len(pairings)
    logger.info(f"Total debates to run: {total_debates}")

    # 7. Store partial results in "debates_inprogress.json"
    partial_file = f'debates_inprogress_{now_str}.json'
    logger.info(f"Storing partial results in {partial_file}")

    # 8. For each pairing, run the debate
    debate_count = 0
    for pairing in pairings:
        debate_count += 1
        t_idx = pairing["topic_index"]
        topic = pairing["topic"]
        pro = pairing["pro"]
        con = pairing["con"]

        logger.info(f"Debate {debate_count}/{total_debates}: topic_idx={t_idx}, pro={pro}, con={con}")
        try:
            # run the debate
            debate_data = run_debate(topic, pro, con, rounds=rounds)
            # log the result
            logger.log_debate(debate_data, pro, con, topic_index=t_idx)
            # save partial JSON
            logger.save_partial_json(filename=partial_file)
        except Exception as e:
            logger.error(f"Debate failed: topic_idx={t_idx}, pro={pro}, con={con}, error={str(e)}")

    logger.info(f"All debates completed. Partial results in records/debates_inprogress_{now_str}.json")

    # 9. Final rename with model & range info
    short_models = "-".join([m.split("-")[0] for m in model_list])
    final_filename = f"debates_{short_models}_{start_idx}to{end_idx}_{now_str}.json"
    inprog_path = os.path.join("records", partial_file)
    final_path = os.path.join("records", final_filename)
    if os.path.exists(inprog_path):
        os.rename(inprog_path, final_path)
        logger.info(f"Renamed partial log to final: {final_path}")
    else:
        logger.error("No partial log found at the end of run")

    logger.info("=== Debate Batch Experiment FINISHED ===")

if __name__ == "__main__":
    main()
