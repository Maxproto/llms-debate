"""
experiments/demo_test.py

Demonstrates the full debate pipeline:
- Runs a debate via run_debate (from src/debate_runner)
- Logs the results using DebateLogger (from src/debate_logger)
- Prints the final debate data as JSON
"""

import sys
import os
import json

# Make sure Python can find src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from runner import run_debate
from logger import DebateLogger

def main():
    # Example topic and model names
    topic = "Should we adopt universal basic income?"
    pro_model = "gpt-4o"
    con_model = "gemini-2.0-flash"

    # 1. Run the debate with 2 rounds of rebuttals
    debate_data = run_debate(topic, pro_model, con_model, rounds=2)

    # 2. Log the result
    logger = DebateLogger(output_dir="logs")
    logger.log_debate(debate_data, pro_model, con_model, topic_index=1)
    logger.save_to_json()  # creates a file like logs/debates_20250101_123000.json

    # 3. Print the debate record to console for inspection
    print("=== Final Debate Data ===")
    print(json.dumps(debate_data, indent=2))

if __name__ == "__main__":
    main()
