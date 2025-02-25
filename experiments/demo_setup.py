"""
experiments/demo_setup.py

Quick demo to test loading topics and generating pairings.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_debate_topics, generate_debate_pairings

def main():
    # 1. Load topics
    topic_file = os.path.join("data", "debate_topics.txt")
    topics = load_debate_topics(topic_file)
    print(f"Loaded {len(topics)} topics from {topic_file}")

    # 2. Define models
    models = [
        'gpt-4o',
        'claude-3.5-haiku',
        'mistral-small-latest',
        'llama-3.2-3b',
        'gemini-2.0-flash'
    ]

    # 3. Generate pairings for demonstration
    pairings = generate_debate_pairings(models, topics)
    print(f"Total debates to schedule: {len(pairings)}")

    # Print the first 2 pairings as a sample
    for i in range(min(2, len(pairings))):
        print(pairings[i])

if __name__ == "__main__":
    main()
