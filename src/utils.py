"""
src/utils.py

Utility functions for loading debate topics and generating pairings.
"""

import os

# Utility functions for loading debate topics.
def load_debate_topics(file_path: str) -> list:
    """
    Load debate topics from a text file, returning a list of non-empty lines.
    Example: each line in 'data/debate_topics.txt' is one topic.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Topics file not found: {file_path}")
    
    topics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            topic = line.strip()
            if topic:
                topics.append(topic)
    return topics


# Utility functions for generating debate pairings among multiple models.
def generate_debate_pairings(models: list, topics: list) -> list:
    """
    For EACH topic, every pair of models debates twice:
      1) modelA = pro, modelB = con
      2) modelB = pro, modelA = con

    Returns a list of dictionaries. Each dictionary has:
    {
      "topic": <string>,
      "pro": <model_name>,
      "con": <model_name>
    }
    """
    n = len(models)
    # All unique pairs (round-robin combos)
    pairs = [
        (models[i], models[j]) 
        for i in range(n) 
        for j in range(i + 1, n)
    ]
    
    pairings = []
    for topic in topics:
        for A, B in pairs:
            pairings.append({"topic": topic, "pro": A, "con": B})
            pairings.append({"topic": topic, "pro": B, "con": A})
    
    return pairings