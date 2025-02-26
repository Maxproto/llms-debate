"""
src/utils.py

1) load_debate_topics(): Load topics from file.
2) generate_debate_pairings(): Build all (pro, con) combos for each selected topic range.
"""

import os
from typing import List, Dict, Any

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

def generate_debate_pairings(models: List[str], topics: List[str], start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
    """
    For each topic in topics[start_idx..end_idx], generate debate pairings of all distinct model pairs.
    Each pair (A,B) yields 2 debates: (pro=A, con=B) and (pro=B, con=A).

    Returns a list of dicts:
    [
      {
        "topic_index": ...,
        "topic": "some topic",
        "pro": "modelA",
        "con": "modelB"
      },
      ...
    ]
    """
    selected_topics = topics[start_idx : end_idx + 1]

    # Build all unique model pairs
    pairings = []
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            # forward
            pairings.append((models[i], models[j]))
            # reverse
            pairings.append((models[j], models[i]))

    # Combine each selected topic with each model pair
    result = []
    for idx, topic in enumerate(selected_topics, start=start_idx):
        for (A, B) in pairings:
            result.append({
                "topic_index": idx,
                "topic": topic,
                "pro": A,
                "con": B
            })
    return result