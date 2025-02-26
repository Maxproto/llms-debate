"""
src/eval_metrics/consistency.py

Checks if a debater contradicts themselves.
Naive approach: search for contradictory phrases in the same agent's text.
"""

def compute_agent_consistency(turns, agent_name):
    agent_text = ""
    for t in turns:
        if t.get("agent", "") == agent_name:
            agent_text += " " + t.get("content", "").lower()
    contradictory_pairs = [
        ("athens is better", "athens is worse"),
        ("sparta is better", "sparta is worse")
    ]
    penalty = 0
    for (p1, p2) in contradictory_pairs:
        if p1 in agent_text and p2 in agent_text:
            penalty += 1
    if penalty == 0:
        return 1.0
    return max(0.0, 1.0 - 0.5 * penalty)
