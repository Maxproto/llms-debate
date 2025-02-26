"""
src/metrics/coherence.py

Computes a simple embedding-based coherence score for a multi-turn debate
by measuring average similarity between consecutive turns.
"""

from numpy import dot, linalg

def compute_agent_coherence(turns, agent_name, embedder):
    """
    Compute coherence for that agent's consecutive turns.
    """
    
    if embedder is None:
        return -1.0

    # Gather agent's consecutive turns
    agent_turns = [t for t in turns if t.get("agent", "") == agent_name]
    if len(agent_turns) < 2:
        return 1.0

    texts = [at["content"] for at in agent_turns]
    embs = embedder.encode(texts, show_progress_bar=False)
    sims = []
    for i in range(len(embs) - 1):
        v1, v2 = embs[i], embs[i+1]
        denom = (linalg.norm(v1) * linalg.norm(v2))
        if denom < 1e-9:
            sims.append(0.0)
        else:
            sims.append(float(dot(v1, v2) / denom))
    if not sims:
        return 1.0
    return float(sum(sims) / len(sims))
