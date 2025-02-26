"""
src/metrics/coverage.py

Computes a simple 'rebuttal coverage' metric: how often a rebuttal references
the last opponent statement.
"""

def compute_agent_coverage(turns, agent_name):
    """
    Compute coverage for that agent's rebuttal turns.
    """
    coverage_count = 0
    rebuttals = 0
    last_speaker = None
    for turn in turns:
        stage = turn.get("stage", "")
        content = turn.get("content", "")
        agent = turn.get("agent", "")
        if agent == agent_name and stage.startswith("rebuttal"):
            rebuttals += 1
            if last_speaker and (last_speaker.lower() in content.lower() or
                                 "opponent" in content.lower() or
                                 "said" in content.lower()):
                coverage_count += 1
        last_speaker = agent
    if rebuttals == 0:
        return 1.0
    return coverage_count / rebuttals