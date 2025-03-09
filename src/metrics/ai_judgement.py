"""
src/eval_metrics/ai_judgement.py

AI judge that reads a debate, removes references to agent identity.
"""

import logging
from typing import List, Dict, Any

from src.models import create_langchain_llm

def ai_judgement(
    turns: List[Dict[str, Any]],
    judge_model_name: str,
    judge_max_tokens: int = 400,
    topic: str = ""
) -> str:
    logger = logging.getLogger("evaluation")

    # Build debate text & Map each agent name to Debater 1, Debater 2, etc.
    agent_map = {}
    next_label_id = 1
    lines = []
    for turn in turns:
        agent_name = turn.get("agent", "")
        if agent_name not in agent_map:
            agent_map[agent_name] = f"Debater {next_label_id}"
            next_label_id += 1
        label = agent_map[agent_name]
        stage = turn.get("stage", "")
        text = turn.get("content", "")
        lines.append(f"{label} ({stage}): {text}")

    debate_text = "\n".join(lines)

    messages = [
        (
            "system",
            f"You are an impartial judge of a debate.\n"
            f"Topic: {topic}\n"
            f"Decide which debater made the stronger argument or if it's a tie. Provide your reasoning.\n"
            f"Answers in the following format: 'Winner: Debater 1' or 'Winner: Debater 2' or 'Winner: Tie'. Because...",
        ),
        ("human", "Debate Transcript:\n" + debate_text + "\n\nWho won and why?"),
    ]

    try:
        llm = create_langchain_llm(
            model_name=judge_model_name,
            temperature=0.0, # temperature=0 for deterministic judgement
            max_tokens=judge_max_tokens
        )
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        logger.error(f"Error in ai_judgement with model={judge_model_name}: {e}")
        return f"Error: {e}"
