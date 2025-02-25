"""
src/runner.py

Orchestrates a single debate between two models for a given topic.
"""

from src.agent import DebateAgentLC

def run_debate(topic: str, pro_model: str, con_model: str, rounds: int = 3):
    """
    Conduct a multi-turn debate and return a transcript record.
    :param topic: The debate topic
    :param pro_model: model name for the Pro side
    :param con_model: model name for the Con side
    :param rounds: how many back-and-forth rebuttals
    :return: a dict describing the entire debate transcript
    """
    debate_record = {
        "topic": topic,
        "pro_model": pro_model,
        "con_model": con_model,
        "turns": []
    }

    # Create Agents
    pro_agent = DebateAgentLC("ProAgent", pro_model, f"You are PRO on the topic: '{topic}'...")
    con_agent = DebateAgentLC("ConAgent", con_model, f"You are CON on the topic: '{topic}'...")

    # 1. Openings
    pro_opening = pro_agent.respond("Please present your opening statement.")
    debate_record["turns"].append({
        "stage": "opening",
        "agent": "ProAgent",
        "content": pro_opening
    })

    con_opening = con_agent.respond(f"ProAgent said: {pro_opening}\nPlease present your opening statement.")
    debate_record["turns"].append({
        "stage": "opening",
        "agent": "ConAgent",
        "content": con_opening
    })

    # 2. Rebuttal rounds
    last_con_msg = con_opening
    for r in range(1, rounds + 1):
        pro_rebuttal = pro_agent.respond(f"ConAgent's last argument: {last_con_msg}\nYour rebuttal:")
        debate_record["turns"].append({
            "stage": f"rebuttal_{r}",
            "agent": "ProAgent",
            "content": pro_rebuttal
        })

        con_rebuttal = con_agent.respond(f"ProAgent said: {pro_rebuttal}\nYour rebuttal:")
        debate_record["turns"].append({
            "stage": f"rebuttal_{r}",
            "agent": "ConAgent",
            "content": con_rebuttal
        })
        last_con_msg = con_rebuttal

    # 3. Closing
    pro_closing = pro_agent.respond("Now give your closing statement.")
    debate_record["turns"].append({
        "stage": "closing",
        "agent": "ProAgent",
        "content": pro_closing
    })

    con_closing = con_agent.respond(f"ProAgent's closing: {pro_closing}\nNow your closing:")
    debate_record["turns"].append({
        "stage": "closing",
        "agent": "ConAgent",
        "content": con_closing
    })

    return debate_record
