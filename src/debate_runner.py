'''
src/debate_runner.py

Defines a run_debate function that orchestrates a debate between two agents.
'''

from src.debate_agent_lc import DebateAgentLC

def run_debate(topic: str, pro_model: str, con_model: str, rounds: int = 3):
    debate_record = {
        "topic": topic,
        "pro_model": pro_model,
        "con_model": con_model,
        "turns": []
    }

    pro_agent = DebateAgentLC("ProAgent", pro_model, f"You are PRO on the topic: '{topic}'...")
    con_agent = DebateAgentLC("ConAgent", con_model, f"You are CON on the topic: '{topic}'...")

    # 1. Openings
    pro_opening = pro_agent.respond("Please present your opening statement.")
    debate_record["turns"].append({
        "stage": "opening",
        "agent": "ProAgent",
        "content": pro_opening
    })

    # Con sees Pro's statement as user input in next turn
    con_opening = con_agent.respond(f"ProAgent said: {pro_opening}\nPlease present your opening statement.")
    debate_record["turns"].append({
        "stage": "opening",
        "agent": "ConAgent",
        "content": con_opening
    })

    # 2. Rebuttals
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