"""
src/debate_logger.py

Logs each debate's transcript to JSON.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List

class DebateLogger:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.debates: List[Dict[str, Any]] = []

    def log_debate(self, debate_data: Dict[str, Any], pro_model: str, con_model: str, topic_index: int):
        """
        Store the debate record plus metadata about models, topic, etc.
        debate_data is the dict returned by run_debate().
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "topic_index": topic_index,
            "pro_model": pro_model,
            "con_model": con_model,
            "debate": debate_data
        }
        self.debates.append(record)

    def save_to_json(self, filename: str = None):
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"debates_{timestamp}.json"
        out_path = os.path.join(self.output_dir, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.debates, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(self.debates)} debates to {out_path}")
        self.debates.clear()