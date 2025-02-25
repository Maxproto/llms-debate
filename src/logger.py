"""
src/logger.py

Uses Python's logging module for info/debug messages,
while also storing debate transcripts in memory to save as JSON.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List

class DebateLogger:
    def __init__(self, output_dir: str = "logs", log_filename: str = None):
        """
        Initialize a logger. If log_filename is None, we name it 'debates.log' by default.
        Also track in-memory debate records for JSON dumping.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if not log_filename:
            log_filename = "debates.log"
        self.log_path = os.path.join(self.output_dir, log_filename)

        # Setup python logging
        self.logger = logging.getLogger("DebateLogger")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicating handlers if re-init
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # (Optional) console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.debates: List[Dict[str, Any]] = []

    def info(self, msg: str):
        """
        Shortcut to log an INFO-level message.
        """
        self.logger.info(msg)

    def error(self, msg: str):
        """
        Shortcut to log an ERROR-level message.
        """
        self.logger.error(msg)

    def debug(self, msg: str):
        """
        Shortcut to log a DEBUG-level message.
        """
        self.logger.debug(msg)

    def log_debate(self, debate_data: Dict[str, Any], pro_model: str, con_model: str, topic_index: int):
        """
        Store the debate record plus metadata about models, topic, etc.
        Then log an INFO message about it.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "topic_index": topic_index,
            "pro_model": pro_model,
            "con_model": con_model,
            "debate": debate_data
        }
        self.debates.append(record)
        self.logger.info(f"Logged debate: topic_index={topic_index}, pro={pro_model}, con={con_model}")

    def save_partial_json(self, filename: str = None):
        """
        Writes all stored debates to a JSON file, but does not clear them.
        Good for partial/in-progress saving.
        """
        if not filename:
            filename = "debates_inprogress.json"
        out_path = os.path.join(self.output_dir, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.debates, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved partial {len(self.debates)} debates to {out_path}")

    def finalize_json(self, filename: str = None):
        """
        Writes all stored debates to a JSON file and clears them.
        For final output after all runs are done.
        """
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"debates_{timestamp}.json"
        out_path = os.path.join(self.output_dir, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.debates, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Finalized {len(self.debates)} debates to {out_path}")
        self.debates.clear()
