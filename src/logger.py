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

class GlobalLogger:
    _loggers = {}

    @classmethod
    def get_logger(cls, name: str = "global", level: int = logging.INFO):
        if name in cls._loggers:
            return cls._loggers[name]
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(console_handler)
        cls._loggers[name] = logger
        logger.info(f"Initialized logger {name}, output to {log_file}.")
        return logger

class DebateLogger:
    def __init__(self, record_dir: str = "records"):
        self.record_dir = record_dir
        self.logger = GlobalLogger.get_logger("Debate")
        self.debates: List[Dict[str, Any]] = []

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
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
        """
        if not filename:
            filename = "debates_inprogress.json"
        out_path = os.path.join(self.record_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.debates, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved partial {len(self.debates)} debates to {out_path}")

    def finalize_json(self, filename: str = None):
        """
        Writes all stored debates to a JSON file and clears them.
        """
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"debates_{timestamp}.json"
        out_path = os.path.join(self.record_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.debates, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Finalized {len(self.debates)} debates to {out_path}")
        self.debates.clear()
