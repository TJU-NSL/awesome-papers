from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Final, List
from dotenv import load_dotenv

load_dotenv()  # Load .env for local runs (no-op in CI when env vars already exist)

# --- Basic Setup --------------------------------------------------------------
README_FILE = Path("daily-arxiv-llm.md")

# FIXME: Adding cs.LG/cs.AI can bring in many irrelevant papers (need stronger LLM filtering)
ARXIV_CATEGORIES: List[str] = ["cs.DC", "cs.OS"]

DEFAULT_START_DATE = "20250101"  # YYYYMMDD

# Use SiliconFlow by default; keep names consistent with original script
API_KEY = os.environ["API_KEY"]  # required
BASE_URL = "https://api.siliconflow.cn/v1"
# MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MODEL = "deepseek-ai/DeepSeek-V3.1-Terminus"

# TODO: Subscribers (e-mail) ---------------------------------------------------
SUBSCRIBER: Final[dict[str, list[str]]] = {
    "zhixin@abc.com": ["thinking", "serving"],
    "test@tju.edu.cn": ["offloading", "RL"]
}


# --- Logging ------------------------------------------------------------------
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
# logger = logging.getLogger("daily-arxiv-llm")

class LOGGER:
    # fallback to print since logging will output too much information
    def debug(self, msg: str):
        print("[DEBUG] ({}): {}".format(time.strftime("%Y%m%d-%H%M%S", time.localtime()), msg))

    def info(self, msg: str):
        print("[INFO] ({}): {}".format(time.strftime("%Y%m%d-%H%M%S", time.localtime()), msg))

    def warning(self, msg: str):
        print("[WARNING] ({}): {}".format(time.strftime("%Y%m%d-%H%M%S", time.localtime()), msg))

    def error(self, msg: str):
        print("[ERROR] ({}): {}".format(time.strftime("%Y%m%d-%H%M%S", time.localtime()), msg))


logger = LOGGER()
