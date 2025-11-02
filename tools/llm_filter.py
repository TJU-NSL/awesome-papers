from __future__ import annotations

import json
import time
from typing import List

from openai import OpenAI, RateLimitError, APIConnectionError

from config import API_KEY, BASE_URL, MODEL, logger
from utils import SYSTEM_PROMPT, USER_PROMPT, TAGS


def llm_filter(papers: List[dict]) -> List[dict]:
    """Use an LLM to determine relevance/tags and generate TL;DR.

    Input papers must include: title, abstract, link.
    Adds keys: `relevant` (bool), `tags` (list[str]|absent), `tldr` (str|absent).
    """
    logger.info(f"LLM filter started for {len(papers)} papers")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for p in papers:
        message = [
            {"role": "system", "content": SYSTEM_PROMPT.replace("{tag_descriptions}", json.dumps(TAGS))},
            {"role": "user", "content": USER_PROMPT.format(title=p.get("title", ""), abstract=p.get("abstract", ""))},
        ]
        while True:
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=message,
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                break
            except RateLimitError as e:
                logger.warning(f"RateLimitError; retrying in 10s: {e}")
                time.sleep(10)
            except APIConnectionError as e:
                logger.warning(f"APIConnectionError; retrying in 5s: {e}")
                time.sleep(5)

        raw = resp.choices[0].message.content
        logger.debug(f"[LLM] title={p.get('title')} | output={raw}")

        try:
            result = json.loads(raw)
            p.update(result)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for paper {p.get('title')}: {raw}")
            p.update({"relevant": False})

    return papers
