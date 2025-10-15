from __future__ import annotations

from datetime import datetime, timedelta

from arxiv_fetcher import fetch_arxiv_papers
from config import ARXIV_CATEGORIES, logger
from llm_filter import llm_filter
from updater import get_date_range, update_daily_arxiv


def _daterange(start: str, end: str):
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    for i in range((e - s).days + 1):
        yield (s + timedelta(days=i)).strftime("%Y%m%d")


def run() -> None:
    start, end = get_date_range()
    if not (start and end):
        return

    logger.info(f"Processing dates from {start} to {end}")
    for day in _daterange(start, end):
        papers = fetch_arxiv_papers(categories=ARXIV_CATEGORIES, start_date=day)
        results = llm_filter(papers)
        update_daily_arxiv(papers=results, date=day)


if __name__ == "__main__":
    run()
