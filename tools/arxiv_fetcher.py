from __future__ import annotations

import time
import arxiv
from typing import Iterable, List
from config import logger


def _ensure_list(categories: Iterable[str] | str) -> list[str]:
    return [categories] if isinstance(categories, str) else list(categories)


def fetch_arxiv_papers(
    categories: Iterable[str] | str,
    start_date: str,
    end_date: str | None = None,
) -> List[dict]:
    """Fetch papers from arXiv within a date window (inclusive of start day).

    Args:
        categories: e.g. ["cs.DC", "cs.OS"] or "cs.DC".
        start_date: YYYYMMDD.
        end_date: YYYYMMDD. If None, equals start_date.

    Returns:
        List of dicts with fields: title, link, abstract, authors, categories, id
    """
    cats = _ensure_list(categories)
    end = end_date or start_date

    logger.info(f"Fetching arXiv papers for {cats} from {start_date} to {end}")

    papers: list[dict] = []
    client = arxiv.Client(page_size=200)

    for cat in cats:
        # arXiv range filter uses HHMM; we construct [YYYYMMDD0000, (end+1)0000)
        search = arxiv.Search(
            query=f"cat:{cat} AND submittedDate:[{start_date}0000 TO {str(int(end) + 1)}0000]",
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            max_results=5000,
        )
        logger.debug(f"Search: {search}")

        results: list[arxiv.Result] = []
        # Retry modestly to tolerate transient network hiccups
        for attempt in range(5):
            try:
                results = list(client.results(search))
                break
            except Exception as exc:
                logger.warning(f"Attempt {attempt+1} failed: {exc}")
                time.sleep(3)

        for r in results:
            logger.debug(f"Found paper: {r.title}")
            papers.append(
                {
                    "title": r.title,
                    "link": r.entry_id,
                    "abstract": r.summary,
                    "authors": [a.name for a in r.authors],
                    "categories": r.categories,
                    "id": r.entry_id,
                }
            )

    logger.info(f"Fetched {len(papers)} papers total")
    return papers
