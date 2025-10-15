from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from config import DEFAULT_START_DATE, README_FILE, logger
from utils import README_HEADER


def _parse_date(text: str, fmt: str) -> datetime:
    return datetime.strptime(text, fmt)


def get_date_range() -> tuple[str | None, str | None]:
    """Determine the [start, end] date window to fetch.
      - Start = the day after the latest date in README (or DEFAULT_START_DATE if the file is missing)
      - End = yesterday (local machine time) in YYYYMMDD
    """
    start = DEFAULT_START_DATE
    end = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")

    # Read README to find the latest date header
    if README_FILE.exists():
        for line in README_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("### "):  # slice YYYY-MM-DD then +1 day
                start = (_parse_date(line[4:14], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
                break

    if start > end:
        logger.error(f"start_date ({start}) is later than end_date ({end}); skip update")
        return None, None

    return start, end


def update_daily_arxiv(papers: List[dict], date: str) -> None:
    """Prepend a new section into README with filtered papers for a given date.
    - Skips creating an empty section so the next run can retry the same date
    """
    logger.info(f"Updating README for date={date}")

    date_str = _parse_date(date, "%Y%m%d").strftime("%Y-%m-%d")

    lines: list[str] = []
    if README_FILE.exists():
        lines = README_FILE.read_text(encoding="utf-8").splitlines(keepends=True)

    # 1) Drop the previous header block (everything before first ###)
    for i, line in enumerate(lines):
        if line.startswith("###"):
            lines = lines[i:]
            break

    # 2) Build new section (only if we have at least one relevant paper)
    new_section: list[str] = []
    if papers:
        section_lines: list[str] = [f"### {date_str}\n"]
        for p in papers:
            if not p.get("relevant", False):
                continue
            tags = p.get("tags", []) or []
            title = p["title"]
            link = p["link"]
            if tags:
                section_lines.append(
                    "* " + " ".join(f"`{t}`" for t in tags) + f" [{title}]({link})\n"
                )
            else:
                section_lines.append(f"* [{title}]({link})\n")
            if tldr := p.get("tldr", ""):
                section_lines.append(f"  > **TL;DR**: {tldr}\n")
        if len(section_lines) > 1:  # at least header + 1 paper
            new_section = section_lines + ["\n"]

    # 3) Count papers currently in content & rebuild header
    paper_count = sum(1 for l in (new_section + lines) if l.startswith("* "))
    header = README_HEADER.format(papers=paper_count, update=date_str.replace("-", "."))

    # 5) Write back
    README_FILE.write_text(header + "".join(new_section + lines), encoding="utf-8")

    logger.info(f"README updated: {paper_count} papers total")
