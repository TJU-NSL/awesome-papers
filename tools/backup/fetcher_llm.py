import os
import json
import time
import arxiv
import dotenv
import logging
from openai import OpenAI, RateLimitError
from typing import List
from pprint import pprint
from datetime import timedelta, datetime
from utils import *

# logging.basicConfig(level=logging.DEBUG)
stime = datetime.strptime
dotenv.load_dotenv()  # for local testing

# TODO: Add "cs.LG" or "cs.AI" will bring too many irrelevant papers (requires stronger LLM filter)
ARXIV_CATEGORIES = ["cs.DC", "cs.OS"]

# TODO: keyword subscription (email notification)
SUBSCRIBER = {"zhixin@abc.com": ["video-generation", "RL", "parallelism", "thinking", "serving", "offloading"]}

README_FILE = 'daily-arxiv-llm.md'
API_KEY = os.environ['API_KEY']  # use siliconflow API by default, set the key through github secrets
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


def get_date_range():
    start_date = "20250101"
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

    lines = open(README_FILE, 'r', encoding='utf-8').readlines()
    for line in lines:
        if line.startswith("### "):
            start_date = (stime(line[4:14], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
            break

    # check start date is earlier than end date
    if start_date > end_date:
        print("[ERROR] start_date is later than end_date, skip update papers: ", start_date, end_date)
        return None, None

    return start_date, end_date


def update_daily_arxiv(papers: List[dict], date: str):
    # papers: papers with title, link, authors, abstract, relevant, [tags, tldr] properties
    # date: format "YYYYMMDD"

    print('[INFO] start update_daily_arxiv for date:', date)

    date = stime(date, "%Y%m%d").strftime("%Y-%m-%d")
    content = open(README_FILE, 'r', encoding='utf-8').readlines()

    # step-1: remove previous headers
    for i, line in enumerate(content):
        if line.startswith("###"):
            content = content[i:]
            break

    # step-2: add new papers
    """ format:
    ### YYYY-MM-DD
    * `tag-1` `tag-2` [title](link)
      > tldr
    """
    if len(papers) > 0:  # do not add empty section to prevent fetch failure (then the next day could fetch again)
        new_section = [f"### {date}\n"]
        for p in papers:
            if not p.get("relevant", False):
                continue
            pprint(p)
            if p.get("tags", []):
                new_section.append(f"* " + " ".join([f"`{tag}`" for tag in p.get("tags", [])]) + f" [{p['title']}]({p['link']})\n")
            else:
                new_section.append(f"* [{p['title']}]({p['link']})\n")
            if p.get("tldr", ""):
                new_section.append(f"  > **TL;DR**: {p['tldr']}\n")
        content = new_section + ["\n"] + content

    # step-3: count papers
    number_papers = sum(1 for line in content if line.startswith("* "))

    # step-4: update headers
    content = [README_HEADER.format(papers=number_papers, update=date.replace('-', '.'))] + content

    # step-5: write back to file
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.writelines(content)

    print(f'[INFO] Successfully updated {number_papers} papers')


def llm_filter(papers: List[dict]) -> List[dict]:
    """
    Use LLM to filter papers relevant to LLM systems, and provide tags+TLDR for papers.
    :param papers: list of papers with title, abstract and link properties.
    :return: papers with relevant (true/false), tags (None or List[str]) and TLDR (str) properties.
    """
    print('[INFO] start llm_filter for', len(papers), 'papers')
    client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
    for p in papers:
        # handle rate limit
        while True:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT.replace("{tag_descriptions}", json.dumps(TAGS))},
                        {"role": "user", "content": USER_PROMPT.format(title=p["title"], abstract=p["abstract"])}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                break
            except RateLimitError as e:
                print("[WARN] RateLimitError, retry after 10 seconds...", e)
                time.sleep(10)
        # process results
        result = response.choices[0].message.content
        print('[LLM filter]:', result, p['title'])
        try:
            result_json = json.loads(result)
            p.update(result_json)
        except json.JSONDecodeError:
            print(f"ERROR: Failed to parse JSON response for paper: {p['title']}: {result}")
            p.update({"relevant": False})
    return papers


def fetch_arxiv_papers(categories: str | List, start_date: str, end_date: str = None):
    # categories example: ["cs.DC", "cs.OS"] or "cs.DC"
    # start_date and end_date format: "YYYYMMDD"

    print('[INFO] start fetch_arxiv_papers for categories:', categories, 'from', start_date, 'to', end_date)
    if isinstance(categories, str):
        categories = [categories]
    if end_date is None:
        end_date = start_date

    papers = []
    client = arxiv.Client(page_size=200)
    for category in categories:
        search = arxiv.Search(
            query=f"cat:{category} AND submittedDate:[{start_date}0000 TO {str(int(end_date) + 1)}0000]",
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            max_results=5000,
        )
        print('[INFO] search function: ', search)

        # surround with try-except to handle potential network errors
        arxiv_results = []
        for attempt in range(5):
            try:
                arxiv_results = list(client.results(search))
                break  # if successful, exit the retry loop
            except Exception as e:
                print(f"[WARN] Attempt {attempt + 1} failed: {e}")
                time.sleep(3)

        for paper in arxiv_results:
            print('[INFO] searched paper:', paper.title)
            papers.append({
                "title": paper.title,
                "link": paper.entry_id,
                "abstract": paper.summary,
                "authors": [a.name for a in paper.authors],
                "categories": paper.categories,
                "id": paper.entry_id,
            })

    print(f"[INFO] fetched {len(papers)} papers from arxiv submitted from {start_date} to {end_date}.")

    return papers


def test():
    # test get date range()
    print(get_date_range())

    # test fetch papers
    papers = fetch_arxiv_papers(categories=["cs.DC", "cs.OS"], start_date="20251009")
    print(f"Total: {len(papers)}")
    for p in papers:
        print(p["title"])
        print("  link:", p["link"])
        print("  authors:", ", ".join(p["authors"]))
        print("  abstract:", p["abstract"].replace("\n", " "), '\n')

    # test llm filter
    results = llm_filter(papers=SAMPLE_PAPERS)
    for result in results:
        print(result['title'])
        print("  relevant:", result['relevant'])
        if result['relevant']:
            print("  tags:", result['tags'])
            print("  tldr:", result['tldr'])

    # test write
    update_daily_arxiv(papers=results, date="20251009")


if __name__ == "__main__":
    # test()

    start_date, end_date = get_date_range()
    print('fetch papers from', start_date, 'to', end_date) if start_date and end_date else exit(0)
    total_days = (stime(end_date, "%Y%m%d") - stime(start_date, "%Y%m%d")).days + 1
    # enumerate date range for testing
    for date in [(stime(start_date, "%Y%m%d") + timedelta(n)).strftime("%Y%m%d") for n in range(total_days)]:
        papers = fetch_arxiv_papers(categories=ARXIV_CATEGORIES, start_date=date)
        results = llm_filter(papers)
        update_daily_arxiv(papers=results, date=date)
