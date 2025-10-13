import os
import json
import time
import arxiv
import datetime as dt
from openai import OpenAI, RateLimitError
from typing import List
from pprint import pprint
from datetime import datetime, timedelta

API_KEY = os.environ['API_KEY']
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

SYSTEM_PROMPT = """
You are an expert in computer science, especially in distributed systems, machine learning and large language models. You will be given an academic paper title and abstract.

Your task is to:
1. Determine if the paper is focusing on LLM systems (e.g., training, inference, RAG system, RLHF framework, etc.), including systems for diffusion models and video generation models.
   The relevant papers should aim at improving the system performance of LLMs or diffusion models (e.g., latency, throughput, cost, scalability, etc.).
   Mark it as {"relevant": false} if it only uses LLM or focuses other domains like traditional deep learning system, LLM security or federated learning.
2. If the paper is relevant, mark it as {"relevant": true}, and assign the most relevant tags (do not include weak relevant tags) from the provided tag list. If a paper is strong relevant to multiple tags, assign the most relevant tags （up to 3）.
3. If the paper is relevant, Provide a concise summary (TLDR) of the paper in no more than 50 words. The TLDR should be informative and include (1) key question, (2) key designs, and (3) ONE key metric.

Make sure your response is a valid JSON object with the following format and DO NOT include any explanations or additional text outside the JSON object:
{"relevant": true, "tags": ["tag1", "tag2", ...], "tldr": "..."} or {"relevant": false}.

Use the following tags and their descriptions to help you decide the tags for each paper:
{tag_descriptions}

Here are examples response for several papers:
{"relevant": true, "tags": ["serving"], "tldr": "Investigates how to schedule LLM inference requests with diverse SLOs. Proposes a simulated-annealing-based scheduler that prioritizes requests using SLOs and input/output lengths. Achieves up to 5× higher SLO attainment than vLLM."}
{"relevant": true, "tags": ["serving", "offloading"], "tldr": "Addresses how to efficiently serve heterogeneous LLM inference requests. Proposes Llumnix, a dynamic scheduler with live request offloading across instances for load balancing and SLO compliance. Achieves up to 10× lower tail latency."}
{"relevant": false}
"""

USER_PROMPT = """
Here is the paper:

Title: {title}

Abstract: {abstract}
"""

TAGS = {  # tags that relate to LLM systems
    "serving":          "targeted at LLM serving or online inference",
    "training":         "designed for training LLMs",
    "offline":          "targeted at offline LLM inference or batch processing",
    "thinking":         "designed for reasoning or thinking LLMs",
    "RL":               "designed for reinforcement learning or post training",
    "MoE":              "designed for mixture-of-experts models",
    "RAG":              "designed for retrieval-augmented generation",
    "video":            "designed for video generation models",
    "multi-modal":      "designed for multi-modal models",
    "sparse":           "leverage or introduce new sparsity techniques",
    "quantization":     "leverage or introduce new quantization techniques",
    # "parallelism":      "leverage or introduce new parallelism techniques",
    "offloading":       "leverage or introduce new KV cache or model weight offloading techniques",
    "hardware":         "targeted at LLM hardware or accelerators",
    "storage":          "leverage or introduce new storage techniques",
    "kernel":           "targeted at LLM operator (CUDA kernel) optimizations",
    "diffusion":        "designed for diffusion models",
    "agentic":          "designed for agentic models",
    "edge":             "designed for LLM inference on edge or mobile devices",
    "networking":       "leverage or introduce new networking or transfer techniques",
    # "others":           "other LLM system topics not covered above",
}

# TODO: keyword subscription (email notification)
# SUBSCRIBER = {
#     "zhixin@abc.com": ["video-generation", "RL", "parallelism", "thinking", "serving", "offloading", "kernel", "hardware", "storage"],
# }

README_FILE = 'daily-arxiv-llm.md'
README_HEADER = """
<div align="center">\n
# Daily Arxiv Papers (LMSys)\n
![Static Badge](https://img.shields.io/badge/total_papers-{papers}-blue?logo=gitbook)\n
![Static Badge](https://img.shields.io/badge/update-{update}-red?logo=fireship)\n
`Fetch from arxiv` → `LLM Filter` → `GitHub workflow update`\n
</div>\n
**⚠️NOTE**: Update papers up to last day every morning (8:00 UTC+8) automatically.\n
**🙋WANT**: keyword subscription (email notification) and more feature.\n
**🔖TAGS**:`serving` `training` `offline` `thinking` `RL` `MoE` `RAG` `video` `multi-modal` `sparse` `quantization` `offloading` `hardware` `storage` `kernel` `diffusion` `agentic` `edge` `networking`\n
---
"""


def get_date_range():
    start_date = "20250101"
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

    lines = open(README_FILE, 'r', encoding='utf-8').readlines()
    for line in lines:
        if line.startswith("### "):
            start_date = (datetime.strptime(line[4:14], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
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

    date = dt.datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
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
    content = [README_HEADER.format(papers=number_papers, update=date)] + content

    # step-5: write back to file
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.writelines(content)


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


def fetch_arxiv_papers(categories: str or List, start_date: str, end_date: str = None):
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
        for paper in client.results(search):
            print('[INFO] searched paper:', paper.title)
            papers.append({
                "title": paper.title,
                "link": paper.entry_id,
                "abstract": paper.summary,
                "authors": [a.name for a in paper.authors],
                "categories": paper.categories,
                "id": paper.entry_id,
            })

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
    results = llm_filter(papers=[
        {
            "title": "Scrooge: A Cost-Effective Deep Learning Inference System",
            "abstract": "Advances in deep learning (DL) have prompted the development of cloud-hosted DL-based media applications that process video and audio streams in real-time. Such applications must satisfy throughput and latency objectives and adapt to novel types of dynamics, while incurring minimal cost. Scrooge, a system that provides media applications as a service, achieves these objectives by packing computations efficiently into GPU-equipped cloud VMs, using an optimization formulation to find the lowest cost VM allocations that meet the performance objectives, and rapidly reacting to variations in input complexity (e.g., changes in participants in a video). Experiments show that Scrooge can save serving cost by 16-32% (which translate to tens of thousands of dollars per year) relative to the state-of-the-art while achieving latency objectives for over 98% under dynamic workloads.",
            "link": "https://arxiv.org/abs/2310.09410"
        }, {
            "title": "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills",
            "abstract": "Large Language Model (LLM) inference consists of two distinct phases - prefill phase which processes the input prompt and decode phase which generates output tokens autoregressively. While the prefill phase effectively saturates GPU compute at small batch sizes, the decode phase results in low compute utilization as it generates one token at a time per request. The varying prefill and decode times also lead to imbalance across micro-batches when using pipeline parallelism, resulting in further inefficiency due to bubbles.\nWe present SARATHI to address these challenges. SARATHI employs chunked-prefills, which splits a prefill request into equal sized chunks, and decode-maximal batching, which constructs a batch using a single prefill chunk and populates the remaining slots with decodes. During inference, the prefill chunk saturates GPU compute, while the decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch. Chunked-prefills allows constructing multiple decode-maximal batches from a single prefill request, maximizing coverage of decodes that can piggyback. Furthermore, the uniform compute design of these batches ameliorates the imbalance between micro-batches, significantly reducing pipeline bubbles.\nOur techniques yield significant improvements in inference performance across models and hardware. For the LLaMA-13B model on A6000 GPU, SARATHI improves decode throughput by up to 10x, and accelerates end-to-end throughput by up to 1.33x. For LLaMa-33B on A100 GPU, we achieve 1.25x higher end-to-end-throughput and up to 4.25x higher decode throughput. When used with pipeline parallelism on GPT-3, SARATHI reduces bubbles by 6.29x, resulting in an end-to-end throughput improvement of 1.91x.",
            "link": "https://arxiv.org/abs/2405.00428"
        }
    ])
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
    if start_date is None or end_date is None:
        exit(0)

    papers = fetch_arxiv_papers(categories=["cs.DC", "cs.OS"], start_date=start_date, end_date=end_date)
    print(f"[INFO] fetched {len(papers)} papers from arxiv from {start_date} to {end_date}")

    if len(papers) == 0:
        print("[INFO] no new papers, exit")
        exit(0)

    results = llm_filter(papers)
    update_daily_arxiv(papers=results, date=end_date)
    print("[INFO] done")