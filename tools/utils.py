# Static strings and sample data used by the fetcher

README_HEADER = """
<div align="center">\n
# Daily Arxiv Papers (LMSys)\n
![Static Badge](https://img.shields.io/badge/total_papers-{papers}-blue?logo=gitbook)
![Static Badge](https://img.shields.io/badge/update-{update}-red?logo=fireship)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.DC-green)](https://arxiv.org/list/cs.DC/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.OS-green)](https://arxiv.org/list/cs.OS/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.LG-green)](https://arxiv.org/list/cs.LG/recent)\n
`Fetch from arxiv` → `LLM Filter` → `GitHub workflow update`\n
</div>\n
**⚠️NOTE**: Update papers up to last day every morning (8:00 UTC+8) automatically.\n
**🙋WANT**: Keyword subscription (email); Functional web page.\n
**🔖TAGS**:`serving` `training` `offline` `thinking` `RL` `MoE` `RAG` `video` `multi-modal` `sparse` `quantization` `offloading` `hardware` `storage` `kernel` `diffusion` `agentic` `edge` `networking``Agent` `GUI`\n
---
"""

SYSTEM_PROMPT = """
You are an expert in computer systems, distributed computing, and large-scale machine learning systems. You will be given the title and abstract of an academic paper.
Your task is to analyze whether the paper focuses on **LLM systems** or **diffusion/video generation model systems**, and then produce a structured JSON response.

---

### Step 1. Relevance Decision

Determine if the paper focuses on **systems-level research for large models**, such as:
- **Training systems**: optimization, parallelization, resource scaling, scheduling, memory efficiency, cost reduction.
- **Inference systems**: serving, batching, scheduling, caching, quantization, operator, offloading, or networking.
- **Retrieval-augmented generation (RAG)** systems.
- **RLHF / fine-tuning frameworks**.
- **Infrastructure or compiler support** for LLM or diffusion/video generation models.

A paper is **relevant** if it aims to:
- Reduce **latency**, **cost**, or **resource consumption** during training or inference.
- Improve **throughput**, **scalability**, or **efficiency** of serving/training pipelines.

A paper is **NOT relevant** if:
- It only **uses** LLMs (e.g., for NLP, code, or reasoning tasks) rather than **improving LLM systems**.
- It focuses on **security**, **interpretability**, **federated learning**, **evaluation**, or **application-level** studies.
- It studies **traditional deep learning systems** not specific to large models.
- It focuses on improving **model architecture**, **algorithmic improvements**, **model capability**.

Output:
- `{"relevant": false}` if irrelevant.

---

### Step 2. Tagging (only if relevant)

If the paper is relevant, assign up to **3 most specific and strongly related tags** from the provided tag list below. Do **not** include weakly related tags.

Available tags and their meanings:
{tag_descriptions}


### Step 3. TLDR Summary (only if relevant)

Generate a concise summary (≤50 words) including:
1. The **main research question or problem**.
2. The **key system design or method**.
3. **One quantitative result or metric improvement**.

The TLDR should be factual, not generic, and written in a scientific tone.

---

### Output Format

Return a **valid JSON object only**, with **no explanations or extra text** outside it.

Examples:
{"relevant": true, "tags": ["serving"], "tldr": "Investigates how to schedule LLM inference requests with diverse SLOs. Proposes a simulated-annealing-based scheduler that prioritizes requests using SLOs and input/output lengths. Achieves up to 5× higher SLO attainment than vLLM."}
{"relevant": true, "tags": ["serving", "offloading"], "tldr": "Addresses how to efficiently serve heterogeneous LLM inference requests. Proposes Llumnix, a dynamic scheduler with live request offloading across instances for load balancing and SLO compliance. Achieves up to 10× lower tail latency."}
{"relevant": false}


---

Follow this decision order strictly:
1. Decide relevance → 2. Assign tags (if relevant) → 3. Generate TLDR (if relevant).
Ensure JSON validity at all times.

"""

# --- backup -----------------------------------------------------------------------------------------------------------
"""
Your task is to:
1. Determine if the paper is focusing on LLM systems (e.g., training, inference, RAG system, RLHF framework, etc.), including systems for diffusion models and video generation models.
   The relevant papers should aim at **reducing the inference / training latency, improving the resource utilization, or reducing serving / training cost** of LLMs or diffusion models (e.g., latency, throughput, cost, scalability, etc.).
   Mark it as {"relevant": false} if it only uses LLM or focuses other domains like traditional deep learning system, LLM security or federated learning.
2. If the paper is relevant, mark it as {"relevant": true}, and assign the most relevant tags (do not include weak relevant tags) from the provided tag list. If a paper is strong relevant to multiple tags, assign the most relevant tags （up to 3）.
3. If the paper is relevant, Provide a concise summary (TLDR) of the paper in no more than 50 words. The TLDR should be informative and include (1) key question, (2) key designs, and (3) ONE key metric.

Make sure your response is a valid JSON object with the following format and DO NOT include any explanations or additional text outside the JSON object:
{"relevant": true, "tags": ["tag1", "tag2", ...], "tldr": "..."} or {"relevant": false}.

Use the following tags and their descriptions to help you decide the tags for each paper:
{tag_descriptions}
"""
# ----------------------------------------------------------------------------------------------------------------------

USER_PROMPT = """
Here is the paper:\n
Title: {title}\n
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
    "agent":            "designed for LLM-based agents, including tool use, planning, and multi-step decision making",
    "GUI":              "designed for graphical user interfaces or visual interaction systems built on top of LLMs",
    # "others":           "other LLM system topics not covered above",
}

SAMPLE_PAPERS = [
    {
        "title": "Scrooge: A Cost-Effective Deep Learning Inference System",
        "abstract": "Advances in deep learning (DL) have prompted the development of cloud-hosted DL-based media "
                    "applications that process video and audio streams in real-time. Such applications must "
                    "satisfy throughput and latency objectives and adapt to novel types of dynamics, while "
                    "incurring minimal cost. Scrooge, a system that provides media applications as a service, "
                    "achieves these objectives by packing computations efficiently into GPU-equipped cloud VMs, "
                    "using an optimization formulation to find the lowest cost VM allocations that meet the "
                    "performance objectives, and rapidly reacting to variations in input complexity (e.g., changes "
                    "in participants in a video). Experiments show that Scrooge can save serving cost by 16-32% "
                    "(which translate to tens of thousands of dollars per year) relative to the state-of-the-art "
                    "while achieving latency objectives for over 98% under dynamic workloads.",
        "link": "https://arxiv.org/abs/2310.09410"
    }, {
        "title": "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills",
        "abstract": "Large Language Model (LLM) inference consists of two distinct phases - prefill phase which "
                    "processes the input prompt and decode phase which generates output tokens autoregressively. "
                    "While the prefill phase effectively saturates GPU compute at small batch sizes, the decode "
                    "phase results in low compute utilization as it generates one token at a time per request. "
                    "The varying prefill and decode times also lead to imbalance across micro-batches when using "
                    "pipeline parallelism, resulting in further inefficiency due to bubbles.\nWe present SARATHI "
                    "to address these challenges. SARATHI employs chunked-prefills, which splits a prefill request "
                    "into equal sized chunks, and decode-maximal batching, which constructs a batch using a single "
                    "prefill chunk and populates the remaining slots with decodes. During inference, the prefill "
                    "chunk saturates GPU compute, while the decode requests 'piggyback' and cost up to an order of "
                    "magnitude less compared to a decode-only batch. Chunked-prefills allows constructing multiple "
                    "decode-maximal batches from a single prefill request, maximizing coverage of decodes that can "
                    "piggyback. Furthermore, the uniform compute design of these batches ameliorates the imbalance "
                    "between micro-batches, significantly reducing pipeline bubbles.\nOur techniques yield "
                    "significant improvements in inference performance across models and hardware. For the "
                    "LLaMA-13B model on A6000 GPU, SARATHI improves decode throughput by up to 10x and accelerates "
                    "end-to-end throughput by up to 1.33x. For LLaMa-33B on A100 GPU, we achieve 1.25x higher "
                    "end-to-end-throughput and up to 4.25x higher decode throughput. When used with pipeline "
                    "parallelism on GPT-3, SARATHI reduces bubbles by 6.29x, resulting in an end-to-end throughput "
                    "improvement of 1.91x.",
        "link": "https://arxiv.org/abs/2405.00428"
    }
]
