
<div align="center">

# Daily Arxiv Papers (LMSys)

![Static Badge](https://img.shields.io/badge/total_papers-863-blue?logo=gitbook)
![Static Badge](https://img.shields.io/badge/update-2026.01.02-red?logo=fireship)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.DC-green)](https://arxiv.org/list/cs.DC/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.OS-green)](https://arxiv.org/list/cs.OS/recent)
[![Static Badge](https://img.shields.io/badge/arXiv-cs.LG-green)](https://arxiv.org/list/cs.LG/recent)

`Fetch from arxiv` → `LLM Filter` → `GitHub workflow update`

</div>

**⚠️NOTE**: Update papers up to last day every morning (8:00 UTC+8) automatically.

**🙋WANT**: Keyword subscription (email); Functional web page.

**🔖TAGS**:`serving` `training` `offline` `thinking` `RL` `MoE` `RAG` `video` `multi-modal` `sparse` `quantization` `offloading` `hardware` `storage` `kernel` `diffusion` `agentic` `edge` `networking` `Agent` `GUI`

---
### 2025-12-30
* `training` `storage` [Understanding LLM Checkpoint/Restore I/O Strategies and Patterns](http://arxiv.org/abs/2512.24511v1)
  > **TL;DR**: Investigates I/O bottlenecks in checkpointing large-scale LLM training workloads. Designs a checkpointing engine using file system-aware aggregation and coalescing strategies with 	exttt{liburing}. Achieves up to 7.6x higher write throughput than TorchSnapshot.
* `serving` `offloading` `kernel` [PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression](http://arxiv.org/abs/2512.24449v1)
  > **TL;DR**: Addresses the large memory footprint of the KV cache during LLM inference. Proposes PackKV, a framework using LLM-aware lossy compression algorithms for the KV cache. Achieves 75.7% higher throughput for K and 171.7% for V compared to cuBLAS kernels.

### 2025-12-29
* `edge` `serving` `kernel` [Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL](http://arxiv.org/abs/2512.23310v1)
  > **TL;DR**: Proposes Splitwise, a DRL framework for adaptive partitioning of transformer layers between edge and cloud to optimize LLM inference. Minimizes latency and energy under fluctuating bandwidth by splitting attention heads and feed-forward blocks. Achieves 1.4x-2.8x lower latency and 41% energy reduction vs. baselines.

### 2025-12-28
* `serving` `MoE` `quantization` [Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware](http://arxiv.org/abs/2512.23029v1)
  > **TL;DR**: Assesses if private LLM serving for SMBs is viable using consumer hardware. Benchmarks a quantized Qwen3-30B MoE model, measuring latency, throughput, and scalability under load. Achieves performance comparable to cloud services with a significant reduction in cost.
* `serving` `edge` `offloading` [Argus: Token Aware Distributed LLM Inference Optimization](http://arxiv.org/abs/2512.22925v1)
  > **TL;DR**: Presents Argus, a token-aware distributed LLM inference framework for edge-cloud systems. It predicts output token lengths for precise cost estimation and uses Lyapunov optimization for dynamic task offloading. In evaluations, it improves inference throughput by up to 2.7x under dynamic workloads.
* `networking` `training` [OptiNIC: A Resilient and Tail-Optimal RDMA NIC for Distributed ML Workloads](http://arxiv.org/abs/2512.22743v1)
  > **TL;DR**: Addresses tail latency in collective communication for distributed ML workloads. Proposes OptiNIC, an RDMA transport that removes retransmissions and in-order delivery, shifting loss recovery to the ML pipeline. Reduces 99th-percentile latency by 3.5x and improves time-to-accuracy by 2x.

### 2025-12-27
* `serving` `multi-modal` `kernel` [Modality Inflation: Energy Characterization and Optimization Opportunities for MLLM Inference](http://arxiv.org/abs/2512.22695v1)
  > **TL;DR**: Characterizes energy inefficiency in multimodal LLM inference, identifying modality inflation bottlenecks via stage-level analysis. Demonstrates that stage-wise DVFS optimization saves energy with modest performance loss, achieving up to 94% energy reduction overhead compared to text-only models.
* `RL` `training` `MoE` [RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure](http://arxiv.org/abs/2512.22560v1)
  > **TL;DR**: Proposes RollArc, a distributed system to improve throughput for multi-task agentic RL training on disaggregated infrastructure via hardware-affinity mapping and fine-grained asynchrony. Achieves 1.35-2.05x end-to-end training time reduction.
* `RL` `training` [Role-Based Fault Tolerance System for LLM RL Post-Training](http://arxiv.org/abs/2512.22492v1)
  > **TL;DR**: Presents RobustRL, a fault tolerance system for RL post-training of LLMs that uses role-based isolation for trainer, rollout, and management components. It enables non-disruptive recovery via role-aware monitoring, warm standbys, and dynamic communication, achieving over 80% Effective Training Time Ratio (vs. 60% baseline) on a 256-GPU cluster.
* `serving` `kernel` [Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving](http://arxiv.org/abs/2512.22420v1)
  > **TL;DR**: Addresses the performance trade-off of static speculative decoding length under varying server load. Introduces Nightjar, a learning-based algorithm that dynamically adapts the speculative length. Achieves up to 14.8% higher throughput and 20.2% lower latency.

### 2025-12-26
* `agent` `RAG` [Agentic Structured Graph Traversal for Root Cause Analysis of Code-related Incidents in Cloud Applications](http://arxiv.org/abs/2512.22113v1)
  > **TL;DR**: Presents PRAXIS, an orchestrator for root cause analysis (RCA) of cloud incidents using an LLM-driven agentic workflow that traverses service and code dependency graphs. The system improves RCA accuracy by up to 3.1× and reduces token consumption by 3.8× compared to ReAct baselines.
* `MoE` `training` `networking` [FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion](http://arxiv.org/abs/2512.22036v1)
  > **TL;DR**: Addresses the overhead of distributed data shuffling for Mixture-of-Experts (MoE) model training. Proposes FUSCO, a library that fuses data transformation and communication via fine-grained layout interpretation and pipelining. Achieves up to 3.84x speedup over NCCL in shuffling benchmarks.
* `serving` `networking` [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](http://arxiv.org/abs/2512.21884v1)
  > **TL;DR**: Studies optimal resource allocation for distributed LLM inference across servers. Formulates block placement and request routing as a MILP problem with an efficient online algorithm. Achieves substantially reduced inference time vs. PETALS in geo-distributed setups.
* `edge` `serving` `offloading` [LIME:Accelerating Collaborative Lossless LLM Inference on Memory-Constrained Edge Devices](http://arxiv.org/abs/2512.21835v1)
  > **TL;DR**: Proposes LIME, a system for lossless LLM inference on memory-constrained edge devices using interleaved pipeline parallelism and model offloading. It features fine-grained scheduling and online memory adaptation. Achieves up to 3.7x speedup over baselines.

### 2025-12-25
* `multi-modal` `offloading` [Hyperion: Low-Latency Ultra-HD Video Analytics via Collaborative Vision Transformer Inference](http://arxiv.org/abs/2512.21730v1)
  > **TL;DR**: Addresses high computational/transmission overhead for Ultra-HD video analytics with vision transformers. Proposes Hyperion, a collaborative cloud-device framework with importance scoring and dynamic scheduling of patches. Increases frame processing rate by up to 1.61x while improving accuracy by 20.2%.
* `kernel` `storage` `quantization` [nncase: An End-to-End Compiler for Efficient LLM Deployment on Heterogeneous Storage Architectures](http://arxiv.org/abs/2512.21571v1)
  > **TL;DR**: Presents nncase, an end-to-end compiler framework for efficient LLM deployment on heterogeneous storage architectures. Uses an e-graph-based term rewriting engine for global optimization and integrates modules for vectorization, distribution, and scheduling. Achieves comparable performance to hand-optimized llama.cpp on CPUs.
* `serving` `MoE` `kernel` [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](http://arxiv.org/abs/2512.21487v1)
  > **TL;DR**: Addresses low throughput in Mixture-of-Experts inference caused by inefficient scheduling in disaggregated expert parallelism. Proposes FinDEP, a fine-grained task scheduling algorithm that optimizes computation/communication overlap through partitioning and ordering. Achieves up to 1.61x higher throughput over prior methods.

### 2025-12-24
* `training` `kernel` `multi-modal` [Mesh-Attention: A New Communication-Efficient Distributed Attention with Improved Data Locality](http://arxiv.org/abs/2512.20968v1)
  > **TL;DR**: Proposes Mesh-Attention, a distributed attention algorithm using 2D tiling to reduce communication overhead during LLM training. Achieves up to 3.4x speedup and reduces communication volume by 85.4% on 256 GPUs compared to Ring-Attention.
* `training` `offloading` [Deadline-Aware Online Scheduling for LLM Fine-Tuning with Spot Market Predictions](http://arxiv.org/abs/2512.20967v1)
  > **TL;DR**: Addresses cost-efficient scheduling of LLM fine-tuning jobs using volatile spot instances. Proposes an online framework mixing spot/on-demand instances with prediction and policy selection algorithms. Achieves up to 54.8% utility improvement over baselines.
* `training` `networking` [Diving into 3D Parallelism with Heterogeneous Spot Instance GPUs: Design and Implications](http://arxiv.org/abs/2512.20953v1)
  > **TL;DR**: Presents AutoHet, a system for automatic parallelism plan generation to optimize distributed LLM training on heterogeneous GPU spot instances. Solves an optimization problem for device grouping and load balancing, achieving up to a 1.79x training throughput speedup.

### 2025-12-23
* `serving` `kernel` [Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs](http://arxiv.org/abs/2512.20573v1)
  > **TL;DR**: Proposes FailFast, a speculative decoding framework using Diffusion LLMs as drafters. Dynamically adapts speculation length to 'fail fast' in hard regions and 'win big' in easy ones. Achieves up to 4.9x speedup over vanilla decoding.
* `serving` `offloading` [Predictive-LoRA: A Proactive and Fragmentation-Aware Serverless Inference System for LLMs](http://arxiv.org/abs/2512.20210v1)
  > **TL;DR**: Addresses high latency and memory fragmentation in serverless LoRA inference. Proposes Predictive-LoRA with a traffic predictor for proactive adapter prefetching and a page-based memory manager. Reduces cold start latency by 68% and improves throughput by 1.52x compared to S-LoRA.
* `agentic` `serving` [Reaching Agreement Among Reasoning LLM Agents](http://arxiv.org/abs/2512.20184v1)
  > **TL;DR**: Proposes a formal distributed consensus model and protocol (Aegean) for multi-agent reasoning LLM systems to replace static workflows. Aegean-Serve enables incremental quorum detection for early termination, reducing latency by 1.2-20x while maintaining answer quality within 2.5%.

### 2025-12-22
* `training` `serving` `networking` [RAPID-LLM: Resilience-Aware Performance analysis of Infrastructure for Distributed LLM Training and Inference](http://arxiv.org/abs/2512.19606v1)
  > **TL;DR**: Presents RAPID-LLM, a performance modeling framework for LLM training/inference on GPU clusters. It combines a hardware-aware trace generator with an extended network simulator. Predicts latency and training times within 10.4% of real measurements on A100 clusters.
* `serving` `networking` [L4: Low-Latency and Load-Balanced LLM Serving via Length-Aware Scheduling](http://arxiv.org/abs/2512.19179v1)
  > **TL;DR**: Addresses performance degradation in multi-instance LLM serving due to request-length heterogeneity. Proposes L4, a runtime system that partitions instances into length-specialized groups for dynamic load balancing. Reduces end-to-end latency by up to 67% and improves throughput by 2.89x.

### 2025-12-21
* `edge` `serving` [QoS-Aware Load Balancing in the Computing Continuum via Multi-Player Bandits](http://arxiv.org/abs/2512.18915v1)
  > **TL;DR**: Proposes QEdgeProxy, a decentralized load balancer for edge-AI workloads that formulates instance selection as a Multi-Player Bandit problem to maximize per-client QoS. It uses Kernel Density Estimation for QoS probability estimation and adaptive exploration. Achieves significantly higher QoS satisfaction than baseline methods under dynamic conditions.
* `serving` `MoE` `offloading` [Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing](http://arxiv.org/abs/2512.18674v1)
  > **TL;DR**: Proposes Remoe, a serverless MoE inference system that assigns expert modules to CPUs and offloads infrequently used experts to reduce memory overhead. Employs activation prediction and optimization algorithms to reduce inference cost by up to 57% and cold start latency by 47% versus baselines.

### 2025-12-20
* `video` `serving` `kernel` [Asynchronous Pipeline Parallelism for Real-Time Multilingual Lip Synchronization in Video Communication Systems](http://arxiv.org/abs/2512.18318v1)
  > **TL;DR**: Proposes an asynchronous pipeline-parallel Transformer framework for real-time multilingual lip sync in video conferencing. Integrates modules via message queues and optimizes inference with graph compilation, quantization, and kernel fusion. Reduces end-to-end latency by 3.1x versus sequential pipelines.
* `serving` `offloading` `networking` [TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale](http://arxiv.org/abs/2512.18194v1)
  > **TL;DR**: Presents TraCT, a rack-scale LLM serving system that uses CXL shared memory for KV cache transfer and storage instead of RDMA. This eliminates the NIC hop, addressing the KV transfer bottleneck. Achieves up to 9.8x lower time-to-first-token and 1.6x higher throughput.

### 2025-12-19
* `serving` `multi-modal` [Enabling Disaggregated Multi-Stage MLLM Inference via GPU-Internal Scheduling and Resource Sharing](http://arxiv.org/abs/2512.17574v1)
  > **TL;DR**: Proposes FlashCodec and UnifiedServe to optimize end-to-end Multi-modal LLM serving, addressing bottlenecks in video preprocessing/encoding and LLM inference. Uses multi-GPU decoding and logical/physical resource sharing to eliminate inter-stage blocking. Achieves up to 4.4× higher throughput vs. state-of-the-art.
* `storage` `serving` `RAG` [Scalable Distributed Vector Search via Accuracy Preserving Index Construction](http://arxiv.org/abs/2512.17264v1)
  > **TL;DR**: Investigates how to scale approximate nearest neighbor search for billion-scale vectors in distributed settings. Proposes SPIRE, an index with balanced partition granularity and an accuracy-preserving recursive construction method. Achieves up to 9.64x higher throughput than state-of-the-art systems on 8 billion vectors.

### 2025-12-18
* `MoE` `offloading` `serving` [Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems](http://arxiv.org/abs/2512.16473v1)
  > **TL;DR**: Proposes a CPU-GPU collaborative inference framework for MoE-based LLMs on memory-limited systems using GPU expert caching and CPU offloading for cache misses. Achieves up to 5.4x higher throughput compared to state-of-the-art offloading methods.
* `kernel` `serving` [Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference](http://arxiv.org/abs/2512.16391v1)
  > **TL;DR**: Addresses attention latency for long-context LLM inference. Proposes Kascade, a training-free sparse attention method that reuses Top-k indices across layers selected via dynamic programming. Achieves up to 4.1x decode attention speedup over FlashAttention-3 while matching dense accuracy.
* `serving` `scheduling` [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](http://arxiv.org/abs/2512.16134v1)
  > **TL;DR**: Addresses the scheduling challenge in DP+EP LLM serving architectures where immediate dispatching causes queuing bubbles and degrades TTFT. Proposes Staggered Batch Scheduling (SBS) to buffer requests for optimal batching and a load-aware allocation strategy. Reduces TTFT by 30-40% and improves throughput by 15-20% compared to baselines.
* `serving` `kernel` [An Online Fragmentation-Aware Scheduler for Managing GPU-Sharing Workloads on Multi-Instance GPUs](http://arxiv.org/abs/2512.16099v1)
  > **TL;DR**: Proposes an online scheduler for Multi-Instance GPUs (MIG) to mitigate resource contention and fragmentation. The framework integrates load balancing, dynamic partitioning, and job migration for GPU-sharing workloads. Achieves up to a 35% improvement in makespan.
* `serving` `offloading` `networking` [MultiPath Transfer Engine: Breaking GPU and Host-Memory Bandwidth Bottlenecks in LLM Services](http://arxiv.org/abs/2512.16056v1)
  > **TL;DR**: Addresses PCIe bandwidth bottlenecks for LLM serving tasks like prefetching and model switching. Proposes MMA, a multipath data transfer scheme between GPU and host memory via dynamic library injection. Achieves a peak bandwidth of 245 GB/s, a 4.62x speedup, reducing TTFT by up to 2.38x.
* `serving` `RAG` [Trustworthy and Controllable Professional Knowledge Utilization in Large Language Models with TEE-GPU Execution](http://arxiv.org/abs/2512.16238v1)
  > **TL;DR**: Presents PKUS, a system for trustworthy LLM serving that isolates proprietary knowledge adapters in TEEs while keeping the backbone model on GPU. Combines split-execution scheduling with adapter pruning to preserve utility while achieving 8.1-11.9x lower latency than baseline TEE inference.

### 2025-12-17
* `serving` `kernel` [Dynamic Rebatching for Efficient Early-Exit Inference with DREX](http://arxiv.org/abs/2512.15705v1)
  > **TL;DR**: Addresses inefficiency in batching for Early-Exit LLM inference. Proposes DREX, a system using dynamic rebatching with a copy-free buffer and an SLA-aware scheduler. Achieves 2-12% higher throughput while eliminating involuntary exits to preserve quality.
* `training` `quantization` `offloading` [LLMQ: Efficient Lower-Precision Pretraining for Consumer GPUs](http://arxiv.org/abs/2512.15306v1)
  > **TL;DR**: Presents LLMQ, a system for efficient medium-sized LLM training on consumer GPUs. Uses 8-bit quantization, activation checkpointing, and copy-engine based collectives to overcome memory/cpu bottlenecks. Achieves 50% FLOP utilization training a 7B model on a single 16GB GPU.

### 2025-12-16
* `serving` `multi-modal` [Cornserve: Efficiently Serving Any-to-Any Multimodal Models](http://arxiv.org/abs/2512.14098v1)
  > **TL;DR**: Presents Cornserve, a serving system for any-to-any multimodal models with heterogeneous components. The system uses a planner to generate optimized deployment plans and a distributed runtime to execute them. Achieves up to 3.81x throughput improvement over baselines.

### 2025-12-15
* `kernel` `hardware` `training` [Design in Tiles: Automating GEMM Deployment on Tile-Based Many-PE Accelerators](http://arxiv.org/abs/2512.13638v1)
  > **TL;DR**: Proposes an automated framework for deploying GEMM on tile-based accelerators by connecting a deployment toolchain with a configurable model. Achieves 1.2-2.0x speedup over expert-tuned libraries on a GH200-scale configuration via higher PE utilization.
* `serving` `MoE` `offloading` [Janus: Disaggregating Attention and Experts for Scalable MoE Inference](http://arxiv.org/abs/2512.13525v1)
  > **TL;DR**: Addresses scalable Mixture-of-Experts (MoE) inference by disaggregating attention and expert modules onto separate GPU clusters. Proposes Janus, which uses an adaptive two-phase communication scheme and a lightweight GPU-kernel scheduler for expert balancing. Achieves up to 3.9x higher per-GPU throughput.
* `training` `MoE` `hardware` [SIGMA: An AI-Empowered Training Stack on Early-Life Hardware](http://arxiv.org/abs/2512.13488v1)
  > **TL;DR**: Presents SIGMA, a training stack for improving reliability and efficiency on early-life AI accelerators. Uses the LUCIA platform for cluster management and a framework for distributed MoE model training. Achieved 94.45% accelerator utilization and 21.08% MFU when training a 200B MoE model.
* `kernel` `training` [FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection](http://arxiv.org/abs/2512.12949v1)
  > **TL;DR**: Presents FlashFuser, a compiler that exploits GPU inter-core connection (DSM) for kernel fusion beyond local scratchpad limits. It uses a DSM communication abstraction, dataflow analyzer, and search engine to optimize execution plans. Achieves 3.3x kernel speedup and reduces memory access by 58%.
* `serving` [PROSERVE: Unified Multi-Priority Request Scheduling for LLM Serving](http://arxiv.org/abs/2512.12928v1)
  > **TL;DR**: Addresses the challenge of LLM serving with diverse client priorities. Proposes PROSERVE, a two-tier scheduler with SlideBatching for adaptive engine-level batching and GoRouting for service-level dispatching. Increases SLO attainment by up to 52% and system gain by 35% over baselines.

### 2025-12-14
* `serving` `kernel` [Fine-Grained Energy Prediction For Parallellized LLM Inference With PIE-P](http://arxiv.org/abs/2512.12801v1)
  > **TL;DR**: Proposes PIE-P, a framework for fine-grained energy prediction of parallelized LLM inference across multiple GPUs. It uses precise sampling and models inter-GPU communication to account for parallelism overheads. Outperforms baselines in prediction accuracy, demonstrating significant improvement in energy estimation.
* `training` `kernel` [gpu_ext: Extensible OS Policies for GPUs via eBPF](http://arxiv.org/abs/2512.12615v1)
  > **TL;DR**: Presents gpu_ext, an eBPF-based runtime for programmable GPU resource management policies. Extends GPU drivers with safe hooks and a device-side eBPF runtime within GPU kernels. Evaluated on inference/training workloads, achieves up to 4.8x throughput improvement.

### 2025-12-13
* `RL` `training` [HetRL: Efficient Reinforcement Learning for LLMs in Heterogeneous Environments](http://arxiv.org/abs/2512.12476v1)
  > **TL;DR**: Presents HetRL, a distributed system for efficient LLM reinforcement learning training in infrastructures with heterogeneous GPUs. It formulates scheduling as a constrained optimization problem solved with a multi-level search algorithm. Achieves up to 9.17x higher throughput versus state-of-the-art systems.
* `training` `sparse` [BOOST: BOttleneck-Optimized Scalable Training Framework for Low-Rank Large Language Models](http://arxiv.org/abs/2512.12131v1)
  > **TL;DR**: Addresses poor scalability of low-rank bottleneck LLMs under standard tensor parallelism. Proposes BOOST, a framework with bottleneck-aware tensor parallelism and kernel/memory optimizations. Achieves 1.46-1.91x training speedup over full-rank baselines.

### 2025-12-12
* `training` `RL` [RollMux: Phase-Level Multiplexing for Disaggregated RL Post-Training](http://arxiv.org/abs/2512.11306v1)
  > **TL;DR**: Addresses the dependency bubble and low hardware utilization in disaggregated RL post-training. Proposes RollMux, a cluster scheduling framework that multiplexes rollout and training phases of different jobs using co-execution groups. Achieves a 1.84x improvement in cost efficiency over standard disaggregation.

### 2025-12-11
* `serving` `offloading` [ESS: An Offload-Centric Latent-Cache Management Architecture for DeepSeek-V3.2-Exp](http://arxiv.org/abs/2512.10576v1)
  > **TL;DR**: Addresses the GPU memory bottleneck from Latent-Cache growth in DeepSeek-V3.2-Exp inference. Proposes ESS, a system that offloads Latent-Cache to CPU memory to decouple batch size from GPU limits. Achieves up to 123% throughput improvement at 128K context length.
* `training` `kernel` `networking` [Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap](http://arxiv.org/abs/2512.10236v1)
  > **TL;DR**: Proposes FiCCO, a method for finer-grain compute-communication overlap in distributed ML training. It explores execution schedules using GPU DMA engines and designs heuristics to minimize operation inefficiencies. Achieves up to 1.6x speedup over coarse-grain approaches.

### 2025-12-10
* `training` `networking` [Straggler Tolerant and Resilient DL Training on Homogeneous GPUs](http://arxiv.org/abs/2512.09685v1)
  > **TL;DR**: Proposes STAR, a straggler-tolerant training system for homogeneous GPUs. Introduces new synchronization modes with heuristic/ML-based selection and proactive resource management to minimize Time-To-Accuracy. Reduces TTA by 48-84% in PS and 51-70% in all-reduce architectures compared to state-of-the-art.
* `serving` `offloading` [WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving](http://arxiv.org/abs/2512.09472v1)
  > **TL;DR**: Proposes WarmServe, a multi-LLM serving system that uses workload prediction to prewarm universal GPU workers, employing evict-aware placement and zero-overhead memory switching. Improves time-to-first-token by up to 50.8x compared to autoscaling systems.
* `RAG` `storage` `networking` [Passing the Baton: High Throughput Distributed Disk-Based Vector Search with BatANN](http://arxiv.org/abs/2512.09331v1)
  > **TL;DR**: Addresses the challenge of high-throughput vector search for billion-scale datasets. Proposes BatANN, a distributed disk-based ANN system that passes full query state to remote machines for locality. Achieves 2.5-6.49x higher throughput than baselines while maintaining sub-6ms mean latency.
* `serving` `MoE` `kernel` [Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens](http://arxiv.org/abs/2512.09277v1)
  > **TL;DR**: Identifies that token-balancing token-routing harms MoE serving performance under memory constraints. Proposes METRO, an algorithm that balances activated experts per GPU using a novel allGather scheme. Reduces decode latency by 11-22% and improves throughput up to 4.11x over baselines.

### 2025-12-09
* `serving` `training` [Magneton: Optimizing Energy Efficiency of ML Systems via Differential Energy Debugging](http://arxiv.org/abs/2512.08365v1)
  > **TL;DR**: Presents Magneton, a differential energy debugging tool for ML systems. It profiles operator-level energy consumption across similar systems to pinpoint wasteful code/configurations. Applied to LLM inference systems, it detected 8 new energy inefficiency cases.
* `training` `kernel` `hardware` [Chopper: A Multi-Level GPU Characterization Tool & Derived Insights Into LLM Training Inefficiency](http://arxiv.org/abs/2512.08242v1)
  > **TL;DR**: Presents Chopper, a multi-level GPU profiling tool, to characterize inefficiencies in distributed LLM training. Analyzes hardware counters and kernel traces on AMD MI300X, revealing frequency overhead (DVFS) as the largest performance gap contributor, exceeding MFMA underutilization.

### 2025-12-08
* `video` `serving` `networking` [Communication-Efficient Serving for Video Diffusion Models with Latent Parallelism](http://arxiv.org/abs/2512.07350v1)
  > **TL;DR**: Addresses the communication bottleneck in parallel serving of Video Diffusion Models (VDMs). Proposes Latent Parallelism, which rotates partitioning dimensions in latent space to exploit local dependencies. Reduces communication overhead by up to 97% while maintaining generation quality.
* `edge` `RAG` `multi-modal` [Venus: An Efficient Edge Memory-and-Retrieval System for VLM-based Online Video Understanding](http://arxiv.org/abs/2512.07344v1)
  > **TL;DR**: Addresses high system overhead for VLM-based online video understanding deployments. Proposes Venus, an edge-cloud architecture with on-device keyframe memory construction and a progressive sampling algorithm for retrieval. Achieves a 15x-131x speedup in total response latency.
* `hardware` `kernel` `serving` [DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management](http://arxiv.org/abs/2512.07312v1)
  > **TL;DR**: Proposes DCO, a predictive cache management system for multi-core AI accelerators running LLMs. It uses dataflow-aware replacement policies, dead-block prediction, and thrashing mitigation. Achieves up to 1.80x speedup over conventional cache architectures in cycle-accurate simulation.

### 2025-12-07
* `video` `serving` `offline` [Optimizing video analytics inference pipelines: a case study](http://arxiv.org/abs/2512.07009v1)
  > **TL;DR**: Investigates how to optimize the computational cost of a video analytics pipeline for livestock monitoring. Proposes system-level optimizations including GPU acceleration, multi-level parallelization, and vectorized clustering. Achieves a 2x speedup on real-world farm video footage.
* `MoE` `edge` `training` [Stable-MoE: Lyapunov-based Token Routing for Distributed Mixture-of-Experts Training over Edge Networks](http://arxiv.org/abs/2512.06784v1)
  > **TL;DR**: Presents Stable-MoE, a Lyapunov-based framework for stable token routing and resource allocation in distributed MoE training over heterogeneous edge networks. Achieves up to 40% higher system throughput and 5% better accuracy on SVHN/CIFAR-100.

### 2025-12-06
* `training` `RL` [A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation](http://arxiv.org/abs/2512.06547v1)
  > **TL;DR**: Addresses training overhead of decoupled loss algorithms like GRPO for asynchronous RLHF. Proposes A-3PO, which approximates the proximal policy via interpolation instead of an extra forward pass. Reduces training time by 18% while maintaining performance.
* `kernel` `quantization` `edge` [Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge Devices](http://arxiv.org/abs/2512.06443v1)
  > **TL;DR**: Addresses low memory bandwidth utilization in parallel, ultra-low-bit LLM inference on edge devices. Proposes Vec-LUT, a vectorized lookup paradigm with a new tensor layout and cache-aware streaming. Achieves up to 4.2x speedup over state-of-the-art baselines.

### 2025-12-05
* `kernel` `quantization` `offloading` [Compiler-supported reduced precision and AoS-SoA transformations for heterogeneous hardware](http://arxiv.org/abs/2512.05516v1)
  > **TL;DR**: Investigates data layout (AoS-to-SoA) and precision transformations for GPU kernels on heterogeneous hardware. Proposes compiler annotations to orchestrate CPU/GPU data conversions and offloading strategies. Achieves a 2.6x speedup on Nvidia G200 platforms for particle simulation kernels.

### 2025-12-04
* `kernel` `thinking` [Counting Without Running: Evaluating LLMs' Reasoning About Code Complexity](http://arxiv.org/abs/2512.04355v1)
  > **TL;DR**: Introduces gpuFLOPBench, a benchmark to evaluate if LLMs can predict FLOP counts for CUDA kernels without running them. Tests models' ability to reason about code complexity and hidden performance characteristics. Analysis shows newer models achieve perfect accuracy on straightforward kernels but large errors on cases with implicit FLOPs from intrinsics or compiler behavior.

### 2025-12-03
* `MoE` `serving` `edge` [OD-MoE: On-Demand Expert Loading for Cacheless Edge-Distributed MoE Inference](http://arxiv.org/abs/2512.03927v1)
  > **TL;DR**: Presents OD-MoE, a distributed MoE inference framework that eliminates the need for expert caching via on-demand loading and a predictive pre-fetcher. Enables inference on edge nodes with <1GB GPU memory, achieving 75% of the decoding speed of a fully cached deployment while using only 1/3 of the GPU memory.
* `training` `offloading` `storage` [FFTrainer: Fast Failover in Large-Language Model Training with Almost-Free State Management](http://arxiv.org/abs/2512.03644v1)
  > **TL;DR**: Addresses slow recovery from node failures in distributed LLM training. Proposes FFTrainer, which uses surplus network capacity for lightweight, continuous state management. Reduces recovery time by up to 98% and mitigates GPU utilization loss by 68%.
* `serving` `offloading` `networking` [TokenScale: Timely and Accurate Autoscaling for Disaggregated LLM Serving with Token Velocity](http://arxiv.org/abs/2512.03416v1)
  > **TL;DR**: Presents TokenScale, an autoscaling framework for disaggregated LLM serving that uses Token Velocity as a predictive metric and Convertible Decoders for dynamic resource use. This approach improves SLO attainment from 50-88% to 80-96% and reduces costs by 4-14% over state-of-the-art systems.

### 2025-12-02
* `serving` `kernel` [TokenPowerBench: Benchmarking the Power Consumption of LLM Inference](http://arxiv.org/abs/2512.03024v1)
  > **TL;DR**: Addresses the lack of tools for measuring LLM inference power consumption. Presents TokenPowerBench, a benchmark with declarative configuration, power measurement, and phase-aligned metrics attribution. Shows how varying batch size and quantization affects joules per token for models up to 405B parameters.
* `offloading` `storage` `training` [Offloading Artificial Intelligence Workloads across the Computing Continuum by means of Active Storage Systems](http://arxiv.org/abs/2512.02646v1)
  > **TL;DR**: Proposes an architecture using active storage systems to offload AI workloads across heterogeneous devices. Embeds computation in storage to minimize data movement, implemented with dataClay. Achieves significant reductions in memory consumption and training times while maintaining accuracy.

### 2025-12-01
* `serving` `offloading` [Tangram: Accelerating Serverless LLM Loading through GPU Memory Reuse and Affinity](http://arxiv.org/abs/2512.01357v1)
  > **TL;DR**: Addresses the cold-start latency bottleneck in serverless LLM serving due to model loading. Proposes Tangram, a system with a unified GPU memory pool for tensor reuse, on-demand KV cache allocation, and affinity-aware scheduling. Achieves up to 6.2x faster model loading.

### 2025-11-29
* `serving` `edge` `offloading` [IslandRun: Privacy-Aware Multi-Objective Orchestration for Distributed AI Inference](http://arxiv.org/abs/2512.00595v1)
  > **TL;DR**: Addresses the challenge of orchestrating LLM inference across heterogeneous devices (edge, cloud) with conflicting goals of performance, privacy, and cost. Proposes IslandRun, a system using agent-based routing and reversible anonymization to route compute to data. Achieves improved privacy and multi-objective optimization.

### 2025-11-28
* `training` `offloading` `edge` [Communication-Computation Pipeline Parallel Split Learning over Wireless Edge Networks](http://arxiv.org/abs/2511.23167v1)
  > **TL;DR**: Proposes a pipeline-parallel split learning method to overlap communication and computation in distributed training over wireless edge networks. Uses micro-batching and jointly optimizes task split and resource allocation. Reduces total training time by over 38% while maintaining accuracy.
* `serving` `RL` `networking` [Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems](http://arxiv.org/abs/2511.22880v1)
  > **TL;DR**: Addresses performance skew from serving heterogeneous LoRA adapters. Proposes LoRAServe, a dynamic adapter placement and routing framework that rebalances adapters across GPUs using RDMA for remote access. Achieves up to 2× higher throughput and 50% fewer GPUs under SLOs compared to SOTA.

### 2025-11-27
* `serving` `edge` `networking` [DisCEdge: Distributed Context Management for Large Language Models at the Edge](http://arxiv.org/abs/2511.22599v1)
  > **TL;DR**: Proposes DisCEdge, a distributed context management system for edge LLM serving. It stores/replicates user context as token sequences instead of raw text to reduce synchronization overhead. Improves median response time by 14.46% and reduces client request size by 90%.
* `serving` `MoE` `sparse` [OmniInfer: System-Wide Acceleration Techniques for Optimizing LLM Serving Throughput and Latency](http://arxiv.org/abs/2511.22481v1)
  > **TL;DR**: Proposes OmniInfer, a unified LLM serving system that optimizes throughput and latency via Mixture-of-Experts scheduling and sparse attention acceleration. Built on vLLM, it reduces TPOT by 36% and TTFT by 38% in cluster evaluations.
* `kernel` `serving` [PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel](http://arxiv.org/abs/2511.22333v1)
  > **TL;DR**: Proposes PAT, a prefix-aware attention kernel for LLM decoding that packs queries by shared prefix and uses a multi-tile design to minimize KV cache memory accesses. Achieves 67.4% average attention latency reduction over state-of-the-art kernels.

### 2025-11-26
* `serving` `edge` `offloading` [DSD: A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving](http://arxiv.org/abs/2511.21669v1)
  > **TL;DR**: Addresses the challenge of speculative decoding for LLM inference in distributed edge-cloud environments. Introduces DSD, a framework for coordinated draft-target execution across devices, evaluated via a custom simulator and adaptive window control. Achieves up to 1.1x speedup and 9.7% higher throughput.
* `training` `MoE` `offloading` [MemFine: Memory-Aware Fine-Grained Scheduling for MoE Training](http://arxiv.org/abs/2511.21431v1)
  > **TL;DR**: Proposes MemFine, a memory-aware scheduling framework to solve the memory bottleneck in MoE training caused by dynamic token routing. It uses chunk-level decomposition and optimized recomputation guided by a memory model. Reduces activation memory by 48.03% and increases throughput by 4.42% versus full recomputation.
* `serving` `edge` [Automated Dynamic AI Inference Scaling on HPC-Infrastructure: Integrating Kubernetes, Slurm and vLLM](http://arxiv.org/abs/2511.21413v1)
  > **TL;DR**: Proposes a system to dynamically scale LLM inference on HPC infrastructure by integrating vLLM, Slurm, and Kubernetes to handle synchronous user-facing workloads. Achieves efficient scaling for up to 1000 concurrent requests with only ~500 ms end-to-end latency overhead.
* `serving` [A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving](http://arxiv.org/abs/2511.20982v1)
  > **TL;DR**: Proposes DOPD, a dynamic LLM inference system that adjusts prefill-to-decoding instance allocations based on real-time load to resolve workload imbalance. Achieves up to 1.5x higher goodput and reduces P90 TTFT by 67.5% compared to vLLM and DistServe.
* `serving` `agentic` `offline` [Aragog: Just-in-Time Model Routing for Scalable Serving of Agentic Workflows](http://arxiv.org/abs/2511.20975v1)
  > **TL;DR**: Proposes Aragog, a system for cost-aware LLM serving of agentic workflows using just-in-time model routing. It decouples configuration into an offline accuracy-preserving step and an online per-stage scheduler. At peak load, achieves 50-217% higher throughput and 32.5-78.9% lower median latency.

### 2025-11-25
* `serving` `offloading` `hardware` [Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management](http://arxiv.org/abs/2511.20172v1)
  > **TL;DR**: Introduces Beluga, a CXL-based memory architecture for efficient KVCache management in LLM serving. It enables GPUs to natively access a shared memory pool via CXL switches, reducing access latency. Achieves 89.6% lower TTFT and 7.35x higher throughput vs. RDMA-based solutions.
* `training` `kernel` [QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](http://arxiv.org/abs/2511.20100v1)
  > **TL;DR**: Proposes Macro Thinking Micro Coding (MTMC), a hierarchical LLM framework for generating high-performance GPU kernels. Decouples optimization strategy (reinforcement learning) from implementation (LLM coding). Achieves up to 7.3x speedup over LLMs and 2.2x over expert-optimized PyTorch kernels.
* `training` `networking` `sparse` [ParaBlock: Communication-Computation Parallel Block Coordinate Federated Learning for Large Language Models](http://arxiv.org/abs/2511.19959v1)
  > **TL;DR**: Addresses high communication latency in federated fine-tuning of large LLM blocks. Proposes ParaBlock, a scheme with parallel communication and computation threads to overlap these phases. Reduces communication costs by up to 42% while maintaining model convergence and performance.
* `diffusion` `edge` [Batch Denoising for AIGC Service Provisioning in Wireless Edge Networks](http://arxiv.org/abs/2511.19847v1)
  > **TL;DR**: Optimizes image generation service quality in wireless edge networks under latency constraints. Proposes STACKING, an algorithm for batch denoising that exploits step importance for parallelism. Achieves lower per-step delay and higher service quality compared to baseline methods.

### 2025-11-24
* `edge` `storage` `RAG` [AME: An Efficient Heterogeneous Agentic Memory Engine for Smartphones](http://arxiv.org/abs/2511.19192v1)
  > **TL;DR**: Proposes AME, a hardware-aware vector database engine for on-device agents on smartphones. It co-designs an efficient matrix pipeline and workload scheduler with mobile SoC constraints. Achieves 1.4x higher query throughput and 7x faster index construction.
* `kernel` [Low-Rank GEMM: Efficient Matrix Multiplication via Low-Rank Approximation with FP8 Acceleration](http://arxiv.org/abs/2511.18674v1)
  > **TL;DR**: Proposes Low-Rank GEMM, a matrix multiplication method using low-rank approximations and FP8 precision to reduce complexity and improve hardware efficiency. Achieves a 7.8x speedup and 75% memory savings over PyTorch FP32 on large matrices.

### 2025-11-22
* `edge` `multi-modal` `offloading` [AVERY: Adaptive VLM Split Computing through Embodied Self-Awareness for Efficient Disaster Response Systems](http://arxiv.org/abs/2511.18151v1)
  > **TL;DR**: Presents AVERY, an adaptive split-computing framework for Vision-Language Models on UAVs. It uses a dual-stream split (context/insight) and a self-aware controller to dynamically offload processing. Outperforms baselines with 93.98% lower energy consumption versus full-edge execution.
* `training` `networking` [Pier: Efficient Large Language Model pretraining with Relaxed Global Communication](http://arxiv.org/abs/2511.17849v1)
  > **TL;DR**: Addresses the high cost of global communication (e.g., all-reduce) in LLM pretraining. Proposes Pier, an optimizer with relaxed communication via momentum warmup/decay. Achieves a 2.7x-3.7x speedup in GPT-2 XL training on 256 A100s without performance loss.

### 2025-11-21
* `training` `MoE` `hardware` [Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design](http://arxiv.org/abs/2511.17127v1)
  > **TL;DR**: Presents a full-stack system design for training MoE foundation models on AMD MI300X GPUs with Pollara interconnect. Characterizes cluster networking, memory bandwidth, and introduces hardware-aware transformer sizing rules. Achieves competitive model performance with optimized throughput on pure AMD hardware.
* `kernel` `serving` [Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2511.16964v1)
  > **TL;DR**: Investigates multi-agent LLM systems for optimizing PyTorch inference. Proposes a framework to compare different agent strategies, finding exploit-heavy strategies with error-fixing agents work best. Achieves an average 2.88x speedup on an H100 GPU across the KernelBench suite.
* `training` `MoE` [MicroMoE: Fine-Grained Load Balancing for Mixture-of-Experts with Token Scheduling](http://arxiv.org/abs/2511.16947v1)
  > **TL;DR**: Addresses load imbalance in MoE model training. Proposes MicroMoE, a distributed system with MicroEP, a fine-grained parallelization strategy using token scheduling for load balancing. Improves end-to-end training throughput by up to 47.6%.

### 2025-11-20
* `training` `RL` `kernel` [Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter](http://arxiv.org/abs/2511.16665v1)
  > **TL;DR**: Proposes TLT, a system to accelerate reasoning RL training by using adaptive speculative decoding to overcome the bottleneck of long-tail response generation. Achieves over 1.7x end-to-end training speedup while preserving model accuracy.
* `RL` `serving` [Fast LLM Post-training via Decoupled and Best-of-N Speculation](http://arxiv.org/abs/2511.16193v1)
  > **TL;DR**: Addresses the slow rollout phase in LLM post-training (e.g., RLHF). Introduces SpecActor, which uses dynamic decoupled speculation and Best-of-N drafting to maximize GPU efficiency and draft accuracy. Achieves 1.3–1.7× speedup over common post-training baselines.
* `kernel` `hardware` [Can Asymmetric Tile Buffering Be Beneficial?](http://arxiv.org/abs/2511.16041v1)
  > **TL;DR**: Proposes Asymmetric Tile Buffering (ATB) to decouple input and output operand tile dimensions in GEMM, increasing arithmetic intensity. A performance model balances ATB benefits against kernel switching overhead. On AMD XDNA2 AI Engine, achieves 4.54x speedup (4.8 to 24.6 TFLOPS) for BFP16-BF16 GEMM.
* `hardware` `serving` `offloading` [A Scalable NorthPole System with End-to-End Vertical Integration for Low-Latency and Energy-Efficient LLM Inference](http://arxiv.org/abs/2511.15950v1)
  > **TL;DR**: Presents a vertically integrated system with NorthPole accelerators for scalable, low-latency LLM inference. Combines custom hardware, a runtime stack, and containerized pipeline. Delivers 2.8 ms per-user inter-token latency for an 8B model while consuming only 30 kW.

### 2025-11-19
* `kernel` `hardware` `serving` [A Tensor Compiler for Processing-In-Memory Architectures](http://arxiv.org/abs/2511.15503v1)
  > **TL;DR**: Proposes DCC, a data-centric compiler optimizing data layouts and compute code jointly for ML kernels on PIM architectures. It uses a unified tuning process with performance prediction. Achieves up to 7.71x speedup for LLM inference over GPU-only execution.
* `networking` `MoE` `training` [GPU-Initiated Networking for NCCL](http://arxiv.org/abs/2511.15076v1)
  > **TL;DR**: Investigates how to reduce GPU communication latency for AI workloads like MoE models. Proposes GPU-Initiated Networking (GIN), an NCCL extension enabling direct, CPU-bypass GPU-to-GPU communication via RDMA. Achieves lower latency device-initiated operations within NCCL's collective runtime.

### 2025-11-18
* `RL` `serving` [Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning](http://arxiv.org/abs/2511.14617v1)
  > **TL;DR**: Addresses performance bottlenecks in synchronous LLM reinforcement learning. Proposes Seer, a system using online context learning with dynamic load balancing and grouped speculative decoding to exploit prompt similarities. Achieves up to 97% higher rollout throughput and 93% lower long-tail latency.
* `serving` `edge` `networking` [Hyperion: Hierarchical Scheduling for Parallel LLM Acceleration in Multi-tier Networks](http://arxiv.org/abs/2511.14450v1)
  > **TL;DR**: This paper tackles minimizing end-to-end latency for LLM inference in multi-tier networks. Hyperion uses a two-stage framework that jointly optimizes offline model partitioning and online request scheduling. It reduces latency by up to 52.1% compared to GPipe while increasing GPU utilization.
* `training` `offloading` `storage` [10Cache: Heterogeneous Resource-Aware Tensor Caching and Migration for LLM Training](http://arxiv.org/abs/2511.14124v1)
  > **TL;DR**: Addresses memory bottlenecks and tensor migration latency in LLM training. Proposes 10Cache, a system for intelligent tensor caching/migration across GPU, CPU, and NVMe tiers using prefetching and buffer reuse. Achieves up to 2x training speedup and an 86.6x higher GPU cache hit rate.
* `serving` `networking` [FailSafe: High-performance Resilient Serving](http://arxiv.org/abs/2511.14116v1)
  > **TL;DR**: Addresses how to sustain LLM serving performance under GPU failures in tensor-parallel settings. Proposes FailSafe with cyclic KVCache placement, hybrid attention, and dynamic routing, plus proactive backup and recovery. Achieves 2x higher throughput and 100x lower recovery latency vs. standard fault handling.
* `MoE` `offloading` `serving` [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts](http://arxiv.org/abs/2511.14102v1)
  > **TL;DR**: Addresses the I/O bottleneck of expert offloading in MoE model inference. Proposes MoE-SpeQ, a system that uses a draft model for speculative expert prefetching and an adaptive governor to overlap I/O with computation. Achieves a 2.34x speedup over the state-of-the-art offloading framework.

### 2025-11-17
* `kernel` `inference` `sparse` [MACKO: Sparse Matrix-Vector Multiplication for Low Sparsity](http://arxiv.org/abs/2511.13061v1)
  > **TL;DR**: Proposes MACKO-SpMV, a GPU-optimized kernel and storage format for efficient sparse matrix-vector multiplication at low (30-90%) sparsity levels common in pruned LLMs. Applied to Llama2-7B, it achieves 1.5x memory reduction and 1.5x faster inference at 50% sparsity compared to a dense baseline.

### 2025-11-16
* `kernel` `training` [Iris: First-Class Multi-GPU Programming Experience in Triton](http://arxiv.org/abs/2511.12500v1)
  > **TL;DR**: Presents Iris, a multi-GPU communication library implemented in Python/Triton to simplify and optimize compute-communication overlap. Introduces tile-based symmetric memory abstractions for single-source kernels. Achieves up to 1.79x speedup over PyTorch and RCCL in GEMM+All-Scatter workloads.

### 2025-11-15
* `video` `serving` [PipeDiT: Accelerating Diffusion Transformers in Video Generation with Task Pipelining and Model Decoupling](http://arxiv.org/abs/2511.12056v1)
  > **TL;DR**: Proposes PipeDiT, a pipelining framework to reduce inference latency and memory consumption for DiT-based video generation. Key innovations include pipeline sequence parallelism, decoupling diffusion/VAE modules, and attention co-processing. Achieves 1.06x-4.02x speedups over baseline frameworks.
* `serving` `kernel` `offloading` [Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding](http://arxiv.org/abs/2511.12031v1)
  > **TL;DR**: Proposes BMC, a KV cache allocation method balancing memory copies and compute redundancy for LLM inference. It pre-allocates tensors with extra rows for in-place updates, repurposing redundancy for speculative decoding. Achieves up to 3.2x throughput acceleration over baseline HuggingFace.

### 2025-11-14
* `agentic` `edge` `serving` [UFO$^3$: Weaving the Digital Agent Galaxy](http://arxiv.org/abs/2511.11332v1)
  > **TL;DR**: Presents UFO3, a system for orchestrating LLM-powered agents across heterogeneous devices. Models user requests as mutable distributed DAGs with asynchronous execution and dynamic optimization. Achieves 31% lower end-to-end latency compared to a sequential baseline.
* `edge` `offloading` `networking` [SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices](http://arxiv.org/abs/2511.11038v1)
  > **TL;DR**: Addresses how to perform accurate AI inference offloading for resource-constrained IoT devices over unreliable networks. Proposes SemanticNN, a semantic codec with a BER-aware decoder and soft quantization, that is resilient to bit errors. Reduces feature transmission volume by 56.82-344.83x while maintaining accuracy.

### 2025-11-13
* `training` `kernel` [Scalable Synthesis of distributed LLM workloads through Symbolic Tensor Graphs](http://arxiv.org/abs/2511.10480v1)
  > **TL;DR**: Presents STAGE, a framework to synthesize high-fidelity execution traces for modeling distributed LLM training workloads using symbolic tensor graphs. It enables systematic exploration of parallelization strategies and scales to model configurations spanning over 32K GPUs with tensor-level accuracy.
* `training` `hardware` [Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs](http://arxiv.org/abs/2511.09861v1)
  > **TL;DR**: Identifies Lit Silicon, a thermal imbalance causing straggler GPUs in multi-GPU LLM training systems. Proposes detection/mitigation techniques and power management models, including GPU/CPU power optimization. Achieves up to 6% performance and 4% power improvements on two LLM training frameworks.
* `training` [MoFa: A Unified Performance Modeling Framework for LLM Pretraining](http://arxiv.org/abs/2511.09837v1)
  > **TL;DR**: Presents MoFa, a performance modeling framework for distributed LLM pretraining that integrates optimization features and fault tolerance overhead. It uses an enhanced cost model and fault tolerance analysis to guide system tuning. Achieves high prediction accuracy across various scenarios.
* `hardware` `offloading` [Taiji: A DPU Memory Elasticity Solution for In-production Cloud Environments](http://arxiv.org/abs/2511.09936v1)
  > **TL;DR**: Proposes Taiji, a resource-elasticity architecture for Data Processing Units (DPUs) using hybrid virtualization and parallel memory swapping to enable memory overcommitment. Achieves over 50% memory expansion with ~5% virtualization overhead and 90% of swap-ins under 10 microseconds.

### 2025-11-12
* `training` `networking` [TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](http://arxiv.org/abs/2511.09741v1)
  > **TL;DR**: Proposes TawPipe, a topology-aware weight pipeline parallelism method for LLM training that optimizes communication hierarchy to reduce cross-node traffic. Groups devices by topology, assigns fixed weight shards, and overlaps communication. Achieves 1.56x higher throughput than WeiPipe on 24 GPUs with long sequences.
* `serving` `networking` [LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication](http://arxiv.org/abs/2511.09557v2)
  > **TL;DR**: Analyzes bottlenecks in multi-node LLM inference. Proposes NVRAR, a low-latency hierarchical all-reduce algorithm using NVSHMEM and recursive doubling. Reduces communication latency by up to 3.6x and end-to-end batch latency by 1.72x for large models like Llama 3.1 405B.
* `training` `kernel` [No Cords Attached: Coordination-Free Concurrent Lock-Free Queues](http://arxiv.org/abs/2511.09410v1)
  > **TL;DR**: Addresses high coordination overhead in concurrent queues for massively parallel AI workloads. Proposes Cyclic Memory Protection (CMP), a coordination-free lock-free queue using bounded protection windows. Outperforms state-of-the-art queues by 1.72-4x under high contention with hundreds of threads.
* `hardware` `serving` [Flex-MIG: Enabling Distributed Execution on MIG](http://arxiv.org/abs/2511.09143v2)
  > **TL;DR**: Addresses GPU underutilization from rigid NVIDIA MIG partitioning. Proposes Flex-MIG, a software framework enabling one-to-many allocation and shared-memory collectives for distributed execution across MIG instances. Improves cluster makespan by up to 17% on diverse traces.

### 2025-11-11
* `kernel` `sparse` [\uline{LO}w-c\uline{O}st yet High-\uline{P}erformant \uline{S}parse Matrix-Matrix Multiplication on Arm SME Architectures](http://arxiv.org/abs/2511.08158v1)
  > **TL;DR**: Proposes LOOPS, a hybrid framework for efficient SpMM on Arm SME architectures. Combines row-wise and vector-wise layouts to co-utilize NEON and SME units via an adaptive parallelization scheme. Achieves up to 14.4x (FP64) speedup over the baseline TACO on CPU.

### 2025-11-06
* `quantization` `sparse` `edge` [Enabling Dynamic Sparsity in Quantized LLM Inference](http://arxiv.org/abs/2511.04477v1)
  > **TL;DR**: Proposes techniques to enable dynamic sparsity in quantized LLM inference for edge devices. Key designs include a zigzag-patterned quantization layout and a specialized GEMV kernel for commodity GPUs. Achieves up to 1.55x faster decoding throughput while maintaining accuracy.

### 2025-11-05
* `edge` `serving` `kernel` [UMDAM: A Unified Data Layout and DRAM Address Mapping for Heterogenous NPU-PIM](http://arxiv.org/abs/2511.03293v1)
  > **TL;DR**: Proposes UMDAM, a unified data layout and DRAM mapping scheme for NPU-PIM co-execution to optimize LLM inference on edge devices. Uses a column-major, tile-based layout to eliminate bandwidth loss and redundant storage. Reduces TTFT by up to 3.0x and TTLT by 2.18x.
* `serving` `offloading` [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](http://arxiv.org/abs/2511.03092v1)
  > **TL;DR**: Proposes SnapStream, a KV cache compression method for efficient long-sequence LLM inference with continuous batching. It integrates sparsification into static-graph dataflow accelerators. Achieves 4× improved on-chip memory usage while maintaining throughput of 1832 tokens/sec for DeepSeek-671B at 128k context length.

### 2025-11-04
* `serving` `edge` `networking` [Federated Attention: A Distributed Paradigm for Collaborative LLM Inference over Edge Networks](http://arxiv.org/abs/2511.02647v1)
  > **TL;DR**: Proposes Federated Attention (FedAttn), a distributed LLM inference framework for edge networks that performs local self-attention and periodic KV matrix aggregation to preserve privacy and reduce communication. Achieves a 3.2x reduction in communication cost while maintaining response quality.
* `serving` `kernel` [From Models to Operators: Rethinking Autoscaling Granularity for Large Generative Models](http://arxiv.org/abs/2511.02248v1)
  > **TL;DR**: Proposes operator-level autoscaling for large generative models to replace inefficient model-level scaling. The framework profiles individual model operators and independently scales resources for each. Achieves up to 40% fewer GPUs while preserving SLOs, or 1.6x higher throughput under fixed resources.
* `training` `kernel` `networking` [Eliminating Multi-GPU Performance Taxes: A Systems Approach to Efficient Distributed LLMs](http://arxiv.org/abs/2511.02168v1)
  > **TL;DR**: Addresses performance inefficiencies in distributed LLM execution across multiple GPUs due to the Bulk Synchronous Parallel (BSP) model. Proposes fine-grained programming patterns using in-kernel communication primitives to replace global barriers with dataflow synchronization. Achieves a 10-20% speedup in end-to-end latency.
* `serving` `offloading` `agentic` [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](http://arxiv.org/abs/2511.02230v1)
  > **TL;DR**: Proposes Continuum, a serving system optimizing multi-turn LLM agent job completion time by predicting tool call durations to set KV cache time-to-live and enable program-level scheduling. Reduces average job completion time and prevents scheduling bubbles for real-world agent workloads.

### 2025-11-03
* `kernel` `training` `serving` [Optimizing Attention on GPUs by Exploiting GPU Architectural NUMA Effects](http://arxiv.org/abs/2511.02132v1)
  > **TL;DR**: Addresses performance degradation of attention kernels on disaggregated GPUs due to NUMA effects. Proposes Swizzled Head-first Mapping, a scheduling strategy aligning attention heads with NUMA domains for cache reuse. Achieves 50% higher performance on AMD MI300X with L2 cache hit rates of 80-97%.

### 2025-11-02
* `serving` `networking` [FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs](http://arxiv.org/abs/2511.00807v1)
  > **TL;DR**: Explores fair, energy/carbon-efficient LLM serving on distributed, heterogeneous GPUs. Proposes FREESH, a joint routing and scheduling system that optimizes query placement and GPU frequency based on spatiotemporal power/carbon variability. Achieves 45.45% emissions reduction and 28.6% energy saving while improving SLO attainment.
* `RL` `training` [AReaL-Hex: Accommodating Asynchronous RL Training over Heterogeneous GPUs](http://arxiv.org/abs/2511.00796v1)
  > **TL;DR**: Introduces AReaL-Hex, a heterogeneity-aware system for asynchronous RL training on heterogeneous GPUs. It uses a two-phase scheduler (MILP-based planning and graph partitioning) to assign stages optimally. On reasoning tasks, it achieves 1.50x higher throughput or 1.46x cost reduction versus homogeneous deployments.

### 2025-11-01
* `serving` `edge` `multi-modal` [EPARA: Parallelizing Categorized AI Inference in Edge Clouds](http://arxiv.org/abs/2511.00603v1)
  > **TL;DR**: Proposes EPARA, an end-to-end parallel inference framework for edge clouds that categorizes AI tasks by latency sensitivity and GPU needs. It uses a task allocator, distributed handler, and state-aware scheduler to improve resource allocation. Achieves up to 2.1x higher goodput compared to prior frameworks.

### 2025-10-31
* `networking` `serving` `RL` [RDMA Point-to-Point Communication for LLM Systems](http://arxiv.org/abs/2510.27656v1)
  > **TL;DR**: Addresses the need for flexible point-to-point communication in LLM systems using diverse NICs. Proposes TransferEngine, a uniform interface that exposes one-sided WriteImm operations with completion primitives. Achieves 400 Gbps throughput and enables 1.3-second RL weight updates for trillion-parameter models.
* `training` `multi-modal` [Synergistic Tensor and Pipeline Parallelism](http://arxiv.org/abs/2510.27257v1)
  > **TL;DR**: Proposes a synergistic schedule for tensor and pipeline parallelism to reduce communication and synchronization bubbles during LLM/MLLM training. Decouples forward/backward passes into fine-grained units and braids them into a composite sequence. Improves training throughput by up to 16% over baselines.
* `serving` `offloading` [SERFLOW: A Cross-Service Cost Optimization Framework for SLO-Aware Dynamic ML Inference](http://arxiv.org/abs/2510.27182v1)
  > **TL;DR**: Addresses cost-efficient, SLO-aware serving for dynamic ML inference where requests may exit early. Proposes SERFLOW, a framework that offloads model stages between IaaS VMs and FaaS functions using stage-specific provisioning and adaptive load balancing. Reduces cloud costs by over 23%.
* `serving` `RL` [Glia: A Human-Inspired AI for Automated Systems Design and Optimization](http://arxiv.org/abs/2510.27176v1)
  > **TL;DR**: Presents Glia, an automated AI system for designing computer systems using a multi-agent LLM workflow with reasoning and experimentation. Applied to distributed GPU clusters for LLM inference, it generates novel algorithms for routing and scheduling, achieving human-expert level performance in less time.

### 2025-10-30
* `training` `agentic` `RL` [FlowMesh: A Service Fabric for Composable LLM Workflows](http://arxiv.org/abs/2510.26913v1)
  > **TL;DR**: Proposes FlowMesh, a service fabric for fine-grained execution of composite LLM workflows (e.g., RLHF, agent workflows). It decomposes workflows into operators for cross-user deduplication, batching, and global scheduling. Achieves up to 3.8x cost reduction compared to baseline pipelines.
* `serving` `MoE` `offloading` [ExpertFlow: Adaptive Expert Scheduling and Memory Coordination for Efficient MoE Inference](http://arxiv.org/abs/2510.26730v1)
  > **TL;DR**: Proposes ExpertFlow, a runtime system for MoE inference that uses adaptive expert prefetching and cache-aware routing to reduce latency from parameter transfers. It dynamically predicts expert needs to minimize cache misses, reducing model stall time to <0.1% of the baseline.
* `training` `sparse` `networking` [An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning](http://arxiv.org/abs/2510.26709v2)
  > **TL;DR**: Proposes ARC-Top-K, an All-Reduce-compatible gradient compressor for distributed training. It aligns sparsity patterns via a lightweight sketch to enable index-free communication. Achieves up to 60.7% reduction in wall-clock training time while matching Top-K accuracy.
* `RL` `training` `serving` [ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems](http://arxiv.org/abs/2510.26475v1)
  > **TL;DR**: Adapts speculative decoding to accelerate generation in RL training systems. Proposes ReSpec with dynamic SD tuning, drafter distillation, and reward-weighted updates to mitigate staleness and policy drift. Achieves up to 4.5x speedup on Qwen models (3B-14B) while preserving reward convergence.

### 2025-10-29
* `MoE` `serving` `networking` [MoEntwine: Unleashing the Potential of Wafer-scale Chips for Large-scale Expert Parallel Inference](http://arxiv.org/abs/2510.25258v1)
  > **TL;DR**: Proposes ER-Mapping and NI-Balancer to optimize MoE inference on wafer-scale chips by co-designing expert and attention layer mappings and overlapping expert migration with idle network links, achieving 62% communication reduction and 39% higher per-device performance than NVL72.

### 2025-10-27
* `edge` `offloading` `serving` [Bayes-Split-Edge: Bayesian Optimization for Constrained Collaborative Inference in Wireless Edge Systems](http://arxiv.org/abs/2510.23503v1)
  > **TL;DR**: Proposes Bayes-Split-Edge, a Bayesian optimization framework for jointly optimizing neural network split points and transmission power in edge-served inference to meet energy and latency constraints. Achieves 2.4x reduction in evaluation cost and near-linear convergence with as few as 20 function evaluations.
* `offloading` `edge` `serving` [Rethinking Inference Placement for Deep Learning across Edge and Cloud Platforms: A Multi-Objective Optimization Perspective and Future Directions](http://arxiv.org/abs/2510.22909v1)
  > **TL;DR**: Addresses optimal placement of DL model inference across edge and cloud to balance latency, cost, and privacy. Proposes a multi-objective optimization framework for partitioning and offloading, enabling tailored deployment for latency-sensitive applications like chatbots.

### 2025-10-23
* `training` `networking` `hardware` [Collective Communication for 100k+ GPUs](http://arxiv.org/abs/2510.20171v1)
  > **TL;DR**: Addresses communication bottlenecks in training LLMs at 100k+ GPU scales. Proposes NCCLX, a collective communication framework optimizing all-reduce and all-gather for ultra-scale clusters, achieving significant throughput and latency improvements over standard NCCL on Llama4 training.
* `training` `MoE` `offloading` [AsyncHZP: Hierarchical ZeRO Parallelism with Asynchronous Scheduling for Scalable LLM Training](http://arxiv.org/abs/2510.20111v1)
  > **TL;DR**: Proposes AsyncHZP, an asynchronous hierarchical ZeRO variant that reduces communication overhead and improves memory efficiency in large-scale LLM training. By adaptively resharding and overlapping communication with computation via multi-stream scheduling, it achieves superior scalability over ND parallelism on both dense and MoE models.

### 2025-10-22
* `MoE` `training` `networking` [HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid Expert/Data Transmission](http://arxiv.org/abs/2510.19470v1)
  > **TL;DR**: Addresses scalability limits of MoE training across datacenters due to low cross-DC bandwidth. Proposes HybridEP, a framework that dynamically optimizes expert and data communication patterns via modeling-guided hybrid transmission, achieving up to 5.6x speedup under constrained bandwidth.
* `training` `MoE` `networking` [RailS: Load Balancing for All-to-All Communication in Distributed Mixture-of-Experts Training](http://arxiv.org/abs/2510.19262v2)
  > **TL;DR**: Addresses all-to-all communication bottlenecks in distributed MoE training by leveraging Rail topology symmetry to enable local, topology-aware load balancing. RailS uses multipath spraying and LPT scheduling to reduce iteration time by up to 40% and boost bus bandwidth by up to 78%.
* `RL` `serving` `offloading` [RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs](http://arxiv.org/abs/2510.19225v1)
  > **TL;DR**: RLBoost improves cost-efficiency of LLM reinforcement learning by harvesting preemptible GPUs for rollout stages, using adaptive offloading, pull-based weight transfer, and token-level response migration. It achieves 1.51x–1.97x higher throughput and 28%–49% better cost efficiency than on-demand-only setups.

### 2025-10-21
* `training` `sparse` `distributed` [MTraining: Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training](http://arxiv.org/abs/2510.18830v1)
  > **TL;DR**: Addresses inefficient training of LLMs with ultra-long contexts due to computational imbalance. Proposes MTraining, a distributed method with dynamic sparse attention and balanced ring attention, achieving 6× higher training throughput while scaling context from 32K to 512K tokens on 32 A100 GPUs.
* `serving` `offloading` [Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications](http://arxiv.org/abs/2510.18586v1)
  > **TL;DR**: Addresses KV-cache inefficiencies in LLM-based multi-agent serving due to memory contention and idle stalls. Tokencake introduces agent-aware space and time schedulers with proactive KV-cache offloading and predictive upload, reducing end-to-end latency by 47.06% and boosting GPU memory utilization by 16.9% over vLLM.
* `serving` `edge` `thinking` [SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices](http://arxiv.org/abs/2510.18544v1)
  > **TL;DR**: Addresses SLO-aware LLM inference scheduling on edge devices by optimizing TTFT, TPOT, and end-to-end latency. Proposes SLICE, a utility-driven scheduler with dynamic generation rate control, achieving up to 35× higher SLO attainment than Orca and FastServe.

### 2025-10-20
* `training` `kernel` `sparse` [Efficient Long-context Language Model Training by Core Attention Disaggregation](http://arxiv.org/abs/2510.18121v1)
  > **TL;DR**: Proposes Core Attention Disaggregation (CAD) to eliminate load imbalance in long-context LLM training by offloading attention computation to dedicated devices. Leverages stateless, composable attention kernels with dynamic rebatching, achieving up to 1.35x training throughput on 512k context lengths.

### 2025-10-19
* `serving` `offloading` `thinking` [Justitia: Fair and Efficient Scheduling for LLM Applications](http://arxiv.org/abs/2510.17015v1)
  > **TL;DR**: Addresses unfair and inefficient scheduling of LLM applications in shared GPU environments. Proposes Justitia, a memory-centric scheduler using neural demand prediction and virtual-time fair queuing to balance efficiency and worst-case guarantees, reducing completion time by up to 3.2× while ensuring fairness compared to vLLM.

### 2025-10-18
* `networking` `training` [Reimagining RDMA Through the Lens of ML](http://arxiv.org/abs/2510.16606v1)
  > **TL;DR**: Celeris redesigns RDMA transport for distributed ML training by eliminating strict reliability and in-order delivery, leveraging ML's fault tolerance to reduce tail latency. By removing retransmissions and using software-level recovery (e.g., Hadamard Transform), it cuts 99th-percentile latency by 2.3x and reduces BRAM usage by 67%.
* `serving` `offloading` `quantization` [FourierCompress: Layer-Aware Spectral Activation Compression for Efficient and Accurate Collaborative LLM Inference](http://arxiv.org/abs/2510.16418v1)
  > **TL;DR**: Addresses communication bottlenecks in collaborative LLM inference by compressing intermediate activations across edge-client and server boundaries. Proposes FourierCompress, a layer-aware FFT-based method retaining low-frequency coefficients, achieving 7.6x compression with <0.3% accuracy loss and 32x faster compression than Top-k.
* `training` `offloading` `sparse` [MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](http://arxiv.org/abs/2510.16415v1)
  > **TL;DR**: Proposes MeCeFO, a fault-tolerant LLM training system that transfers failed node workloads to neighbors with minimal overhead using skip-connections, recomputation, and low-rank gradient approximation. Achieves 5.0–6.7× higher resilience than SOTA with only 4.18% throughput drop under failures.

### 2025-10-17
* `training` `serving` `hardware` [GOGH: Correlation-Guided Orchestration of GPUs in Heterogeneous Clusters](http://arxiv.org/abs/2510.15652v1)
  > **TL;DR**: Proposes GOGH, a learning-based system for adaptive GPU resource orchestration in heterogeneous clusters to minimize energy and meet performance targets. Uses two neural networks to predict model-hardware compatibility and co-location effects, improving allocation over time. Reduces energy consumption by up to 27% while maintaining SLOs.
* `training` `kernel` `sparse` [PRISM: Probabilistic Runtime Insights and Scalable Performance Modeling for Large-Scale Distributed Training](http://arxiv.org/abs/2510.15596v1)
  > **TL;DR**: PRISM models stochastic performance variability in large-scale distributed training, offering probabilistic guarantees on training time. It identifies communication kernels (e.g., AllGather, ReduceScatter) as primary variability sources and enables 1.26x performance gains via placement-aware optimization, with 20.8% KS distance in prediction accuracy.
* `serving` `thinking` [BeLLMan: Controlling LLM Congestion](http://arxiv.org/abs/2510.15330v1)
  > **TL;DR**: Addresses LLM inference congestion due to uncontrolled autoregressive token generation. Proposes beLLMan, a feedback controller that dynamically adjusts output length based on system load, reducing end-to-end latency by up to 8× and energy use by 25% while increasing request throughput by 19%.

### 2025-10-16
* `serving` `multi-modal` `offloading` [xLLM Technical Report](http://arxiv.org/abs/2510.14686v1)
  > **TL;DR**: Designs xLLM, a high-performance LLM serving framework with decoupled service-engine architecture for multimodal inference, featuring dynamic PD/EPD disaggregation and global KV cache management. Achieves up to 2.2x higher throughput than vLLM-Ascend under identical TPOT constraints on Qwen models.
* `training` `offloading` `hardware` [ScalePool: Hybrid XLink-CXL Fabric for Composable Resource Disaggregation in Unified Scale-up Domains](http://arxiv.org/abs/2510.14580v1)
  > **TL;DR**: Proposes ScalePool, a hybrid XLink-CXL fabric for disaggregated memory and accelerator interconnection in LLM training clusters. Achieves 1.22x average speedup and 4.5x lower latency for memory-intensive workloads by enabling coherent, tiered memory pooling beyond RDMA.
* `serving` `offline` [FairBatching: Fairness-Aware Batch Formation for LLM Inference](http://arxiv.org/abs/2510.14392v1)
  > **TL;DR**: Addresses computational unfairness in LLM inference batching by redesigning batch formation to balance prefill and decode tasks. Introduces adaptive budgeting and dynamic scheduling to reduce TTFT tail latency by up to 2.29x while improving single-node and cluster capacity by 20.0% and 54.3%, respectively.

### 2025-10-15
* `serving` `offloading` `networking` [FIRST: Federated Inference Resource Scheduling Toolkit for Scientific AI Model Access](http://arxiv.org/abs/2510.13724v1)
  > **TL;DR**: FIRST enables federated LLM inference across distributed HPC clusters using a cloud-like API, auto-scaling resources and maintaining hot nodes for low-latency serving. It achieves scalable, on-premises generation of billions of tokens daily without commercial cloud reliance.
* `serving` `offloading` `thinking` [Adaptive Rescheduling in Prefill-Decode Disaggregated LLM Inference](http://arxiv.org/abs/2510.13668v1)
  > **TL;DR**: Addresses workload imbalance in disaggregated LLM inference due to unpredictable decode lengths. Proposes ARES, a system using LLM-internal state to predict output length and dynamically reschedule decode tasks, reducing P99 TPOT by 74.77% and improving goodput by 2.24×.
* `quantization` `hardware` `edge` [F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs](http://arxiv.org/abs/2510.13401v1)
  > **TL;DR**: Designs a hardware accelerator (F-BFQ) to efficiently execute mixed block-floating-point quantized LLMs on edge devices, dynamically switching between BFP variants without reconfiguration. Achieves 1.4x faster inference than Arm NEON CPU and 5.2 tokens/s on AMD Kria.
* `serving` `offloading` `RAG` [BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure](http://arxiv.org/abs/2510.13223v1)
  > **TL;DR**: Addresses load imbalance in disaggregated LLM serving by dynamically migrating KV cache and model layers between prefill and decode nodes. Enables cache-agnostic scheduling with overlapped transmission, achieving 1.2x-3.9x higher throughput and up to 78.4% lower latency than vLLM.

### 2025-10-14
* `RL` `training` `offloading` [Laminar: A Scalable Asynchronous RL Post-Training Framework](http://arxiv.org/abs/2510.12633v1)
  > **TL;DR**: Addresses scalability bottlenecks in RL post-training of LLMs caused by trajectory latency skew. Proposes Laminar, a fully decoupled architecture with tiered relay workers for asynchronous parameter updates and dynamic trajectory repackaging. Achieves up to 5.48× training throughput speedup on a 1024-GPU cluster.

### 2025-10-13
* `training` `serving` `offloading` [An Explorative Study on Distributed Computing Techniques in Training and Inference of Large Language Models](http://arxiv.org/abs/2510.11211v1)
  > **TL;DR**: Explores distributed computing techniques for training and serving LLMs, including system modifications to enable consumer-grade deployment and a comparative analysis of three serving frameworks. Implements a metaheuristic-based offloading method that reduces memory usage by up to 40% on consumer hardware.

### 2025-10-12
* `training` `sparse` `offloading` [DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism](http://arxiv.org/abs/2510.10620v1)
  > **TL;DR**: Addresses dynamic sequence length and attention pattern variability in long-context LLM training. Proposes DCP, a fine-grained dynamic context parallelism framework that adapts data and computation blocks to device resources, achieving up to 1.46x end-to-end training speed-up over static methods.

### 2025-10-11
* `serving` `offloading` `MoE` [SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference](http://arxiv.org/abs/2510.10302v1)
  > **TL;DR**: Proposes SP-MoE, an SD-aware offloading framework for MoE-based LLM inference that prefetches experts using draft-target model correspondence and pipelines I/O to reduce latency. Achieves 1.07–3.5× speedup in tokens-per-token (TPOT) over state-of-the-art methods.

### 2025-10-10
* `serving` `hardware` [SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference](http://arxiv.org/abs/2510.08544v1)
  > Designs specialized hardware chips for disaggregated LLM inference, tailoring prefill (compute-heavy) and decode (memory-heavy) stages to their distinct workloads. SPAD achieves 19%-41% lower hardware cost and 2%-17% lower TDP than GPU-based systems while maintaining performance.

### 2025-10-09
* `serving` `MoE` `offloading` [From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Prefill](http://arxiv.org/abs/2510.08055v1)
  > Addresses high energy and latency in MoE LLM serving due to redundant expert weight loads during chunked prefill. Proposes layered prefill, which schedules by layer groups instead of tokens to eliminate reloads. Reduces TTFT by up to 70% and per-token energy by 22% while maintaining stall-free decoding.

### 2025-10-08
* `kernel` [Vectorized FlashAttention with Low-cost Exponential Computation in RISC-V Vector Processors](http://arxiv.org/abs/2510.06834v1)
  > **TL;DR**: Vectorizes FlashAttention on RISC-V vector processors using low-cost exponential approximations and tiling to optimize attention kernels. Achieves significant performance gains without custom ISA extensions.
* `RL` `training` [EARL: Efficient Agentic Reinforcement Learning Systems for Large Language Models](http://arxiv.org/abs/2510.05943v1)
  > **TL;DR**: Addresses scalability bottlenecks in agentic RL training for LLMs by dynamically adapting parallelism and decentralizing intermediate data exchange. Achieves 2.1× higher throughput and eliminates OOM failures during long-context RL training.

### 2025-10-01 ~ 2025-10-07
* `agentic` `offloading` `serving` [Toward Systems Foundations for Agentic Exploration](http://arxiv.org/abs/2510.05556v1)
  > **TL;DR**: Addresses the need for efficient execution branching in LLM agents. Proposes foundational support for fork semantics, side-effect isolation, and microsecond-scale forking. Enables scalable agentic exploration with order-of-magnitude faster snapshot/restore than existing tools.
* `MoE` `serving` [Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting](http://arxiv.org/abs/2510.05497v1)
  > **TL;DR**: Addresses data movement bottlenecks in large-scale MoE LLM serving by forecasting expert selection patterns via massive profiling, enabling architectural optimizations that achieve up to 6.3X speedup on DeepSeek V3.
* `training` `offloading` [OptPipe: Memory- and Scheduling-Optimized Pipeline Parallelism for LLM Training](http://arxiv.org/abs/2510.05186v1)
  > **TL;DR**: Optimizes pipeline parallelism for LLM training by formulating scheduling as a constrained optimization problem that balances memory, computation, and bubble minimization. Dynamically adjusts activation offloading and scheduling to reduce idle time by up to 50% under fixed memory constraints.
* `agentic` `serving` `thinking` [Speculative Actions: A Lossless Framework for Faster Agentic Systems](http://arxiv.org/abs/2510.04371v1)
  > **TL;DR**: Proposes speculative actions to accelerate agentic systems by predicting future actions with faster models, enabling parallel execution. Achieves up to 55% accuracy in action prediction and significantly reduces end-to-end latency in real-world environments.
* `serving` `thinking` `MoE` [SATER: A Self-Aware and Token-Efficient Approach to Routing and Cascading](http://arxiv.org/abs/2510.05164v1)
  > **TL;DR**: How to reduce cost and latency of LLM inference while maintaining performance? SATER introduces a self-aware routing and cascade framework with confidence-aware rejection and preference optimization, cutting computational cost by >50% and cascade latency by >80%.
* `diffusion` `MoE` [Paris: A Decentralized Trained Open-Weight Diffusion Model](http://arxiv.org/abs/2510.03434v1)
  > **TL;DR**: Can high-quality diffusion models be trained decentralized without gradient synchronization? Paris uses 8 isolated expert diffusion models with a router for inference, achieving comparable quality to centralized baselines using 16× less compute and 14× less data.
* `serving` `diffusion` [TridentServe: A Stage-level Serving System for Diffusion Pipelines](http://arxiv.org/abs/2510.02838v1)
  > **TL;DR**: Addresses inefficient static serving of diffusion pipelines by introducing dynamic stage-level resource allocation. TridentServe co-optimizes stage placement and request routing, achieving up to 4.1x lower P95 latency while improving SLO attainment.
* `MoE` `serving` [ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models](http://arxiv.org/abs/2510.02613v1)
  > **TL;DR**: Enables fine-grained, zero-downtime scaling of MoE LLMs during inference by decoupling execution from memory operations and using zero-copy remapping. Achieves up to 9× lower scale-up latency and 2× higher throughput during scaling.
* `serving` `diffusion` [TetriServe: Efficient DiT Serving for Heterogeneous Image Generation](http://arxiv.org/abs/2510.01565v1)
  > **TL;DR**: Addresses efficient serving of DiT models under heterogeneous SLOs by introducing step-level sequence parallelism and round-based scheduling. TetriServe dynamically adjusts parallelism per request step, achieving up to 32% higher SLO attainment while maintaining image quality.
* `agentic` `serving` [FlashResearch: Real-time Agent Orchestration for Efficient Deep Research](http://arxiv.org/abs/2510.05145v1)
  > **TL;DR**: How to accelerate deep research agents by parallelizing sequential reasoning? FlashResearch dynamically decomposes queries into tree-structured tasks and orchestrates parallel execution across breadth and depth, achieving 5x speedup while maintaining report quality.
* `training` `networking` [An Efficient, Reliable and Observable Collective Communication Library in Large-scale GPU Training Clusters](http://arxiv.org/abs/2510.00991v1)
  > **TL;DR**: Designs ICCL, a collective communication library for large-scale LLM training, to improve P2P efficiency, tolerate NIC failures, and enable microsecond-level anomaly observability, achieving 28.5% lower latency and 6.02% higher training throughput than NCCL.
* `training` `offloading` [ElasWave: An Elastic-Native System for Scalable Hybrid-Parallel Training](http://arxiv.org/abs/2510.00606v3)
  > **TL;DR**: Designs an elastic-native LLM training system that maintains parameter consistency and low recovery time during dynamic scaling. Introduces multi-dimensional scheduling, online resharding, and asynchronous migration with in-memory snapshots. Achieves 1.6× higher throughput and 51% lower MTTR than baselines.
* `MoE` `training` [FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training](http://arxiv.org/abs/2510.00207v2)
  > **TL;DR**: FlowMoE develops a unified pipeline scheduling framework for distributed MoE training by integrating MHA, gating, expert computation, and communication. It uses tensor chunk-based priority scheduling to overlap all-reduce with computing, reducing training time by up to 57%.
* `training` `kernel` [LoRAFusion: Efficient LoRA Fine-Tuning for LLMs](http://arxiv.org/abs/2510.00206v1)
  > **TL;DR**: Improves LoRA fine-tuning efficiency by fusing memory-bound ops via kernel optimization and scheduling multiple LoRA adapters with adaptive batching. Achieves up to 1.96× speedup over Megatron-LM and 1.46× over mLoRA.
* `networking` `training` `serving` [Lattica: A Decentralized Cross-NAT Communication Framework for Scalable AI Inference and Training](http://arxiv.org/abs/2510.00183v2)
  > **TL;DR**: Designs a decentralized cross-NAT framework to enable scalable AI training and inference without centralized infrastructure. Uses NAT traversal, CRDTs, and DHT-based discovery for peer-to-peer model synchronization. Achieves reliable communication in permissionless environments with low latency and high throughput.
* `serving` [TASP: Topology-aware Sequence Parallelism](http://arxiv.org/abs/2509.26541v2)
  > **TL;DR**: Addresses inefficient communication in sequence parallelism for long-context LLMs by decomposing Ring AllGather into topology-aware concurrent ring paths. TASP exploits AlltoAll accelerator topology to boost communication efficiency, achieving up to 3.58x speedup over Ring Attention.
* `serving` `offloading` [Parallax: Efficient LLM Inference Service over Decentralized Environment](http://arxiv.org/abs/2509.26182v1)
  > **TL;DR**: How to efficiently serve LLMs over decentralized, heterogeneous GPU pools? Parallax uses a two-phase scheduler for layer-wise model allocation and dynamic pipeline construction, reducing latency by up to 40% and improving throughput versus decentralized baselines.

### 2025-09-16 ~ 2025-09-30
* `serving` `storage` [Accelerating LLM Inference with Precomputed Query Storage](http://arxiv.org/abs/2509.25919v1)
  > **TL;DR**: Reduces LLM inference latency by precomputing and storing response pairs for predictable queries. Uses LLM-driven query generation and disk-backed vector indexing for efficient retrieval. Achieves up to 17.3% latency reduction without compromising response quality.
* `edge` `hardware` `kernel` [Context-Driven Performance Modeling for Causal Inference Operators on Neural Processing Units](http://arxiv.org/abs/2509.25155v1)
  > **TL;DR**: How to enable efficient long-context LLM inference on edge NPUs? Analyzes quadratic vs. sub-quadratic attention operators on NPUs, identifying memory-bound vs. compute-bound bottlenecks, and proposes hardware-aware model co-design. Achieves up to 95% reduction in pipeline stalls for long contexts.
* `MoE` `serving` [GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference](http://arxiv.org/abs/2509.25041v1)
  > **TL;DR**: Addresses high communication overhead and load imbalance in distributed MoE inference. Proposes GRACE-MoE with expert grouping, dynamic replication, and locality-aware routing to co-optimize communication and computation. Achieves up to 3.79x latency reduction over SOTA systems.
* `MoE` `serving` `load-balancing` [From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing](http://arxiv.org/abs/2510.03293v1)
  > **TL;DR**: How to improve MoE inference efficiency without retraining? LASER dynamically balances expert load using gate score distributions, routing tokens to least-loaded experts when scores are ambiguous. Achieves up to 30% lower latency and higher throughput on Mixtral and DeepSeek-MoE with near-zero accuracy drop.
* `training` [HAPT: Heterogeneity-Aware Automated Parallel Training on Heterogeneous Clusters](http://arxiv.org/abs/2509.24859v1)
  > **TL;DR**: HAPT automates parallel training on heterogeneous GPU clusters by optimizing inter-operator parallel strategies and adaptive 1F1B scheduling to maximize computation-communication overlap, achieving 1.3x-1.6x higher training throughput than existing frameworks.
* `serving` `offloading` [SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving](http://arxiv.org/abs/2509.24626v1)
  > **TL;DR**: How to enable efficient serving of long-context LLMs using dynamic sparse attention? SparseServe introduces hierarchical HBM-DRAM KV cache management with fragmentation-aware transfers, working-set-aware batching, and layer-segmented prefill, achieving up to 9.26x lower TTFT and 3.14x higher throughput.
* `serving` `multi-modal` [RServe: Overlapping Encoding and Prefill for Efficient LMM Inference](http://arxiv.org/abs/2509.24381v1)
  > **TL;DR**: Addressing high latency in LMM inference, REDServe overlaps multimodal encoding with language model prefill via disaggregation and fine-grained scheduling, achieving up to 66% lower latency and 109% higher throughput.
* `RL` `training` [RL in the Wild: Characterizing RLVR Training in LLM Deployment](http://arxiv.org/abs/2509.25279v1)
  > **TL;DR**: Characterizes system challenges in RL with verifiable rewards (RLVR) for LLM training, identifying issues like GPU idling and load imbalance due to skewed workloads; proposes PolyTrace benchmark achieving 94.7% accuracy in workload simulation.
* `serving` `training` `edge` [MACE: A Hybrid LLM Serving System with Colocated SLO-aware Continuous Retraining Alignment](http://arxiv.org/abs/2510.03283v1)
  > **TL;DR**: How to jointly serve LLM inference and fine-tuning on edge devices without violating SLOs? MACE collocates inference and iteration-level retraining with dynamic GPU resource allocation, achieving up to 63% lower latency while maintaining >85% GPU utilization on NVIDIA AGX Orin.
* `training` `sparse` [AdaPtis: Reducing Pipeline Bubbles with Adaptive Pipeline Parallelism on Heterogeneous Models](http://arxiv.org/abs/2509.23722v1)
  > **TL;DR**: Addresses pipeline bubbles in LLM training caused by model heterogeneity by jointly optimizing partition, placement, and scheduling. AdaPtis uses a performance model to guide adaptive pipeline parallelism, achieving up to 2.14x speedup over Megatron-LM.
* `serving` [A Predictive and Synergistic Two-Layer Scheduling Framework for LLM Serving](http://arxiv.org/abs/2509.23384v3)
  > **TL;DR**: Addresses inefficient two-layer LLM serving by introducing predictive, synergistic scheduling to bridge cluster- and engine-layer information gaps. Uses a performancemodel for adaptive batching and state-driven routing, improving SLO attainment by 43% and throughput by 3x.
* `serving` `quantization` `edge` [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](http://arxiv.org/abs/2509.23324v1)
  > **TL;DR**: Can smaller LLMs match larger models' accuracy on smartphones by leveraging underused NPU compute via test-time scaling? Proposes hardware-aware quantization and LUT optimizations for NPU-efficient inference, achieving up to 2.2× speedup and matching larger model accuracy.
* `training` [A Flexible Programmable Pipeline Parallelism Framework for Efficient DNN Training](http://arxiv.org/abs/2510.05112v2)
  > **TL;DR**: Designs FlexPipe, a programmable framework for automated and customizable pipeline parallelism in DNN training, using a DSL and scheduler to explore efficient schedules; achieves up to 2.28× speedup over Megtron-LM.
* `MoE` `quantization` `offloading` [Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](http://arxiv.org/abs/2510.02345v1)
  > **TL;DR**: Addresses the MoE trilemma by dynamically clustering experts and applying structured compression with hierarchical routing, mixed-precision storage (FP16/INT4), and dynamic offloading, reducing parameters by 80% while improving throughput by 10–20% and load balance by 3×.
* `training` `hardware` [Efficient Fine-Grained GPU Performance Modeling for Distributed Deep Learning of LLM](http://arxiv.org/abs/2509.22832v1)
  > **TL;DR**: Predicts end-to-end LLM training time across distributed GPUs by decomposing models into primitives and using lightweight hardware-aware models. Achieves <10% prediction error on 20B models across 128 GPUs while running entirely on CPU.
* `training` `sparse` [Zeppelin: Balancing Variable-length Workloads in Data Parallel Large Model Training](http://arxiv.org/abs/2509.21841v2)
  > **TL;DR**: Addresses load imbalance in data-parallel LLM training due to variable sequence lengths. Introduces hierarchical sequence partitioning, dynamic NIC routing, and module-aware remapping to balance computation and communication. Achieves 2.80x speedup over state-of-the-art.
* `multi-modal` `serving` `edge` `kernel` [Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices](http://arxiv.org/abs/2510.05109v1)
  > **TL;DR**: Designs a software-hardware co-designed system to efficiently run multimodal models on battery-powered edge devices by modularly scheduling vision, audio, and language components across heterogeneous accelerators. Uses optimized low-bit kernels and token-aware buffering, reducing energy consumption by 42.3% and enabling all-day LMM inference on-device.
* `training` `sparse` [Data-Centric Elastic Pipeline Parallelism for Efficient Long-Context LLM Training](http://arxiv.org/abs/2509.21275v1)
  > **TL;DR**: Addresses inefficient pipeline parallelism in long-context LLM training by adaptively switching between token- and batch-level partitioning. InfiniPipe uses workload-aware sequence packing and stage-aware checkpointing to balance load and reduce memory. Achieves 1.69x speedup over SOTA.
* `training` `offloading` [SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips](http://arxiv.org/abs/2509.21271v1)
  > **TL;DR**: How to optimize large-scale LLM training on Superchips using offloading? SuperOffload introduces Superchip-aware techniques like adaptive weight offloading and CPU-optimized Adam, achieving 2.5x higher throughput and enabling 25B model training on a single GH200.
* `training` `networking` [Go With The Flow: Churn-Tolerant Decentralized Training of Large Language Models](http://arxiv.org/abs/2509.21221v1)
  > **TL;DR**: Proposes GWTF, a decentralized framework for LLM training tolerant to node churn and network instability. Uses a novel flow-based routing algorithm to optimize microbatch scheduling across heterogeneous clients. Reduces training time by up to 45% under high churn.
* `RL` `serving` [RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training](http://arxiv.org/abs/2509.21009v1)
  > **TL;DR**: Addresses GPU underutilization in synchronous RL post-training caused by long-tail response delays. Introduces tail batching to schedule long responses separately, enabling balanced workloads across training stages. Achieves up to 2.56x faster training time on 128 H800 GPUs.
* `serving` [PARS: Low-Latency LLM Serving via Pairwise Learning-to-Rank](http://arxiv.org/abs/2510.03243v1)
  > **TL;DR**: How to reduce LLM serving latency caused by Head-of-Line blocking? PARS uses pairwise learning-to-rank to predict optimal task ordering by response length, integrated into vLLM. Achieves up to 35% lower average latency without sacrificing throughput.
* `training` `serving` [Kant: An Efficient Unified Scheduling System for Large-Scale AI Clusters](http://arxiv.org/abs/2510.01256v1)
  > **TL;DR**: Designs Kant, a unified scheduler for co-scheduling LLM training and inference in large AI clusters. Uses Backfill and E-Binpack to improve GPU utilization and reduce fragmentation. Achieves up to 30% higher GPU Allocation Ratio (GAR) compared to baseline schedulers.
* `serving` `hardware` `offline` [Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute](http://arxiv.org/abs/2509.20241v1)
  > **TL;DR**: Estimates energy use per LLM inference query at scale, accounting for real-world GPU utilization and PUE. Proposes a bottom-up methodology to quantify efficiency gains at model, platform, and hardware levels, achieving up to 20x reduction in energy per query for 1B queries/day.
* `training` `offloading` [BurstEngine: an Efficient Distributed Framework for Training Transformers on Extremely Long Sequences of over 1M Tokens](http://arxiv.org/abs/2509.19836v1)
  > **TL;DR**: Designs BurstEngine to efficiently train LLMs on sequences >1M tokens by introducing topology-aware BurstAttention, selective checkpointing, and fused loss computation, achieving 1.2× speedup and reduced memory overhead compared to state-of-the-art methods.
* `serving` `offloading` [Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference](http://arxiv.org/abs/2509.19729v1)
  > **TL;DR**: Addresses dynamic context length variance in LLM serving by adaptively transforming parallelism strategies across instances. Proposes a header-centric KV cache layout, weight padding, and transformation-aware scheduler, achieving up to 6.57× higher throughput than state-of-the-art systems.
* `RAG` [On The Reproducibility Limitations of RAG Systems](http://arxiv.org/abs/2509.18869v1)
  > **TL;DR**: Addresses the reproducibility limitations of RAG systems by introducing ReproRAG, a benchmarking framework that quantifies non-determinism across embedding models, retrieval algorithms, and hardware. Evaluates trade-offs using metrics like Exact Match Rate and Jaccard Similarity.
* `MoE` `serving` [Expert-as-a-Service: Towards Efficient, Scalable, and Robust Large-scale MoE Serving](http://arxiv.org/abs/2509.17863v1)
  > **TL;DR**: Addresses efficient serving of large-scale MoE models by disaggregating experts into stateless services. Uses peer-to-peer communication for low-overhead routing and dynamic resource scaling. Achieves 37.5% resource savings with <2% throughput loss under failures.
* `serving` `hardware` [Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs](http://arxiv.org/abs/2509.17542v2)
  > **TL;DR**: Designs a disaggregated LLM inference system using heterogeneous GPUs to separate prefill and decoding stages, enabling cost-efficient deployment. Introduces a heterogeneous-compatible transmission module and joint optimization for parallelism and instance allocation. Achieves 38% higher resource utilization compared to homogeneous setups.
* `agentic` `serving` `RAG` [Asteria: Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access](http://arxiv.org/abs/2509.17360v1)
  > **TL;DR**: Asteria improves agentic LLM performance by introducing semantic-aware cross-region caching for tool access. It uses semantic embeddings and a lightweight LLM judger for precise retrieval, achieving 3.6× higher throughput with 85%+ cache hit rates while maintaining accuracy.
* `serving` `heterogeneous` [Cronus: Efficient LLM inference on Heterogeneous GPU Clusters via Partially Disaggregated Prefill](http://arxiv.org/abs/2509.17357v1)
  > **TL;DR**: Cronus improves LLM inference throughput on heterogeneous GPU clusters by partially disaggregating prefill across low- and high-end GPUs, overlapping stages to balance load. It reduces P99 TTFT and TBT by up to 40% while maintaining high throughput.
* `multi-modal` `offloading` `edge` [MoA-Off: Adaptive Heterogeneous Modality-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference](http://arxiv.org/abs/2509.16995v1)
  > **TL;DR**: How to efficiently infer multimodal LLMs on edge devices? MoA-Off proposes a modality-aware offloading framework that dynamically splits computation between edge and cloud based on input complexity, reducing latency by 30% and resource overhead by 30%-65% while preserving accuracy.
* `serving` `offloading` `hardware` [ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching](http://arxiv.org/abs/2509.16857v1)
  > **TL;DR**: Addresses KV cache fetch interference in distributed prefix caching for LLM serving. Proposes ShadowServe, a SmartNIC-offloaded system with chunked pipelining and minimal-copy memory management. Achieves up to 2.2x lower TPOT and 1.38x lower TTFT in low-bandwidth scenarios.
* `serving` [Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads](http://arxiv.org/abs/2509.16495v1)
  > **TL;DR**: Addresses the latency-throughput tradeoff in LLM serving by introducing Shift Parallelism, which dynamically switches between tensor and sequence parallelism to leverage KV cache invariance. Achieves 1.51× lower latency in interactive workloads and 50% higher throughput in batch workloads than tensor parallelism alone.
* `training` [Robust LLM Training Infrastructure at ByteDance](http://arxiv.org/abs/2509.16293v2)
  > **TL;DR**: How to ensure stable large-scale LLM training amid frequent failures? ByteRobust introduces data-driven fault detection and recovery tailored to LLM parallelism, achieving 97% effective training time ratio (ETTR) over 9,600 GPUs.
* `RL` `training` [RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation](http://arxiv.org/abs/2509.15965v1)
  > **TL;DR**: Addresses low hardware utilization in RL training by introducing M2Flow, a macro-to-micro flow transformation that optimizes workflow execution via context switching and elastic pipelining. Achieves 1.1x–2.13x speedup in end-to-end training throughput.
* `networking` `training` [Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](http://arxiv.org/abs/2509.15940v1)
  > **TL;DR**: Addresses communication inefficiencies in large-scale LLM training by aligning communication patterns with data center topology. Proposes Arnold, a scheduling system that reduces communication spread and improves end-to-end training throughput by 10.6% on 9600+ GPUs.
* `hardware` `networking` [PCCL: Photonic circuit-switched collective communication for distributed ML](http://arxiv.org/abs/2509.15450v1)
  > **TL;DR**: Addresses communication bottlenecks in distributed ML training by reconfiguring photonic networks to eliminate congestion. Proposes PCCL, a hardware-agnostic system that creates direct circuits for collective operations, achieving up to 3X faster communication and 1.3X end-to-end training speedup.
* `RAG` `agentic` [LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology](http://arxiv.org/abs/2509.13978v2)
  > **TL;DR**: Explores using LLM agents to interpret and query scientific workflow provenance via natural language. Combines metadata-driven design and RAG to translate prompts into structured queries, achieving high accuracy on real-world chemistry workflows.
* `serving` `kernel` `hardware` [FLAME: A Serving System Optimized for Large-Scale Generative Recommendation with Efficiency](http://arxiv.org/abs/2509.22681v1)
  > **TL;DR**: Designs a production-grade serving system for large-scale generative recommendation models by decoupling pre-processing and computation, optimizing memory with PDA, and accelerating inference via TensorRT-based fused kernels. Achieves up to 6.3x throughput gain and 2.3x latency reduction.
* `diffusion` `hardware` [AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions](http://arxiv.org/abs/2509.13523v1)
  > **TL;DR**: Designs AERIS, a billion-parameter Swin diffusion transformer for high-resolution weather prediction, using SWiPe to enable efficient window, sequence, and pipeline parallelism; achieves 10.21 ExaFLOPS on Aurora with 95.5% weak scaling efficiency.
* `serving` `offloading` [Scaling Up Throughput-oriented LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management](http://arxiv.org/abs/2509.13201v1)
  > **TL;DR**: Addresses how to improve throughput of non-latency-sensitive LLM inference on opportunistic GPU clusters. Introduces pervasive context management to reuse computational context across dynamic resources, enabling 98.1% reduction in execution time.
* `agentic` `serving` `offloading` [Toward Systems Foundations for Agentic Exploration](http://arxiv.org/abs/2510.05556v1)
  > **TL;DR**: Addresses the lack of efficient system support for LLM agentic exploration. Proposes fundamental challenges in fork semantics, side-effect handling, and microsecond-level native forking. Achieves scalable branching with minimal overhead in multi-agent deployments.
* `agentic` `serving` `offloading` [Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization](http://arxiv.org/abs/2509.21301v1)
  > **TL;DR**: Nova enables efficient real-time serving of agentic vision-language models by adaptively partitioning GPU resources across vision, prefill, and decode stages, plus lightweight vision encoder offloading. It achieves up to 23.3% lower max latency while maintaining high throughput.

### 2025-09-15

* [Characterizing the Efficiency of Distributed Training: A Power, Performance, and Thermal Perspective](https://arxiv.org/abs/2509.10371)
* [Ordered Consensus with Equal Opportunity](https://arxiv.org/abs/2509.09868)
* [The (R)evolution of Scientific Workflows in the Agentic AI Era: Towards Autonomous Science](https://arxiv.org/abs/2509.09915)
* [DBOS Network Sensing: A Web Services Approach to Collaborative Awareness](https://arxiv.org/abs/2509.09898)
* [XBOF: A Cost-Efficient CXL JBOF with Inter-SSD Compute Resource Sharing](https://arxiv.org/abs/2509.10251)
* [DBOS Network Sensing: A Web Services Approach to Collaborative Awareness](https://arxiv.org/abs/2509.09898)


### 2025-09-12

* [TrEnv: Transparently Share Serverless Execution Environments Across Different Functions and Nodes](https://arxiv.org/abs/2509.09525)
* [Weaker Assumptions for Asymmetric Trust](https://arxiv.org/abs/2509.09493)
* [Barycentric Coded Distributed Computing with Flexible Recovery Threshold for Collaborative Mobile Edge Computing](https://arxiv.org/abs/2509.09435)
* [WebAssembly and Unikernels: A Comparative Study for Serverless at the Edge](https://arxiv.org/abs/2509.09400)
* [Coherence-Aware Task Graph Modeling for Realistic Application](https://arxiv.org/abs/2509.09094)
* [Optimizing the Variant Calling Pipeline Execution on Human Genomes Using GPU-Enabled Machines](https://arxiv.org/abs/2509.09058)
* [A Comparative Analysis of Identifier Schemes: UUIDv4, UUIDv7, and ULID for Distributed Systems](https://arxiv.org/abs/2509.08969)
* [Towards A High-Performance Quantum Data Center Network Architecture](https://arxiv.org/abs/2509.09653)
* [HARD: A Performance Portable Radiation Hydrodynamics Code based on FleCSI Framework](https://arxiv.org/abs/2509.08971)
* [μFork: Supporting POSIX fork Within a Single-Address-Space OS](https://arxiv.org/abs/2509.09439)
* [TrEnv: Transparently Share Serverless Execution Environments Across Different Functions and Nodes](https://arxiv.org/abs/2509.09525)


### 2025-09-11

* [Reconfigurable Holographic Surfaces and Near Field Communication for Non-Terrestrial Networks: Potential and Challenges](https://arxiv.org/abs/2509.08770)
* [A 410GFLOP/s, 64 RISC-V Cores, 204.8GBps Shared-Memory Cluster in 12nm FinFET with Systolic Execution Support for Efficient B5G/6G AI-Enhanced O-RAN](https://arxiv.org/abs/2509.08608)
* [An HPC Benchmark Survey and Taxonomy for Characterization](https://arxiv.org/abs/2509.08347)
* [Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism](https://arxiv.org/abs/2509.08309)
* [Design and Implementation of Code Completion System Based on LLM and CodeBERT Hybrid Subsystem](https://arxiv.org/abs/2509.08215)
* [Aurora: Architecting Argonne's First Exascale Supercomputer for Accelerated Scientific Discovery](https://arxiv.org/abs/2509.08207)
* [Towards Scalable Proteomics: Opportunistic SMC Samplers on HTCondor](https://arxiv.org/abs/2509.08020)


### 2025-09-10

* [Scaling atomic ordering in shared memory](https://arxiv.org/abs/2509.07781)
* [AgentX: Towards Orchestrating Robust Agentic Workflow Patterns with FaaS-hosted MCP Services](https://arxiv.org/abs/2509.07595)
* [Navigating Energy Doldrums: Modeling the Impact of Energy Price Volatility on HPC Cost of Ownership](https://arxiv.org/abs/2509.07567)
* [Astra: A Multi-Agent System for GPU Kernel Performance Optimization](https://arxiv.org/abs/2509.07506)
* [DREAMS: Decentralized Resource Allocation and Service Management across the Compute Continuum Using Service Affinity](https://arxiv.org/abs/2509.07497)
* [Dependency-Aware Execution Mechanism in Hyperledger Fabric Architecture](https://arxiv.org/abs/2509.07425)
* [DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference](https://arxiv.org/abs/2509.07379)
* [Optimizing Task Scheduling in Fog Computing with Deadline Awareness](https://arxiv.org/abs/2509.07378)
* [A Study on Messaging Trade-offs in Data Streaming for Scientific Workflows](https://arxiv.org/abs/2509.07199)
* [Bodega: Serving Linearizable Reads Locally from Anywhere at Anytime via Roster Leases](https://arxiv.org/abs/2509.07158)
* [Crossword: Adaptive Consensus for Dynamic Data-Heavy Workloads](https://arxiv.org/abs/2509.07157)
* [MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?](https://arxiv.org/abs/2509.07727)
* [HYLU: Hybrid Parallel Sparse LU Factorization](https://arxiv.org/abs/2509.07690)
* [veScale: Consistent and Efficient Tensor Programming with Eager-Mode SPMD](https://arxiv.org/abs/2509.07003)


### 2025-09-09

* [IM-PIR: In-Memory Private Information Retrieval](https://arxiv.org/abs/2509.06514)
* [MaaSO: SLO-aware Orchestration of Heterogeneous Model Instances for MaaS](https://arxiv.org/abs/2509.06362)
* [FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving](https://arxiv.org/abs/2509.06261)
* [20 Years in Life of a Smart Building: A retrospective](https://arxiv.org/abs/2509.06229)
* [Gathering in Non-Vertex-Transitive Graphs Under Round Robin](https://arxiv.org/abs/2509.06064)
* [DISTRIBUTEDANN: Efficient Scaling of a Single DISKANN Graph Across Thousands of Computers](https://arxiv.org/abs/2509.06046)
* [A Simple and Robust Protocol for Distributed Counting](https://arxiv.org/abs/2509.05870)
* [Multi-IaC-Eval: Benchmarking Cloud Infrastructure as Code Across Multiple Formats](https://arxiv.org/abs/2509.05303)
* [Distributed Automatic Generation Control subject to Ramp-Rate-Limits: Anytime Feasibility and Uniform Network-Connectivity](https://arxiv.org/abs/2509.06588)
* [Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing](https://arxiv.org/abs/2509.06552)
* [Several Performance Bounds on Decentralized Online Optimization are Highly Conservative and Potentially Misleading](https://arxiv.org/abs/2509.06466)
* [Social Dynamics of DAOs: Power, Onboarding, and Inclusivity](https://arxiv.org/abs/2509.06163)
* [Introduction to Number Theoretic Transform](https://arxiv.org/abs/2509.05884)
* [Tiga: Accelerating Geo-Distributed Transactions with Synchronized Clocks [Technical Report]](https://arxiv.org/abs/2509.05759)
* [Distributed Deep Learning using Stochastic Gradient Staleness](https://arxiv.org/abs/2509.05679)
* [Workflow for High-Fidelity Dynamic Analysis of Structures with Pile Foundation](https://arxiv.org/abs/2509.05675)
* [Efficient Fault Localization in a Cloud Stack Using End-to-End Application Service Topology](https://arxiv.org/abs/2509.05511)
* [MambaLite-Micro: Memory-Optimized Mamba Inference on MCUs](https://arxiv.org/abs/2509.05488)


### 2025-09-08

* [Scaling Performance of Large Language Model Pretraining](https://arxiv.org/abs/2509.05258)
* [Dynamic reconfiguration for malleable applications using RMA](https://arxiv.org/abs/2509.05248)
* [Toward Distributed 3D Gaussian Splatting for High-Resolution Isosurface Visualization](https://arxiv.org/abs/2509.05216)
* [VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving](https://arxiv.org/abs/2509.04827)
* [STADI: Fine-Grained Step-Patch Diffusion Parallelism for Heterogeneous GPUs](https://arxiv.org/abs/2509.04719)


### 2025-09-05

* [On the impact of unlimited computational power in OBLOT: consequences for synchronous robots on graphs](https://arxiv.org/abs/2509.04383)
* [Trustworthy Second-hand Marketplace for Built Environment](https://arxiv.org/abs/2509.04085)
* [LowDiff: Efficient Frequent Checkpointing via Low-Cost Differential for High-Performance Distributed Training Systems](https://arxiv.org/abs/2509.04084)
* [Counterfactual simulations for large scale systems with burnout variables](https://arxiv.org/abs/2509.04038)
* [Gathering of asynchronous robots on circle with limited visibility using finite communication](https://arxiv.org/abs/2509.04004)
* [Distributed Download from an External Data Source in Asynchronous Faulty Settings](https://arxiv.org/abs/2509.03755)
* [Combining Performance and Productivity: Accelerating the Network Sensing Graph Challenge with GPUs and Commodity Data Science Software](https://arxiv.org/abs/2509.03653)
* [Massively-Parallel Implementation of Inextensible Elastic Rods Using Inter-block GPU Synchronization](https://arxiv.org/abs/2509.04277)
* [Cloud-Assisted Remote Control for Aerial Robots: From Theory to Proof-of-Concept Implementation](https://arxiv.org/abs/2509.04095)
* [Prob-GParareal: A Probabilistic Numerical Parallel-in-Time Solver for Differential Equations](https://arxiv.org/abs/2509.03945)
* [Towards Deterministic Sub-0.5 us Response on Linux through Interrupt Isolation](https://arxiv.org/abs/2509.03855)


### 2025-09-04

* [CloudFormer: An Attention-based Performance Prediction for Public Clouds with Unknown Workload](https://arxiv.org/abs/2509.03394)
* [Efficient and Secure Sleepy Model for BFT Consensus](https://arxiv.org/abs/2509.03145)
* [The High Cost of Keeping Warm: Characterizing Overhead in Serverless Autoscaling Policies](https://arxiv.org/abs/2509.03104)
* [FlashRecovery: Fast and Low-Cost Recovery from Failures for Large-Scale Training of LLMs](https://arxiv.org/abs/2509.03047)
* [Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training](https://arxiv.org/abs/2509.03018)
* [A Novel IaaS Tax Model as Leverage Towards Green Cloud Computing](https://arxiv.org/abs/2509.02767)
* [DPQuant: Efficient and Differentially-Private Model Training via Dynamic Quantization Scheduling](https://arxiv.org/abs/2509.03472)
* [A description of the radio astronomy data processing tool DDF Pipeline](https://arxiv.org/abs/2509.03075)
* [Treasure Hunt in Anonymous Graphs with Quantum Pebbles by Oblivious Agents](https://arxiv.org/abs/2509.02909)
* [\textit{In Silico} Benchmarking of Detectable Byzantine Agreement in Noisy Quantum Networks](https://arxiv.org/abs/2509.02629)
* [On the Optimization of Methods for Establishing Well-Connected Communities](https://arxiv.org/abs/2509.02590)
* [Safe Sharing of Fast Kernel-Bypass I/O Among Nontrusting Applications](https://arxiv.org/abs/2509.02899)


### 2025-09-03

* [Energy-Efficient Split Learning for Resource-Constrained Environments: A Smart Farming Solution](https://arxiv.org/abs/2509.02549)
* [MLP-Offload: Multi-Level, Multi-Path Offloading for LLM Pre-training to Break the GPU Memory Wall](https://arxiv.org/abs/2509.02480)
* [Safe Memory Reclamation Techniques](https://arxiv.org/abs/2509.02457)
* [KubeIntellect: A Modular LLM-Orchestrated Agent Framework for End-to-End Kubernetes Management](https://arxiv.org/abs/2509.02449)
* [An Efficient and Adaptive Watermark Detection System with Tile-based Error Correction](https://arxiv.org/abs/2509.02447)
* [Efficient Pyramidal Analysis of Gigapixel Images on a Decentralized Modest Computer Cluster](https://arxiv.org/abs/2509.02440)
* [A Continuous Energy Ising Machine Leveraging Difference-of-Convex Programming](https://arxiv.org/abs/2509.01928)
* [Optimal Parallel Scheduling under Concave Speedup Functions](https://arxiv.org/abs/2509.01811)
* [STZ: A High Quality and High Speed Streaming Lossy Compression Framework for Scientific Data](https://arxiv.org/abs/2509.01626)
* [HiCR, an Abstract Model for Distributed Heterogeneous Programming](https://arxiv.org/abs/2509.01425)
* [LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving](https://arxiv.org/abs/2509.01229)
* [LobRA: Multi-tenant Fine-tuning over Heterogeneous Data](https://arxiv.org/abs/2509.01193)
* [DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving](https://arxiv.org/abs/2509.01083)
* [Parallelizing Drug Discovery: HPC Pipelines for Alzheimer's Molecular Docking and Simulation](https://arxiv.org/abs/2509.00937)
* [Accelerating Latency-Critical Applications with AI-Powered Semi-Automatic Fine-Grained Parallelization on SMT Processors](https://arxiv.org/abs/2509.00883)
* [HADIS: Hybrid Adaptive Diffusion Model Serving for Efficient Text-to-Image Generation](https://arxiv.org/abs/2509.00642)
* [KVComp: A High-Performance, LLM-Aware, Lossy Compression Framework for KV Cache](https://arxiv.org/abs/2509.00579)
* [HydroGAT: Distributed Heterogeneous Graph Attention Transformer for Spatiotemporal Flood Prediction](https://arxiv.org/abs/2509.02481)
* [Online Identification of IT Systems through Active Causal Learning](https://arxiv.org/abs/2509.02130)
* [Batch Query Processing and Optimization for Agentic Workflows](https://arxiv.org/abs/2509.02121)
* [OASIS: Object-based Analytics Storage for Intelligent SQL Query Offloading in Scientific Tabular Workloads](https://arxiv.org/abs/2509.01966)
* [AdaptCache: KV Cache Native Storage Hierarchy for Low-Delay and High-Quality Language Model Serving](https://arxiv.org/abs/2509.00105)
* [Towards Agentic OS: An LLM Agent Framework for Linux Schedulers](https://arxiv.org/abs/2509.01245)


### 2025-09-01

* [Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding](https://arxiv.org/abs/2508.21706)
* [Odyssey: Adaptive Policy Selection for Resilient Distributed Training](https://arxiv.org/abs/2508.21613)
* [Unpacking Maximum Extractable Value on Polygon: A Study on Atomic Arbitrage](https://arxiv.org/abs/2508.21473)
* [Addressing Reproducibility Challenges in HPC with Continuous Integration](https://arxiv.org/abs/2508.21289)
* [Fast and Scalable Mixed Precision Euclidean Distance Calculations Using GPU Tensor Cores](https://arxiv.org/abs/2508.21230)
* [An Optimistic Gradient Tracking Method for Distributed Minimax Optimization](https://arxiv.org/abs/2508.21431)


### 2025-08-29

* [Collaborative Evolution of Intelligent Agents in Large-Scale Microservice Systems](https://arxiv.org/abs/2508.20508)
* [pdGRASS: A Fast Parallel Density-Aware Algorithm for Graph Spectral Sparsification](https://arxiv.org/abs/2508.20403)
* [CoFormer: Collaborating with Heterogeneous Edge Devices for Scalable Transformer Inference](https://arxiv.org/abs/2508.20375)
* [Predictable LLM Serving on GPU Clusters](https://arxiv.org/abs/2508.20274)
* [SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization](https://arxiv.org/abs/2508.20258)
* [SpeedMalloc: Improving Multi-threaded Applications via a Lightweight Core for Memory Allocation](https://arxiv.org/abs/2508.20253)
* [A Hybrid Stochastic Gradient Tracking Method for Distributed Online Optimization Over Time-Varying Directed Networks](https://arxiv.org/abs/2508.20645)
* [High performance visualization for Astronomy and Cosmology: the VisIVO's pathway toward Exascale systems](https://arxiv.org/abs/2508.20603)
* [Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs](https://arxiv.org/abs/2508.20333)


### 2025-08-28

* [HPC Digital Twins for Evaluating Scheduling Policies, Incentive Structures and their Impact on Power and Cooling](https://arxiv.org/abs/2508.20016)
* [Separation of Three or More Autonomous Mobile Models under Hierarchical Schedulers](https://arxiv.org/abs/2508.19805)
* [Beyond the Bermuda Triangle of Contention: IOMMU Interference in Mixed Criticality Systems](https://arxiv.org/abs/2508.19670)
* [Taming the Chaos: Coordinated Autoscaling for Heterogeneous and Disaggregated LLM Inference](https://arxiv.org/abs/2508.19559)
* [Formal Modeling and Verification of the Algorand Consensus Protocol in CADP](https://arxiv.org/abs/2508.19452)
* [HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2508.19373)
* [New Tools, Programming Models, and System Support for Processing-in-Memory Architectures](https://arxiv.org/abs/2508.19868)
* [Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents](https://arxiv.org/abs/2508.19504)


### 2025-08-27

* [Ab-initio Quantum Transport with the GW Approximation, 42,240 Atoms, and Sustained Exascale Performance](https://arxiv.org/abs/2508.19138)
* [CARMA: Collocation-Aware Resource Manager with GPU Memory Estimator](https://arxiv.org/abs/2508.19073)
* [Deep Learning-Enabled Supercritical Flame Simulation at Detailed Chemistry and Real-Fluid Accuracy Towards Trillion-Cell Scale](https://arxiv.org/abs/2508.18969)
* [SIREN: Software Identification and Recognition in HPC Systems](https://arxiv.org/abs/2508.18950)
* [ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive](https://arxiv.org/abs/2508.18850)
* [Examining MPI and its Extensions for Asynchronous Multithreaded Communication](https://arxiv.org/abs/2508.18667)
* [Strata: Hierarchical Context Caching for Long Context Language Model Serving](https://arxiv.org/abs/2508.18572)
* [Managing Multi Instance GPUs for High Throughput and Energy Savings](https://arxiv.org/abs/2508.18556)
* [Experiences with Model Context Protocol Servers for Science and High Performance Computing](https://arxiv.org/abs/2508.18489)
* [Architecting Distributed Quantum Computers: Design Insights from Resource Estimation](https://arxiv.org/abs/2508.19160)
* [History Rhymes: Accelerating LLM Reinforcement Learning with RhymeRL](https://arxiv.org/abs/2508.18588)
* [DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction](https://arxiv.org/abs/2508.18376)


### 2025-08-26

* [Flash Sparse Attention: An Alternative Efficient Implementation of Native Sparse Attention Kernel](https://arxiv.org/abs/2508.18224)
* [Practical GPU Choices for Earth Observation: ResNet-50 Training Throughput on Integrated, Laptop, and Cloud Accelerators](https://arxiv.org/abs/2508.18206)
* [Wait-free Replicated Data Types and Fair Reconciliation](https://arxiv.org/abs/2508.18193)
* [Scalable Engine and the Performance of Different LLM Models in a SLURM based HPC architecture](https://arxiv.org/abs/2508.17814)
* [ExpertWeave: Efficiently Serving Expert-Specialized Fine-Tuned Adapters at Scale](https://arxiv.org/abs/2508.17624)
* [Zen-Attention: A Compiler Framework for Dynamic Attention Folding on AMD NPUs](https://arxiv.org/abs/2508.17593)
* [Easy Acceleration with Distributed Arrays](https://arxiv.org/abs/2508.17493)
* [Bine Trees: Enhancing Collective Operations by Optimizing Communication Locality](https://arxiv.org/abs/2508.17311)
* [TokenLake: A Unified Segment-level Prefix Cache Pool for Fine-grained Elastic Long-Context LLM Serving](https://arxiv.org/abs/2508.17219)
* [PICO: Performance Insights for Collective Operations](https://arxiv.org/abs/2508.16809)
* [Neuromorphic Simulation of Drosophila Melanogaster Brain Connectome on Loihi 2](https://arxiv.org/abs/2508.16792)
* [Equinox: Holistic Fair Scheduling in Serving Large Language Models](https://arxiv.org/abs/2508.16646)
* [GPU Acceleration for Faster Evolutionary Spatial Cyclic Game Systems](https://arxiv.org/abs/2508.16639)
* [Performance measurements of modern Fortran MPI applications with Score-P](https://arxiv.org/abs/2508.16592)
* [Views: A Hardware-friendly Graph Database Model For Storing Semantic Information](https://arxiv.org/abs/2508.18123)
* [Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective](https://arxiv.org/abs/2508.16712)
* [GPT-OSS-20B: A Comprehensive Deployment-Centric Analysis of OpenAI's Open-Weight Mixture of Experts Model](https://arxiv.org/abs/2508.16700)
* [Scalable Hybrid quantum Monte Carlo simulation of U(1) gauge field coupled to fermions on GPU](https://arxiv.org/abs/2508.16298)
* [Iridescent: A Framework Enabling Online System Implementation Specialization](https://arxiv.org/abs/2508.16690)
* [Puzzle: Scheduling Multiple Deep Learning Models on Mobile Device with Heterogeneous Processors](https://arxiv.org/abs/2508.17764)


### 2025-08-25

* [Generalizing Brooks' theorem via Partial Coloring is Hard Classically and Locally](https://arxiv.org/abs/2508.16308)
* [HyperFlexis: Joint Design of Algorithms and Systems for Multi-SLO Serving and Fast Scaling](https://arxiv.org/abs/2508.15919)
* [On the Duality of Task and Actor Programming Models](https://arxiv.org/abs/2508.16522)
* [Hybrid Classical-Quantum Supercomputing: A demonstration of a multi-user, multi-QPU and multi-GPU environment](https://arxiv.org/abs/2508.16297)
* [Self-Healing Network of Interconnected Edge Devices Empowered by Infrastructure-as-Code and LoRa Communication](https://arxiv.org/abs/2508.16268)
* [Towards Integrated Energy-Communication-Transportation Hub: A Base-Station-Centric Design in 5G and Beyond](https://arxiv.org/abs/2508.15833)
* [CXLAimPod: CXL Memory is all you need in AI era](https://arxiv.org/abs/2508.15980)


### 2025-08-22

* [CausalMesh: A Formally Verified Causal Cache for Stateful Serverless Computing](https://arxiv.org/abs/2508.15647)
* [Efficient Mixed-Precision Large Language Model Inference with TurboMind](https://arxiv.org/abs/2508.15601)
* [Lower Bounds for $k$-Set Agreement in Fault-Prone Networks](https://arxiv.org/abs/2508.15562)
* [Universal Dancing by Luminous Robots under Sequential Schedulers](https://arxiv.org/abs/2508.15484)
* [Databelt: A Continuous Data Path for Serverless Workflows in the 3D Compute Continuum](https://arxiv.org/abs/2508.15351)
* [Declarative Data Pipeline for Large Scale ML Services](https://arxiv.org/abs/2508.15105)
* [Mitigating context switching in densely packed Linux clusters with Latency-Aware Group Scheduling](https://arxiv.org/abs/2508.15703)
* [On the Effectiveness of Graph Reordering for Accelerating Approximate Nearest Neighbor Search on GPU](https://arxiv.org/abs/2508.15436)
* [Optimizing Compilation for Distributed Quantum Computing via Clustering and Annealing](https://arxiv.org/abs/2508.15267)
* [Reliable Multi-view 3D Reconstruction for `Just-in-time' Edge Environments](https://arxiv.org/abs/2508.15158)
* [TOAST: Fast and scalable auto-partitioning based on principled static analysis](https://arxiv.org/abs/2508.15010)
* [Scalable FPGA Framework for Real-Time Denoising in High-Throughput Imaging: A DRAM-Optimized Pipeline using High-Level Synthesis](https://arxiv.org/abs/2508.14917)
* [Mitigating context switching in densely packed Linux clusters with Latency-Aware Group Scheduling](https://arxiv.org/abs/2508.15703)


### 2025-08-21

* [The Cost Advantage of Virtual Machine Migrations: Empirical Insights into Amazon's EC2 Marketspace](https://arxiv.org/abs/2508.14883)
* [Leveraging Hardware-Aware Computation in Mixed-Precision Matrix Multiply: A Tile-Centric Approach](https://arxiv.org/abs/2508.14848)
* [MOHAF: A Multi-Objective Hierarchical Auction Framework for Scalable and Fair Resource Allocation in IoT Ecosystems](https://arxiv.org/abs/2508.14830)
* [DAG it off: Latency Prefers No Common Coins](https://arxiv.org/abs/2508.14716)
* [A Systematic Evaluation of the Potential of Carbon-Aware Execution for Scientific Workflows](https://arxiv.org/abs/2508.14625)
* [Boosting Payment Channel Network Liquidity with Topology Optimization and Transaction Selection](https://arxiv.org/abs/2508.14524)
* [Auditable Shared Objects: From Registers to Synchronization Primitives](https://arxiv.org/abs/2508.14506)
* [SSSP-Del: Fully Dynamic Distributed Algorithm for Single-Source Shortest Path](https://arxiv.org/abs/2508.14319)
* [Pure Data Spaces](https://arxiv.org/abs/2508.14271)
* [Time-optimal Asynchronous Minimal Vertex Covering by Myopic Robots](https://arxiv.org/abs/2508.14247)
* [Cooperative SGD with Dynamic Mixing Matrices](https://arxiv.org/abs/2508.14565)
* [Lagrangian Simulation Volume-Based Contour Tree Simplification](https://arxiv.org/abs/2508.14339)
* [Power Stabilization for AI Training Datacenters](https://arxiv.org/abs/2508.14318)
* [A High Performance GPU CountSketch Implementation and Its Application to Multisketching and Least Squares Problems](https://arxiv.org/abs/2508.14209)


### 2025-08-20

* [Is RISC-V ready for High Performance Computing? An evaluation of the Sophon SG2044](https://arxiv.org/abs/2508.13840)
* [Estimating CO$_2$ emissions of distributed applications and platforms with SimGrid/Batsim](https://arxiv.org/abs/2508.13693)
* [LUNDIsim: model meshes for flow simulation and scientific data compression benchmarks](https://arxiv.org/abs/2508.13636)
* [LAMMPS-KOKKOS: Performance Portable Molecular Dynamics Across Exascale Architectures](https://arxiv.org/abs/2508.13523)
* [DDoS Attacks in Cloud Computing: Detection and Prevention](https://arxiv.org/abs/2508.13522)
* [Optimizing Allreduce Operations for Heterogeneous Architectures with Multiple Processes per GPU](https://arxiv.org/abs/2508.13397)
* [OrbitChain: Orchestrating In-orbit Real-time Analytics of Earth Observation Data](https://arxiv.org/abs/2508.13374)
* [Persistent and Partitioned MPI for Stencil Communication](https://arxiv.org/abs/2508.13370)
* [Harnessing the Full Potential of RRAMs through Scalable and Distributed In-Memory Computing with Integrated Error Correction](https://arxiv.org/abs/2508.13298)
* [Analog computation with transcriptional networks](https://arxiv.org/abs/2508.14017)
* [PennyLane-Lightning MPI: A massively scalable quantum circuit simulator based on distributed computing in CPU clusters](https://arxiv.org/abs/2508.13615)
* [X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms](https://arxiv.org/abs/2508.13337)
* [Sustainable AI Training via Hardware-Software Co-Design on NVIDIA, AMD, and Emerging GPU Architectures](https://arxiv.org/abs/2508.13163)
* [Towards Timing Isolation for Mixed-Criticality Communication in Software-Defined Vehicles](https://arxiv.org/abs/2508.13652)


### 2025-08-19

* [Team Formation and Applications](https://arxiv.org/abs/2508.13084)
* [Congested Clique Counting for Local Gibbs Distributions](https://arxiv.org/abs/2508.13083)
* [WANify: Gauging and Balancing Runtime WAN Bandwidth for Geo-distributed Data Analytics](https://arxiv.org/abs/2508.12961)
* [Accelerating Edge Inference for Distributed MoE Models with Latency-Optimized Expert Placement](https://arxiv.org/abs/2508.12851)
* [Dissecting CPU-GPU Unified Physical Memory on AMD MI300A APUs](https://arxiv.org/abs/2508.12743)
* [DIT: Dimension Reduction View on Optimal NFT Rarity Meters](https://arxiv.org/abs/2508.12671)
* [Proceedings 18th Interaction and Concurrency Experience](https://arxiv.org/abs/2508.12308)
* [Data-driven Trust Bootstrapping for Mobile Edge Computing-based Industrial IoT Services](https://arxiv.org/abs/2508.12560)
* [Attack Graph Generation on HPC Clusters](https://arxiv.org/abs/2508.12161)
* [OS-R1: Agentic Operating System Kernel Tuning with Reinforcement Learning](https://arxiv.org/abs/2508.12551)


### 2025-08-18

* [Efficient GPU-Centered Singular Value Decomposition Using the Divide-and-Conquer Method](https://arxiv.org/abs/2508.11467)
* [Time, Fences and the Ordering of Events in TSO](https://arxiv.org/abs/2508.11415)
* [Space-efficient population protocols for exact majority in general graphs](https://arxiv.org/abs/2508.11384)
* [Inter-APU Communication on AMD MI300A Systems via Infinity Fabric: a Deep Dive](https://arxiv.org/abs/2508.11298)
* [Element and Everything Tokens: Two-Tier Architecture for Mobilizing Alternative Assets](https://arxiv.org/abs/2508.11266)
* [EMLIO: Minimizing I/O Latency and Energy Consumption for Large-Scale AI Training](https://arxiv.org/abs/2508.11035)
* [OpenCXD: An Open Real-Device-Guided Hybrid Evaluation Framework for CXL-SSDs](https://arxiv.org/abs/2508.11477)


### 2025-08-15

* [Minimmit: Fast Finality with Even Faster Blocks](https://arxiv.org/abs/2508.10862)
* [Introducing CQ: A C-like API for Quantum Accelerated HPC](https://arxiv.org/abs/2508.10854)
* [Dalek: An Unconventional and Energy-Aware Heterogeneous Cluster](https://arxiv.org/abs/2508.10481)
* [GPZ: GPU-Accelerated Lossy Compressor for Particle Data](https://arxiv.org/abs/2508.10305)
* [Mixed-Precision Performance Portability of FFT-Based GPU-Accelerated Algorithms for Block-Triangular Toeplitz Matrices](https://arxiv.org/abs/2508.10202)
* [Hard Shell, Reliable Core: Improving Resilience in Replicated Systems with Selective Hybridization](https://arxiv.org/abs/2508.10141)
* [Leveraging OS-Level Primitives for Robotic Action Management](https://arxiv.org/abs/2508.10259)


### 2025-08-14

* [Closing the HPC-Cloud Convergence Gap: Multi-Tenant Slingshot RDMA for Kubernetes](https://arxiv.org/abs/2508.09663)
* [HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap](https://arxiv.org/abs/2508.09591)
* [Verify Distributed Deep Learning Model Implementation Refinement with Iterative Relation Inference](https://arxiv.org/abs/2508.09505)
* [Distributed Diamond Formation of Sliding Squares](https://arxiv.org/abs/2508.09638)
* [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)
* [Semantic-Aware LLM Orchestration for Proactive Resource Management in Predictive Digital Twin Vehicular Networks](https://arxiv.org/abs/2508.09149)
* [Holistic Heterogeneous Scheduling for Autonomous Applications using Fine-grained, Multi-XPU Abstraction](https://arxiv.org/abs/2508.09503)
* [A Limits Study of Memory-side Tiering Telemetry](https://arxiv.org/abs/2508.09351)


### 2025-08-13

* [P/D-Device: Disaggregated Large Language Model between Cloud and Devices](https://arxiv.org/abs/2508.09035)
* [A Reinforcement Learning-Driven Task Scheduling Algorithm for Multi-Tenant Distributed Systems](https://arxiv.org/abs/2508.08525)
* [Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson -- Extended](https://arxiv.org/abs/2508.08430)
* [Ultra Ethernet's Design Principles and Architectural Innovations](https://arxiv.org/abs/2508.08906)
* [Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2508.08744)
* [Two for One, One for All: Deterministic LDC-based Robust Computation in Congested Clique](https://arxiv.org/abs/2508.08740)
* [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](https://arxiv.org/abs/2508.08712)
* [Vector-Centric Machine Learning Systems: A Cross-Stack Approach](https://arxiv.org/abs/2508.08469)
* [Towards Efficient and Practical GPU Multitasking in the Era of LLM](https://arxiv.org/abs/2508.08448)
* [Extremely Scalable Distributed Computation of Contour Trees via Pre-Simplification](https://arxiv.org/abs/2508.08433)
* [XDMA: A Distributed, Extensible DMA Architecture for Layout-Flexible Data Movements in Heterogeneous Multi-Accelerator SoCs](https://arxiv.org/abs/2508.08396)
* [Towards Efficient and Practical GPU Multitasking in the Era of LLM](https://arxiv.org/abs/2508.08448)
* [Ultra Ethernet's Design Principles and Architectural Innovations](https://arxiv.org/abs/2508.08906)
* [Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference](https://arxiv.org/abs/2508.08438)


### 2025-08-12

* [On the Operational Resilience of CBDC: Threats and Prospects of Formal Validation for Offline Payments](https://arxiv.org/abs/2508.08064)
* [Performance Evaluation of Brokerless Messaging Libraries](https://arxiv.org/abs/2508.07934)
* [Towards Lock Modularization for Heterogeneous Environments](https://arxiv.org/abs/2508.07756)
* [Over-the-Top Resource Broker System for Split Computing: An Approach to Distribute Cloud Computing Infrastructure](https://arxiv.org/abs/2508.07744)
* [Perpetual exploration in anonymous synchronous networks with a Byzantine black hole](https://arxiv.org/abs/2508.07703)
* [Taming Cold Starts: Proactive Serverless Scheduling with Model Predictive Control](https://arxiv.org/abs/2508.07640)
* [Coordinated Power Management on Heterogeneous Systems](https://arxiv.org/abs/2508.07605)
* [An Experimental Exploration of In-Memory Computing for Multi-Layer Perceptrons](https://arxiv.org/abs/2508.07317)
* [FlashMP: Fast Discrete Transform-Based Solver for Preconditioning Maxwell's Equations on GPUs](https://arxiv.org/abs/2508.07193)
* [The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU Libraries](https://arxiv.org/abs/2508.07071)
* [Convergence Sans Synchronization](https://arxiv.org/abs/2508.06949)
* [Kairos: Low-latency Multi-Agent Serving with Shared LLMs and Excessive Loads in the Public Cloud](https://arxiv.org/abs/2508.06948)
* [PiKV: KV Cache Management System for Mixture of Experts](https://arxiv.org/abs/2508.06526)
* [Fully-Fluctuating Participation in Sleepy Consensus](https://arxiv.org/abs/2508.08068)
* [GPU-Accelerated Syndrome Decoding for Quantum LDPC Codes below the 63 $μ$s Latency Threshold](https://arxiv.org/abs/2508.07879)
* [Enhancing Privacy in Decentralized Min-Max Optimization: A Differentially Private Approach](https://arxiv.org/abs/2508.07505)
* [Real-Time Analysis of Unstructured Data with Machine Learning on Heterogeneous Architectures](https://arxiv.org/abs/2508.07423)
* [DSperse: A Framework for Targeted Verification in Zero-Knowledge Machine Learning](https://arxiv.org/abs/2508.06972)
* [A Portable Multi-GPU Solver for Collisional Plasmas with Coulombic Interactions](https://arxiv.org/abs/2508.06771)
* [PANAMA: A Network-Aware MARL Framework for Multi-Agent Path Finding in Digital Twin Ecosystems](https://arxiv.org/abs/2508.06767)


### 2025-08-11

* [Performant Unified GPU Kernels for Portable Singular Value Computation Across Hardware and Precision](https://arxiv.org/abs/2508.06339)
* [KV Cache Compression for Inference Efficiency in LLMs: A Review](https://arxiv.org/abs/2508.06297)
* [EC2MoE: Adaptive End-Cloud Pipeline Collaboration Enabling Scalable Mixture-of-Experts Inference](https://arxiv.org/abs/2508.06024)
* [KnapFormer: An Online Load Balancer for Efficient Diffusion Transformers Training](https://arxiv.org/abs/2508.06001)
* [Snowpark: Performant, Secure, User-Friendly Data Engineering and AI/ML Next To Your Data](https://arxiv.org/abs/2508.05904)
* [A Dynamic Approach to Load Balancing in Cloud Infrastructure: Enhancing Energy Efficiency and Resource Utilization](https://arxiv.org/abs/2508.05821)
* [Accelerating Data Chunking in Deduplication Systems using Vector Instructions](https://arxiv.org/abs/2508.05797)
* [Voting-Based Semi-Parallel Proof-of-Work Protocol](https://arxiv.org/abs/2508.06489)


### 2025-08-08

* [Simulating LLM training workloads for heterogeneous compute and network infrastructure](https://arxiv.org/abs/2508.05370)


### 2025-08-07

* [S2M3: Split-and-Share Multi-Modal Models for Distributed Multi-Task Inference on the Edge](https://arxiv.org/abs/2508.04271)


### 2025-08-06

* [Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling](https://arxiv.org/abs/2508.03611)
* [Frontier: Simulating the Next Generation of LLM Inference Systems](https://arxiv.org/abs/2508.03148)


### 2025-08-05

* [PUSHtap: PIM-based In-Memory HTAP with Unified Data Storage Format](https://arxiv.org/abs/2508.02309)
* [Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving](https://arxiv.org/abs/2508.01989)


### 2025-08-04

* [SwarnRaft: Leveraging Consensus for Robust Drone Swarm Coordination in GNSS-Degraded Environments](https://arxiv.org/abs/2508.00622)
* [Adacc: Adaptive Compression and Activation Checkpointing for LLM Memory Management](https://arxiv.org/abs/2508.00806)
* [Quality-of-Service Aware LLM Routing for Edge Computing with Multiple Experts](https://arxiv.org/abs/2508.00234)


### 2025-07-31

* [DSPE: Profit Maximization in Edge-Cloud Storage System using Dynamic Space Partitioning with Erasure Code](https://arxiv.org/abs/2507.22801)
* [Leveraging Caliper and Benchpark to Analyze MPI Communication Patterns: Insights from AMG2023, Kripke, and Laghos](https://arxiv.org/abs/2507.22372)


### 2025-07-30

* [LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems](https://arxiv.org/abs/2507.21276)
* [Advancing Compositional LLM Reasoning with Structured Task Relations in Interactive Multimodal Communications](https://arxiv.org/abs/2507.21199)


### 2025-07-29

* [MegatronApp: Efficient and Comprehensive Management on Distributed LLM Training](https://arxiv.org/abs/2507.19845)


### 2025-07-28

* [RailX: A Flexible, Scalable, and Low-Cost Network Architecture for Hyper-Scale LLM Training Systems](https://arxiv.org/abs/2507.18889)


### 2025-07-25

* [Cloud Native System for LLM Inference Serving](https://arxiv.org/abs/2507.18007)
* [Unlock the Potential of Fine-grained LLM Serving via Dynamic Module Scaling](https://arxiv.org/abs/2507.18006)
* [Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving](https://arxiv.org/abs/2507.18454)


### 2025-07-24

* [BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs](https://arxiv.org/abs/2507.17133)
* [BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving](https://arxiv.org/abs/2507.17120)


### 2025-07-23

* [Cooling Matters: Benchmarking Large Language Models and Vision-Language Models on Liquid-Cooled Versus Air-Cooled H100 GPU Systems](https://arxiv.org/abs/2507.16781)
* [Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey of Algorithms, Execution, and Open Challenges](https://arxiv.org/abs/2507.16731)
* [Reducing GPU Memory Fragmentation via Spatio-Temporal Planning for Efficient Large-Scale Model Training](https://arxiv.org/abs/2507.16274)


### 2025-07-22

* [Efficient Routing of Inference Requests across LLM Instances in Cloud-Edge Computing](https://arxiv.org/abs/2507.15553)
* [GALE: Leveraging Heterogeneous Systems for Efficient Unstructured Mesh Data Analysis](https://arxiv.org/abs/2507.15230)
* [Byzantine-Robust Decentralized Coordination of LLM Agents](https://arxiv.org/abs/2507.14928)
* [Characterizing Communication Patterns in Distributed Large Language Model Inference](https://arxiv.org/abs/2507.14392)
* [IDSS, a Novel P2P Relational Data Storage Service](https://arxiv.org/abs/2507.14682)
* [A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering](https://arxiv.org/abs/2507.14179)


### 2025-07-21

* [DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training](https://arxiv.org/abs/2507.13833)
* [Leveraging Multi-Instance GPUs through moldable task scheduling](https://arxiv.org/abs/2507.13601)
* [An End-to-End DNN Inference Framework for the SpiNNaker2 Neuromorphic MPSoC](https://arxiv.org/abs/2507.13736)


### 2025-07-18

* [BootSeer: Analyzing and Mitigating Initialization Bottlenecks in Large-Scale LLM Training](https://arxiv.org/abs/2507.12619)


### 2025-07-17

* [Toward Efficient SpMV in Sparse LLMs via Block Extraction and Compressed Storage](https://arxiv.org/abs/2507.12205)
* [Arctic Inference with Shift Parallelism: Fast and Efficient Open Source Inference System for Enterprise AI](https://arxiv.org/abs/2507.11830)


### 2025-07-16

* [Quantifying the Energy Consumption and Carbon Emissions of LLM Inference via Simulations](https://arxiv.org/abs/2507.11417)
* [MIRAGE: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving](https://arxiv.org/abs/2507.11507)


### 2025-07-15

* [Zorse: Optimizing LLM Training Efficiency on Heterogeneous GPU Clusters](https://arxiv.org/abs/2507.10392)
* [Cross-Timeslot Optimization for Distributed GPU Inference Using Reinforcement Learning](https://arxiv.org/abs/2507.10259)
* [Past-Future Scheduler for LLM Serving under SLA Guarantees](https://arxiv.org/abs/2507.10150)
* [ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism](https://arxiv.org/abs/2507.10069)
* [EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning](https://arxiv.org/abs/2507.10026)
* [Green-LLM: Optimal Workload Allocation for Environmentally-Aware Distributed Inference](https://arxiv.org/abs/2507.09942)
* [SLIM: A Heterogeneous Accelerator for Edge Inference of Sparse Large Language Model via Adaptive Thresholding](https://arxiv.org/abs/2507.09201)
* [On Evaluating Performance of LLM Inference Serving Systems](https://arxiv.org/abs/2507.09019)


### 2025-07-11

* [KIS-S: A GPU-Aware Kubernetes Inference Simulator with RL-Based Auto-Scaling](https://arxiv.org/abs/2507.07932)
* [KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](https://arxiv.org/abs/2507.07400)
* [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/abs/2507.07120)
* [Analysing semantic data storage in Distributed Ledger Technologies for Data Spaces](https://arxiv.org/abs/2507.07116)


### 2025-07-10

* [Nexus: Taming Throughput-Latency Tradeoff in LLM Serving via Efficient GPU Sharing](https://arxiv.org/abs/2507.06608)
* [SlimCaching: Edge Caching of Mixture-of-Experts for Distributed Inference](https://arxiv.org/abs/2507.06567)


### 2025-07-08

* [On Fault Tolerance of Data Storage Systems: A Holistic Perspective](https://arxiv.org/abs/2507.03849)
* [Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning](https://arxiv.org/abs/2507.03305)
* [Symbiosis: Multi-Adapter Inference and Fine-Tuning](https://arxiv.org/abs/2507.03220)
* [ZettaLith: An Architectural Exploration of Extreme-Scale AI Inference Acceleration](https://arxiv.org/abs/2507.02871)
* [Performance Evaluation of General Purpose Large Language Models for Basic Linear Algebra Subprograms Code Generation](https://arxiv.org/abs/2507.04697)


### 2025-07-04

* [FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference](https://arxiv.org/abs/2507.02620)
* [Dissecting the Impact of Mobile DVFS Governors on LLM Inference Performance and Energy Efficiency](https://arxiv.org/abs/2507.02135)


### 2025-07-03

* [Deep Recommender Models Inference: Automatic Asymmetric Data Flow Optimization](https://arxiv.org/abs/2507.01676)
* [EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices](https://arxiv.org/abs/2507.01438)


### 2025-07-02

* [Accelerating Loading WebGraphs in ParaGrapher](https://arxiv.org/abs/2507.00716)
* [DynoStore: A wide-area distribution system for the management of data over heterogeneous storage](https://arxiv.org/abs/2507.00576)
* [LLM-Mesh: Enabling Elastic Sharing for Serverless LLM Inference](https://arxiv.org/abs/2507.00507)
* [Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and High-Performance GPUs](https://arxiv.org/abs/2507.00418)
* [Toward Edge General Intelligence with Multiple-Large Language Model (Multi-LLM): Architecture, Trust, and Orchestration](https://arxiv.org/abs/2507.00672)
* [HelixPipe: Efficient Distributed Training of Long Sequence Transformers with Attention Parallel Pipeline Parallelism](https://arxiv.org/abs/2507.00394)


### 2025-07-01

* [Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC](https://arxiv.org/abs/2506.24045)
* [QPART: Adaptive Model Quantization and Dynamic Workload Balancing for Accuracy-aware Edge Inference](https://arxiv.org/abs/2506.23934)
* [Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model](https://arxiv.org/abs/2506.23635)


### 2025-06-30

* [MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism](https://arxiv.org/abs/2506.22175)
* [SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference](https://arxiv.org/abs/2506.22033)


### 2025-06-27

* [ParEval-Repo: A Benchmark Suite for Evaluating LLMs with Repository-level HPC Translation Tasks](https://arxiv.org/abs/2506.20938)


### 2025-06-26

* [Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU](https://arxiv.org/abs/2506.20187)
* [MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection](https://arxiv.org/abs/2506.19884)


### 2025-06-25

* [Shelby: Decentralized Storage Designed to Serve](https://arxiv.org/abs/2506.19233)
* [Vertex addition to a ball graph with application to reliability and area coverage in autonomous swarms](https://arxiv.org/abs/2506.19197)
* [Binsparse: A Specification for Cross-Platform Storage of Sparse Matrices and Tensors](https://arxiv.org/abs/2506.19175)


### 2025-06-24

* [Leveraging Cloud-Fog Automation for Autonomous Collision Detection and Classification in Intelligent Unmanned Surface Vehicles](https://arxiv.org/abs/2506.18024)
* [Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model-Based Recommendation Systems](https://arxiv.org/abs/2506.17551)
* [Leveraging Large Language Model for Intelligent Log Processing and Autonomous Debugging in Cloud AI Platforms](https://arxiv.org/abs/2506.17900)
* [VeriLocc: End-to-End Cross-Architecture Register Allocation via LLM](https://arxiv.org/abs/2506.17506)


### 2025-06-23

* [TrainVerify: Equivalence-Based Verification for Distributed LLM Training](https://arxiv.org/abs/2506.15961)


### 2025-06-19

* [All is Not Lost: LLM Recovery without Checkpoints](https://arxiv.org/abs/2506.15461)
* [eLLM: Elastic Memory Management Framework for Efficient LLM Serving](https://arxiv.org/abs/2506.15155)
* [Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching](https://arxiv.org/abs/2506.14852)
* [Efficient Serving of LLM Applications with Probabilistic Demand Modeling](https://arxiv.org/abs/2506.14851)


### 2025-06-18

* [Keigo: Co-designing Log-Structured Merge Key-Value Stores with a Non-Volatile, Concurrency-aware Storage Hierarchy (Extended Version)](https://arxiv.org/abs/2506.14630)


### 2025-06-17

* [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/abs/2506.12708)
* [HarMoEny: Efficient Multi-GPU Inference of MoE Models](https://arxiv.org/abs/2506.12417)
* [NaSh: Guardrails for an LLM-Powered Natural Language Shell](https://arxiv.org/abs/2506.13028)
* [Semantic Scheduling for LLM Inference](https://arxiv.org/abs/2506.12204)


### 2025-06-16

* [A retrospective on DISPEED -- Leveraging heterogeneity in a drone swarm for IDS execution](https://arxiv.org/abs/2506.11800)
* [SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding](https://arxiv.org/abs/2506.11309)


### 2025-06-13

* [TD-Pipe: Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference](https://arxiv.org/abs/2506.10470)
* [HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration](https://arxiv.org/abs/2506.10401)


### 2025-06-12

* [Understanding the Performance and Power of LLM Inferencing on Edge Accelerators](https://arxiv.org/abs/2506.09554)
* [SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving](https://arxiv.org/abs/2506.09397)
* [ScalableHD: Scalable and High-Throughput Hyperdimensional Computing Inference on Multi-Core CPUs](https://arxiv.org/abs/2506.09282)
* [EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model](https://arxiv.org/abs/2506.09061)


### 2025-06-11

* [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)


### 2025-06-10

* [Addressing tokens dynamic generation, propagation, storage and renewal to secure the GlideinWMS pilot based jobs and system](https://arxiv.org/abs/2506.07379)
* [Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage](https://arxiv.org/abs/2506.06472)
* [Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques](https://arxiv.org/abs/2506.06579)


### 2025-06-09

* [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://arxiv.org/abs/2506.05508)


### 2025-06-06

* [FlashDMoE: Fast Distributed MoE in a Single Kernel](https://arxiv.org/abs/2506.04667)
* [SkimROOT: Accelerating LHC Data Filtering with Near-Storage Processing](https://arxiv.org/abs/2506.04507)
* [Knowledge-Guided Attention-Inspired Learning for Task Offloading in Vehicle Edge Computing](https://arxiv.org/abs/2506.04456)
* [Inference economics of language models](https://arxiv.org/abs/2506.04645)


### 2025-06-05

* [Cascadia: A Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)
* [Crowd-SFT: Crowdsourcing for LLM Alignment](https://arxiv.org/abs/2506.04063)
* [Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs](https://arxiv.org/abs/2506.03296)


### 2025-06-04

* [Adaptive Configuration Selection for Multi-Model Inference Pipelines in Edge Computing](https://arxiv.org/abs/2506.02814)
* [Simplifying Root Cause Analysis in Kubernetes with StateGraph and LLM](https://arxiv.org/abs/2506.02490)
* [D-Rex: Heterogeneity-Aware Reliability Framework and Adaptive Algorithms for Distributed Storage](https://arxiv.org/abs/2506.02026)
* [Evaluating the Efficacy of LLM-Based Reasoning for Multiobjective HPC Job Scheduling](https://arxiv.org/abs/2506.02025)
* [NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs](https://arxiv.org/abs/2506.02024)
* [DistMLIP: A Distributed Inference Platform for Machine Learning Interatomic Potentials](https://arxiv.org/abs/2506.02023)
* [Efficient and Workload-Aware LLM Serving via Runtime Layer Swapping and KV Cache Resizing](https://arxiv.org/abs/2506.02006)


### 2025-06-03

* [Adaptive, Efficient and Fair Resource Allocation in Cloud Datacenters leveraging Weighted A3C Deep Reinforcement Learning](https://arxiv.org/abs/2506.00929)
* [Advancing AI-assisted Hardware Design with Hierarchical Decentralized Training and Personalized Inference-Time Optimization](https://arxiv.org/abs/2506.00002)


### 2025-06-02

* [Distributed Intelligence in the Computing Continuum with Active Inference](https://arxiv.org/abs/2505.24618)
* [SkyLB: A Locality-Aware Cross-Region Load Balancer for LLM Inference](https://arxiv.org/abs/2505.24095)
* [EmbAdvisor: Adaptive Cache Management for Sustainable LLM Serving](https://arxiv.org/abs/2505.23970)


### 2025-05-30

* [Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters](https://arxiv.org/abs/2505.23554)
* [MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning](https://arxiv.org/abs/2505.23254)
* [Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism](https://arxiv.org/abs/2505.23219)
* [Accelerating AllReduce with a Persistent Straggler](https://arxiv.org/abs/2505.23523)


### 2025-05-29

* [Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference](https://arxiv.org/abs/2505.21919)

### 2025-05-27

* [DGRAG: Distributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems](https://arxiv.org/abs/2505.19847)
* [Win Fast or Lose Slow: Balancing Speed and Accuracy in Latency-Sensitive Decisions of LLMs](https://arxiv.org/abs/2505.19481)

### 2025-05-26

* [H2:Towards Efficient Large-Scale LLM Training on Hyper-Heterogeneous Cluster over 1,000 Chips](https://arxiv.org/abs/2505.17548)
* [Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models](https://arxiv.org/abs/2505.17826)

### 2025-05-23

* [Edge-First Language Model Inference: Models, Metrics, and Tradeoffs](https://arxiv.org/abs/2505.16508)
* [Recursive Offloading for LLM Serving in Multi-tier Networks](https://arxiv.org/abs/2505.16502)

### 2025-05-22

* [Balanced and Elastic End-to-end Training of Dynamic LLMs](https://arxiv.org/abs/2505.14864)


### 2025-05-21

* [ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs](https://arxiv.org/abs/2505.14468)

### 2025-05-20

* [HydraInfer: Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving](https://arxiv.org/abs/2505.12658)
* [Arrow: Adaptive Scheduling Mechanisms for Disaggregated LLM Inference Architecture](https://arxiv.org/abs/2505.11916)
* [Occult: Optimizing Collaborative Communication across Experts for Accelerated Parallel MoE Training and Inference](https://arxiv.org/abs/2505.13345)

### 2025-05-19

* [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://arxiv.org/abs/2505.11329)
* [MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production](https://arxiv.org/abs/2505.11432)
* [MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems](https://arxiv.org/abs/2505.11415)


### 2025-05-16

* [ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production](https://arxiv.org/abs/2505.09999)


### 2025-05-15

* [ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor](https://arxiv.org/abs/2505.09142)

### 2025-05-14

* [Fused3S: Fast Sparse Attention on Tensor Cores](https://arxiv.org/abs/2505.08098)
* [Patchwork: A Unified Framework for RAG Serving](https://arxiv.org/abs/2505.07833)

### 2025-05-13

* [PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications](https://arxiv.org/abs/2505.07203)
* [SneakPeek: Data-Aware Model Selection and Scheduling for Inference Serving on the Edge](https://arxiv.org/abs/2505.06641)
* [Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference](https://arxiv.org/abs/2505.06461)
* [SpecRouter: Adaptive Routing for Multi-Level Speculative Decoding in Large Language Models](https://arxiv.org/abs/2505.07680)
* [QoS-Efficient Serving of Multiple Mixture-of-Expert LLMs Using Partial Runtime Reconfiguration](https://arxiv.org/abs/2505.06481)
* [Towards Efficient LLM Storage Reduction via Tensor Deduplication and Delta Compression](https://arxiv.org/abs/2505.06252)


### 2025-05-12

* [Understanding Stragglers in Large Model Training Using What-if Analysis](https://arxiv.org/abs/2505.05713)


### 2025-05-09

* [Walrus: An Efficient Decentralized Storage Network](https://arxiv.org/abs/2505.05370)
* [Exploring Influence Factors on LLM Suitability for No-Code Development of End User IoT Applications](https://arxiv.org/abs/2505.04710)
* [HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights](https://arxiv.org/abs/2505.04846)


### 2025-05-08

* [Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving](https://arxiv.org/abs/2505.04021)
* [Rollbaccine : Herd Immunity against Storage Rollback Attacks in TEEs [Technical Report]](https://arxiv.org/abs/2505.04014)
* [Splitwiser: Efficient LM inference with constrained resources](https://arxiv.org/abs/2505.03763)


### 2025-05-06

* [Large Language Model Partitioning for Low-Latency Inference at the Edge](https://arxiv.org/abs/2505.02533)
* [Opt-GPTQ: An Optimized GPTQ Combining Sparse Attention and Quantization Techniques](https://arxiv.org/abs/2505.02351)
* [HAS-GPU: Efficient Hybrid Auto-scaling with Fine-grained GPU Allocation for SLO-aware Serverless Inferences](https://arxiv.org/abs/2505.01968)
* [HSplitLoRA: A Heterogeneous Split Parameter-Efficient Fine-Tuning Framework for Large Language Models](https://arxiv.org/abs/2505.02795)

### 2025-05-05

* [CaGR-RAG: Context-aware Query Grouping for Disk-based Vector Search in RAG Systems](https://arxiv.org/abs/2505.01164)


### 2025-04-30

* [Leveraging Neural Graph Compilers in Machine Learning Research for Edge-Cloud Systems](https://arxiv.org/abs/2504.20198)
* [GenTorrent: Scaling Large Language Model Serving with An Overley Network](https://arxiv.org/abs/2504.20101)
* [Tempo: Application-aware LLM Serving with Mixed SLO Requirements](https://arxiv.org/abs/2504.20068)
* [OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification](https://arxiv.org/abs/2504.20964)


### 2025-04-29

* [Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration](https://arxiv.org/abs/2504.19516)
* [Adaptra: Straggler-Resilient Hybrid-Parallel Training with Pipeline Adaptation](https://arxiv.org/abs/2504.19232)
* [semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage](https://arxiv.org/abs/2504.19867)
* [Taming the Titans: A Survey of Efficient LLM Inference Serving](https://arxiv.org/abs/2504.19720)


### 2025-04-28

* [EcoServe: Enabling Cost-effective LLM Serving with Proactive Intra- and Inter-Instance Orchestration](https://arxiv.org/abs/2504.18154)


### 2025-04-24

* [Preemption Aware Task Scheduling for Priority and Deadline Constrained DNN Inference Task Offloading in Homogeneous Mobile-Edge Networks](https://arxiv.org/abs/2504.16792)
* [Real-time Bayesian inference at extreme scale: A digital twin for tsunami early warning applied to the Cascadia subduction zone](https://arxiv.org/abs/2504.16344)
* [HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing](https://arxiv.org/abs/2504.16112)

### 2025-04-23

* [SeaLLM: Service-Aware and Latency-Optimized Resource Sharing for Large Language Model Inference](https://arxiv.org/abs/2504.15720)
* [High-Throughput LLM inference on Heterogeneous Clusters](https://arxiv.org/abs/2504.15303)
* [RAGDoll: Efficient Offloading-based Online RAG System on a Single GPU](https://arxiv.org/abs/2504.15302)
* [D$^{2}$MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving](https://arxiv.org/abs/2504.15299)
* [Scalability Optimization in Cloud-Based AI Inference Services: Strategies for Real-Time Load Balancing and Automated Scaling](https://arxiv.org/abs/2504.15296)
* [StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation](https://arxiv.org/abs/2504.15930)
* [RAGDoll: Efficient Offloading-based Online RAG System on a Single GPU](https://arxiv.org/abs/2504.15302)

### 2025-04-22

* [SLO-Aware Scheduling for Large Language Model Inferences](https://arxiv.org/abs/2504.14966)
* [gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling](https://arxiv.org/abs/2504.14775)
* [Joint Optimization of Offloading, Batching and DVFS for Multiuser Co-Inference](https://arxiv.org/abs/2504.14611)
* [MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core](https://arxiv.org/abs/2504.14960)
* [Optimizing SLO-oriented LLM Serving with PD-Multiplexing](https://arxiv.org/abs/2504.14489)

### 2025-04-18

* [You Don't Need All Attentions: Distributed Dynamic Fine-Tuning for Foundation Models](https://arxiv.org/abs/2504.12471)

### 2025-04-17

* [Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures](https://arxiv.org/abs/2504.11750)
* [Cost-Efficient LLM Serving in the Cloud: VM Selection with KV Cache Offloading](https://arxiv.org/abs/2504.11816)
* [70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float](https://arxiv.org/abs/2504.11651)

### 2025-04-16

* [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

### 2025-04-15

* [Optimal Graph Stretching for Distributed Averaging](https://arxiv.org/abs/2504.10289)
* [Training LLMs on HPC Systems: Best Practices from the OpenGPT-X Project](https://arxiv.org/abs/2504.10013)
* [MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints](https://arxiv.org/abs/2504.09345)
* [Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training](https://arxiv.org/abs/2504.09307)
* [DynaServe: Unified and Elastic Tandem-Style Execution for Dynamic Disaggregated LLM Serving](https://arxiv.org/abs/2504.09285)
* [SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting](https://arxiv.org/abs/2504.08850)
* [DARIS: An Oversubscribed Spatio-Temporal Scheduler for Real-Time DNN Inference on GPUs](https://arxiv.org/abs/2504.08795)
* [PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters](https://arxiv.org/abs/2504.08791)
* [SLOs-Serve: Optimized Serving of Multi-SLO LLMs](https://arxiv.org/abs/2504.08784)
* [MigGPT: Harnessing Large Language Models for Automated Migration of Out-of-Tree Linux Kernel Patches Across Versions](https://arxiv.org/abs/2504.09474)

### 2025-04-14

* [Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices](https://arxiv.org/abs/2504.08242)

### 2025-04-11

* [Token Level Routing Inference System for Edge Devices](https://arxiv.org/abs/2504.07878)

### 2025-04-09

* [Nonuniform-Tensor-Parallelism: Mitigating GPU failure impact for Scaled-up LLM Training](https://arxiv.org/abs/2504.06095)
* [HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference](https://arxiv.org/abs/2504.05897)

### 2025-04-08

* [IntentContinuum: Using LLMs to Support Intent-Based Computing Across the Compute Continuum](https://arxiv.org/abs/2504.04429)
* [HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs](https://arxiv.org/abs/2504.03871)
* [FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling](https://arxiv.org/abs/2504.03775)
* [Adaptive Orchestration for Inference of Large Foundation Models at the Edge](https://arxiv.org/abs/2504.03668)
* [LLM & HPC:Benchmarking DeepSeek's Performance in High-Performance Computing Tasks](https://arxiv.org/abs/2504.03665)
* [PIPO: Pipelined Offloading for Efficient Inference on Consumer Devices](https://arxiv.org/abs/2504.03664)

### 2025-04-07

* [LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications](https://arxiv.org/abs/2504.03444)

### 2025-04-04

* [FT-Transformer: Resilient and Reliable Transformer with End-to-End Fault Tolerant Attention](https://arxiv.org/abs/2504.02211)

### 2025-04-02

* [AMP4EC: Adaptive Model Partitioning Framework for Efficient Deep Learning Inference in Edge Computing Environments](https://arxiv.org/abs/2504.00407)

### 2025-04-01

* [OrchMLLM: Orchestrate Multimodal Data with Batch Post-Balancing to Accelerate Multimodal Large Language Model Training](https://arxiv.org/abs/2503.23830)
* [MVDRAM: Enabling GeMV Execution in Unmodified DRAM for Low-Bit LLM Acceleration](https://arxiv.org/abs/2503.23817)


### 2025-03-31

* [Niyama : Breaking the Silos of LLM Inference Serving](https://arxiv.org/abs/2503.22562)


### 2025-03-28

* [Robust DNN Partitioning and Resource Allocation Under Uncertain Inference Time](https://arxiv.org/abs/2503.21476)
* [Optimizing Multi-DNN Inference on Mobile Devices through Heterogeneous Processor Co-Execution](https://arxiv.org/abs/2503.21109)
* [Scalability Evaluation of HPC Multi-GPU Training for ECG-based LLMs](https://arxiv.org/abs/2503.21033)


### 2025-03-27

* [Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation](https://arxiv.org/abs/2503.20552)
* [Harmonia: A Multi-Agent Reinforcement Learning Approach to Data Placement and Migration in Hybrid Storage Systems](https://arxiv.org/abs/2503.20507)
* [L4: Diagnosing Large-scale LLM Training Failures via Automated Log Analysis](https://arxiv.org/abs/2503.20263)


### 2025-03-26

* [Mist: Efficient Distributed Training of Large Language Models via Memory-Parallelism Co-Optimization](https://arxiv.org/abs/2503.19050)


### 2025-03-25

* [Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://arxiv.org/abs/2503.18292)
* [Risk Management for Distributed Arbitrage Systems: Integrating Artificial Intelligence](https://arxiv.org/abs/2503.18265)
* [WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training](https://arxiv.org/abs/2503.17924)
* [PipeBoost: Resilient Pipelined Architecture for Fast Serverless LLM Scaling](https://arxiv.org/abs/2503.17707)
* [A Generative Caching System for Large Language Models](https://arxiv.org/abs/2503.17603)


### 2025-03-24

* [Improving the End-to-End Efficiency of Offline Inference for Multi-LLM Applications Based on Sampling and Simulation](https://arxiv.org/abs/2503.16893)
* [Distributed LLMs and Multimodal Large Language Models: A Survey on Advances, Challenges, and Future Directions](https://arxiv.org/abs/2503.16585)


### 2025-03-21

* [SPIN: Accelerating Large Language Model Inference with Heterogeneous Speculative Models](https://arxiv.org/abs/2503.15921)
* [ATTENTION2D: Communication Efficient Distributed Self-Attention Mechanism](https://arxiv.org/abs/2503.15758)


### 2025-03-20

* [Efficient allocation of image recognition and LLM tasks on multi-GPU system](https://arxiv.org/abs/2503.15252)
* [Prada: Black-Box LLM Adaptation with Private Data on Resource-Constrained Devices](https://arxiv.org/abs/2503.14932)
* [RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving](https://arxiv.org/abs/2503.14649)


### 2025-03-19

* [Do Large Language Models Understand Performance Optimization?](https://arxiv.org/abs/2503.13772)


### 2025-03-18

* [Adaptive Fault Tolerance Mechanisms of Large Language Models in Cloud Computing Environments](https://arxiv.org/abs/2503.12228)
* [FAILS: A Framework for Automated Collection and Analysis of LLM Service Incidents](https://arxiv.org/abs/2503.12185)


### 2025-03-17

* [Beyond A Single AI Cluster: A Survey of Decentralized LLM Training](https://arxiv.org/abs/2503.11023)
* [LLMPerf: GPU Performance Modeling meets Large Language Models](https://arxiv.org/abs/2503.11244)
* [Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores](https://arxiv.org/abs/2503.10725)


### 2025-03-14

* [SPPO:Efficient Long-sequence LLM Training via Adaptive Sequence Pipeline Parallel Offloading](https://arxiv.org/abs/2503.10377)
* [Collaborative Speculative Inference for Efficient LLM Inference Serving](https://arxiv.org/abs/2503.10325)
* [MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching](https://arxiv.org/abs/2503.09716)


### 2025-03-13

* [Performance Models for a Two-tiered Storage System](https://arxiv.org/abs/2503.08966)
* [Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference](https://arxiv.org/abs/2503.09304)
* [Sometimes Painful but Certainly Promising: Feasibility and Trade-offs of Language Model Inference at the Edge](https://arxiv.org/abs/2503.09114)


### 2025-03-12

* [TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems](https://arxiv.org/abs/2503.08415)
* [Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference](https://arxiv.org/abs/2503.08311)
* [Will LLMs Scaling Hit the Wall? Breaking Barriers via Distributed Resources on Massive Edge Devices](https://arxiv.org/abs/2503.08223)
* [Accelerating MoE Model Inference with Expert Sharding](https://arxiv.org/abs/2503.08467)
* [FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework](https://arxiv.org/abs/2503.08461)
* [DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems](https://arxiv.org/abs/2503.07675)


### 2025-03-11

* [Seesaw: High-throughput LLM Inference via Model Re-sharding](https://arxiv.org/abs/2503.06433)
* [eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference](https://arxiv.org/abs/2503.06823)
* [Distributed Graph Neural Network Inference With Just-In-Time Compilation For Industry-Scale Graphs](https://arxiv.org/abs/2503.06208)


### 2025-03-10

* [Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching](https://arxiv.org/abs/2503.05248)
* [Linear-MoE: Linear Sequence Modeling Meets Mixture-of-Experts](https://arxiv.org/abs/2503.05447)


### 2025-03-07

* [Dynamic Pricing for On-Demand DNN Inference in the Edge-AI Market](https://arxiv.org/abs/2503.04521)
* [Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling](https://arxiv.org/abs/2503.04398)
* [Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation](https://arxiv.org/abs/2503.04302)


### 2025-03-06

* [Enhancing Memory Efficiency in Large Language Model Training Through Chronos-aware Pipeline Parallelism](https://arxiv.org/abs/2503.03182)
* [Environment-Aware Dynamic Pruning for Pipelined Edge Inference](https://arxiv.org/abs/2503.03070)


### 2025-03-05

* [SpecInF: Exploiting Idle GPU Resources in Distributed DL Training via Speculative Inference Filling](https://arxiv.org/abs/2503.02550)
* [CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory](https://arxiv.org/abs/2503.02354)
* [VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference](https://arxiv.org/abs/2503.02236)
### 2025-03-04

* [Improving inference time in multi-TPU systems with profiled model segmentation](https://arxiv.org/abs/2503.01025)

### 2025-03-03

* [ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs](https://arxiv.org/abs/2502.21231)
* [TeleRAG: Efficient Retrieval-Augmented Generation Inference with Lookahead Retrieval](https://arxiv.org/abs/2502.20969)
* [Cicada: A Pipeline-Efficient Approach to Serverless Inference with Decoupled Management](https://arxiv.org/abs/2502.20959)
* [SkyStore: Cost-Optimized Object Storage Across Regions and Clouds](https://arxiv.org/abs/2502.20818)
* [LADs: Leveraging LLMs for AI-Driven DevOps](https://arxiv.org/abs/2502.20825)

### 2025-02-28

* [SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks](https://arxiv.org/abs/2502.19913)

### 2025-02-27

* [CLLoRA: An Approach to Measure the Effects of the Context Length for LLM Fine-Tuning](https://arxiv.org/abs/2502.18910)

### 2025-02-25

* [FairKV: Balancing Per-Head KV Cache for Fast Multi-GPU Inference](https://arxiv.org/abs/2502.15804)
* [Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization](https://arxiv.org/abs/2502.15763)
* [LoXR: Performance Evaluation of Locally Executing LLMs on XR Devices](https://arxiv.org/abs/2502.15761)
* [DistrEE: Distributed Early Exit of Deep Neural Network Inference on Edge Devices](https://arxiv.org/abs/2502.15735)

### 2025-02-24

* [Towards Swift Serverless LLM Cold Starts with ParaServe](https://arxiv.org/abs/2502.15524)
* [FlexPie: Accelerate Distributed Inference on Edge Devices with Flexible Combinatorial Optimization[Technical Report]](https://arxiv.org/abs/2502.15312)

### 2025-02-21

* [Serving Models, Fast and Slow:Optimizing Heterogeneous LLM Inferencing Workloads at Scale](https://arxiv.org/abs/2502.14617)
* [Optimizing the Longhorn Cloud-native Software Defined Storage Engine for High Performance](https://arxiv.org/abs/2502.14419)
* [CarbonEdge: Leveraging Mesoscale Spatial Carbon-Intensity Variations for Low Carbon Edge Computing](https://arxiv.org/abs/2502.14076)
* [LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention](https://arxiv.org/abs/2502.14866)
* [LLM4FaaS: No-Code Application Development using LLMs and FaaS](https://arxiv.org/abs/2502.14450)

### 2025-02-20

* [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965)

### 2025-02-19

* [SparkAttention: High-Performance Multi-Head Attention for Large Models on Volta GPU Architecture](https://arxiv.org/abs/2502.12784)
* [Distributed On-Device LLM Inference With Over-the-Air Computation](https://arxiv.org/abs/2502.12559)
* [Understanding Silent Data Corruption in LLM Training](https://arxiv.org/abs/2502.12340)
* [Semantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay](https://arxiv.org/abs/2502.10151)

### 2025-02-18

* [Scalable and Cost-Efficient ML Inference: Parallel Batch Processing with Serverless Functions](https://arxiv.org/abs/2502.12017)
* [BagChain: A Dual-functional Blockchain Leveraging Bagging-based Distributed Learning](https://arxiv.org/abs/2502.11464)
* [DreamDDP: Accelerating Data Parallel Distributed LLM Training with Layer-wise Scheduled Partial Synchronization](https://arxiv.org/abs/2502.11058)
* [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
* [DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services](https://arxiv.org/abs/2502.11417)
* [Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings](https://arxiv.org/abs/2502.11007)

### 2025-02-17

* [λScale: Enabling Fast Scaling for Serverless Large Language Model Inference](https://arxiv.org/abs/2502.09922)

### 2025-02-14

* [ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments](https://arxiv.org/abs/2502.09334)

### 2025-02-13

* [Memory Offloading for Large Language Model Inference with Latency SLO Guarantees](https://arxiv.org/abs/2502.08182)
* [HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment](https://arxiv.org/abs/2502.07903)
* [Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers](https://arxiv.org/abs/2502.08145)

### 2025-02-11

* [MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643)
* [fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving](https://arxiv.org/abs/2502.05370)

### 2025-02-10

* [EcoServe: Designing Carbon-Aware AI Inference Systems](https://arxiv.org/abs/2502.05043)
* [WaferLLM: A Wafer-Scale LLM Inference System](https://arxiv.org/abs/2502.04563)

### 2025-02-07

* [HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference](https://arxiv.org/abs/2502.03589)
* [InfinitePOD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers](https://arxiv.org/abs/2502.03885)

### 2025-02-05

* [LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models](https://arxiv.org/abs/2502.02406)
* [Longer Attention Span: Increasing Transformer Context Length with Sparse Graph Processing Techniques](https://arxiv.org/abs/2502.01659)

### 2025-02-04

* [OCTOPINF: Workload-Aware Inference Serving for Edge Video Analytics](https://arxiv.org/abs/2502.01277)
* [Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs](https://arxiv.org/abs/2502.00722)
* [General Coded Computing in a Probabilistic Straggler Regime](https://arxiv.org/abs/2502.00645)
* [Leveraging InfiniBand Controller to Configure Deadlock-Free Routing Engines for Dragonflies](https://arxiv.org/abs/2502.01214)

### 2025-02-03

* [Infer-EDGE: Dynamic DNN Inference Optimization in 'Just-in-time' Edge-AI Implementations](https://arxiv.org/abs/2501.18842)

### 2025-01-30

* [Dual-Lagrange Encoding for Storage and Download in Elastic Computing for Resilience](https://arxiv.org/abs/2501.17275)

### 2025-01-29

* [On the Shape Containment Problem within the Amoebot Model with Reconfigurable Circuits](https://arxiv.org/abs/2501.16892)

### 2025-01-28

* [Static Batching of Irregular Workloads on GPUs: Framework and Application to Efficient MoE Model Inference](https://arxiv.org/abs/2501.16103)
* [Aging-aware CPU Core Management for Embodied Carbon Amortization in Cloud LLM Inference](https://arxiv.org/abs/2501.15829)
* [HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](https://arxiv.org/abs/2501.14808)
* [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](https://arxiv.org/abs/2501.14794)
* [DeServe: Towards Affordable Offline LLM Inference via Decentralization](https://arxiv.org/abs/2501.14784)
* [Dynamic Adaptation in Data Storage: Real-Time Machine Learning for Enhanced Prefetching](https://arxiv.org/abs/2501.14771)

### 2025-01-27

* [Locality-aware Fair Scheduling in LLM Serving](https://arxiv.org/abs/2501.14312)

### 2025-01-22

* [Accelerating End-Cloud Collaborative Inference via Near Bubble-free Pipeline Optimization](https://arxiv.org/abs/2501.12388)
* [DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference](https://arxiv.org/abs/2501.10375)
* [AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding](https://arxiv.org/abs/2501.12162)
* [Glinthawk: A Two-Tiered Architecture for High-Throughput LLM Inference](https://arxiv.org/abs/2501.11779)

### 2025-01-20

* [Over-the-Air Multi-Sensor Inference with Neural Networks Using Memristor-Based Analog Computing](https://arxiv.org/abs/2501.10245)

### 2025-01-17

* [PICE: A Semantic-Driven Progressive Inference System for LLM Serving in Cloud-Edge Networks](https://arxiv.org/abs/2501.09367)

### 2025-01-15

* [PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving](https://arxiv.org/abs/2501.08192)
* [HgPCN: A Heterogeneous Architecture for E2E Embedded Point Cloud Inference](https://arxiv.org/abs/2501.07767)

### 2025-01-14

* [CoCoI: Distributed Coded Inference System for Straggler Mitigation](https://arxiv.org/abs/2501.06856)
* [Ladder-residual: parallelism-aware architecture for accelerating large model inference with communication overlapping](https://arxiv.org/abs/2501.06589)

### 2025-01-13

* [A Practical Cross-Layer Approach for ML-Driven Storage Placement in Warehouse-Scale Computers](https://arxiv.org/abs/2501.05651)

### 2025-01-10

* [Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing](https://arxiv.org/abs/2501.05313)

### 2025-01-09

* [Collaborative Inference Acceleration with Non-Penetrative Tensor Partitioning](https://arxiv.org/abs/2501.04489)
* [Scalable Data Notarization Leveraging Hybrid DLTs](https://arxiv.org/abs/2501.04571)

### 2025-01-07

* [TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms](https://arxiv.org/abs/2501.02600)

### 2025-01-06

* [Efficient LLM Inference with Activation Checkpointing and Hybrid Caching](https://arxiv.org/abs/2501.01792)

### 2025-01-03

* [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
* [Dynamic Optimization of Storage Systems Using Reinforcement Learning Techniques](https://arxiv.org/abs/2501.00068)

