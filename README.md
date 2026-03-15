# 🚀 Advanced RLHF & DPO Alignment Pipeline
**Enterprise-Grade Post-Training Infrastructure for Southeast Asian Large Language Models**

[![Python](https://img.shields.io/badge/Language-Python%203.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/Optimization-DeepSpeed-8A2BE2?style=for-the-badge)](https://www.deepspeed.ai/)
[![HuggingFace](https://img.shields.io/badge/Tools-Hugging_Face_TRL-F9AB00?style=for-the-badge&logo=huggingface)](https://huggingface.co/docs/trl/index)

## 🌟 Executive Summary
This repository houses a state-of-the-art **Post-Training and Alignment Pipeline** designed to transform foundational Large Language Models (LLMs) into safe, instruction-following, and culturally aligned agents. Specifically architected for the nuances of **Southeast Asian (SEA) languages**, it implements robust Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) methodologies at scale.

## 🏗️ System Architecture
The pipeline is designed for distributed multi-GPU environments (A100/H100 clusters) utilizing **DeepSpeed ZeRO-3** for aggressive memory optimization.

### The Alignment Lifecycle:
1. **`01_sft.py` (Supervised Fine-Tuning):** Bootstraps the base model (e.g., SEA-LION, Llama-3) to follow explicit instructions using high-quality SEA datasets.
2. **`02_reward_modeling.py` (RM):** Trains a specialized classifier to score responses based on human preference data (Helpful, Honest, Harmless - 3H).
3. **`03_dpo_alignment.py` (DPO):** Directly optimizes the policy model using preference data, bypassing the instability of traditional PPO while achieving superior alignment.
4. **`llm_judge.py` (Evaluation):** An automated framework utilizing GPT-4 as an impartial judge to score the aligned model's outputs against baselines.

## 📂 Project Topology
```text
├── configs/
│   ├── deepspeed_zero3.json    # Distributed training optimization
│   └── dpo_sea_lion.yaml       # Hyperparameters for DPO phase
├── src/
│   ├── data_prep/              # Custom tokenizers & SEA normalizers
│   ├── pipeline/               # Core training scripts (SFT, RM, DPO)
│   └── evaluation/             # LLM-as-a-judge and benchmarking metrics
├── docs/
│   └── MATHEMATICAL_BACKGROUND.md # Deep-dive into DPO/PPO formulations
├── Makefile                    # Standardized MLOps commands
├── Dockerfile                  # GPU-ready containerization
└── requirements.txt            # Pinned dependencies
```

## 🚀 MLOps Workflow

We utilize a `Makefile` to standardize operations across research and production environments.

```bash
# 1. Setup Environment
make install

# 2. Run Data Preprocessing (SEA Normalization)
make prepare-data

# 3. Launch Distributed DPO Training (Requires Multi-GPU)
make train-dpo

# 4. Evaluate Aligned Model
make evaluate
```

## 📊 Evaluation & Metrics
All training runs are seamlessly tracked using **Weights & Biases (W&B)**. We monitor:
- Implicit Reward Accuracy
- Policy/Reference KL Divergence (Ensuring the model doesn't collapse)
- Generative Perplexity

---
**Architected by [Leong Wai Yi](https://github.com/LeongWaiYiw)**  
*AI Engineer @ AI Singapore | Specializing in LLM Post-Training & SEA NLP*
