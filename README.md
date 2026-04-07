# Multimodal-Oran-RAG

Research codebase for multimodal Retrieval-Augmented Generation (RAG) on O-RAN technical documents.

## Current Status

This project is active and has moved beyond prototyping into measurable retrieval evaluation.

Recent milestone completed:
- Full 3-phase sweep (model knockout, hyperparameter grid, final pipeline comparison)
- Reproducible outputs for fixed-alpha experiments in `results/full_sweep_fixed_alpha`
- Figure generation pipeline and report artifacts for analysis

## What This Project Does

`multimodal-oran-rag` focuses on extracting and retrieving both text and visual information from O-RAN specifications so downstream LLM workflows can answer technical questions with better context.

Core capabilities:
- Parse and segment O-RAN documents
- Retrieve with grounded and unified multimodal pipelines
- Evaluate retrieval settings across model, `k`, and alpha (`α`) configurations
- Generate sweep reports and plots for comparison

## Latest Sweep Progress

From `results/full_sweep_fixed_alpha` on `dataset/fin_E.json`:

- **Phase 1 (model knockout, grounded, k=3, α=0.75):**
  - `qwen2.5:3b` ranked #1 (0.7278), `gemma3:4b` ranked #2 (0.6822)
- **Phase 2 (k x α sweep):**
  - Grounded best: `qwen2.5:3b`, `k=7`, `α=0.25`, accuracy `0.8270`
  - Unified best: `gemma3:4b`, `k=5`, `α=1.00`, accuracy `0.6760`
- **Phase 3 (final comparison):**
  - Winner: **grounded** pipeline with `qwen2.5:3b`, `k=7`, `α=0.25`, accuracy `0.8191`

Artifacts:
- Report: `results/full_sweep_fixed_alpha/sweep_report.md`
- Summary JSON: `results/full_sweep_fixed_alpha/sweep_summary.json`
- Figures: `results/full_sweep_fixed_alpha/figures`

## Roadmap

| Feature / Component | Status | Notes |
|---|---|---|
| PDF/Text Preprocessing | ✅ Completed | Parsing and segmentation are in use. |
| Diagram/Image Extraction | ✅ Completed | Extraction and indexing are in place. |
| Multimodal Embeddings | ✅ Completed | Grounded and unified strategies are implemented and benchmarked. |
| Retrieval Pipelines | ✅ Completed | Full sweep completed with comparable pipeline metrics. |
| Evaluation Framework | ✅ Completed | Multi-phase accuracy benchmarking and plotting are implemented. |
| LLM Integration (Q&A/Summarization) | ✅ Completed | Contextual Q&A and summarization workflows are integrated. |

## Installation

```bash
git clone <your-repo-url>
cd multimodal-oran-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Sweep Analysis

If you have sweep results in a directory containing `sweep_summary.json`, generate plots with:

```bash
python scripts/plot_full_sweep.py --sweep-dir results/full_sweep_fixed_alpha
```

This writes figures to `<sweep-dir>/figures`.
