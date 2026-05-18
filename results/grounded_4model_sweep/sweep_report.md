# Sweep Report

**Generated:** 2026-04-27 09:00:07  
**Dataset:** `dataset/fin_E.json`  
**Phases run:** [0, 1, 2, 3]  

## Phase 0 — Control Baseline  `(no RAG, LLM-only)`

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | `qwen2.5vl:32b` | 0.8191 |
| 2 | `gemma2:27b` | 0.7981 |
| 3 | `gemma2:latest` | 0.7735 |
| 4 | `mistral-nemo:latest` | 0.7419 |

## Phase 1 — Model Knockout  `(k=3, α=0.75, grounded)`

| Rank | Model | Accuracy | Carried Forward |
|------|-------|----------|-----------------|
| 1 | `gemma2:27b` | 0.7682 | ✓ |
| 2 | `gemma2:latest` | 0.7296 | ✓ |

## Phase 2 — Hyperparam Sweep  `(k × α grid)`

### Grounded Pipeline

| Rank | Model | k | α | Accuracy | |
|------|-------|---|---|----------|---|
| 1 | `gemma2:27b` | 7 | 0.25 | 0.8586 | ★ best |
| 2 | `gemma2:27b` | 7 | 0.00 | 0.8507 |  |
| 3 | `gemma2:27b` | 5 | 0.25 | 0.8376 |  |
| 4 | `gemma2:27b` | 5 | 0.00 | 0.8306 |  |
| 5 | `gemma2:27b` | 3 | 0.25 | 0.8165 |  |
| 6 | `gemma2:27b` | 3 | 0.00 | 0.8147 |  |
| 7 | `gemma2:27b` | 7 | 0.75 | 0.7893 |  |
| 8 | `gemma2:27b` | 5 | 0.75 | 0.7805 |  |
| 9 | `gemma2:27b` | 3 | 0.75 | 0.7752 |  |
| 10 | `gemma2:27b` | 5 | 1.00 | 0.7726 |  |
| 11 | `gemma2:27b` | 7 | 1.00 | 0.7656 |  |
| 12 | `gemma2:27b` | 3 | 1.00 | 0.7612 |  |

## Phase 3 — Final Pipeline Comparison

| Rank | Pipeline | Model | k | α | Accuracy |
|------|----------|-------|---|---|----------|
| 1 | grounded | `gemma2:27b` | 7 | 0.25 | 0.8586 |

**Winner:** `grounded` — model `gemma2:27b`, k=7, α=0.25, accuracy **0.8586**
