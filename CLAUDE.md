# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research codebase for multimodal Retrieval-Augmented Generation on O-RAN technical specifications. It evaluates two competing RAG pipelines against the [ORAN-Bench-13K](dataset/README.md) multiple-choice question dataset (`dataset/fin_E.json` easy, `fin_M.json` medium, `fin_H.json` hard).

## Infrastructure Requirements

All pipelines depend on Docker-hosted services. Default bridge IPs (hardcoded in `eval/scripts/sweep.py`):

| Service | Host | Port |
|---|---|---|
| Ollama (LLM inference + embeddings) | 172.17.0.4 | 11434 |
| Weaviate (vector DB) | 172.17.0.2 | 8080, gRPC 50051 |
| multi2vec-clip (multimodal embeddings) | 172.17.0.5 | 8080 |
| Phoenix/OTEL tracing (optional) | localhost | 6006 |

## Common Commands

**Run the full 3-phase sweep:**
```bash
python sweep.py --dataset dataset/fin_E.json --output-dir results/my_sweep
```

**Run a single eval (direct, bypasses sweep orchestration):**
```bash
# LLM-only baseline
python eval/scripts/eval.py --pipeline none --model llama3.2 --dataset dataset/fin_E.json

# Grounded pipeline
python eval/scripts/eval.py --pipeline grounded --model qwen2.5:3b --dataset dataset/fin_E.json \
  --grounded-collection Grounded_nomic_full --top-k 7 --grounded-weaviate-host 172.17.0.2

# Unified pipeline
python eval/scripts/eval.py --pipeline unified --model gemma3:4b --dataset dataset/fin_E.json \
  --unified-collection unified_embedding --unified-weaviate-host 172.17.0.2 \
  --multi2vec-host 172.17.0.5 --unified-query-alpha 1.0 --top-k 5
```

**Generate plots from a completed sweep:**
```bash
python scripts/plot_full_sweep.py --sweep-dir results/full_sweep_fixed_alpha
```

**Launch interactive Streamlit UIs:**
```bash
streamlit run weaviate_v3/grounded/grounded_ui.py
streamlit run weaviate_v3/unified/unified_ui.py
```

**Index chunks into Weaviate (unified pipeline):**
```bash
python weaviate_v3/unified/batch_embedder.py <data_dir>
```

## Architecture

### Two RAG Pipelines

**Grounded** ([weaviate_v3/grounded/](weaviate_v3/grounded/))
- Embedder: `text2vec-ollama` with `nomic-embed-text`
- Weaviate collection: `Grounded_nomic_full`
- Query strategy: hybrid BM25 + vector, blended by `α`
- Generation: `ollama.generate()` with text context
- Best known config: `qwen2.5:3b`, k=7, α=0.25 → **81.91% accuracy** on `fin_E`

**Unified** ([weaviate_v3/unified/](weaviate_v3/unified/))
- Embedder: `multi2vec-clip` with image weight 0.6, text_preview weight 0.4
- Weaviate collection: `unified_embedding`
- Query strategy: auto-summarizes queries >70 tokens before embedding, then reranks by modality
- Generation: `ollama.chat()` for multimodal content
- Best known config: `gemma3:4b`, k=5, α=1.00 → 67.60% accuracy on `fin_E`

### `α` (alpha) Parameter
Controls BM25 vs. vector blend in Weaviate hybrid search:
- `α=0.0` — pure vector similarity
- `α=0.25–0.75` — hybrid
- `α=1.0` — pure BM25 keyword

### 3-Phase Sweep Orchestration (`eval/scripts/sweep.py`)

1. **Phase 1 — Model Knockout**: Tests 4 models at fixed k=3, α=0.75 (grounded only). Top-N models advance.
2. **Phase 2 — Hyperparameter Grid**: Each pipeline sweeps k ∈ [3,5,7] × α ∈ [1.0, 0.75, 0.25, 0.0] independently with top models from Phase 1.
3. **Phase 3 — Fair Comparison**: Each pipeline enters with its own Phase 2 best config for a head-to-head comparison.

Sweep output: `sweep_summary.json` (structured results per phase) and `sweep_report.md` (human-readable). Raw per-run JSON files are gitignored.

### Data Flow

```
PDFs → Marker OCR → chunks/ (raw JSON)
                  → json_preprocessor/ + ChunkCaptioner.py → clean_chunks/ (captions added)
                  → weaviate_v3/unified/batch_embedder.py → Weaviate (unified_embedding)
                  → weaviate_v3/grounded/ indexing → Weaviate (Grounded_nomic_full)
```

Evaluation datasets live in `dataset/` as line-delimited JSON: `["question", ["1. A", "2. B", ...], "correct_answer"]`.

### Key Classes

| Class | File | Role |
|---|---|---|
| `ChatRAG` | [weaviate_v3/grounded/grounded_rag.py](weaviate_v3/grounded/grounded_rag.py) | Grounded retrieval + generation |
| `UnifiedChatRAG` | [weaviate_v3/unified/unified_rag.py](weaviate_v3/unified/unified_rag.py) | Unified multimodal retrieval + generation |
| `NoRAGRunner` / `GroundedRunner` / `UnifiedRunner` | [eval/scripts/eval.py](eval/scripts/eval.py) | Eval harness runners |
| `RunConfig` / `PipelineBest` / `SweepData` | [eval/scripts/sweep.py](eval/scripts/sweep.py) | Sweep state dataclasses |
| `ImageCaptioner` | [ChunkCaptioner.py](ChunkCaptioner.py) | Caption generation via SAIL-VL |
