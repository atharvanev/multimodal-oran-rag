# Evaluation Reference

## Purpose

`evaluate_grounded.py` evaluates multiple-choice O-RAN questions from datasets such as `dataset/fin_E.json`.

It supports three pipeline modes:

- `none`: LLM-only control, no retrieval
- `grounded`: grounded Weaviate retrieval pipeline
- `unified`: unified multimodal retrieval pipeline

## Dataset Format

The evaluator expects line-delimited JSON where each line looks like:

```json
["Question text", ["1. Option A", "2. Option B", "3. Option C", "4. Option D"], "3"]
```

## Main Arguments

- `--dataset`: path to the evaluation dataset
- `--pipeline`: `none`, `grounded`, or `unified`
- `--model`: Ollama generation model to use
- `--top-k`: number of retrieved chunks for RAG pipelines
- `--limit`: optionally evaluate only the first N questions
- `--output`: optional JSON output path for detailed results
- `--ollama-host`: Ollama host for generation
- `--ollama-port`: Ollama port for generation

## Grounded Pipeline Arguments

- `--grounded-collection`
- `--grounded-weaviate-host`
- `--grounded-weaviate-port`
- `--grounded-weaviate-grpc-port`

## Unified Pipeline Arguments

- `--unified-collection`
- `--unified-weaviate-host`
- `--unified-weaviate-port`
- `--unified-weaviate-grpc-port`
- `--multi2vec-host`
- `--multi2vec-port`
- `--unified-query-alpha`
- `--unified-modality-balance`
- `--unified-summarize-threshold`

## Example Commands

LLM-only control:

```bash
python3 eval/evaluate_grounded.py   --pipeline none   --model llama3.2   --dataset dataset/fin_E.json
```

Grounded pipeline:

```bash
python3 eval/evaluate_grounded.py   --pipeline grounded   --model llama3.2   --dataset dataset/fin_E.json   --grounded-collection Grounded_nomic_full   --grounded-weaviate-host 172.17.0.5
```

Unified pipeline:

```bash
python3 eval/eval.py   --pipeline unified   --model llama3.2   --dataset dataset/fin_E.json   --unified-collection Unified_embedding   --unified-weaviate-host 172.17.0.5   --multi2vec-host 172.17.0.7 --output unified.json
```

Save detailed output:

```bash
python3 eval/evaluate_grounded.py   --pipeline grounded   --model llama3.2   --dataset dataset/fin_E.json   --output eval/fin_E_grounded_results.json
```

## Output Summary

The script prints a compact JSON summary with:

- `dataset`
- `pipeline`
- `model`
- `top_k`
- `total`
- `correct`
- `accuracy`

If `--output` is provided, it also writes per-question details including predictions, raw answers, and retrieved sources.
