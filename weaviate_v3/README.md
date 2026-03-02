# Weaviate v3: Grounded + Unified

This document lists the two Weaviate collections currently used by the v3 chat interfaces, and shows how each path interacts with Ollama using code-backed stubs.

## Active Collections

1. `Grounded_nomic_full`
- Source of default value in UI: `weaviate_v3/grounded/grounded_ui.py` (line 36).
- Passed into `ChatRAG(collection_name=...)`: `weaviate_v3/grounded/grounded_ui.py` (lines 55-60).

2. `unified_embedding`
- Source of default value in Unified UI: `weaviate_v3/unified/unified_ui.py` (line 62).
- Source of schema class used by embedder: `weaviate_v3/unified/batch_embedder.py` (line 27, line 50).

## Grounded Flow and Ollama Interaction

### Purpose
- Grounded uses Weaviate `text2vec-ollama` for embedding/search + direct Ollama generation for answers.

### Proof Stubs from Code

```python
# weaviate_v3/grounded/grounded_rag.py
class ChatRAG:
    def __init__(self, collection_name: str, weaviate_host: str, ollama_model: str, ...):
        schema = {
            "class": self.collection_name,
            "vectorizer": "text2vec-ollama",
            "moduleConfig": {
                "text2vec-ollama": {"apiEndpoint": "...:11434", "model": "nomic-embed-text"},
                "generative-ollama": {"apiEndpoint": "...:11434", "model": self.ollama_model},
            },
        }

    def chat(self, user_message: str, use_rag: bool = True, top_k: int = 3, ...):
        response = ollama.chat(...)      # multimodal branch
        response = ollama.generate(...)  # text branch
```

### Key Lines
- `import ollama`: `grounded_rag.py:4`
- Weaviate vectorizer config (`text2vec-ollama`, `generative-ollama`): `grounded_rag.py:60-69`
- Chat-time calls to Ollama:
- `ollama.chat(...)`: `grounded_rag.py:297-304`
- `ollama.generate(...)`: `grounded_rag.py:309-312`

## Unified Flow and Ollama Interaction

### Purpose
- Unified stores multimodal vectors (`multi2vec-clip`) and uses Ollama at chat-time for:
- automatic query summarization above token threshold
- final answer generation

### Proof Stubs from Code

```python
# weaviate_v3/unified/batch_embedder.py
COLLECTION_NAME = "unified_embedding"
schema = {
    "class": COLLECTION_NAME,
    "vectorizer": "multi2vec-clip",
    "moduleConfig": {"multi2vec-clip": {"inferenceUrl": "...", "imageFields": [...], "textFields": [...]}},
}

# weaviate_v3/unified/unified_rag.py
class UnifiedChatRAG:
    def _summarize_if_needed(self, user_message: str):
        response = ollama.generate(model=self.ollama_model, prompt=summarize_prompt)

    def chat(self, user_message: str, ...):
        response = ollama.chat(...)      # multimodal branch
        response = ollama.generate(...)  # text branch
```

### Key Lines
- `COLLECTION_NAME = "unified_embedding"`: `batch_embedder.py:27`
- Unified schema class/vectorizer: `batch_embedder.py:50-55`
- Unified chat class import + constructor: `unified_rag.py:3`, `unified_rag.py:8-18`
- Auto-summarization via Ollama: `unified_rag.py:43-63`
- Chat-time Ollama calls:
- `ollama.chat(...)`: `unified_rag.py:197-206`
- `ollama.generate(...)`: `unified_rag.py:209`

## Notes

- `Grounded_nomic_full` is the current default collection in Grounded UI; it is configurable from the sidebar.
- `unified_embedding` is used by Unified UI and Unified embedder scripts.
- There is also a non-primary script named `weaviate_v3/unified/unified_embedder("dont use").py` that still references `unified_embedding` and the same `multi2vec-clip` schema pattern.
