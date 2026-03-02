# %% [markdown]
# # Simplified Unified Embedder - Image + Text Preview
#
# This notebook-style script ingests cleaned JSON chunks into Weaviate using:
# - images (weight 0.6)
# - text_preview (weight 0.4)
#
# Notes:
# - No LLM summarization is used during ingestion.
# - text_preview is read directly from JSON (fallback to full text).
# - chunk_id is used instead of reserved property name id.

# %%
import json
import os
from pathlib import Path

import clip
import requests
import torch
import weaviate
from clip.simple_tokenizer import SimpleTokenizer

# %% [markdown]
# ## Configuration

# %%
WEAVIATE_HOST = "172.17.0.2"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "unified_embedding"
MULTI2VEC_CLIP_INFERENCE_URL = os.getenv(
    "MULTI2VEC_CLIP_INFERENCE_URL",
    "http://multi2vec-clip:8080",
)

# %% [markdown]
# ## CLIP Token Utilities

# %%
_clip_model = None
_clip_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def init_clip():
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        _clip_model, _ = clip.load("ViT-B/32", device=_device)
        _clip_tokenizer = SimpleTokenizer()
        print(f"CLIP loaded on {_device}")
    return _clip_model


def count_tokens(text):
    if not text:
        return 0
    init_clip()
    try:
        # Exact CLIP token count: start + encoded text + end.
        return len(_clip_tokenizer.encode(text)) + 2
    except Exception:
        # Conservative fallback.
        tokens = clip.tokenize([text], truncate=True)
        return int((tokens[0] != 0).sum().item())


def truncate_to_clip_limit(text, max_tokens=77):
    if not text:
        return ""
    init_clip()
    available = max(1, max_tokens - 2)
    token_ids = _clip_tokenizer.encode(text)
    if len(token_ids) <= available:
        return text
    return _clip_tokenizer.decode(token_ids[:available]).strip()


print(f"Token test (Hello world): {count_tokens('Hello world')}")

# %% [markdown]
# ## Schema: images + text_preview

# %%
def create_schema(delete_existing=True):
    base_url = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"

    if delete_existing:
        try:
            requests.delete(f"{base_url}/v1/schema/{COLLECTION_NAME}")
            print("Deleted existing collection")
        except Exception:
            pass

    schema = {
        "class": COLLECTION_NAME,
        "vectorizer": "multi2vec-clip",
        "moduleConfig": {
            "multi2vec-clip": {
                "inferenceUrl": MULTI2VEC_CLIP_INFERENCE_URL,
                "imageFields": ["images"],
                "textFields": ["text_preview"],
                "weights": {
                    "imageFields": [0.6],
                    "textFields": [0.4],
                },
            }
        },
        "properties": [
            {
                "name": "chunk_id",
                "dataType": ["text"],
                "moduleConfig": {"multi2vec-clip": {"skip": True}},
            },
            {
                "name": "block_type",
                "dataType": ["text"],
                "moduleConfig": {"multi2vec-clip": {"skip": True}},
            },
            {
                "name": "page",
                "dataType": ["int"],
                "moduleConfig": {"multi2vec-clip": {"skip": True}},
            },
            {
                "name": "text_preview",
                "dataType": ["text"],
                "description": "Short text for embedding",
                "moduleConfig": {"multi2vec-clip": {"skip": False}},
            },
            {
                "name": "text",
                "dataType": ["text"],
                "description": "Full text (not vectorized)",
                "moduleConfig": {"multi2vec-clip": {"skip": True}},
            },
            {
                "name": "trace",
                "dataType": ["text"],
                "moduleConfig": {"multi2vec-clip": {"skip": True}},
            },
            {
                "name": "filename",
                "dataType": ["text"],
                "moduleConfig": {"multi2vec-clip": {"skip": True}},
            },
            {
                "name": "images",
                "dataType": ["blob"],
                "moduleConfig": {"multi2vec-clip": {"skip": False}},
            },
        ],
    }

    response = requests.post(
        f"{base_url}/v1/schema",
        json=schema,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        print("Schema created")
        print("Embedding: images (0.6) + text_preview (0.4)")
        return True

    print(f"Schema creation failed: {response.status_code}")
    print(response.text)
    return False


# %% [markdown]
# ## Import One JSON File

# %%
def import_data(filepath):
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )

    with open(filepath, "r") as f:
        data = json.load(f)

    default_filename = Path(filepath).stem.replace("_cleaned", "")
    print(f"Importing {default_filename}: {len(data)} entries")

    collection = client.collections.get(COLLECTION_NAME)
    stats = {
        "total": len(data),
        "submitted": 0,
        "success": 0,
        "failed": 0,
        "clamped": 0,
    }

    with collection.batch.dynamic() as batch:
        for i, entry in enumerate(data, 1):
            text = entry.get("text", "")
            text_preview = entry.get("text_preview") or text

            # Hard guard: never send text_preview above CLIP max context length.
            if count_tokens(text_preview) > 77:
                text_preview = truncate_to_clip_limit(text_preview, max_tokens=77)
                stats["clamped"] += 1

            images = entry.get("images") or entry.get("image")
            if images == "":
                images = None

            batch.add_object(
                properties={
                    "chunk_id": entry.get("id", ""),
                    "block_type": entry.get("block_type") or entry.get("type") or "Unknown",
                    "page": entry.get("page", 0),
                    "text_preview": text_preview,
                    "text": text,
                    "trace": entry.get("trace", ""),
                    "filename": entry.get("filename", default_filename),
                    "images": images,
                }
            )
            stats["submitted"] += 1

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(data)}")

    # Weaviate v4 reports batch failures here.
    failed_objects = getattr(collection.batch, "failed_objects", []) or []
    stats["failed"] = len(failed_objects)
    stats["success"] = stats["submitted"] - stats["failed"]

    print("=" * 80)
    print(f"Submitted: {stats['submitted']}/{stats['total']}")
    print(f"Success:   {stats['success']}")
    print(f"Failed:    {stats['failed']}")
    print(f"Clamped:   {stats['clamped']}")
    print("=" * 80)

    client.close()
    return stats


# %% [markdown]
# ## Batch Import (clean_chunks)

# %%
def import_all_clean_chunks(data_dir="../../clean_chunks"):
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*_cleaned.json"))
    print(f"Found {len(json_files)} files")

    all_stats = []
    for json_file in json_files:
        print("#" * 80)
        print(f"Processing: {json_file.name}")
        print("#" * 80)
        all_stats.append(import_data(str(json_file)))

    summary = {
        "total": sum(s["total"] for s in all_stats),
        "submitted": sum(s["submitted"] for s in all_stats),
        "success": sum(s["success"] for s in all_stats),
        "failed": sum(s["failed"] for s in all_stats),
        "clamped": sum(s["clamped"] for s in all_stats),
    }

    print("=" * 80)
    print("OVERALL SUMMARY")
    print(summary)
    print("=" * 80)
    return summary


# %% [markdown]
# ## Run Batch Import

# %%
if __name__ == "__main__":
    if create_schema(delete_existing=True):
        import_all_clean_chunks("../../clean_chunks")
