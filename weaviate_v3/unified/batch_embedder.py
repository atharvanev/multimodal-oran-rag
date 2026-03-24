"""
Batch Unified Embedder
======================
Ingests all *_cleaned.json files from a directory into Weaviate using
multi2vec-clip with image (0.6) + text_preview (0.4) embeddings.

Usage:
    python batch_embedder.py                        # uses default data_dir
    python batch_embedder.py ../../clean_chunks     # custom data_dir
"""

import json
import os
import sys
from pathlib import Path

import requests
import weaviate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEAVIATE_HOST = "172.17.0.2"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "unified_embedding"
DEFAULT_DATA_DIR = "clean_chunks"
MULTI2VEC_CLIP_INFERENCE_URL = os.getenv(
    "MULTI2VEC_CLIP_INFERENCE_URL",
    "http://172.17.0.7:8080",
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-file import
# ---------------------------------------------------------------------------

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
    }

    with collection.batch.dynamic() as batch:
        for i, entry in enumerate(data, 1):
            text = entry.get("text", "")
            text_preview = entry.get("text_preview") or text

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

    failed_objects = getattr(collection.batch, "failed_objects", []) or []
    stats["failed"] = len(failed_objects)
    stats["success"] = stats["submitted"] - stats["failed"]

    print("=" * 80)
    print(f"Submitted: {stats['submitted']}/{stats['total']}")
    print(f"Success:   {stats['success']}")
    print(f"Failed:    {stats['failed']}")
    print("=" * 80)

    client.close()
    return stats


# ---------------------------------------------------------------------------
# Batch import
# ---------------------------------------------------------------------------

def import_all_clean_chunks(data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*_cleaned.json"))

    if not json_files:
        print(f"No *_cleaned.json files found in {data_dir}")
        return None

    print(f"Found {len(json_files)} files in {data_dir}")

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
    }

    print("=" * 80)
    print("OVERALL SUMMARY")
    for k, v in summary.items():
        print(f"  {k:<12}: {v}")
    print("=" * 80)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR

    print(f"Data directory : {data_dir}")
    print(f"Weaviate       : {WEAVIATE_HOST}:{WEAVIATE_PORT}")
    print(f"Collection     : {COLLECTION_NAME}")
    print(f"CLIP endpoint  : {MULTI2VEC_CLIP_INFERENCE_URL}")
    print()

    if create_schema(delete_existing=True):
        import_all_clean_chunks(data_dir)
