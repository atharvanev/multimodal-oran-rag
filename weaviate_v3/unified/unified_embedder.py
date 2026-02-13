# %% [markdown]
# # Simplified Unified Embedder - Image + Text Only
# 
# **Simplified approach:** Since CLIP embeds the image itself, we don't need a separate description field.
# 
# **What gets embedded:**
# - 🖼️ Image (0.6 weight)
# - 📝 Text (0.4 weight)
# 
# **Prerequisites:**
# ```bash
# ollama serve
# ollama pull llama3.2
# ```

# %%
import weaviate
import json
import clip
from clip.simple_tokenizer import SimpleTokenizer
import torch
import requests
from pathlib import Path

# %% [markdown]
# ## Configuration

# %%
OLLAMA_URL = "http://localhost:11434"
WEAVIATE_HOST = "172.17.0.2"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
TARGET_TOKENS = 70

# %% [markdown]
# ## Check Ollama

# %%
try:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    if response.status_code == 200:
        models = [m.get("name", "") for m in response.json().get("models", [])]
        if any("llama3.2" in name for name in models):
            print("✓ Ollama + llama3.2 ready")
        else:
            print(f"✗ llama3.2 not found. Run: ollama pull llama3.2")
    else:
        print(f"✗ Ollama error: {response.status_code}")
except Exception as e:
    print(f"✗ Cannot connect: {e}. Run: ollama serve")

# %% [markdown]
# ## CLIP Token Counting

# %%
_clip_model = None
_clip_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def init_clip():
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        _clip_model, _ = clip.load("ViT-B/32", device=_device)
        _clip_tokenizer = SimpleTokenizer()
        print(f"✓ CLIP loaded on {_device}")
    return _clip_model

def count_tokens(text):
    if not text:
        return 0
    init_clip()
    try:
        # Exact CLIP token count: start + encoded text + end
        return len(_clip_tokenizer.encode(text)) + 2
    except Exception:
        # Conservative fallback
        tokens = clip.tokenize([text], truncate=True)
        return (tokens[0] != 0).sum().item()

def truncate_to_clip_limit(text, max_tokens=77):
    if not text:
        return ""
    init_clip()
    available = max(1, max_tokens - 2)
    token_ids = _clip_tokenizer.encode(text)
    if len(token_ids) <= available:
        return text
    return _clip_tokenizer.decode(token_ids[:available]).strip()

print(f"Test: {count_tokens('Hello world')} tokens")

# %% [markdown]
# ## Text Shortening

# %%
def shorten_with_llm(text, target_tokens=70, max_iterations=5, verbose=True):
    if not text:
        return "", 0, 0
    
    current_tokens = count_tokens(text)
    if current_tokens <= target_tokens:
        if verbose:
            print(f"✓ Under limit: {current_tokens} tokens")
        return text, current_tokens, 0
    
    current_text = text
    best_text = text
    best_tokens = current_tokens
    if verbose:
        print(f"Starting: {current_tokens} tokens")
    
    for iteration in range(max_iterations):
        prompt = f"""Make this shorter to fit under {target_tokens} tokens (currently {current_tokens}).
Keep core meaning. Remove redundant words.

{current_text}

Return ONLY the shortened text."""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 200}
                },
                timeout=60
            )
            
            if response.status_code != 200:
                if verbose:
                    print(f"✗ Error: {response.status_code}")
                break
            
            result = response.json()
            shortened = result.get("response", "").strip()
            
            # Clean formatting
            for quote in ['"', "'"]:
                if shortened.startswith(quote) and shortened.endswith(quote):
                    shortened = shortened[1:-1]
            
            # Remove prefixes
            for prefix in ["Here is", "Here's", "Shortened"]:
                if shortened.lower().startswith(prefix.lower()):
                    parts = shortened.split(":", 1)
                    if len(parts) > 1:
                        shortened = parts[1].strip()
                    break
            
            # Clean again
            for quote in ['"', "'"]:
                if shortened.startswith(quote) and shortened.endswith(quote):
                    shortened = shortened[1:-1]
            
            if not shortened:
                continue

            new_tokens = count_tokens(shortened)
            
            if verbose:
                print(f"  Iter {iteration + 1}: {new_tokens} tokens")
            
            if new_tokens <= target_tokens:
                if verbose:
                    print(f"✓ Success in {iteration + 1} iterations")
                return shortened, new_tokens, iteration + 1
            
            if new_tokens < best_tokens:
                best_text = shortened
                best_tokens = new_tokens
                current_text = shortened
                current_tokens = new_tokens
            else:
                if verbose:
                    print("⚠ No progress")
                # Retry with the best-so-far text instead of exiting early
                current_text = best_text
                current_tokens = best_tokens
                
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            break
    
    # Final hard clamp to guarantee model-safe input
    final_text = best_text
    final_tokens = best_tokens
    if final_tokens > target_tokens:
        final_text = truncate_to_clip_limit(final_text, max_tokens=target_tokens)
        final_tokens = count_tokens(final_text)

    if verbose:
        print(f"⚠ Max iterations. Final: {final_tokens} tokens")
    return final_text, final_tokens, max_iterations

# %% [markdown]
# ## Create Simplified Schema
# 
# Only embeds: **image** (0.6) + **text_preview** (0.4)
# 
# No description field needed!

# %%
def create_schema():
    base_url = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"
    
    try:
        requests.delete(f"{base_url}/v1/schema/unified_embedding")
        print("Deleted existing collection")
    except:
        pass
    
    schema = {
        "class": "unified_embedding",
        "vectorizer": "multi2vec-clip",
        "moduleConfig": {
            "multi2vec-clip": {
                "imageFields": ["images"],
                "textFields": ["text_preview"],  # Only text, no description
                "weights": {
                    "imageFields": [0.6],  # Image more important
                    "textFields": [0.4]    # Text for context
                }
            }
        },
        "properties": [
            {"name": "id", "dataType": ["text"],
             "moduleConfig": {"multi2vec-clip": {"skip": True}}},
            {"name": "block_type", "dataType": ["text"],
             "moduleConfig": {"multi2vec-clip": {"skip": True}}},
            {"name": "page", "dataType": ["int"],
             "moduleConfig": {"multi2vec-clip": {"skip": True}}},
            {"name": "text_preview", "dataType": ["text"],
             "description": "Short text for embedding (≤70 tokens)",
             "moduleConfig": {"multi2vec-clip": {"skip": False}}},
            {"name": "text", "dataType": ["text"],
             "description": "Full text (not vectorized)",
             "moduleConfig": {"multi2vec-clip": {"skip": True}}},
            {"name": "trace", "dataType": ["text"],
             "moduleConfig": {"multi2vec-clip": {"skip": True}}},
            {"name": "filename", "dataType": ["text"],
             "moduleConfig": {"multi2vec-clip": {"skip": True}}},
            {"name": "images", "dataType": ["blob"],
             "moduleConfig": {"multi2vec-clip": {"skip": False}}}
        ]
    }
    
    response = requests.post(
        f"{base_url}/v1/schema",
        json=schema,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print("✓ Simplified schema created")
        print("  Embedding: image (0.6) + text_preview (0.4)")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False

create_schema()

# %% [markdown]
# ## Import Data (Simplified)

# %%
def import_data(filepath, target_tokens=70):
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )
    
    collection = client.collections.get("unified_embedding")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    default_filename = Path(filepath).stem.replace("_cleaned", "")
    print(f"\nImporting {default_filename}: {len(data)} entries\n")
    
    stats = {'total': len(data), 'success': 0, 'shortened': 0, 'iterations': 0}
    
    with collection.batch.dynamic() as batch:
        for i, entry in enumerate(data):
            try:
                text = entry.get('text', '')
                
                # Only process text (no description field needed)
                if count_tokens(text) > target_tokens:
                    print(f"[{i+1}] Shortening...")
                    text_preview, t, iters = shorten_with_llm(text, target_tokens, verbose=False)
                    stats['shortened'] += 1
                    stats['iterations'] += iters
                    print(f"  → {t} tokens ({iters} iter)")
                else:
                    text_preview = text
                
                # Hard guard: never send text_preview above CLIP max context length
                if count_tokens(text_preview) > 77:
                    text_preview = truncate_to_clip_limit(text_preview, max_tokens=77)

                # Simplified: no description fields
                batch.add_object({
                    "id": entry.get("id", ""),
                    "block_type": entry.get("block_type") or entry.get("type") or "Unknown",
                    "page": entry.get('page', 0),
                    "text_preview": text_preview,  # Short version for embedding
                    "text": text,                  # Full version for display
                    "trace": entry.get('trace', ''),
                    "filename": entry.get("filename", default_filename),
                    "images": entry.get("images") or entry.get("image")
                })
                stats['success'] += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{len(data)}")
                
            except Exception as e:
                print(f"✗ Error on entry {i+1}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Complete: {stats['success']}/{stats['total']}")
    print(f"Shortened: {stats['shortened']} texts")
    print(f"Ollama calls: {stats['iterations']}")
    print(f"{'='*80}\n")
    
    client.close()
    return stats

# %% [markdown]
# ## Run Import

# %%
#filepath = "../clean_chunks/O-RAN-WG1-CCIN-TR-R004-v01.00_cleaned.json"
#stats = import_data(filepath, target_tokens=70)

# %% [markdown]
# ## Batch Import

# %%
data_dir = Path("../../clean_chunks")
json_files = list(data_dir.glob("*_cleaned.json"))

print(f"Found {len(json_files)} files\n")

all_stats = []
for json_file in json_files:
    print(f"\n{'#'*80}")
    print(f"Processing: {json_file.name}")
    print(f"{'#'*80}\n")
    
    stats = import_data(str(json_file), 70)
    all_stats.append(stats)

print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}")
print(f"Total: {sum(s['total'] for s in all_stats)}")
print(f"Success: {sum(s['success'] for s in all_stats)}")
print(f"Shortened: {sum(s['shortened'] for s in all_stats)}")
print(f"Ollama calls: {sum(s['iterations'] for s in all_stats)}")

# %% [markdown]
# ## Review Shortened Text
# 
# Inspect a sample of rows where `text_preview` differs from `text` to verify shortened outputs are sensible.

# %%
SAMPLE_POOL = 500
MAX_SHOW = 12

client = weaviate.connect_to_local(
    host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT
)

try:
    collection = client.collections.get("unified_embedding")
    response = collection.query.fetch_objects(
        limit=SAMPLE_POOL,
        return_properties=["filename", "page", "block_type", "text", "text_preview"]
    )

    changed = []
    for obj in response.objects:
        props = obj.properties or {}
        original = (props.get("text") or "").strip()
        preview = (props.get("text_preview") or "").strip()
        if original and preview and preview != original:
            changed.append(props)

    print(f"Scanned: {len(response.objects)} objects")
    print(f"Shortened candidates: {len(changed)}\n")

    if not changed:
        print("No shortened examples found in this sample. Increase SAMPLE_POOL or run import first.")
    else:
        for i, props in enumerate(changed[:MAX_SHOW], 1):
            original = props.get("text", "").strip()
            preview = props.get("text_preview", "").strip()
            original_tokens = count_tokens(original)
            preview_tokens = count_tokens(preview)

            print("=" * 100)
            print(f"Example {i}")
            print(f"File/Page: {props.get('filename', '?')} / {props.get('page', '?')}")
            print(f"Type: {props.get('block_type', '?')}")
            print(f"Tokens: {original_tokens} -> {preview_tokens}")
            print("\nOriginal:")
            print(original[:500] + ("..." if len(original) > 500 else ""))
            print("\nShortened (text_preview):")
            print(preview[:500] + ("..." if len(preview) > 500 else ""))
            print()

finally:
    client.close()

# %% [markdown]
# ## Test Query

# %%
client = weaviate.connect_to_local(
    host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT
)

collection = client.collections.get("unified_embedding")

response = collection.query.near_text(
    query="5G network architecture",
    limit=5,
    return_properties=["block_type", "page", "filename", "text", "text_preview"]
)

print(f"Found {len(response.objects)} results\n")

for i, obj in enumerate(response.objects, 1):
    print(f"{'='*80}")
    print(f"Result {i}")
    print(f"Type: {obj.properties['block_type']}")
    print(f"Page: {obj.properties['page']}")
    print(f"File: {obj.properties['filename']}")
    print(f"\nSearched with: {obj.properties['text_preview'][:100]}...")
    print(f"Tokens: {count_tokens(obj.properties['text_preview'])}")
    print(f"\nFull text: {obj.properties['text'][:200]}...\n")

client.close()
