import argparse
import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
TARGET_TOKENS = 70
CLIP_MAX_TOKENS = 77

_clip_model = None
_clip_tokenizer = None
_device = "cpu"


def init_clip():
    global _clip_model, _clip_tokenizer, _device
    if _clip_model is None:
        try:
            import clip
            import torch
            from clip.simple_tokenizer import SimpleTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Missing CLIP dependencies. Install with: pip install git+https://github.com/openai/CLIP.git torch"
            ) from exc
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _ = clip.load("ViT-B/32", device=_device)
        _clip_tokenizer = SimpleTokenizer()
        print(f"CLIP loaded on {_device}")
    return _clip_model


def count_tokens(text: str) -> int:
    if not text:
        return 0
    init_clip()
    try:
        return len(_clip_tokenizer.encode(text)) + 2
    except Exception:
        import clip

        tokens = clip.tokenize([text], truncate=True)
        return (tokens[0] != 0).sum().item()


def truncate_to_clip_limit(text: str, max_tokens: int = CLIP_MAX_TOKENS) -> str:
    if not text:
        return ""
    init_clip()
    available = max(1, max_tokens - 2)
    token_ids = _clip_tokenizer.encode(text)
    if len(token_ids) <= available:
        return text
    return _clip_tokenizer.decode(token_ids[:available]).strip()


def shorten_with_llm(
    text: str,
    target_tokens: int = TARGET_TOKENS,
    max_iterations: int = 5,
    ollama_url: str = OLLAMA_URL,
    model: str = OLLAMA_MODEL,
) -> tuple[str, int, int]:
    if not text:
        return "", 0, 0

    current_tokens = count_tokens(text)
    if current_tokens <= target_tokens:
        return text, current_tokens, 0

    current_text = text
    best_text = text
    best_tokens = current_tokens

    for iteration in range(max_iterations):
        prompt = f"""Make this shorter to fit under {target_tokens} tokens (currently {current_tokens}).
Keep core meaning. Remove redundant words.

{current_text}

Return ONLY the shortened text."""

        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 200},
                },
                timeout=60,
            )
            if response.status_code != 200:
                break

            shortened = response.json().get("response", "").strip()

            for quote in ['"', "'"]:
                if shortened.startswith(quote) and shortened.endswith(quote):
                    shortened = shortened[1:-1]

            for prefix in ["Here is", "Here's", "Shortened"]:
                if shortened.lower().startswith(prefix.lower()):
                    parts = shortened.split(":", 1)
                    if len(parts) > 1:
                        shortened = parts[1].strip()
                    break

            for quote in ['"', "'"]:
                if shortened.startswith(quote) and shortened.endswith(quote):
                    shortened = shortened[1:-1]

            if not shortened:
                continue

            new_tokens = count_tokens(shortened)

            if new_tokens <= target_tokens:
                return shortened, new_tokens, iteration + 1

            if new_tokens < best_tokens:
                best_text = shortened
                best_tokens = new_tokens
                current_text = shortened
                current_tokens = new_tokens
            else:
                current_text = best_text
                current_tokens = best_tokens

        except Exception:
            break

    final_text = best_text
    final_tokens = best_tokens
    if final_tokens > target_tokens:
        final_text = truncate_to_clip_limit(final_text, max_tokens=target_tokens)
        final_tokens = count_tokens(final_text)
    return final_text, final_tokens, max_iterations


def _resolve_output_path(input_file: Path, output: Path | None, in_place: bool) -> Path:
    if in_place:
        return input_file

    if output is None:
        return input_file.with_name(f"{input_file.stem}_preview{input_file.suffix}")

    if output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    output.mkdir(parents=True, exist_ok=True)
    return output / input_file.name


def process_file(
    input_file: Path,
    output_path: Path,
    target_tokens: int,
    clip_max_tokens: int,
    ollama_url: str,
    model: str,
) -> dict:
    with open(input_file, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of blocks in {input_file}")

    stats = {
        "file": str(input_file),
        "total": len(data),
        "shortened": 0,
        "llm_calls": 0,
        "updated": 0,
    }

    for block in data:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "") or ""

        if count_tokens(text) > target_tokens:
            text_preview, _, iters = shorten_with_llm(
                text=text,
                target_tokens=target_tokens,
                ollama_url=ollama_url,
                model=model,
            )
            stats["shortened"] += 1
            stats["llm_calls"] += iters
        else:
            text_preview = text

        if count_tokens(text_preview) > clip_max_tokens:
            text_preview = truncate_to_clip_limit(text_preview, max_tokens=clip_max_tokens)

        if block.get("text_preview") != text_preview:
            stats["updated"] += 1
        block["text_preview"] = text_preview

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    return stats


def iter_input_files(input_path: Path, pattern: str):
    if input_path.is_file():
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    for file_path in sorted(input_path.glob(pattern)):
        if file_path.is_file():
            yield file_path


def main():
    parser = argparse.ArgumentParser(
        description="Add text_preview to cleaned JSON blocks using unified_embedder logic."
    )
    parser.add_argument("--input", required=True, help="Input JSON file or directory")
    parser.add_argument(
        "--pattern",
        default="*_cleaned.json",
        help="Glob pattern used when --input is a directory",
    )
    parser.add_argument(
        "--output",
        help="Output file or directory. If omitted and not --in-place, writes *_preview.json beside input",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file(s) instead of writing new file(s)",
    )
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--clip-max-tokens", type=int, default=CLIP_MAX_TOKENS)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    files = list(iter_input_files(input_path, args.pattern))
    if not files:
        print("No files matched.")
        return

    print(f"Processing {len(files)} file(s)")
    totals = {"total": 0, "shortened": 0, "llm_calls": 0, "updated": 0}

    for input_file in files:
        out_file = _resolve_output_path(input_file, output_path, args.in_place)
        stats = process_file(
            input_file=input_file,
            output_path=out_file,
            target_tokens=args.target_tokens,
            clip_max_tokens=args.clip_max_tokens,
            ollama_url=args.ollama_url,
            model=args.model,
        )
        print(
            f"- {input_file.name}: blocks={stats['total']}, "
            f"updated={stats['updated']}, shortened={stats['shortened']}, "
            f"llm_calls={stats['llm_calls']} -> {out_file}"
        )
        totals["total"] += stats["total"]
        totals["shortened"] += stats["shortened"]
        totals["llm_calls"] += stats["llm_calls"]
        totals["updated"] += stats["updated"]

    print(
        "Done: "
        f"blocks={totals['total']}, updated={totals['updated']}, "
        f"shortened={totals['shortened']}, llm_calls={totals['llm_calls']}"
    )


if __name__ == "__main__":
    main()
