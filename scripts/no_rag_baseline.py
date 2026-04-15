#!/usr/bin/env python3
"""
No-RAG baseline sweep: evaluate all foundational LLMs with pipeline=none.
Mirrors Phase 1 of the strategic sweep but removes retrieval entirely so
results reflect pure model knowledge on the ORAN-Bench-13K MCQ dataset.

Usage:
    python scripts/no_rag_baseline.py [--dataset dataset/fin_E.json]
                                      [--output-dir results/no_rag_baseline]
                                      [--limit N]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT   = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "eval" / "scripts" / "eval.py"

MODELS = [
    "llama3.2:latest",
    "qwen2.5:3b",
    "phi4-mini:3.8b",
    "gemma3:4b",
]

OLLAMA_HOST = "172.17.0.4"
OLLAMA_PORT = 11434
DEFAULT_DS  = str(REPO_ROOT / "dataset" / "fin_E.json")


# ── runner ────────────────────────────────────────────────────────────────────

def run_model(model: str, args: argparse.Namespace) -> dict | None:
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--dataset",     args.dataset,
        "--pipeline",    "none",
        "--model",       model,
        "--ollama-host", args.ollama_host,
        "--ollama-port", str(args.ollama_port),
        "--disable-phoenix",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.output_dir:
        out_path = args.output_dir / f"no_rag_{model.replace(':', '-').replace('/', '-')}.json"
        cmd += ["--output", str(out_path)]

    print(f"\n▶  {model}")
    print("   " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print("   ✗ TIMEOUT")
        return None

    if result.returncode != 0:
        print(f"   ✗ FAILED (exit {result.returncode})")
        print(result.stderr[-500:])
        return None

    # parse last JSON object from stdout (tolerates extra prints)
    decoder = json.JSONDecoder()
    last_obj = None
    s = result.stdout
    i = 0
    while i < len(s):
        j = s.find("{", i)
        if j == -1:
            break
        try:
            obj, end = decoder.raw_decode(s[j:])
            if isinstance(obj, dict):
                last_obj = obj
        except json.JSONDecodeError:
            pass
        i = j + 1

    if not last_obj:
        print("   ✗ Could not parse JSON from stdout")
        print(result.stdout[-300:])
        return None

    acc = last_obj.get("accuracy", 0.0)
    print(f"   ✔  accuracy={acc:.4f}  ({last_obj.get('correct')}/{last_obj.get('total')})")
    return last_obj


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plot(results: list[tuple[str, float]], output_dir: Path, dataset: str) -> None:
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160

    df = pd.DataFrame(results, columns=["model", "accuracy"])
    df = df.sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(df))))
    sns.barplot(data=df, x="accuracy", y="model", hue="model",
                palette="viridis", dodge=False, ax=ax, legend=False)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("")
    ax.set_title("Phase 1 Model Knockout")
    for i, row in enumerate(df.itertuples()):
        ax.text(row.accuracy + 0.01, i, f"{row.accuracy:.3f}", va="center", fontsize=10)

    fig.tight_layout()
    plot_path = output_dir / "no_rag_baseline.png"
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"\n📊  Plot saved → {plot_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-RAG baseline sweep + plot")
    p.add_argument("--dataset",     default=DEFAULT_DS)
    p.add_argument("--output-dir",  default=str(REPO_ROOT / "results" / "no_rag_baseline"), type=Path)
    p.add_argument("--models",      nargs="+", default=MODELS)
    p.add_argument("--ollama-host", default=OLLAMA_HOST)
    p.add_argument("--ollama-port", type=int, default=OLLAMA_PORT)
    p.add_argument("--limit",       type=int, default=None)
    p.add_argument("--timeout",     type=int, default=1800)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁  Results will be saved to: {args.output_dir.resolve()}")
    print(f"📋  Models: {args.models}")
    print(f"📄  Dataset: {args.dataset}")
    print("\n" + "═" * 60)
    print("No-RAG Baseline Sweep  (pipeline=none)")
    print("═" * 60)

    results: list[tuple[str, float]] = []
    for model in args.models:
        report = run_model(model, args)
        if report:
            results.append((model, report["accuracy"]))

    if not results:
        print("\n✗ No results collected — check Ollama connectivity.")
        sys.exit(1)

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n── Final Ranking ──")
    for rank, (model, acc) in enumerate(results, 1):
        print(f"  {rank}. {model:30s}  {acc:.4f}")

    # save summary JSON
    summary = {
        "generated": datetime.now().isoformat(),
        "dataset": args.dataset,
        "pipeline": "none",
        "results": [{"model": m, "accuracy": a} for m, a in results],
        "winner": results[0][0],
    }
    summary_path = args.output_dir / "no_rag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n📄  Summary written → {summary_path}")

    make_plot(results, args.output_dir, args.dataset)
    print("\n✅  Done.")


if __name__ == "__main__":
    main()
