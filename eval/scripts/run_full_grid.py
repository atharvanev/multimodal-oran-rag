"""
Full grid evaluation: all models × all (k, alpha) combos × medium + hard datasets.
Grounded pipeline only. No-RAG baselines included for comparison.
Run with: conda run -n ui python eval/scripts/run_full_grid.py
"""

import json
import subprocess
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "eval" / "scripts" / "eval.py"
OUTPUT_DIR = REPO_ROOT / "results" / "grounded_medium_hard_full_grid"

MODELS = [
    "gemma2:27b",
    "qwen2.5vl:32b",
    "mistral-nemo:latest",
    "gemma2:latest",
]
K_VALUES = [3, 5, 7]
ALPHA_VALUES = [1.0, 0.75, 0.25, 0.0]
DATASETS = {
    "medium": str(REPO_ROOT / "dataset" / "fin_M.json"),
    "hard":   str(REPO_ROOT / "dataset" / "fin_H.json"),
}

OLLAMA_HOST       = "172.17.0.4"
OLLAMA_PORT       = "11434"
WV_HOST           = "172.17.0.2"
WV_PORT           = "8080"
WV_GRPC_PORT      = "50051"
GROUNDED_COLLECTION = "Grounded_nomic_full"


def safe_model(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def alpha_str(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "")


def run_eval(cmd: list[str], label: str) -> dict | None:
    print(f"\n▶  {label}")
    print("   " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ✗ FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-500:])
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("   ✗ Could not parse JSON output")
        print(result.stdout[:300])
        return None


def base_cmd(pipeline: str, model: str, dataset_path: str, output_path: Path) -> list[str]:
    return [
        "/home/prime/miniconda3/envs/ui/bin/python", str(EVAL_SCRIPT),
        "--pipeline",    pipeline,
        "--model",       model,
        "--dataset",     dataset_path,
        "--ollama-host", OLLAMA_HOST,
        "--ollama-port", OLLAMA_PORT,
        "--grounded-weaviate-host",      WV_HOST,
        "--grounded-weaviate-port",      WV_PORT,
        "--grounded-weaviate-grpc-port", WV_GRPC_PORT,
        "--grounded-collection",         GROUNDED_COLLECTION,
        "--disable-phoenix",
        "--output",      str(output_path),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total_runs = len(MODELS) * len(DATASETS) + len(MODELS) * len(K_VALUES) * len(ALPHA_VALUES) * len(DATASETS)
    run_num = 0

    # No-RAG baselines
    for ds_name, ds_path in DATASETS.items():
        for model in MODELS:
            run_num += 1
            label = f"[{run_num}/{total_runs}] norag | {ds_name} | {model}"
            out = OUTPUT_DIR / f"norag_{ds_name}_{safe_model(model)}.json"
            cmd = base_cmd("none", model, ds_path, out)
            report = run_eval(cmd, label)
            if report:
                report.update({"dataset_name": ds_name, "k": None, "alpha": None})
                results.append(report)
                print(f"   ✓ accuracy={report['accuracy']:.3f} ({report['correct']}/{report['total']})")

    # Grounded RAG full grid
    for ds_name, ds_path in DATASETS.items():
        for model, k, alpha in product(MODELS, K_VALUES, ALPHA_VALUES):
            run_num += 1
            label = f"[{run_num}/{total_runs}] grounded | {ds_name} | {model} | k={k} α={alpha}"
            out = OUTPUT_DIR / f"grounded_{ds_name}_{safe_model(model)}_k{k}_a{alpha_str(alpha)}.json"
            cmd = base_cmd("grounded", model, ds_path, out) + [
                "--top-k",              str(k),
                "--grounded-query-alpha", str(alpha),
            ]
            report = run_eval(cmd, label)
            if report:
                report.update({"dataset_name": ds_name, "k": k, "alpha": alpha})
                results.append(report)
                print(f"   ✓ accuracy={report['accuracy']:.3f} ({report['correct']}/{report['total']})")

    _write_summary(results)
    _write_report(results)
    print(f"\nAll done. Results in {OUTPUT_DIR}")


def _write_summary(results: list[dict]) -> None:
    summary: dict = {"runs": results, "best_per_dataset": {}}
    for ds_name in DATASETS:
        ds_results = [r for r in results if r.get("dataset_name") == ds_name]
        if ds_results:
            best = max(ds_results, key=lambda r: r["accuracy"])
            summary["best_per_dataset"][ds_name] = {
                "pipeline": best.get("pipeline"),
                "model":    best.get("model"),
                "k":        best.get("k"),
                "alpha":    best.get("alpha"),
                "accuracy": best.get("accuracy"),
                "correct":  best.get("correct"),
                "total":    best.get("total"),
            }
    path = OUTPUT_DIR / "full_grid_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")


def _write_report(results: list[dict]) -> None:
    lines = ["# Grounded Full Grid Eval — Medium & Hard Datasets\n"]
    lines.append(
        f"Models: {', '.join(MODELS)}  \n"
        f"k values: {K_VALUES}  \n"
        f"α values: {ALPHA_VALUES}  \n"
    )

    for ds_name in DATASETS:
        lines.append(f"\n## {ds_name.capitalize()} Dataset\n")
        ds_results = [r for r in results if r.get("dataset_name") == ds_name]

        # No-RAG section
        norag = sorted(
            [r for r in ds_results if r.get("pipeline") == "none"],
            key=lambda r: r["accuracy"],
            reverse=True,
        )
        lines.append("### No-RAG Baseline\n")
        lines.append("| Model | Correct | Total | Accuracy |")
        lines.append("|-------|---------|-------|----------|")
        for r in norag:
            lines.append(f"| {r['model']} | {r['correct']} | {r['total']} | {r['accuracy']:.1%} |")

        # Grounded section — top 20 configs
        grounded = sorted(
            [r for r in ds_results if r.get("pipeline") == "grounded"],
            key=lambda r: r["accuracy"],
            reverse=True,
        )
        lines.append("\n### Grounded RAG — All Configs (ranked by accuracy)\n")
        lines.append("| Rank | Model | k | α | Correct | Total | Accuracy |")
        lines.append("|------|-------|---|---|---------|-------|----------|")
        for i, r in enumerate(grounded, 1):
            lines.append(
                f"| {i} | {r['model']} | {r['k']} | {r['alpha']} "
                f"| {r['correct']} | {r['total']} | {r['accuracy']:.1%} |"
            )

        if grounded:
            best = grounded[0]
            lines.append(
                f"\n**Best grounded config:** `{best['model']}`, k={best['k']}, α={best['alpha']} "
                f"→ **{best['accuracy']:.1%}** ({best['correct']}/{best['total']})\n"
            )

    # Cross-dataset summary table
    lines.append("\n## Summary: Best Config per Dataset\n")
    lines.append("| Dataset | Pipeline | Model | k | α | Accuracy |")
    lines.append("|---------|----------|-------|---|---|----------|")
    for ds_name in DATASETS:
        ds_results = [r for r in results if r.get("dataset_name") == ds_name]
        if ds_results:
            best = max(ds_results, key=lambda r: r["accuracy"])
            k_str    = str(best["k"])    if best["k"]    is not None else "—"
            alpha_str_ = str(best["alpha"]) if best["alpha"] is not None else "—"
            lines.append(
                f"| {ds_name} | {best['pipeline']} | {best['model']} "
                f"| {k_str} | {alpha_str_} | {best['accuracy']:.1%} |"
            )

    path = OUTPUT_DIR / "full_grid_report.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
