#!/usr/bin/env python3
"""
Strategic evaluation sweep.

Phase 0 — Control baseline : all models run with no RAG (LLM-only) at fixed k=0.
                             Establishes a baseline to measure RAG uplift.

Phase 1 — Model knockout   : all models at fixed k=3, α=0.75, grounded only.
                             Eliminates weak models early before the expensive grid search.
                             Top-N models carry forward (default 2).

Phase 2 — Hyperparam sweep : k × α grid run independently for EACH pipeline
                             (grounded and unified get their own sweep).
                             This prevents grounded-tuned hyperparams from
                             disadvantaging unified in the final comparison.

Phase 3 — Pipeline compare : fair fight — each pipeline enters with its own
                             personal-best model + hyperparams from phase 2.
                             Winner is the best overall config across both pipelines.

Total runs: ~57  (4 models × phase0 + 4 models × phase1 + top2 × 3k × 4α × 2pipelines + 2 phase3 finals)
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

# ── defaults (mirror eval.py) ─────────────────────────────────────────────────
EVAL_SCRIPT    = Path(__file__).parent / "eval.py"
DEFAULT_DS     = "dataset/fin_E.json"
OLLAMA_HOST    = "172.17.0.4"
OLLAMA_PORT    = 11434
WV_HOST        = "172.17.0.2"
WV_PORT        = 8080
WV_GRPC_PORT   = 50051
MULTI2VEC_HOST = "172.17.0.5"
MULTI2VEC_PORT = 8080

MODELS = [
    "gemma2:27b",
    "qwen2.5vl:32b",
    "mistral-nemo:latest",
    "gemma2:latest",
]
K_VALUES     = [3, 5, 7]
ALPHA_VALUES = [1.0, 0.75, 0.25, 0.0]
PIPELINES    = ["grounded", "unified"]

TOP_N_MODELS = 2  # models to carry forward after phase 1


# ── config dataclass ──────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    model: str
    k: int
    alpha: float
    pipeline: str
    phase: str
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.phase} | {self.pipeline} | {self.model} | k={self.k} | α={self.alpha}"

    @property
    def output_stem(self) -> str:
        """Filesystem-safe filename stem for this run's JSON output."""
        safe_model = self.model.replace(":", "-").replace("/", "-")
        alpha_str = f"{self.alpha:.2f}".replace(".", "")
        return f"{self.phase.lower()}_{self.pipeline}_{safe_model}_k{self.k}_a{alpha_str}"


# ── helpers ───────────────────────────────────────────────────────────────────
def make_output_path(cfg: RunConfig, output_dir: Optional[Path]) -> Optional[Path]:
    """Return a per-run output path, or None if --output-dir was not set."""
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{cfg.output_stem}.json"


def _extract_last_json_object(mixed_stdout: str) -> Optional[dict]:
    """
    eval.py is supposed to print a single JSON object to stdout, but some
    dependencies occasionally print extra lines (e.g. "Conversation history cleared")
    which breaks a plain json.loads(stdout).

    This scans stdout and returns the last successfully-decoded top-level JSON
    object (dict) found.
    """
    if not mixed_stdout:
        return None
    decoder = json.JSONDecoder()
    last_obj: Optional[dict] = None

    s = mixed_stdout
    i = 0
    n = len(s)
    while i < n:
        # find next plausible JSON object start
        j = s.find("{", i)
        if j == -1:
            break
        try:
            obj, end = decoder.raw_decode(s[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            last_obj = obj
        i = j + max(end, 1)

    return last_obj


# ── runner ────────────────────────────────────────────────────────────────────
def run_eval(cfg: RunConfig, args: argparse.Namespace) -> Optional[dict]:
    """Invoke eval_runner.py as a subprocess and return the parsed JSON report."""
    cmd = [
        args.eval_python, str(args.eval_script),
        "--dataset",      args.dataset,
        "--pipeline",     cfg.pipeline,
        "--model",        cfg.model,
        "--top-k",        str(cfg.k),
        "--ollama-host",  args.ollama_host,
        "--ollama-port",  str(args.ollama_port),
        "--grounded-weaviate-host",      args.wv_host,
        "--grounded-weaviate-port",      str(args.wv_port),
        "--grounded-weaviate-grpc-port", str(args.wv_grpc_port),
        "--grounded-query-alpha",        str(cfg.alpha),
        "--unified-weaviate-host",       args.wv_host,
        "--unified-weaviate-port",       str(args.wv_port),
        "--unified-weaviate-grpc-port",  str(args.wv_grpc_port),
        "--unified-query-alpha",         str(cfg.alpha),
        "--multi2vec-host",              args.multi2vec_host,
        "--multi2vec-port",              str(args.multi2vec_port),
        "--disable-phoenix",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.block_filter:
        cmd += ["--block-filter", args.block_filter]

    # pass a unique output path so eval_runner writes full per-run detail
    out_path = make_output_path(cfg, args.output_dir)
    if out_path:
        cmd += ["--output", str(out_path)]

    print(f"\n▶  {cfg.label}")
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

    # eval.py prints the summary JSON to stdout, but tolerate extra stdout noise.
    report = _extract_last_json_object(result.stdout)
    if not report:
        print("   ✗ Could not parse JSON from stdout. stdout tail:")
        print(result.stdout[-500:])
        return None

    acc = report.get("accuracy", 0.0)
    saved = f"  →  saved {out_path}" if out_path else ""
    print(f"   ✔  accuracy={acc:.3f}  ({report.get('correct')}/{report.get('total')}){saved}")
    return report


# ── phases ────────────────────────────────────────────────────────────────────
def phase1_model_knockout(args: argparse.Namespace) -> tuple[list[str], list[tuple[str, float]]]:
    print("\n" + "═" * 60)
    print("PHASE 1 — Model Knockout  (k=3, α=0.75, grounded)")
    print("═" * 60)
    results: list[tuple[str, float]] = []
    for model in args.models:
        cfg = RunConfig(model=model, k=3, alpha=0.75, pipeline="grounded", phase="P1")
        report = run_eval(cfg, args)
        if report:
            results.append((model, report["accuracy"]))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n── Phase 1 ranking ──")
    for rank, (m, acc) in enumerate(results, 1):
        print(f"  {rank}. {m:30s}  {acc:.3f}")

    top_models = [m for m, _ in results[:args.top_n]]
    print(f"\n  → Carrying forward: {top_models}")
    return top_models, results


@dataclass
class PipelineBest:
    pipeline: str
    model: str
    k: int
    alpha: float
    accuracy: float


@dataclass
class SweepData:
    """Accumulates full result tables across all phases for report generation."""
    phase0_results: list[tuple[str, float]] = field(default_factory=list)
    phase1_results: list[tuple[str, float]] = field(default_factory=list)
    top_models: list[str] = field(default_factory=list)
    # pipeline → sorted list of (model, k, alpha, accuracy)
    phase2_results: dict[str, list[tuple[str, int, float, float]]] = field(default_factory=dict)
    phase3_results: list[tuple[str, str, int, float, float]] = field(default_factory=list)


def phase0_control_baseline(args: argparse.Namespace) -> list[tuple[str, float]]:
    print("\n" + "═" * 60)
    print("PHASE 0 — Control Baseline  (no RAG, LLM-only)")
    print("═" * 60)
    results: list[tuple[str, float]] = []
    for model in args.models:
        cfg = RunConfig(model=model, k=0, alpha=0.0, pipeline="none", phase="P0")
        report = run_eval(cfg, args)
        if report:
            results.append((model, report["accuracy"]))
    results.sort(key=lambda x: x[1], reverse=True)
    print("\n── Phase 0 ranking ──")
    for rank, (m, acc) in enumerate(results, 1):
        print(f"  {rank}. {m:30s}  {acc:.3f}")
    return results


def phase2_hyperparam_sweep(
    top_models: list[str], args: argparse.Namespace
) -> tuple[list[PipelineBest], dict[str, list[tuple[str, int, float, float]]]]:
    """
    Sweep k × alpha independently for EACH pipeline so neither is
    evaluated with hyperparams tuned for the other.
    Returns the best config found per pipeline and the full sorted result
    table per pipeline for report generation.
    """
    pipeline_bests: list[PipelineBest] = []
    all_pipeline_results: dict[str, list[tuple[str, int, float, float]]] = {}

    for pipeline in args.pipelines:
        print("\n" + "═" * 60)
        print(f"PHASE 2 — Hyperparam Sweep  (k × α, {pipeline})")
        print("═" * 60)
        results: list[tuple[str, int, float, float]] = []
        for model, k, alpha in product(top_models, args.k_values, args.alpha_values):
            cfg = RunConfig(model=model, k=k, alpha=alpha, pipeline=pipeline, phase="P2")
            report = run_eval(cfg, args)
            if report:
                results.append((model, k, alpha, report["accuracy"]))

        if not results:
            print(f"   ✗ All runs failed for pipeline={pipeline}; skipping.")
            continue

        results.sort(key=lambda x: x[3], reverse=True)
        all_pipeline_results[pipeline] = results

        print(f"\n── Phase 2 [{pipeline}] top-5 ──")
        for model, k, alpha, acc in results[:5]:
            print(f"  {model:30s}  k={k}  α={alpha:.2f}  {acc:.3f}")

        best_model, best_k, best_alpha, best_acc = results[0]
        print(f"\n  → Best for {pipeline}: {best_model}  k={best_k}  α={best_alpha}  acc={best_acc:.3f}")
        pipeline_bests.append(PipelineBest(pipeline, best_model, best_k, best_alpha, best_acc))

    return pipeline_bests, all_pipeline_results


def phase3_pipeline_compare(
    pipeline_bests: list[PipelineBest], args: argparse.Namespace
) -> list[tuple[str, str, int, float, float]]:
    """
    Fair fight: each pipeline runs with its own best model + hyperparams.
    Returns the sorted results list for report generation.
    """
    print("\n" + "═" * 60)
    print("PHASE 3 — Pipeline Compare  (each pipeline at its personal best)")
    print("═" * 60)

    results: list[tuple[str, str, int, float, float]] = []
    for pb in pipeline_bests:
        cfg = RunConfig(model=pb.model, k=pb.k, alpha=pb.alpha,
                        pipeline=pb.pipeline, phase="P3")
        report = run_eval(cfg, args)
        acc = report["accuracy"] if report else pb.accuracy  # fall back to phase 2 result
        results.append((pb.pipeline, pb.model, pb.k, pb.alpha, acc))

    results.sort(key=lambda x: x[4], reverse=True)
    print("\n── Phase 3 final comparison ──")
    for pipeline, model, k, alpha, acc in results:
        print(f"  {pipeline:10s}  {model:30s}  k={k}  α={alpha:.2f}  acc={acc:.3f}")

    winner = results[0]
    print(f"\n  🏆  Winner: {winner[0]}  (model={winner[1]}, k={winner[2]}, α={winner[3]}, acc={winner[4]:.3f})")
    return results


# ── report generation ─────────────────────────────────────────────────────────
def generate_markdown_report(data: SweepData, args: argparse.Namespace) -> str:
    lines: list[str] = []

    lines.append("# Sweep Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Dataset:** `{args.dataset}`  ")
    lines.append(f"**Phases run:** {args.phases}  ")
    lines.append("")

    # ── Phase 0 ──────────────────────────────────────────────────────────────
    lines.append("## Phase 0 — Control Baseline  `(no RAG, LLM-only)`")
    lines.append("")
    if data.phase0_results:
        lines.append("| Rank | Model | Accuracy |")
        lines.append("|------|-------|----------|")
        for rank, (model, acc) in enumerate(data.phase0_results, 1):
            lines.append(f"| {rank} | `{model}` | {acc:.4f} |")
    else:
        lines.append("_Phase 0 was not run._")
    lines.append("")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    lines.append("## Phase 1 — Model Knockout  `(k=3, α=0.75, grounded)`")
    lines.append("")
    if data.phase1_results:
        lines.append("| Rank | Model | Accuracy | Carried Forward |")
        lines.append("|------|-------|----------|-----------------|")
        for rank, (model, acc) in enumerate(data.phase1_results, 1):
            carried = "✓" if model in data.top_models else ""
            lines.append(f"| {rank} | `{model}` | {acc:.4f} | {carried} |")
    else:
        lines.append("_Phase 1 was not run._")
    lines.append("")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    lines.append("## Phase 2 — Hyperparam Sweep  `(k × α grid)`")
    lines.append("")
    if data.phase2_results:
        for pipeline, results in data.phase2_results.items():
            lines.append(f"### {pipeline.capitalize()} Pipeline")
            lines.append("")
            if results:
                best_key = (results[0][0], results[0][1], results[0][2])
                lines.append("| Rank | Model | k | α | Accuracy | |")
                lines.append("|------|-------|---|---|----------|---|")
                for rank, (model, k, alpha, acc) in enumerate(results, 1):
                    star = "★ best" if (model, k, alpha) == best_key else ""
                    lines.append(f"| {rank} | `{model}` | {k} | {alpha:.2f} | {acc:.4f} | {star} |")
            else:
                lines.append("_No results._")
            lines.append("")
    else:
        lines.append("_Phase 2 was not run._")
        lines.append("")

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    lines.append("## Phase 3 — Final Pipeline Comparison")
    lines.append("")
    if data.phase3_results:
        lines.append("| Rank | Pipeline | Model | k | α | Accuracy |")
        lines.append("|------|----------|-------|---|---|----------|")
        for rank, (pipeline, model, k, alpha, acc) in enumerate(data.phase3_results, 1):
            lines.append(f"| {rank} | {pipeline} | `{model}` | {k} | {alpha:.2f} | {acc:.4f} |")
        winner = data.phase3_results[0]
        lines.append("")
        lines.append(
            f"**Winner:** `{winner[0]}` — model `{winner[1]}`, "
            f"k={winner[2]}, α={winner[3]:.2f}, accuracy **{winner[4]:.4f}**"
        )
    else:
        lines.append("_Phase 3 was not run._")
    lines.append("")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strategic eval sweep")
    p.add_argument("--eval-script",  default=str(EVAL_SCRIPT))
    p.add_argument("--eval-python",  default=sys.executable,
                   help="Python interpreter used to invoke eval.py subprocesses")
    p.add_argument("--dataset",     default=DEFAULT_DS)
    p.add_argument("--limit",       type=int, default=None,
                   help="Cap questions per run (great for quick sanity checks)")
    p.add_argument("--timeout",     type=int, default=1800,
                   help="Seconds before a single run is killed (default 30 min)")
    p.add_argument("--top-n",       type=int, default=TOP_N_MODELS,
                   help="Models to carry from phase 1 → phase 2")
    p.add_argument("--block-filter", default=None, choices=["chunks"])
    p.add_argument("--output-dir",  default=None, type=Path,
                   help="Directory to save per-run JSON results. "
                        "Files named e.g. p1_grounded_llama3.2_k3_a075.json")

    # override defaults if your cluster differs
    p.add_argument("--ollama-host",    default=OLLAMA_HOST)
    p.add_argument("--ollama-port",    type=int, default=OLLAMA_PORT)
    p.add_argument("--wv-host",        default=WV_HOST)
    p.add_argument("--wv-port",        type=int, default=WV_PORT)
    p.add_argument("--wv-grpc-port",   type=int, default=WV_GRPC_PORT)
    p.add_argument("--multi2vec-host", default=MULTI2VEC_HOST)
    p.add_argument("--multi2vec-port", type=int, default=MULTI2VEC_PORT)

    # override sweep grid if needed
    p.add_argument("--models",       nargs="+", default=MODELS)
    p.add_argument("--pipelines",    nargs="+", default=PIPELINES,
                   choices=["grounded", "unified"])
    p.add_argument("--k-values",     nargs="+", type=int,   default=K_VALUES)
    p.add_argument("--alpha-values", nargs="+", type=float, default=ALPHA_VALUES)

    # run only specific phases (default: all)
    p.add_argument("--phases", nargs="+", type=int, choices=[0, 1, 2, 3],
                   default=[0, 1, 2, 3])

    # skip phase 1 by providing pre-chosen top models
    p.add_argument("--top-models", nargs="+", default=None,
                   help="Skip phase 1 and use these models directly in phase 2")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁  Per-run JSON results will be saved to: {args.output_dir.resolve()}")

    top_models: list[str] = args.top_models or []
    pipeline_bests: list[PipelineBest] = []
    sweep_data = SweepData()

    if 0 in args.phases:
        sweep_data.phase0_results = phase0_control_baseline(args)

    if 1 in args.phases and not args.top_models:
        top_models, p1_results = phase1_model_knockout(args)
        sweep_data.phase1_results = p1_results
        sweep_data.top_models = top_models
    elif args.top_models:
        sweep_data.top_models = top_models

    if 2 in args.phases:
        if not top_models:
            print("No top models for phase 2. Run phase 1 first or pass --top-models.")
            sys.exit(1)
        pipeline_bests, p2_results = phase2_hyperparam_sweep(top_models, args)
        sweep_data.phase2_results = p2_results

    if 3 in args.phases:
        if not pipeline_bests:
            print("No pipeline bests for phase 3. Run phase 2 first.")
            sys.exit(1)
        sweep_data.phase3_results = phase3_pipeline_compare(pipeline_bests, args)

    if args.output_dir:
        summary = {
            "dataset": args.dataset,
            "phases_run": args.phases,
            "models_tested": args.models,
            "phase0_control": [
                {"model": m, "accuracy": acc}
                for m, acc in sweep_data.phase0_results
            ],
            "top_models_phase2": top_models,
            "pipeline_bests": [
                {
                    "pipeline": pb.pipeline,
                    "model": pb.model,
                    "k": pb.k,
                    "alpha": pb.alpha,
                    "accuracy": pb.accuracy,
                }
                for pb in pipeline_bests
            ],
            "phase2_full_grid": {
                pipeline: [
                    {"model": m, "k": k, "alpha": a, "accuracy": acc}
                    for m, k, a, acc in rows
                ]
                for pipeline, rows in sweep_data.phase2_results.items()
            },
        }
        summary_path = args.output_dir / "sweep_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n📄  Sweep summary written to: {summary_path}")

        report_md = generate_markdown_report(sweep_data, args)
        report_path = args.output_dir / "sweep_report.md"
        report_path.write_text(report_md)
        print(f"📄  Markdown report written to: {report_path}")

    print("\n✅  Sweep complete.")


if __name__ == "__main__":
    main()