#!/usr/bin/env python3
"""
Generate Seaborn/Matplotlib figures from results/full_sweep/sweep_summary.json
and Phase 1 JSON metadata (line-scanned, no full parse of large files).
Writes PNGs to results/full_sweep/figures/
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO_ROOT / "results" / "full_sweep"
SUMMARY_PATH = SWEEP_DIR / "sweep_summary.json"
OUT_DIR = SWEEP_DIR / "figures"


def _scan_p1_metadata(path: Path) -> tuple[str, float] | None:
    """Read model name and top-level accuracy from a phase-1 result JSON without loading details."""
    head_lines: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for _ in range(120):
            line = f.readline()
            if not line:
                break
            head_lines.append(line)
    head = "".join(head_lines)
    m_model = re.search(r'"model"\s*:\s*"([^"]+)"', head)
    m_acc = re.search(r'"accuracy"\s*:\s*([\d.eE+-]+)', head)
    if not m_model or not m_acc:
        return None
    return m_model.group(1), float(m_acc.group(1))


def load_phase1() -> pd.DataFrame:
    rows = []
    for p in sorted(SWEEP_DIR.glob("p1_*.json")):
        meta = _scan_p1_metadata(p)
        if meta is None:
            continue
        model, acc = meta
        rows.append({"model": model, "accuracy": acc, "file": p.name})
    return pd.DataFrame(rows)


def load_phase2_long(summary: dict) -> pd.DataFrame:
    rows = []
    for pipeline, items in summary["phase2_full_grid"].items():
        for row in items:
            rows.append(
                {
                    "pipeline": pipeline,
                    "model": row["model"],
                    "k": int(row["k"]),
                    "alpha": float(row["alpha"]),
                    "accuracy": float(row["accuracy"]),
                }
            )
    return pd.DataFrame(rows)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def fig_phase1_bar(df: pd.DataFrame) -> plt.Figure:
    df = df.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(df))))
    sns.barplot(data=df, x="accuracy", y="model", hue="model", palette="viridis", dodge=False, ax=ax, legend=False)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("")
    ax.set_title("Phase 1 — Model knockout (grounded, k=3, α=0.75)")
    for i, row in enumerate(df.itertuples()):
        ax.text(row.accuracy + 0.01, i, f"{row.accuracy:.3f}", va="center", fontsize=10)
    fig.tight_layout()
    return fig


def fig_phase2_heatmaps(df: pd.DataFrame) -> plt.Figure:
    pipelines = sorted(df["pipeline"].unique())
    models = sorted(df["model"].unique())
    nrows, ncols = len(pipelines), len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    vmin, vmax = df["accuracy"].min(), df["accuracy"].max()
    for i, pipe in enumerate(pipelines):
        for j, model in enumerate(models):
            ax = axes[i][j]
            sub = df[(df["pipeline"] == pipe) & (df["model"] == model)]
            if sub.empty:
                ax.axis("off")
                continue
            pivot = sub.pivot_table(index="k", columns="alpha", values="accuracy", aggfunc="first")
            pivot = pivot.sort_index(axis=0, ascending=False).sort_index(axis=1)
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="crest",
                vmin=vmin,
                vmax=vmax,
                cbar=i == 0 and j == ncols - 1,
                ax=ax,
                linewidths=0.5,
            )
            ax.set_title(f"{pipe} — {model}")
            ax.set_xlabel("α")
            ax.set_ylabel("k")
    fig.suptitle("Phase 2 — Hyperparameter grid (accuracy)", y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


def fig_phase2_lines(df: pd.DataFrame) -> plt.Figure:
    g = sns.relplot(
        data=df,
        x="alpha",
        y="accuracy",
        hue="k",
        style="k",
        col="pipeline",
        row="model",
        kind="line",
        marker="o",
        height=3.2,
        aspect=1.15,
        palette="tab10",
        linewidth=2,
        markersize=7,
    )
    g.set_axis_labels("α (query blend)", "Accuracy")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("Phase 2 — Accuracy vs α by k (lines)", fontsize=14)
    return g.fig


def fig_phase3_compare(summary: dict) -> plt.Figure:
    bests = summary["pipeline_bests"]
    df = pd.DataFrame(bests)
    df["label"] = df.apply(
        lambda r: f"{r['pipeline']}\n{r['model']}\nk={int(r['k'])}, α={r['alpha']:.2g}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("muted", n_colors=len(df))
    x = range(len(df))
    ax.bar(x, df["accuracy"], color=colors, edgecolor="0.2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase 3 — Best config per pipeline")
    for i, acc in enumerate(df["accuracy"]):
        ax.text(i, acc + 0.02, f"{acc:.4f}", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_grounded_vs_unified(df2: pd.DataFrame) -> plt.Figure:
    """Same (model, k, α): grounded accuracy vs unified accuracy."""
    g = (
        df2[df2["pipeline"] == "grounded"][["model", "k", "alpha", "accuracy"]]
        .rename(columns={"accuracy": "grounded"})
    )
    u = (
        df2[df2["pipeline"] == "unified"][["model", "k", "alpha", "accuracy"]]
        .rename(columns={"accuracy": "unified"})
    )
    merged = g.merge(u, on=["model", "k", "alpha"], how="inner")
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        data=merged,
        x="grounded",
        y="unified",
        hue="model",
        style="k",
        s=120,
        alpha=0.85,
        ax=ax,
    )
    lo = min(merged["grounded"].min(), merged["unified"].min())
    hi = max(merged["grounded"].max(), merged["unified"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=1.5, label="y = x")
    ax.set_xlabel("Grounded accuracy")
    ax.set_ylabel("Unified accuracy")
    ax.set_title("Phase 2 — Grounded vs unified (matched k, α)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def fig_pipeline_gap_strip(df2: pd.DataFrame, summary: dict) -> plt.Figure:
    """Distribution of Phase-2 accuracies by pipeline (shows gap between grounded and unified)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = sorted(df2["pipeline"].unique())
    sns.stripplot(data=df2, x="pipeline", y="accuracy", hue="model", order=order, dodge=True, alpha=0.85, size=8, ax=ax)
    for b in summary["pipeline_bests"]:
        ax.axhline(b["accuracy"], color="0.4", ls="--", lw=1, alpha=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase 2 — All grid points by pipeline (dashed = phase-3 best per pipeline)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="model")
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_PATH.open(encoding="utf-8") as f:
        summary = json.load(f)

    df1 = load_phase1()
    df2 = load_phase2_long(summary)

    fig_phase1_bar(df1).savefig(OUT_DIR / "01_phase1_model_knockout.png", bbox_inches="tight")
    plt.close("all")

    fig_phase2_heatmaps(df2).savefig(OUT_DIR / "02_phase2_k_alpha_heatmaps.png", bbox_inches="tight")
    plt.close("all")

    fig_phase2_lines(df2).savefig(OUT_DIR / "03_phase2_accuracy_vs_alpha_lines.png", bbox_inches="tight")
    plt.close("all")

    fig_pipeline_gap_strip(df2, summary).savefig(OUT_DIR / "04_phase2_strip_by_pipeline.png", bbox_inches="tight")
    plt.close("all")

    fig_grounded_vs_unified(df2).savefig(OUT_DIR / "05_phase2_grounded_vs_unified_scatter.png", bbox_inches="tight")
    plt.close("all")

    fig_phase3_compare(summary).savefig(OUT_DIR / "06_phase3_pipeline_bests.png", bbox_inches="tight")
    plt.close("all")

    meta = {
        "dataset": summary.get("dataset"),
        "phases_run": summary.get("phases_run"),
        "figures": [p.name for p in sorted(OUT_DIR.glob("*.png"))],
    }
    (OUT_DIR / "figures_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
