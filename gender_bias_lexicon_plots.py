#!/usr/bin/env python3
"""
Plotting utility for lexicon-based gender bias outputs.

Expected input files (produced by gender_bias_lexicon_analysis.py):
- model_child_gender_summary.csv
- model_daughter_minus_son_gaps.csv
- model_gap_bootstrap_ci.csv

Usage:
  python3 gender_bias_lexicon_plots.py
  python3 gender_bias_lexicon_plots.py --input-dir Narratives2/visualizations/gender_bias
  python3 gender_bias_lexicon_plots.py --output-dir Narratives2/visualizations/gender_bias/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPOSITE_METRICS = [
    "trait_bias_index",
    "role_bias_index",
    "domain_bias_index",
    "direct_gender_marker_index",
]

COMPOSITE_LABELS: Dict[str, str] = {
    "trait_bias_index": "Trait Bias Index",
    "role_bias_index": "Role Bias Index",
    "domain_bias_index": "Domain Bias Index",
    "direct_gender_marker_index": "Direct Gender Marker Index",
}

PER_TOKEN_METRICS = [
    "agentic_per_1000",
    "communal_per_1000",
    "career_per_1000",
    "family_per_1000",
    "masculine_traits_per_1000",
    "feminine_traits_per_1000",
]

PER_TOKEN_LABELS: Dict[str, str] = {
    "agentic_per_1000": "Agentic",
    "communal_per_1000": "Communal",
    "career_per_1000": "Career",
    "family_per_1000": "Family",
    "masculine_traits_per_1000": "Masculine Traits",
    "feminine_traits_per_1000": "Feminine Traits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots for gender bias lexicon outputs.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="Narratives2/visualizations/gender_bias",
        help="Directory containing analysis CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for plot images. Defaults to <input-dir>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Image DPI (default: 220).",
    )
    return parser.parse_args()


def ensure_paths(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def load_csvs(input_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "summary": input_dir / "model_child_gender_summary.csv",
        "gaps": input_dir / "model_daughter_minus_son_gaps.csv",
        "ci": input_dir / "model_gap_bootstrap_ci.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

    return {k: pd.read_csv(v) for k, v in files.items()}


def infer_model_col(df: pd.DataFrame) -> str:
    for c in ["model", "provider", "llm"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer model column from: {list(df.columns)}")


def save_fig(path: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_composite_heatmap(gaps: pd.DataFrame, model_col: str, out_dir: Path, dpi: int) -> None:
    data = gaps[gaps["metric"].isin(COMPOSITE_METRICS)].copy()
    if data.empty:
        return

    data["metric_label"] = data["metric"].map(COMPOSITE_LABELS)
    pivot = data.pivot(index=model_col, columns="metric_label", values="daughter_minus_son")
    pivot = pivot.sort_index()

    fig_w = max(7.5, 1.7 * max(1, pivot.shape[1]))
    fig_h = max(4.0, 0.6 * max(1, pivot.shape[0]))
    plt.figure(figsize=(fig_w, fig_h))

    arr = pivot.values
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 0.1)

    im = plt.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Daughter - Son")

    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=25, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.title("Composite Bias Indices by Model (Daughter - Son)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = arr[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    save_fig(out_dir / "heatmap_composite_daughter_minus_son.png", dpi)


def plot_gap_bars(gaps: pd.DataFrame, model_col: str, out_dir: Path, dpi: int) -> None:
    data = gaps[gaps["metric"].isin(COMPOSITE_METRICS)].copy()
    if data.empty:
        return

    model_order = sorted(data[model_col].unique())
    metric_order = COMPOSITE_METRICS
    x = np.arange(len(model_order))
    width = 0.18

    plt.figure(figsize=(13, 6.8))
    for idx, metric in enumerate(metric_order):
        m = data[data["metric"] == metric].set_index(model_col)
        vals = [float(m.loc[model, "daughter_minus_son"]) if model in m.index else np.nan for model in model_order]
        plt.bar(x + (idx - 1.5) * width, vals, width=width, label=COMPOSITE_LABELS[metric])

    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, model_order, rotation=25, ha="right")
    plt.ylabel("Daughter - Son")
    plt.title("Bias Gap by Model Across Composite Indices")
    plt.legend(frameon=False)
    save_fig(out_dir / "bar_composite_gaps_by_model.png", dpi)


def plot_metric_means(summary: pd.DataFrame, model_col: str, out_dir: Path, dpi: int) -> None:
    keep_metrics = [m for m in PER_TOKEN_METRICS if m in summary.columns]
    if not keep_metrics:
        return

    for metric in keep_metrics:
        metric_label = PER_TOKEN_LABELS.get(metric, metric)
        data = summary[[model_col, "child_label", metric]].copy()
        pivot = data.pivot(index=model_col, columns="child_label", values=metric)
        pivot = pivot.sort_index()

        x = np.arange(len(pivot.index))
        width = 0.36

        plt.figure(figsize=(12, 6.4))
        son_vals = pivot["son"].values if "son" in pivot.columns else np.zeros(len(pivot.index))
        daughter_vals = pivot["daughter"].values if "daughter" in pivot.columns else np.zeros(len(pivot.index))

        plt.bar(x - width / 2, son_vals, width=width, label="Son", color="#4C78A8")
        plt.bar(x + width / 2, daughter_vals, width=width, label="Daughter", color="#F58518")

        plt.xticks(x, pivot.index, rotation=25, ha="right")
        plt.ylabel("Mean per 1000 tokens")
        plt.title(f"{metric_label}: Son vs Daughter by Model")
        plt.legend(frameon=False)
        save_fig(out_dir / f"bar_{metric}_son_vs_daughter.png", dpi)


def plot_ci_errorbars(ci_df: pd.DataFrame, model_col: str, out_dir: Path, dpi: int) -> None:
    data = ci_df[ci_df["metric"].isin(COMPOSITE_METRICS)].copy()
    if data.empty:
        return

    for metric in COMPOSITE_METRICS:
        m = data[data["metric"] == metric].copy()
        if m.empty:
            continue

        m = m.sort_values(model_col)
        x = np.arange(len(m))
        y = m["gap_mean"].values
        lower = y - m["ci_low_2.5"].values
        upper = m["ci_high_97.5"].values - y

        plt.figure(figsize=(11.5, 6.1))
        plt.errorbar(x, y, yerr=[lower, upper], fmt="o", capsize=4)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(x, m[model_col], rotation=25, ha="right")
        plt.ylabel("Daughter - Son")
        plt.title(f"Bootstrap 95% CI: {COMPOSITE_LABELS.get(metric, metric)}")
        save_fig(out_dir / f"ci_{metric}.png", dpi)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"

    ensure_paths(input_dir, output_dir)
    tables = load_csvs(input_dir)

    summary = tables["summary"]
    gaps = tables["gaps"]
    ci_df = tables["ci"]

    model_col = infer_model_col(summary)

    plot_composite_heatmap(gaps, model_col, output_dir, args.dpi)
    plot_gap_bars(gaps, model_col, output_dir, args.dpi)
    plot_metric_means(summary, model_col, output_dir, args.dpi)
    plot_ci_errorbars(ci_df, model_col, output_dir, args.dpi)

    print("Gender Bias Plot Generation Complete")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Saved plot files:")
    for p in sorted(output_dir.glob("*.png")):
        print(f"- {p}")


if __name__ == "__main__":
    main()
