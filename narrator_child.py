from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PMI_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_pmi.csv")
OUT_DIR = Path("Narratives3/intersectionality/figures/per_model_i6")

FIG_DPI  = 150
N_BOOT   = 1000

OUT_DIR.mkdir(parents=True, exist_ok=True)


PMI_DIMS = [
    ("pmi_role_score",   "Role Bias"),
    ("pmi_domain_score", "Domain Bias"),
    ("pmi_trait_score",  "Trait Bias"),
]

NARRATOR_PALETTE = {
    "female": "#41BAC0",   # rose
    "male":   "#AD702E",   # navy
}

child_order = ["daughter", "son"]
x_pos       = {c: i for i, c in enumerate(child_order)}


# Cell-stats computation 

def compute_cell_stats(df: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """
    For every (model_key × metric × narrator_role × child_label) cell,
    compute the mean PMI score and 95 % bootstrap CI.

    Returns a DataFrame with columns:
        model_key, metric, narrator_role, child_label,
        mean, ci_low, ci_high, n
    """
    rng  = np.random.default_rng(42)
    rows: list[dict] = []

    for model_key in sorted(df["model_key"].dropna().unique()):
        model_df = df[
            (df["model_key"] == model_key) &
            (df["person_gender_role"].isin(["female", "male"]))
        ]
        if len(model_df) < 10:
            continue

        for col, _ in PMI_DIMS:
            if col not in model_df.columns:
                continue
            for narrator_role in ("female", "male"):
                for child_label in ("daughter", "son"):
                    cell = model_df[
                        (model_df["person_gender_role"] == narrator_role) &
                        (model_df["child_label"]        == child_label)
                    ]
                    vals = cell[col].dropna().values
                    if len(vals) < 2:
                        continue
                    boots = [rng.choice(vals, len(vals)).mean()
                             for _ in range(n_boot)]
                    rows.append({
                        "model_key":     model_key,
                        "metric":        col,
                        "narrator_role": narrator_role,
                        "child_label":   child_label,
                        "mean":          float(vals.mean()),
                        "ci_low":        float(np.percentile(boots, 2.5)),
                        "ci_high":       float(np.percentile(boots, 97.5)),
                        "n":             int(len(vals)),
                    })

    return pd.DataFrame(rows)


# Per-model figure 

def plot_model(
    cell_stats: pd.DataFrame,
    model_key:  str,
    fig_dir:    Path,
) -> None:
    label    = model_key.replace("ollama-", "")
    model_df = cell_stats[cell_stats["model_key"] == model_key]

    if model_df.empty:
        print(f"    [{label}] skipped — no narrator-gendered cell stats")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        f"Narrator Gender × Child Gender Interaction — {label}\n"
        "PMI Bias Scores  ·  Parallel lines = no moderation  ·  "
        "Steeper / crossing lines = narrator amplifies bias  "
        "(shading = 95 % bootstrap CI)",
        fontsize=11, fontweight="bold",
    )

    for ax, (col, dim_label) in zip(axes, PMI_DIMS):
        sub_dim = model_df[model_df["metric"] == col]
        if sub_dim.empty:
            ax.set_visible(False)
            continue

        for narrator_role, color in NARRATOR_PALETTE.items():
            sub = (
                sub_dim[sub_dim["narrator_role"] == narrator_role]
                .set_index("child_label")
                .reindex(child_order)
            )
            x  = [x_pos[c] for c in child_order]
            y  = sub["mean"].values.astype(float)
            lo = sub["ci_low"].values.astype(float)
            hi = sub["ci_high"].values.astype(float)

            valid = ~np.isnan(y)
            if valid.sum() < 1:
                continue

            x_valid = [x[i] for i in range(len(x)) if valid[i]]
            ax.plot(
                x_valid, y[valid], "o-",
                color=color, linewidth=2.2, markersize=9,
                label=f"{narrator_role.capitalize()} narrator", zorder=3,
            )
            if valid.all():
                ax.fill_between(x, lo, hi, color=color, alpha=0.12, zorder=2)
                ax.errorbar(x, y,
                            yerr=[y - lo, hi - y],
                            fmt="none", color=color,
                            capsize=5, elinewidth=1.5)

            # Annotate same-gender dyad
            same_child = "daughter" if narrator_role == "female" else "son"
            if same_child in sub.index:
                yi = sub.loc[same_child, "mean"]
                if pd.notna(yi):
                    yi = float(yi)
                    ax.annotate(
                        "same-gender",
                        (x_pos[same_child], yi),
                        xytext=(
                            x_pos[same_child] + 0.07,
                            yi + abs(yi) * 0.05 + 0.01,
                        ),
                        fontsize=7.5, color=color, alpha=0.85,
                    )

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(["Daughter stories", "Son stories"], fontsize=10)
        ax.set_ylabel("Mean PMI score\n(+ = daughter-stereotyped)", fontsize=9)
        ax.set_title(dim_label, fontsize=11, fontweight="bold")
        ax.set_facecolor("#f8f8f8")
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=10, framealpha=0.85)

    fig.tight_layout()
    stem = fig_dir / f"fig_i6_{label}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: fig_i6_{label}.png / .pdf")


# Main 

def main() -> None:
    print("Loading PMI scores …")
    load_cols = (
        ["id", "model_key", "child_label", "person_gender_role"]
        + [col for col, _ in PMI_DIMS]
    )
    df = pd.read_csv(PMI_CSV, usecols=load_cols)
    print(f"  {len(df):,} stories  |  {df['model_key'].nunique()} models")

    narrator_mask = df["person_gender_role"].isin(["female", "male"])
    print(f"  {narrator_mask.sum():,} stories with gendered narrator "
          f"({(~narrator_mask).sum():,} neutral/missing excluded)\n")

    print("Computing narrator × child interaction cell stats …")
    cell_stats = compute_cell_stats(df)
    print(f"  {len(cell_stats)} cells computed\n")

    models = sorted(df["model_key"].dropna().unique())
    print(f"Generating {len(models)} figures …\n")
    for model_key in models:
        label = model_key.replace("ollama-", "")
        print(f"  {label}")
        plot_model(cell_stats, model_key, OUT_DIR)

    print(f"\nDone — all figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
