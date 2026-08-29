from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


PMI_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_pmi.csv")
OUT_DIR = Path("Narratives3/intersectionality/figures")
FIG_DPI = 150
N_BOOT  = 1000

OUT_DIR.mkdir(parents=True, exist_ok=True)


PMI_DIMS = [
    ("pmi_role_score",   "Role Bias"),
    ("pmi_domain_score", "Domain Bias"),
    ("pmi_trait_score",  "Trait Bias"),
]

REGION_ORDER = [
    "North America", "Europe/Oceania", "Latin America",
    "East/SE Asia",  "South Asia",     "MENA", "Sub-Saharan Africa",
]




def build_gap_matrix(
    df: pd.DataFrame,
    col: str,
    models: list[str],
    regions: list[str],
    n_boot: int = N_BOOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (gap_mat, sig_mat) indexed by models × regions.
    gap = mean(daughter) − mean(son)
    sig = True when 95 % bootstrap CI excludes 0
    """
    rng      = np.random.default_rng(42)
    gap_mat  = pd.DataFrame(np.nan,  index=models, columns=regions)
    sig_mat  = pd.DataFrame(False,   index=models, columns=regions)

    for model in models:
        for region in regions:
            cell = df[(df["model_key"] == model) & (df["country_region"] == region)]
            d = cell[cell["child_label"] == "daughter"][col].dropna().values
            s = cell[cell["child_label"] == "son"][col].dropna().values
            if len(d) < 3 or len(s) < 3:
                continue
            gap   = float(d.mean() - s.mean())
            boots = [
                rng.choice(d, len(d)).mean() - rng.choice(s, len(s)).mean()
                for _ in range(n_boot)
            ]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            gap_mat.loc[model, region]  = gap
            sig_mat.loc[model, region]  = bool(lo > 0 or hi < 0)

    return gap_mat, sig_mat


# Main 

def main() -> None:
    print("Loading PMI scores …")
    load_cols = (
        ["model_key", "child_label", "country_region"]
        + [col for col, _ in PMI_DIMS]
    )
    df = pd.read_csv(PMI_CSV, usecols=load_cols)
    print(f"  {len(df):,} stories  |  {df['model_key'].nunique()} models  "
          f"|  {df['country_region'].nunique()} regions")

    regions = [r for r in REGION_ORDER if r in df["country_region"].unique()]
    models  = sorted(df["model_key"].dropna().unique())

    # Build all three gap matrices
    print("Building gap matrices …")
    gap_mats: dict[str, pd.DataFrame] = {}
    sig_mats: dict[str, pd.DataFrame] = {}
    for col, label in PMI_DIMS:
        gap_mats[col], sig_mats[col] = build_gap_matrix(df, col, models, regions)
        print(f"  {label}: gap range [{gap_mats[col].min().min():+.2f}, "
              f"{gap_mats[col].max().max():+.2f}]")

    # Sort models by mean |gap| across all dimensions (most biased first) 
    mean_abs_gap = pd.concat(gap_mats.values()).abs().groupby(level=0).mean().mean(axis=1)
    sorted_models = mean_abs_gap.sort_values(ascending=False).index.tolist()
    model_labels  = [m.replace("ollama-", "") for m in sorted_models]

    # Shared colour scale (symmetric, across all panels) 
    all_vals = pd.concat(gap_mats.values()).values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax     = float(np.percentile(np.abs(all_vals), 95))   # clip outliers
    print(f"  Colour scale: ±{vmax:.2f}  (95th-percentile of |gap|)")

    # Annotation matrix (value + star) 
    def make_annot(gap_mat: pd.DataFrame, sig_mat: pd.DataFrame,
                   row_order: list[str]) -> pd.DataFrame:
        ann = pd.DataFrame("", index=row_order, columns=gap_mat.columns)
        for m in row_order:
            for r in gap_mat.columns:
                v = gap_mat.loc[m, r]
                if pd.notna(v):
                    star = "*" if bool(sig_mat.loc[m, r]) else ""
                    ann.loc[m, r] = f"{float(v):+.2f}{star}"
        return ann

    # Figure 
    n_models  = len(sorted_models)
    cell_h    = 0.72
    fig_h     = n_models * cell_h + 4.0
    fig_w     = len(regions) * 1.85 + 4.0

    fig, axes = plt.subplots(
        1, 3,
        figsize=(fig_w * 3, fig_h),
        sharey=True,
    )

    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.65])

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, (col, dim_label) in zip(axes, PMI_DIMS):
        gm = gap_mats[col].loc[sorted_models, regions]
        sm = sig_mats[col].loc[sorted_models, regions]
        an = make_annot(gm, sm, sorted_models)

        sns.heatmap(
            gm.astype(float),
            annot=an,
            fmt="s",
            cmap="RdBu_r",
            norm=norm,
            linewidths=0.4,
            linecolor="#cccccc",
            ax=ax,
            mask=gm.isna(),
            cbar=(ax is axes[-1]),
            cbar_ax=cbar_ax if ax is axes[-1] else None,
            cbar_kws={},
            annot_kws={"fontsize": 14, "fontweight": "bold"},
            xticklabels=True,
            yticklabels=True,
        )

        ax.set_title(dim_label, fontsize=16, fontweight="bold", pad=12)
        ax.set_xlabel("World Region", fontsize=14, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Model", fontsize=14, fontweight="bold")
            ax.set_yticklabels(model_labels, fontsize=14, fontweight="bold", rotation=0)
        else:
            ax.set_yticklabels([])

        ax.set_xticklabels(
            [r.replace("/", "/\n") for r in regions],
            fontsize=14, fontweight="bold", rotation=25, ha="right",
        )

    cbar_ax.set_ylabel(
        "Daughter − Son mean PMI gap", fontsize=14, fontweight="bold"
    )
    cbar_ax.tick_params(labelsize=12)

    fig.suptitle(
        "Model × Region PMI Gender Bias Gap  (Daughter − Son mean PMI score)\n"
        "Red = daughters more stereotyped · Blue = sons more stereotyped · * = 95 % bootstrap CI ≠ 0\n"
        "Models sorted top → bottom by overall |bias| magnitude",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    stem = OUT_DIR / "fig_model_region_pmi"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {stem}.png / .pdf")

    # ── Per-model figures (regions × PMI dimensions) ──────────────────────────
    per_model_dir = OUT_DIR / "per_model_region"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating per-model figures → {per_model_dir}/")

    for model_key in sorted_models:
        label = model_key.replace("ollama-", "")

        # Build 7-region × 3-dimension gap/sig DataFrames for this model
        row_data: dict[str, list] = {"gap": [], "sig": []}
        mat_gap = pd.DataFrame(index=regions, columns=[lbl for _, lbl in PMI_DIMS],
                               dtype=float)
        mat_sig = pd.DataFrame(False, index=regions, columns=[lbl for _, lbl in PMI_DIMS])

        for col, dim_label in PMI_DIMS:
            gm = gap_mats[col]
            sm = sig_mats[col]
            if model_key in gm.index:
                for region in regions:
                    mat_gap.loc[region, dim_label] = gm.loc[model_key, region]
                    mat_sig.loc[region, dim_label] = sm.loc[model_key, region]

        # Annotation: value + star
        ann = pd.DataFrame("", index=regions, columns=[lbl for _, lbl in PMI_DIMS])
        for region in regions:
            for _, dim_label in PMI_DIMS:
                v = mat_gap.loc[region, dim_label]
                if pd.notna(v):
                    star = "*" if bool(mat_sig.loc[region, dim_label]) else ""
                    ann.loc[region, dim_label] = f"{float(v):+.2f}{star}"

        n_regions = len(regions)
        cell_h_pm = 0.72
        cell_w_pm = 2.2
        fig_h_pm  = n_regions * cell_h_pm + 3.0
        fig_w_pm  = 3 * cell_w_pm + 3.0

        fig_pm, ax_pm = plt.subplots(figsize=(fig_w_pm, fig_h_pm))

        sns.heatmap(
            mat_gap.astype(float),
            annot=ann,
            fmt="s",
            cmap="RdBu_r",
            norm=norm,
            linewidths=0.4,
            linecolor="#cccccc",
            ax=ax_pm,
            mask=mat_gap.isna(),
            cbar=True,
            cbar_kws={},
            annot_kws={"fontsize": 14, "fontweight": "bold"},
            xticklabels=True,
            yticklabels=True,
        )

        ax_pm.set_title(
            f"{label}  —  PMI Gender Bias Gap by Region & Dimension\n"
            "(Daughter − Son mean PMI score · * = 95 % bootstrap CI ≠ 0)",
            fontsize=14, fontweight="bold", pad=12,
        )
        ax_pm.set_xlabel("PMI Dimension", fontsize=14, fontweight="bold")
        ax_pm.set_ylabel("World Region",  fontsize=14, fontweight="bold")
        ax_pm.set_yticklabels(
            ax_pm.get_yticklabels(), fontsize=14, fontweight="bold", rotation=0
        )
        ax_pm.set_xticklabels(
            ax_pm.get_xticklabels(), fontsize=14, fontweight="bold", rotation=0
        )

        cbar = ax_pm.collections[0].colorbar
        cbar.set_label("Daughter − Son mean PMI gap", fontsize=13, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

        fig_pm.tight_layout()
        stem_pm = per_model_dir / f"fig_region_pmi_{label}"
        for ext in ("png", "pdf"):
            fig_pm.savefig(f"{stem_pm}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig_pm)
        print(f"  Saved: fig_region_pmi_{label}.png / .pdf")


if __name__ == "__main__":
    main()
