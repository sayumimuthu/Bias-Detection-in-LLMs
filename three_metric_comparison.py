from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


HS_DIR  = Path("Narratives3/hidden_states_by_model")
PMI_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_pmi.csv")
SEM_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_semantic.csv")
OUT_DIR = Path("Narratives3/hidden_state_surface_corr/three_measures")

FIG_DPI = 150
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAUGHTER_COLOR = "#1EE9CE"   # rose / pink
SON_COLOR      = "#E57E1E"   # steel blue

PMI_DIMS = ["pmi_role_score",  "pmi_domain_score", "pmi_trait_score"]
SEM_DIMS = ["sem_role_bias",   "sem_domain_bias",  "sem_trait_bias"]


# gender-axis projection 
def _gender_axis_projection(hs_layer: np.ndarray,
                             is_female: np.ndarray) -> np.ndarray:
    """
    Project each story's hidden state onto the unit vector pointing from the
    son centroid to the daughter centroid.
    Positive  → story sits on the daughter side.
    Negative  → story sits on the son side.
    """
    f_mean = hs_layer[is_female].mean(axis=0)
    m_mean = hs_layer[~is_female].mean(axis=0)
    axis   = f_mean - m_mean
    norm   = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.zeros(len(hs_layer))
    axis /= norm
    return hs_layer @ axis          # dot product: scalar per story


# 1: Load surface scores 
print("Loading PMI scores …")
pmi_full = pd.read_csv(PMI_CSV, usecols=["id", "model_key", "child_label"] + PMI_DIMS)
print(f"  {len(pmi_full):,} rows  |  models: {pmi_full['model_key'].nunique()}")

print("Loading semantic scores …")
sem_full = pd.read_csv(SEM_CSV, usecols=["id"] + SEM_DIMS)
print(f"  {len(sem_full):,} rows")

# Merge on id: semantic CSV has same rows in same id order
surface = pmi_full.merge(sem_full, on="id", how="left")

# 2: Z-score normalise across the full corpus 
# Computed once here so every model's scores sit on the same scale.
# A story's z-score is relative to all 16 models' stories combined.
print("\nComputing corpus-wide z-scores …")
for col in PMI_DIMS + SEM_DIMS:
    mu              = surface[col].mean()
    sigma           = surface[col].std()
    surface[f"z_{col}"] = (surface[col] - mu) / (sigma + 1e-9)
    print(f"  z_{col:25s}  μ={mu:+.4f}  σ={sigma:.4f}")

# Composite = mean of the three z-scored dimensions
surface["pmi_composite"] = surface[[f"z_{c}" for c in PMI_DIMS]].mean(axis=1)
surface["sem_composite"] = surface[[f"z_{c}" for c in SEM_DIMS]].mean(axis=1)

print(f"\n  pmi_composite  μ={surface['pmi_composite'].mean():+.4f}"
      f"  σ={surface['pmi_composite'].std():.4f}")
print(f"  sem_composite  μ={surface['sem_composite'].mean():+.4f}"
      f"  σ={surface['sem_composite'].std():.4f}")


# 3: Per-model figures 
npy_files = sorted(HS_DIR.glob("hidden_states_*.npy"))
print(f"\nFound {len(npy_files)} model(s) with hidden states\n")

for npy_path in npy_files:
    mk        = npy_path.stem.replace("hidden_states_", "")
    meta_path = HS_DIR / f"extractor_meta_{mk}.csv"
    label     = mk.replace("ollama-", "")

    if not meta_path.exists():
        print(f"  [{label}] skipped — no metadata CSV")
        continue

    # Load hidden states for this model 
    hs   = np.load(npy_path)                    # (n, n_layers, hidden_dim)
    meta = pd.read_csv(meta_path)               # id, protagonist_gender, ...
    print(f"  {label}: {hs.shape[0]} stories × {hs.shape[1]} layers × {hs.shape[2]}-dim")

    is_female = (meta["protagonist_gender"] == "female").values
    if is_female.sum() < 2 or (~is_female).sum() < 2:
        print(f"    skipped — too few stories per gender")
        continue

    # Compute final-layer projection 
    proj = _gender_axis_projection(hs[:, -1, :], is_female)
    meta = meta[["id"]].copy()
    meta["gender_projection"] = proj

    # Get this model's surface scores 
    model_surface = surface[surface["model_key"] == mk][
        ["id", "child_label", "pmi_composite", "sem_composite"]
    ].copy()

    if model_surface.empty:
        print(f"    skipped — no surface scores for {mk}")
        continue

    # Merge surface + projection on id 
    merged = model_surface.merge(meta, on="id", how="inner")
    if merged.empty:
        print(f"    skipped — no ID overlap between surface scores and hidden states")
        continue
    print(f"    {len(merged)} matched stories  "
          f"(daughter={( merged['child_label']=='daughter').sum()}, "
          f"son={(merged['child_label']=='son').sum()})")

    
    measures = [
        ("pmi_composite",    "PMI Composite Score\n(z-normalised mean: role + domain + trait)",
                             "PMI Composite (z-score)"),
        ("sem_composite",    "Semantic Composite Score\n(z-normalised mean: role + domain + trait)",
                             "Semantic Composite (z-score)"),
        ("gender_projection","Hidden-State\nGender Projection",
                             "Projection value (dot product)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        f"Three-Measure Gender Bias Comparison — {label}\n"
        "Positive = daughter-stereotyped  ·  Negative = son-stereotyped",
        fontsize=13, fontweight="bold",
    )

    for ax, (col, panel_title, y_label) in zip(axes, measures):
        sub      = merged[["child_label", col]].dropna()
        d_vals   = sub.loc[sub["child_label"] == "daughter", col].values
        s_vals   = sub.loc[sub["child_label"] == "son",      col].values

        if len(d_vals) < 2 or len(s_vals) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            continue

        # Violin plot  
        parts = ax.violinplot(
            [d_vals, s_vals],
            positions=[0, 1],
            showmedians=True,
            showextrema=True,
            widths=0.55,
        )
        parts["bodies"][0].set_facecolor(DAUGHTER_COLOR)
        parts["bodies"][0].set_alpha(0.45)
        parts["bodies"][1].set_facecolor(SON_COLOR)
        parts["bodies"][1].set_alpha(0.45)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in parts:
                parts[part].set_color("#333333")
                parts[part].set_linewidth(1.3)

      
        all_vals  = np.concatenate([d_vals, s_vals])
        data_span = all_vals.max() - all_vals.min() if all_vals.std() > 1e-6 else 1.0
        offset    = data_span * 0.06

        for x_pos, vals, color in [(0, d_vals, DAUGHTER_COLOR),
                                    (1, s_vals, SON_COLOR)]:
            mean_val = vals.mean()
            ax.scatter(x_pos, mean_val, color="black", s=70, zorder=6,
                       marker="D", edgecolors="white", linewidth=1.2)
            ax.text(x_pos, mean_val + offset, f"μ = {mean_val:.3f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75))

        # Statistics 
        # Pearson r: is_daughter (0/1) vs score
        is_d  = (sub["child_label"] == "daughter").astype(float).values
        score = sub[col].values
        r     = np.corrcoef(is_d, score)[0, 1] if score.std() > 1e-9 else float("nan")

        # Welch t-test
        _, p_val = ttest_ind(d_vals, s_vals, equal_var=False)
        sig_str  = ("***" if p_val < 0.001 else
                    "**"  if p_val < 0.01  else
                    "*"   if p_val < 0.05  else "ns")

        #r_str = f"r = {r:.3f}" if not np.isnan(r) else ""
        #p_str = f"p = {p_val:.2e}  {sig_str}"

        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Daughter\nstories", "Son\nstories"], fontsize=12)
        ax.set_ylabel(y_label, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
        ax.set_facecolor("#f8f8f8")
        ax.spines[["top", "right"]].set_visible(False)

  
    legend_handles = [
        mpatches.Patch(facecolor=DAUGHTER_COLOR, alpha=0.6, label="Daughter stories (F)"),
        mpatches.Patch(facecolor=SON_COLOR,      alpha=0.6, label="Son stories (M)"),
        plt.Line2D([0], [0], marker="D", color="gray", markersize=7,
                   linestyle="None", label="Group mean"),
        plt.Line2D([0], [0], color="black", linewidth=1, linestyle="--",
                   alpha=0.5, label="Zero baseline"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    stem = OUT_DIR / f"three_measures_{label}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: three_measures_{label}.png / .pdf")

print(f"\nAll figures in {OUT_DIR}/")
