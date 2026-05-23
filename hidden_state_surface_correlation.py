from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


HS_DIR   = Path("Narratives3/hidden_states_by_model")
SEM_CSV  = Path("Narratives3/bias_analysis/results/story_level_semantic.csv")
CUL_CSV  = Path("Narratives3/cultural_bias/results/story_level_cultural.csv")
OUT_DIR  = Path("Narratives3/hidden_state_surface_corr")

FIG_DPI      = 150
RANDOM_STATE = 42

# Surface scores to correlate against: (column_name, readable_label)
GENDER_SCORES: list[tuple[str, str]] = [
    ("stereotype_score",  "Lexical Stereotype Score"),
    ("trait_bias_index",  "Trait Bias Index"),
    ("role_bias_index",   "Role Bias Index"),
    ("domain_bias_index", "Domain Bias Index"),
    ("marker_bias_index", "Marker Bias Index"),
    ("sem_stereotype_score", "Semantic Stereotype Score"),
    ("sem_trait_bias",    "Semantic Trait Bias"),
    ("sem_role_bias",     "Semantic Role Bias"),
    ("sem_domain_bias",   "Semantic Domain Bias"),
    ("sem_stereo_align",  "Semantic Stereo Alignment"),
]

CULTURAL_SCORES: list[tuple[str, str]] = [
    ("idv_proxy",        "Individualism Proxy"),
    ("pdi_proxy",        "Power Distance Proxy"),
    ("cultural_richness","Cultural Richness"),
]

GEN_COLORS = {"female": "#E91E8C", "male": "#1E88E5"}
FAM_COLORS = {
    "llama": "#4C72B0", "mistral": "#DD8452",
    "qwen":  "#55A868", "gemma":   "#C44E52", "gpt-oss": "#8172B2",
}



def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def _gender_axis_projection(hs_layer: np.ndarray, is_female: np.ndarray) -> np.ndarray:
    """
    Compute the gender axis (female centroid - male centroid) in a single layer's
    hidden-state space, then project every story onto that axis.

    Returns a 1-D array of signed scalar projections, one per story.
    Positive = closer to the female centroid direction.
    Negative = closer to the male centroid direction.
    """
    female_vecs = hs_layer[is_female]
    male_vecs   = hs_layer[~is_female]
    axis = female_vecs.mean(0) - male_vecs.mean(0)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.zeros(len(hs_layer))
    axis /= norm
    return hs_layer @ axis   # dot product of each story with the unit axis


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (r, p) or (nan, nan) when there's not enough variance."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5 or x[valid].std() < 1e-9 or y[valid].std() < 1e-9:
        return float("nan"), float("nan")
    r, p = pearsonr(x[valid], y[valid])
    return float(r), float(p)



def load_surface_scores() -> pd.DataFrame:
    """
    Load and merge the semantic and cultural surface-level CSV files.
    The semantic CSV already contains all lexical columns, so it is loaded first.
    Cultural columns are joined on 'id'.
    """
    sem = pd.read_csv(SEM_CSV)
    cul_cols = ["id"] + [c for c, _ in CULTURAL_SCORES]
    cul = pd.read_csv(CUL_CSV, usecols=lambda c: c in cul_cols)
    merged = sem.merge(cul, on="id", how="left")
    print(f"  Surface scores: {len(merged):,} stories  ×  {len(merged.columns)} columns")
    return merged


def load_model_hidden_states(model_key: str) -> tuple[np.ndarray, pd.DataFrame] | None:
    """
    Load .npy hidden states and metadata for one model.
    Returns None if files are missing.
    """
    hs_path   = HS_DIR / f"hidden_states_{model_key}.npy"
    meta_path = HS_DIR / f"extractor_meta_{model_key}.csv"
    if not hs_path.exists() or not meta_path.exists():
        return None
    hs   = np.load(hs_path)           # (n_stories, n_layers, hidden_dim)
    meta = pd.read_csv(meta_path)
    return hs, meta


def collect_all_projections(surface: pd.DataFrame,
                             only_model: str | None) -> pd.DataFrame:
    """
    For every available model, compute the final-layer gender-axis projection
    for each story.  Join with surface scores on 'id'.

    Returns a flat DataFrame with one row per story, columns:
      id, model_key, model_family, protagonist_gender, is_female,
      gender_projection_final,
      + all surface score columns
    """
    rows: list[pd.DataFrame] = []

    npy_files = sorted(HS_DIR.glob("hidden_states_*.npy"))
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in {HS_DIR}.\n"
            "Run hidden_state_extractor_by_model.py first."
        )

    for npy_path in npy_files:
        mk = npy_path.stem.replace("hidden_states_", "")
        if only_model and mk != only_model:
            continue

        result = load_model_hidden_states(mk)
        if result is None:
            continue
        hs, meta = result
        print(f"  {mk}: {hs.shape[0]} stories × {hs.shape[1]} layers × {hs.shape[2]}-dim")

        is_female = (meta["protagonist_gender"] == "female").values
        if is_female.sum() < 2 or (~is_female).sum() < 2:
            print(f"    [skip] too few stories per gender")
            continue

        proj_final = _gender_axis_projection(hs[:, -1, :], is_female)
        meta = meta.copy()
        meta["gender_projection_final"] = proj_final
        meta["is_female"] = is_female

        merged = meta.merge(
            surface[[c for c in surface.columns
                      if c in ["id"] + [s for s, _ in GENDER_SCORES + CULTURAL_SCORES]]],
            on="id", how="left",
        )
        rows.append(merged)

    if not rows:
        raise RuntimeError("No model data could be loaded. Check HS_DIR and model filter.")

    combined = pd.concat(rows, ignore_index=True)
    print(f"\n  Combined projection table: {len(combined):,} stories  |  "
          f"{combined['model_key'].nunique()} models")
    return combined


def collect_layer_projections(model_key: str,
                               surface: pd.DataFrame) -> pd.DataFrame | None:
    """
    For one model, compute gender-axis projections at every layer.
    Returns a DataFrame with columns: id, layer, gender_projection, stereotype_score, ...
    """
    result = load_model_hidden_states(model_key)
    if result is None:
        return None
    hs, meta = result
    is_female = (meta["protagonist_gender"] == "female").values

    surface_cols = ["id"] + [s for s, _ in GENDER_SCORES + CULTURAL_SCORES]
    surf_sub = surface[[c for c in surface_cols if c in surface.columns]]

    layer_rows: list[pd.DataFrame] = []
    for l in range(hs.shape[1]):
        proj = _gender_axis_projection(hs[:, l, :], is_female)
        frame = meta[["id", "model_key", "model_family",
                       "protagonist_gender"]].copy()
        frame["layer"] = l
        frame["gender_projection"] = proj
        merged = frame.merge(surf_sub, on="id", how="left")
        layer_rows.append(merged)

    return pd.concat(layer_rows, ignore_index=True)


# Plot 01: Scatter on surface score vs hidden projection 

def plot_scatter_surface_vs_hidden(combined: pd.DataFrame, out: Path) -> None:
    """
    One panel per model. X = stereotype_score (surface lexical bias).
    Y = hidden-state gender projection (final layer).
    Coloured by protagonist gender.  Pearson r in subplot title.
    """
    models   = sorted(combined["model_key"].unique())
    n        = len(models)
    ncols    = min(3, n)
    nrows    = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 4.2),
                              squeeze=False)

    fig.suptitle(
        "Surface Lexical Stereotype Score  vs  Hidden-State Gender Projection\n"
        "(final transformer layer — computed by the model that generated the story)",
        fontsize=13, fontweight="bold",
    )

    score_col = "stereotype_score"
    for idx, model_key in enumerate(models):
        ax   = axes[idx // ncols][idx % ncols]
        sub  = combined[combined["model_key"] == model_key].dropna(
                   subset=[score_col, "gender_projection_final"])

        for gender, color in GEN_COLORS.items():
            mask = sub["protagonist_gender"] == gender
            ax.scatter(
                sub.loc[mask, score_col],
                sub.loc[mask, "gender_projection_final"],
                c=color, alpha=0.30, s=8, linewidths=0, label=gender,
            )

        r, p = _safe_pearson(
            sub[score_col].values,
            sub["gender_projection_final"].values,
        )
        fam   = sub["model_family"].iloc[0] if len(sub) else "?"
        color = FAM_COLORS.get(fam, "gray")
        r_str = f"r = {r:.3f}  (p = {p:.2e})" if not np.isnan(r) else "r = n/a"
        ax.set_title(f"{model_key}\n{r_str}", fontsize=8, fontweight="bold",
                     color=color)
        ax.set_xlabel("Lexical Stereotype Score", fontsize=8)
        ax.set_ylabel("Hidden-State Gender Projection", fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_facecolor("#f8f8f8")
        ax.tick_params(labelsize=7)

    # Hide empty axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [mpatches.Patch(color=c, label=g) for g, c in GEN_COLORS.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=9, ncol=2)
    plt.tight_layout()
    _save(fig, out / "01_scatter_surface_vs_hidden.png")


# Plot 02: Correlation heatmap on model × surface score 

def plot_correlation_heatmap(combined: pd.DataFrame, out: Path) -> pd.DataFrame:
    """
    Heatmap: rows = models, columns = surface-bias scores.
    Cell = Pearson r between hidden-state gender projection (final layer)
    and that surface score for stories from that model.
    Diverging colormap: red = positive r, blue = negative.
    """
    models    = sorted(combined["model_key"].unique())
    all_scores = [(col, lbl) for col, lbl in GENDER_SCORES + CULTURAL_SCORES
                  if col in combined.columns]

    matrix = np.full((len(models), len(all_scores)), np.nan)
    for ri, mk in enumerate(models):
        sub = combined[combined["model_key"] == mk]
        for ci, (col, _) in enumerate(all_scores):
            valid = sub[["gender_projection_final", col]].dropna()
            if len(valid) < 5:
                continue
            r, _ = _safe_pearson(valid["gender_projection_final"].values,
                                  valid[col].values)
            matrix[ri, ci] = r

    score_labels = [lbl for _, lbl in all_scores]
    df_heat = pd.DataFrame(matrix, index=models, columns=score_labels)

    fig_w = max(12, len(all_scores) * 1.3)
    fig_h = max(4, len(models) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        df_heat,
        annot=True, fmt=".2f",
        cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.4,
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(
        "Pearson r : Hidden-State Gender Projection  ×  Surface Bias Scores\n"
        "Red = positive correlation  |  Blue = negative  |  "
        "Higher |r| = geometry and surface score agree more",
        fontweight="bold", fontsize=11,
    )
    ax.set_xlabel("Surface Bias Score")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    _save(fig, out / "02_correlation_heatmap.png")

    return df_heat


# Plot 03: Per-layer correlation curve 

def plot_layer_correlation_curve(surface: pd.DataFrame,
                                  only_model: str | None,
                                  out: Path) -> None:
    """
    For each model, compute Pearson r between gender-axis projection and
    stereotype_score at every transformer layer.
    Plot as one line per model: shows which depth best matches surface bias.
    """
    npy_files = sorted(HS_DIR.glob("hidden_states_*.npy"))
    fig, ax   = plt.subplots(figsize=(12, 5))

    has_data  = False
    max_layer = 0

    for npy_path in npy_files:
        mk = npy_path.stem.replace("hidden_states_", "")
        if only_model and mk != only_model:
            continue

        print(f"  Layer correlations for {mk} …")
        layer_df = collect_layer_projections(mk, surface)
        if layer_df is None:
            continue

        score_col = "stereotype_score"
        if score_col not in layer_df.columns:
            continue

        layers = sorted(layer_df["layer"].unique())
        rs = []
        for l in layers:
            sub   = layer_df[layer_df["layer"] == l][[
                "gender_projection", score_col]].dropna()
            r, _  = _safe_pearson(sub["gender_projection"].values,
                                   sub[score_col].values)
            rs.append(r)

        rs = np.array(rs)
        max_layer = max(max_layer, len(layers))

        # Find model family for color
        result = load_model_hidden_states(mk)
        fam    = result[1]["model_family"].iloc[0] if result else "?"
        color  = FAM_COLORS.get(fam, "gray")

        peak = int(np.nanargmax(np.abs(rs)))
        ax.plot(layers, rs, "o-", color=color, linewidth=1.8,
                markersize=4, alpha=0.85, label=mk)
        ax.axvline(peak, color=color, linestyle=":", alpha=0.25)
        has_data = True

    if not has_data:
        print("  No data for layer correlation curve: skipping.")
        plt.close(fig)
        return

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Transformer Layer  (0 = token embedding)")
    ax.set_ylabel("Pearson r\n(Hidden-State Projection  ×  Lexical Stereotype Score)")
    ax.set_title(
        "At Which Transformer Layer Does Hidden-State Geometry Best Match Surface Bias?\n"
        "Each line = one model  |  Higher |r| = geometry and surface score agree more at that depth",
        fontweight="bold",
    )
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.8)
    ax.set_facecolor("#f8f8f8")
    ax.set_xlim(left=-0.3)
    plt.tight_layout()
    _save(fig, out / "03_layer_correlation_curve.png")


# Plot 04: Three measures comparison (violin) 

def plot_three_measures_comparison(combined: pd.DataFrame, out: Path) -> None:
    """
    Compare three independent ways of measuring gender bias:
      1. Lexical stereotype_score   (surface word counting)
      2. Semantic sem_stereotype_score (embedding cosine similarity)
      3. Hidden-state gender projection (geometric, from this extractor)
    Show distributions by protagonist gender as violin plots.
    If all three agree (female stories higher on all three), bias is robust.
    """
    measures = [
        ("stereotype_score",         "Lexical\nStereotype Score"),
        ("sem_stereotype_score",     "Semantic\nStereotype Score"),
        ("gender_projection_final",  "Hidden-State\nGender Projection"),
    ]
    available = [(col, lbl) for col, lbl in measures if col in combined.columns]
    n = len(available)

    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5.5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Three Independent Measures of Gender Bias: Do They Agree?\n"
        "Female-protagonist stories should score higher on all three if bias is consistent",
        fontsize=12, fontweight="bold",
    )

    for ax, (col, lbl) in zip(axes, available):
        sub = combined[["protagonist_gender", col]].dropna()
        sub = sub[sub["protagonist_gender"].isin(["female", "male"])]

        # Violin
        parts = ax.violinplot(
            [sub.loc[sub["protagonist_gender"] == g, col].values
             for g in ["female", "male"]],
            positions=[0, 1],
            showmedians=True, widths=0.6,
        )
        for pc, color in zip(parts["bodies"], [GEN_COLORS["female"], GEN_COLORS["male"]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
        for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.2)

        r, p = _safe_pearson(
            (sub["protagonist_gender"] == "female").astype(float).values,
            sub[col].values,
        )
        r_str = f"r = {r:.3f}  p = {p:.2e}" if not np.isnan(r) else ""
        ax.set_title(f"{lbl}\n{r_str}", fontweight="bold", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Female\nprotagonist", "Male\nprotagonist"], fontsize=9)
        ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    _save(fig, out / "04_three_measures_comparison.png")


# Plot 05: Per-model alignment bar chart 

def plot_model_alignment_bar(combined: pd.DataFrame, out: Path) -> None:
    """
    Bar chart: for each model, show the Pearson r between the hidden-state
    gender projection (final layer) and the lexical stereotype_score.
    Sorted from highest to lowest.  Color = model family.
    Shows which models' internal geometry most closely mirrors their surface bias.
    """
    models = sorted(combined["model_key"].unique())
    rs, colors, labels = [], [], []

    for mk in models:
        sub = combined[combined["model_key"] == mk][
            ["gender_projection_final", "stereotype_score"]].dropna()
        r, _ = _safe_pearson(sub["gender_projection_final"].values,
                              sub["stereotype_score"].values)
        rs.append(r)
        fam = combined.loc[combined["model_key"] == mk, "model_family"].iloc[0]
        colors.append(FAM_COLORS.get(fam, "gray"))
        labels.append(mk.replace("ollama-", ""))

    order   = np.argsort(rs)[::-1]
    rs      = [rs[i] for i in order]
    colors  = [colors[i] for i in order]
    labels  = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.1), 5))
    bars = ax.bar(range(len(labels)), rs, color=colors, alpha=0.85,
                  edgecolor="white")

    for bar, r in zip(bars, rs):
        if not np.isnan(r):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    r + (0.01 if r >= 0 else -0.03),
                    f"{r:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r  (Hidden Projection × Lexical Stereotype Score)")
    ax.set_title(
        "Which Model's Internal Geometry Best Mirrors Its Surface Language Bias?\n"
        "Higher r = hidden-state gender separation tracks lexical stereotype score more closely",
        fontweight="bold",
    )
    ax.set_facecolor("#f8f8f8")
    # Legend for model families
    seen = {}
    for mk, col in zip(labels, colors):
        fam = next((f for f, c in FAM_COLORS.items() if c == col), "?")
        seen[fam] = col
    handles = [mpatches.Patch(color=c, label=f) for f, c in seen.items()]
    ax.legend(handles=handles, fontsize=8, framealpha=0.8)
    plt.tight_layout()
    _save(fig, out / "05_model_alignment_bar.png")


# Save numeric summary 
def save_metrics(df_heat: pd.DataFrame, out: Path) -> None:
    path = out / "surface_hidden_metrics.csv"
    df_heat.to_csv(path)
    print(f"  saved → {path.name}")


# Main 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="Restrict to a single model key (default: all extracted).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading surface-level bias scores …")
    surface = load_surface_scores()

    print("\nCollecting final-layer hidden-state projections …")
    combined = collect_all_projections(surface, only_model=args.model)

    print("\n Plot 01: Surface score vs hidden projection scatter")
    plot_scatter_surface_vs_hidden(combined, OUT_DIR)

    print("\n Plot 02: Correlation heatmap (model × surface score)")
    df_heat = plot_correlation_heatmap(combined, OUT_DIR)

    print("\n Plot 03: Per-layer correlation curve")
    plot_layer_correlation_curve(surface, only_model=args.model, out=OUT_DIR)

    print("\n Plot 04: Three-measure comparison (violin)")
    plot_three_measures_comparison(combined, OUT_DIR)

    print("\n Plot 05: Per-model alignment bar chart")
    plot_model_alignment_bar(combined, OUT_DIR)

    print("\n Saving numeric summary")
    save_metrics(df_heat, OUT_DIR)

    print(f"\nAll outputs in  {OUT_DIR}/")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
