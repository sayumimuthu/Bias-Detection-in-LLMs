from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


EMBED_DIR  = Path("Narratives3/embedding_viz")
OUT_DIR    = Path("Narratives3/bias_geometry")
META_PATH  = EMBED_DIR / "story_meta_with_projections.csv"

FIG_DPI    = 150
RANDOM_STATE = 42

FAM_COLORS = {
    "llama":   "#4C72B0",
    "mistral": "#DD8452",
    "qwen":    "#55A868",
    "gemma":   "#C44E52",
    "gpt-oss": "#8172B2",
}
GEN_COLORS  = {"female": "#E91E8C", "male": "#1E88E5"}
REG_COLORS  = {
    "North America":      "#4C72B0",
    "Latin America":      "#DD8452",
    "MENA":               "#55A868",
    "Sub-Saharan Africa": "#C44E52",
    "South Asia":         "#8172B2",
    "East/SE Asia":       "#64B5CD",
    "Europe/Oceania":     "#FF800E",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _load() -> tuple[np.ndarray, pd.DataFrame]:
    print("Loading cached vectors and metadata …")
    # Try SBERT vectors first, fall back to Doc2Vec
    sbert_path = EMBED_DIR / "vectors_sbert.npy"
    d2v_path   = EMBED_DIR / "vectors_d2v.npy"
    if sbert_path.exists():
        vectors = np.load(sbert_path)
        tag = "SBERT"
    else:
        vectors = np.load(d2v_path)
        tag = "Doc2Vec"
    meta = pd.read_csv(META_PATH)
    print(f"  {tag}  |  vectors {vectors.shape}  |  meta {meta.shape}")
    return vectors, meta, tag


# Analysis 1: Gender bias axis 

def gender_bias_axis(vectors: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """
    Compute the global gender bias axis: the unit vector pointing from the
    male centroid toward the female centroid in the full embedding space.
    """
    female_mask = (meta["protagonist_gender"] == "female").values
    male_mask   = (meta["protagonist_gender"] == "male").values
    axis = vectors[female_mask].mean(0) - vectors[male_mask].mean(0)
    return axis / (np.linalg.norm(axis) + 1e-9)


def plot_gender_projection_by_model(vectors: np.ndarray, meta: pd.DataFrame,
                                    axis: np.ndarray, tag: str,
                                    out: Path) -> None:
    """
    For each model, show the distribution of story projections onto the global
    gender axis, split by protagonist gender.
    A larger gap between female (positive) and male (negative) projections
    means that model encodes stronger gender-semantic separation.
    """
    meta = meta.copy()
    meta["gender_proj"] = vectors @ axis  # scalar projection per story

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Stories Projected onto the Gender Bias Axis  [{tag}]\n"
        "Positive = more 'female-space', Negative = more 'male-space'",
        fontsize=13, fontweight="bold",
    )

    # Left: box plot per model × gender
    order = sorted(meta["model_key"].unique())
    sns.boxplot(
        data=meta, x="model_key", y="gender_proj",
        hue="protagonist_gender",
        palette=GEN_COLORS,
        order=order, ax=axes[0], fliersize=1.5, linewidth=0.8,
    )
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=40, ha="right", fontsize=8)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Projection scalar")
    axes[0].set_title("Distribution per model", fontweight="bold")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_facecolor("#f8f8f8")

    # Right: per-model gap  (mean_female_proj − mean_male_proj)
    gaps = (
        meta.groupby(["model_key", "protagonist_gender"])["gender_proj"]
        .mean().unstack()
    )
    gaps["gap"] = gaps["female"] - gaps["male"]
    gaps = gaps["gap"].sort_values(ascending=True)
    colors = ["#C44E52" if g > gaps.median() else "#4C72B0" for g in gaps]
    gaps.plot(kind="barh", ax=axes[1], color=colors, alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(gaps.median(), color="gray", linestyle="--",
                    linewidth=0.8, label=f"median={gaps.median():.3f}")
    axes[1].set_title("Gender Axis Gap per Model\n(female mean − male mean)",
                      fontweight="bold")
    axes[1].set_xlabel("Semantic distance on gender axis")
    axes[1].legend(fontsize=8)
    axes[1].set_facecolor("#f8f8f8")

    plt.tight_layout()
    _save(fig, out / "01_gender_bias_axis_by_model.png")


def plot_gender_projection_by_role(vectors: np.ndarray, meta: pd.DataFrame,
                                   axis: np.ndarray, tag: str, out: Path) -> None:
    """
    Same projection, but broken down by narrator role.
    Reveals whether stories told by (e.g.) 'Father' differ from 'Mother'
    in their gender-semantic placement.
    """
    meta = meta.copy()
    meta["gender_proj"] = vectors @ axis
    roles = sorted(meta["person"].unique())
    pal = dict(zip(roles, sns.color_palette("tab20", len(roles))))

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=meta, x="person", y="gender_proj",
        hue="protagonist_gender",
        palette=GEN_COLORS, split=True,
        order=roles, ax=ax, linewidth=0.6,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Projection onto gender axis")
    ax.set_title(
        f"Gender Axis Projection by Narrator Role  [{tag}]",
        fontweight="bold",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_facecolor("#f8f8f8")
    plt.tight_layout()
    _save(fig, out / "02_gender_projection_by_role.png")


def plot_country_gender_gap(vectors: np.ndarray, meta: pd.DataFrame,
                            axis: np.ndarray, tag: str, out: Path) -> None:
    """
    Rank countries by the size of their gender-axis gap.
    Highlights which countries / cultures receive the most gender-stereotyped
    treatment across all models.
    """
    meta = meta.copy()
    meta["gender_proj"] = vectors @ axis
    gaps = (
        meta.groupby(["country", "protagonist_gender"])["gender_proj"]
        .mean().unstack()
    )
    gaps["gap"] = gaps["female"] - gaps["male"]
    gaps = gaps["gap"].sort_values()

    # colour by region
    country_region = meta.drop_duplicates("country").set_index("country")["region"]
    colors = [REG_COLORS.get(country_region.get(c, ""), "#888888") for c in gaps.index]

    fig, ax = plt.subplots(figsize=(10, 14))
    gaps.plot(kind="barh", ax=ax, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Gender axis gap (female − male mean projection)")
    ax.set_title(
        f"Gender Bias Gap by Country  [{tag}]\n(coloured by region)",
        fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=c, label=r) for r, c in REG_COLORS.items()
        if r in country_region.values
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              framealpha=0.8)
    ax.set_facecolor("#f8f8f8")
    plt.tight_layout()
    _save(fig, out / "03_country_gender_gap.png")


# Analysis 2: Cultural drift 

def plot_cultural_centroid_map(vectors: np.ndarray, meta: pd.DataFrame,
                               tag: str, out: Path) -> None:
    """
    Project region centroids onto PCA-2D.
    Large centroid spread → models encode strong cultural differentiation.
    Overlapping centroids → cultures are conflated.
    Annotated arrows show centroid-to-centroid displacement.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    vecs_2d = pca.fit_transform(vectors)

    regions = [r for r in REG_COLORS if (meta["region"] == r).any()]
    centroids = {
        r: vecs_2d[(meta["region"] == r).values].mean(0) for r in regions
    }

    fig, ax = plt.subplots(figsize=(13, 9))

    # Background scatter (all stories, faint)
    ax.scatter(vecs_2d[:, 0], vecs_2d[:, 1],
               c="#dddddd", alpha=0.12, s=5, linewidths=0, zorder=1)

    # Per-region scatter + centroid star
    for region, centroid in centroids.items():
        color = REG_COLORS[region]
        mask  = (meta["region"] == region).values
        ax.scatter(vecs_2d[mask, 0], vecs_2d[mask, 1],
                   c=color, alpha=0.20, s=7, linewidths=0, zorder=2)
        ax.scatter(*centroid, c=color, s=260, marker="*", edgecolors="white",
                   linewidths=1.2, zorder=5)
        ax.annotate(
            region, centroid, fontsize=8.5, fontweight="bold",
            xytext=(6, 6), textcoords="offset points", color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.75, edgecolor=color, linewidth=0.7),
            zorder=6,
        )

    ax.set_title(
        f"Cultural Region Centroids in Embedding Space (PCA-2D)  [{tag}]",
        fontweight="bold",
    )
    ax.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    ax.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    ax.set_facecolor("#f8f8f8")
    plt.tight_layout()
    _save(fig, out / "04_cultural_centroid_map.png")


def plot_region_distance_matrix(vectors: np.ndarray, meta: pd.DataFrame,
                                tag: str, out: Path) -> None:
    """
    Pairwise cosine distance between region centroids.
    High distance = culturally distinct semantic space; low = conflated.
    """
    regions = sorted([r for r in REG_COLORS if (meta["region"] == r).any()])
    centroids = np.array([
        vectors[(meta["region"] == r).values].mean(0) for r in regions
    ])
    dist_matrix = squareform(pdist(centroids, metric="cosine"))

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        dist_matrix,
        xticklabels=regions, yticklabels=regions,
        annot=True, fmt=".3f", cmap="Blues",
        ax=ax, linewidths=0.4, annot_kws={"size": 8},
    )
    ax.set_title(
        f"Cosine Distance Between Cultural Region Centroids  [{tag}]",
        fontweight="bold",
    )
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    _save(fig, out / "05_region_distance_matrix.png")


# Analysis 3: Gender × model family subspace alignment 

def plot_gender_subspace_by_family(vectors: np.ndarray, meta: pd.DataFrame,
                                   tag: str, out: Path) -> None:
    """
    For each model family, compute separate female and male centroids.
    Then plot all 10 centroids (5 families × 2 genders) in PCA-2D.
    Families whose female/male centroids are far apart encode stronger bias.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    vecs_2d = pca.fit_transform(vectors)

    rows = []
    for fam in sorted(meta["model_family"].unique()):
        for gender in ["female", "male"]:
            mask = (
                (meta["model_family"] == fam) &
                (meta["protagonist_gender"] == gender)
            ).values
            if mask.sum() == 0:
                continue
            cx, cy = vecs_2d[mask].mean(0)
            rows.append({"family": fam, "gender": gender, "cx": cx, "cy": cy})

    cdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(vecs_2d[:, 0], vecs_2d[:, 1],
               c="#dddddd", alpha=0.12, s=5, linewidths=0, zorder=1)

    for fam in cdf["family"].unique():
        sub = cdf[cdf["family"] == fam]
        fam_color = FAM_COLORS[fam]
        for _, row in sub.iterrows():
            marker = "^" if row["gender"] == "female" else "v"
            ax.scatter(row["cx"], row["cy"],
                       c=GEN_COLORS[row["gender"]], s=280,
                       marker=marker, edgecolors=fam_color, linewidths=2.5,
                       zorder=5)
            ax.annotate(
                f"{fam}\n{row['gender']}", (row["cx"], row["cy"]),
                fontsize=7.5, xytext=(5, 5), textcoords="offset points",
                color=fam_color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.7, edgecolor=fam_color, linewidth=0.6),
                zorder=6,
            )
        # Arrow from male to female centroid
        male_row   = sub[sub["gender"] == "male"].iloc[0]
        female_row = sub[sub["gender"] == "female"].iloc[0]
        ax.annotate(
            "", xy=(female_row["cx"], female_row["cy"]),
            xytext=(male_row["cx"], male_row["cy"]),
            arrowprops=dict(arrowstyle="->", color=fam_color,
                            lw=1.5, alpha=0.7),
            zorder=4,
        )

    legend_handles = [
        mpatches.Patch(facecolor=c, label=f, edgecolor="black", linewidth=0.5)
        for f, c in FAM_COLORS.items()
    ] + [
        mpatches.Patch(facecolor=GEN_COLORS["female"], label="▲ female centroid"),
        mpatches.Patch(facecolor=GEN_COLORS["male"],   label="▼ male centroid"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best",
              framealpha=0.85)
    ax.set_title(
        f"Gender Subspace Separation per Model Family  [{tag}]\n"
        "Arrow: male centroid → female centroid  |  longer = more bias",
        fontweight="bold",
    )
    ax.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    ax.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    ax.set_facecolor("#f8f8f8")
    plt.tight_layout()
    _save(fig, out / "06_gender_subspace_by_family.png")


# Analysis 4: Interactive gender-axis projection explorer 

def plot_interactive_gender_axis(vectors: np.ndarray, meta: pd.DataFrame,
                                 axis: np.ndarray, tag: str, out: Path) -> None:
    """
    Interactive scatter: t-SNE 2-D coloured by signed gender-axis projection.
    Hover shows the story snippet, model, and country.
    """
    meta = meta.copy()
    meta["gender_proj"] = vectors @ axis

    xy = np.column_stack([meta["tsne_x"].values, meta["tsne_y"].values])

    fig = go.Figure(go.Scatter(
        x=xy[:, 0], y=xy[:, 1],
        mode="markers",
        marker=dict(
            color=meta["gender_proj"].values,
            colorscale="RdBu",
            reversescale=True,
            size=4,
            opacity=0.6,
            colorbar=dict(title="Gender axis<br>projection"),
            cmin=-meta["gender_proj"].abs().quantile(0.95),
            cmax= meta["gender_proj"].abs().quantile(0.95),
        ),
        customdata=np.column_stack([
            meta["model_key"].values,
            meta["country"].values,
            meta["protagonist_gender"].values,
            meta["person"].values,
            meta["gender_proj"].round(4).astype(str).values,
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b>  ·  %{customdata[3]}<br>"
            "Country: %{customdata[1]}  |  Gender: %{customdata[2]}<br>"
            "Bias projection: %{customdata[4]}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Gender Bias Axis Projection · t-SNE space  [{tag}]<br>"
                "<span style='font-size:12px'>"
                "Red = more 'male-space', Blue = more 'female-space'</span>"
            ),
            font=dict(size=14),
        ),
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        width=1300, height=750,
        plot_bgcolor="#f8f8f8",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.write_html(str(out / "07_interactive_gender_axis.html"))
    print(f"  saved → {out / '07_interactive_gender_axis.html'}")


# Main

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vectors, meta, tag = _load()

    print("\n Gender bias axis ")
    axis = gender_bias_axis(vectors, meta)
    print(f"  Global gender axis magnitude: {np.linalg.norm(axis):.4f}  (unit vector)")

    female_proj = (vectors @ axis)[(meta["protagonist_gender"] == "female").values]
    male_proj   = (vectors @ axis)[(meta["protagonist_gender"] == "male").values]
    t, p        = ttest_ind(female_proj, male_proj, equal_var=False)
    print(f"  Welch t-test (female vs male projection):  t={t:.3f},  p={p:.2e}")

    plot_gender_projection_by_model(vectors, meta, axis, tag, OUT_DIR)
    plot_gender_projection_by_role(vectors, meta, axis, tag, OUT_DIR)
    plot_country_gender_gap(vectors, meta, axis, tag, OUT_DIR)

    print("\n Cultural drift ")
    plot_cultural_centroid_map(vectors, meta, tag, OUT_DIR)
    plot_region_distance_matrix(vectors, meta, tag, OUT_DIR)

    print("\n Gender subspace by model family ")
    plot_gender_subspace_by_family(vectors, meta, tag, OUT_DIR)

    print("\n Interactive plots ")
    plot_interactive_gender_axis(vectors, meta, axis, tag, OUT_DIR)

    print(f"\nAll outputs written to  {OUT_DIR}/")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

