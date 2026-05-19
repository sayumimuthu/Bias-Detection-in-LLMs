from __future__ import annotations

import ast
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


USE_SBERT = False  # set to True to use sentence-transformers instead of Doc2Vec


DATA_PATH = Path("Narratives3/clean_stories_for_analysis.csv")
OUT_DIR   = Path("Narratives3/embedding_viz")

# Doc2Vec hyper-parameters (ignored when USE_SBERT = True)
D2V_VECTOR_SIZE = 150
D2V_MIN_COUNT   = 2
D2V_EPOCHS      = 50

# t-SNE hyper-parameters
TSNE_PERPLEXITY = 40
TSNE_ITER       = 1000
RANDOM_STATE    = 42
FIG_DPI         = 150


FAM_COLORS = {
    "llama":   "#4C72B0",
    "mistral": "#DD8452",
    "qwen":    "#55A868",
    "gemma":   "#C44E52",
    "gpt-oss": "#8172B2",
}
GEN_COLORS = {"female": "#E91E8C", "male": "#1E88E5"}

REGION_MAP = {
    "United States": "North America",  "Canada": "North America",
    "Mexico": "Latin America",         "Brazil": "Latin America",
    "Argentina": "Latin America",      "Colombia": "Latin America",
    "United Arab Emirates": "MENA",    "Saudi Arabia": "MENA",
    "Iran": "MENA",                    "Egypt": "MENA",
    "Turkey": "MENA",                  "Morocco": "MENA",
    "Nigeria": "Sub-Saharan Africa",   "Kenya": "Sub-Saharan Africa",
    "Ethiopia": "Sub-Saharan Africa",  "South Africa": "Sub-Saharan Africa",
    "Ghana": "Sub-Saharan Africa",
    "India": "South Asia",             "Sri Lanka": "South Asia",
    "Pakistan": "South Asia",
    "Japan": "East/SE Asia",           "China": "East/SE Asia",
    "South Korea": "East/SE Asia",     "Indonesia": "East/SE Asia",
    "Thailand": "East/SE Asia",        "Vietnam": "East/SE Asia",
    "Philippines": "East/SE Asia",
    "Russia": "Europe/Oceania",        "Germany": "Europe/Oceania",
    "Greece": "Europe/Oceania",        "Italy": "Europe/Oceania",
    "France": "Europe/Oceania",        "Spain": "Europe/Oceania",
    "Poland": "Europe/Oceania",        "Australia": "Europe/Oceania",
}
REG_COLORS = {
    "North America":    "#4C72B0",
    "Latin America":    "#DD8452",
    "MENA":             "#55A868",
    "Sub-Saharan Africa": "#C44E52",
    "South Asia":       "#8172B2",
    "East/SE Asia":     "#64B5CD",
    "Europe/Oceania":   "#FF800E",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def _legend_patches(color_dict: dict) -> list:
    return [mpatches.Patch(color=c, label=l) for l, c in color_dict.items()]


# Load & filter 

def load_data() -> pd.DataFrame:
    print("Loading data …")
    df = pd.read_csv(DATA_PATH)
    df = df[df["word_count_compliant"] == True].reset_index(drop=True)
    df["region"] = df["country"].map(REGION_MAP)
    df["token_list"] = df["tokens"].apply(ast.literal_eval)
    print(f"  {len(df):,} compliant stories  |  "
          f"{df['model_key'].nunique()} models  |  "
          f"{df['country'].nunique()} countries")
    return df


# Build embeddings 

def build_doc2vec(df: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    model_path = cache_dir / "doc2vec.model"
    vec_path   = cache_dir / "vectors_d2v.npy"

    if vec_path.exists() and model_path.exists():
        print("Loading cached Doc2Vec vectors …")
        return np.load(vec_path)

    print("Training Doc2Vec …")
    tagged = [
        TaggedDocument(words=row["token_list"], tags=[str(i)])
        for i, row in df.iterrows()
    ]
    model = Doc2Vec(
        vector_size=D2V_VECTOR_SIZE,
        min_count=D2V_MIN_COUNT,
        epochs=D2V_EPOCHS,
        workers=4,
        dm=1,
        seed=RANDOM_STATE,
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(str(model_path))

    vecs = np.array([model.dv[str(i)] for i in range(len(df))])
    np.save(vec_path, vecs)
    print(f"  Doc2Vec complete: shape {vecs.shape}")
    return vecs


def build_sbert(df: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    vec_path = cache_dir / "vectors_sbert.npy"
    if vec_path.exists():
        print("Loading cached SBERT vectors …")
        return np.load(vec_path)

    print("Encoding with sentence-transformers (all-mpnet-base-v2) …")
    model = SentenceTransformer("all-mpnet-base-v2")
    vecs = model.encode(
        df["story"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    np.save(vec_path, vecs)
    print(f"  SBERT complete: shape {vecs.shape}")
    return vecs


def get_vectors(df: pd.DataFrame, cache_dir: Path) -> tuple[np.ndarray, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if USE_SBERT:
        return build_sbert(df, cache_dir), "SBERT (all-mpnet-base-v2)"
    return build_doc2vec(df, cache_dir), "Doc2Vec"


# Dimensionality reduction 

def reduce(vectors: np.ndarray) -> dict[str, np.ndarray]:
    print("PCA: 50 components …")
    pca50 = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(vectors)

    print("t-SNE 2-D  (perplexity=40, ~1-2 min) …")
    tsne2 = TSNE(
        n_components=2, perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITER, init="pca", learning_rate="auto",
        random_state=RANDOM_STATE,
    ).fit_transform(pca50)

    print("t-SNE 3-D …")
    tsne3 = TSNE(
        n_components=3, perplexity=TSNE_PERPLEXITY,
        max_iter=800, init="pca", learning_rate="auto",
        random_state=RANDOM_STATE,
    ).fit_transform(pca50)

    return {"pca50": pca50, "tsne2": tsne2, "tsne3": tsne3}


def cache_projections(meta: pd.DataFrame, proj: dict, cache_dir: Path) -> pd.DataFrame:
    meta = meta.copy()
    meta["pca_x"]  = proj["pca50"][:, 0]
    meta["pca_y"]  = proj["pca50"][:, 1]
    meta["tsne_x"] = proj["tsne2"][:, 0]
    meta["tsne_y"] = proj["tsne2"][:, 1]
    meta["tsne3_x"] = proj["tsne3"][:, 0]
    meta["tsne3_y"] = proj["tsne3"][:, 1]
    meta["tsne3_z"] = proj["tsne3"][:, 2]
    meta.to_csv(cache_dir / "story_meta_with_projections.csv", index=False)
    return meta


# Static visualisations 

def plot_pca_grid(df: pd.DataFrame, meta: pd.DataFrame, proj: dict,
                  tag: str, out: Path) -> None:
    """2 × 2 PCA grid: model family / gender / region / scree plot."""
    pca2 = proj["pca50"][:, :2]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Story Embeddings: PCA Projection  [{tag}]",
                 fontsize=15, fontweight="bold")

    def _scatter(ax, col, colors, title, xlabel="PC 1", ylabel="PC 2"):
        for val, color in colors.items():
            mask = meta[col] == val
            ax.scatter(pca2[mask, 0], pca2[mask, 1],
                       c=color, alpha=0.30, s=6, linewidths=0, label=val)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(markerscale=3, fontsize=7, loc="best",
                  framealpha=0.7, edgecolor="none")
        ax.set_facecolor("#f8f8f8")

    _scatter(axes[0, 0], "model_family",       FAM_COLORS, "Model Family")
    _scatter(axes[0, 1], "protagonist_gender", GEN_COLORS, "Protagonist Gender")
    _scatter(axes[1, 0], "region",             REG_COLORS, "Cultural Region")

    # Scree plot
    ax = axes[1, 1]
    pca_full = PCA(n_components=20, random_state=RANDOM_STATE)
    pca_full.fit(proj["pca50"])
    ev = pca_full.explained_variance_ratio_ * 100
    ax.bar(range(1, 21), ev, color="#4C72B0", alpha=0.8)
    ax.plot(range(1, 21), np.cumsum(ev), "r-o", markersize=4, label="Cumulative %")
    ax.axhline(80, color="gray", linestyle="--", alpha=0.6, label="80 % threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Scree Plot", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    _save(fig, out / "01_pca_grid.png")


def plot_tsne_grid(df: pd.DataFrame, meta: pd.DataFrame, proj: dict,
                   tag: str, out: Path) -> None:
    """2 × 2 t-SNE grid: model family / gender / region / narrator role."""
    xy = proj["tsne2"]
    person_cats = sorted(meta["person"].unique())
    person_pal  = dict(zip(person_cats,
                           sns.color_palette("tab20", len(person_cats))))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Story Embeddings: t-SNE Projection  [{tag}]",
                 fontsize=15, fontweight="bold")

    schemes = [
        ("model_family",       FAM_COLORS,   "Model Family"),
        ("protagonist_gender", GEN_COLORS,   "Protagonist Gender"),
        ("region",             REG_COLORS,   "Cultural Region"),
        ("person",             person_pal,   "Narrator Role"),
    ]
    for ax, (col, colors, title) in zip(axes.flat, schemes):
        for val, color in colors.items():
            mask = meta[col] == val
            if not mask.any():
                continue
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=color, alpha=0.30, s=6, linewidths=0, label=val)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=3, fontsize=7, loc="best",
                  framealpha=0.7, edgecolor="none")
        ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    _save(fig, out / "02_tsne_grid.png")


def plot_kde_gender(meta: pd.DataFrame, proj: dict, tag: str, out: Path) -> None:
    """KDE density contours comparing female vs male story distributions."""
    xy = proj["tsne2"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Female vs Male Story Density in Embedding Space  [{tag}]",
                 fontsize=14, fontweight="bold")

    # Side-by-side KDE per gender
    for ax, gender in zip(axes, ["female", "male"]):
        mask  = (meta["protagonist_gender"] == gender).values
        color = GEN_COLORS[gender]
        ax.scatter(xy[~mask, 0], xy[~mask, 1],
                   c="#cccccc", alpha=0.15, s=4, linewidths=0)
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=color, alpha=0.4, s=5, linewidths=0, label=gender)
        try:
            sns.kdeplot(
                x=xy[mask, 0], y=xy[mask, 1],
                ax=ax, color=color, levels=6, linewidths=1.2,
                fill=True, alpha=0.18,
            )
        except Exception:
            pass
        ax.set_title(f"Protagonist: {gender}", fontweight="bold", color=color)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_facecolor("#f8f8f8")

    plt.tight_layout()
    _save(fig, out / "03_kde_gender.png")


def plot_per_model_tsne(meta: pd.DataFrame, proj: dict, tag: str, out: Path) -> None:
    """Small-multiples: one t-SNE panel per model, coloured by gender."""
    xy      = proj["tsne2"]
    models  = sorted(meta["model_key"].unique())
    ncols   = 4
    nrows   = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    fig.suptitle(f"Per-Model Embedding Space (gender)  [{tag}]",
                 fontsize=14, fontweight="bold")

    for ax, model in zip(axes.flat, models):
        model_mask = (meta["model_key"] == model).values
        # grey background: all other stories
        ax.scatter(xy[~model_mask, 0], xy[~model_mask, 1],
                   c="#dddddd", alpha=0.12, s=3, linewidths=0)
        for gender, color in GEN_COLORS.items():
            mask = model_mask & (meta["protagonist_gender"] == gender).values
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=color, alpha=0.55, s=5, linewidths=0, label=gender)
        ax.set_title(model, fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f8f8f8")

    # hide unused panels
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)

    handles = _legend_patches(GEN_COLORS)
    fig.legend(handles=handles, loc="lower right", fontsize=9,
               framealpha=0.8, ncol=2)
    plt.tight_layout()
    _save(fig, out / "04_per_model_tsne.png")


# Interactive Plotly visualisations 

def _snippet(df: pd.DataFrame) -> pd.Series:
    return df["story"].str[:180].str.replace("\n", " ")


def plot_interactive_2d(df: pd.DataFrame, meta: pd.DataFrame,
                        proj: dict, tag: str, out: Path) -> None:
    """Three interactive 2-D t-SNE HTML files (family / gender / region)."""
    xy = proj["tsne2"]
    snippets = _snippet(df).values
    meta = meta.copy()
    meta["snippet"] = snippets

    configs = [
        ("model_family",       FAM_COLORS, "Model Family"),
        ("protagonist_gender", GEN_COLORS, "Protagonist Gender"),
        ("region",             REG_COLORS, "Cultural Region"),
    ]

    for col, colors, title in configs:
        fig = go.Figure()
        for val, color in colors.items():
            mask = (meta[col] == val).values
            if not mask.any():
                continue
            sub = meta[mask]
            fig.add_trace(go.Scatter(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                name=val,
                marker=dict(color=color, size=4, opacity=0.5),
                customdata=np.column_stack([
                    sub["snippet"].values,
                    sub["country"].values,
                    sub["protagonist_gender"].values,
                    sub["model_key"].values,
                    sub["person"].values,
                ]),
                hovertemplate=(
                    "<b>%{customdata[3]}</b>  ·  %{customdata[4]}<br>"
                    "Country: %{customdata[1]}  |  Gender: %{customdata[2]}<br>"
                    "<i style='font-size:11px'>%{customdata[0]}…</i>"
                    "<extra></extra>"
                ),
            ))
        fig.update_layout(
            title=dict(
                text=f"Story Embeddings · t-SNE · {title}  [{tag}]",
                font=dict(size=14),
            ),
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            width=1300, height=750,
            plot_bgcolor="#f8f8f8",
            hoverlabel=dict(bgcolor="white", font_size=12),
            legend=dict(itemsizing="constant"),
        )
        fname = f"05_tsne_interactive_{col}.html"
        fig.write_html(str(out / fname))
        print(f"  saved → {out / fname}")


def plot_interactive_3d(df: pd.DataFrame, meta: pd.DataFrame,
                        proj: dict, tag: str, out: Path) -> None:
    """Interactive 3-D t-SNE coloured by model family."""
    xyz = proj["tsne3"]
    snippets = _snippet(df).values

    fig = go.Figure()
    for fam, color in FAM_COLORS.items():
        mask = (meta["model_family"] == fam).values
        if not mask.any():
            continue
        sub = meta[mask]
        fig.add_trace(go.Scatter3d(
            x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
            mode="markers",
            name=fam,
            marker=dict(color=color, size=3, opacity=0.55),
            customdata=np.column_stack([
                snippets[mask],
                sub["country"].values,
                sub["protagonist_gender"].values,
                sub["model_key"].values,
            ]),
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "Country: %{customdata[1]}  |  Gender: %{customdata[2]}<br>"
                "<i style='font-size:11px'>%{customdata[0]}…</i>"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=dict(
            text=f"Story Embeddings · 3-D t-SNE · Model Family  [{tag}]",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3",
            bgcolor="#f8f8f8",
        ),
        width=1300, height=850,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.write_html(str(out / "06_tsne_3d_interactive.html"))
    print(f"  saved: {out / '06_tsne_3d_interactive.html'}")


def plot_embedding_heatmap(vectors: np.ndarray, meta: pd.DataFrame,
                           tag: str, out: Path) -> None:
    """Cosine-similarity heatmap of model-family centroids."""
    from scipy.spatial.distance import cosine

    families = sorted(meta["model_family"].unique())
    n = len(families)
    sim = np.zeros((n, n))
    for i, f1 in enumerate(families):
        c1 = vectors[(meta["model_family"] == f1).values].mean(axis=0)
        for j, f2 in enumerate(families):
            c2 = vectors[(meta["model_family"] == f2).values].mean(axis=0)
            sim[i, j] = 1 - cosine(c1, c2)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        sim, xticklabels=families, yticklabels=families,
        annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.6, vmax=1.0, ax=ax, linewidths=0.5,
        annot_kws={"size": 11},
    )
    ax.set_title(f"Cosine Similarity: Model Family Centroids  [{tag}]",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "07_model_family_cosine_similarity.png")


# Main 

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df      = load_data()
    vectors, tag = get_vectors(df, OUT_DIR)

    print("Reducing dimensions …")
    proj = reduce(vectors)

    meta_cols = [
        "id", "model_key", "model_family", "model_params",
        "protagonist_gender", "country", "region", "person", "word_count",
    ]
    meta = df[meta_cols].copy()
    meta = cache_projections(meta, proj, OUT_DIR)

    print("\nGenerating static plots …")
    plot_pca_grid(df, meta, proj, tag, OUT_DIR)
    plot_tsne_grid(df, meta, proj, tag, OUT_DIR)
    plot_kde_gender(meta, proj, tag, OUT_DIR)
    plot_per_model_tsne(meta, proj, tag, OUT_DIR)
    plot_embedding_heatmap(vectors, meta, tag, OUT_DIR)

    print("\nGenerating interactive HTML plots …")
    plot_interactive_2d(df, meta, proj, tag, OUT_DIR)
    plot_interactive_3d(df, meta, proj, tag, OUT_DIR)

    print(f"\nAll outputs written to  {OUT_DIR}/")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

