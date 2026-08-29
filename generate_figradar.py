from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PMI_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_pmi.csv")
SEM_CSV = Path("Narratives3/bias_analysis/results_pmi/results/story_level_semantic.csv")
OUT_DIR = Path("Narratives3/bias_analysis/figures")
FIG_DPI = 150

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Core drawing helper ────────────────────────────────────────────────────────

def draw_radar(
    values_per_model: dict[str, list[float]],
    categories: list[str],
    colors: dict,
    title: str,
    out_path: Path,
) -> None:
    """
    Spider chart with one polygon per model.
    Each axis is normalised independently to [0, 1].
    Actual numeric values are annotated along each spoke.
    """
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    all_vals = np.array(list(values_per_model.values()), dtype=float)
    a_min    = all_vals.min(axis=0)
    a_max    = all_vals.max(axis=0)
    a_rng    = np.where(a_max > a_min, a_max - a_min, 1.0)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f2f2f2")

    # Concentric grid rings
    ring_pts = np.linspace(0, 2 * np.pi, 300)
    for ring in np.linspace(0.2, 1.0, 5):
        ax.plot(ring_pts, [ring] * 300, color="gray", alpha=0.25, linewidth=0.7, zorder=1)

    # Spoke lines
    for angle in angles:
        ax.plot([angle, angle], [0, 1.05], color="gray", alpha=0.4, linewidth=0.8, zorder=1)

    # Numeric labels at 20 / 40 / 60 / 80 / 100 % of each spoke
    for j, angle in enumerate(angles):
        for k in range(1, 6):
            r_frac = k / 5
            actual = a_min[j] + r_frac * a_rng[j]
            ax.text(
                angle, r_frac, f"{actual:.2f}",
                ha="center", va="center", fontsize=7.5, color="dimgray",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.4),
                zorder=5,
            )

    # One polygon per model
    for model_key, vals in values_per_model.items():
        label = model_key.replace("ollama-", "")
        color = colors.get(model_key, "gray")
        norm  = [(v - a_min[j]) / a_rng[j] for j, v in enumerate(vals)]
        ang_c = list(angles) + [angles[0]]
        nrm_c = norm + [norm[0]]
        ax.plot(ang_c, nrm_c, "o-", linewidth=2.2, markersize=7,
                color=color, label=label, zorder=4)
        ax.fill(ang_c, nrm_c, alpha=0.12, color=color, zorder=3)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, size=14, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(0, 1.18)
    ax.set_title(title, size=15, fontweight="bold", pad=30)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.45, 1.25),
        fontsize=9.5,
        framealpha=0.9,
        title="Model",
        title_fontsize=10,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}.png / .pdf")


# ── PMI radar charts (Fig 14) ──────────────────────────────────────────────────

def generate_pmi_radars(fig_dir: Path) -> None:
    print("Loading PMI scores …")
    needed = [
        "model_key",
        "pmi_role_fem_raw", "pmi_domain_fem_raw", "pmi_trait_fem_raw",
        "pmi_role_masc_raw", "pmi_domain_masc_raw", "pmi_trait_masc_raw",
    ]
    df = pd.read_csv(PMI_CSV, usecols=needed)
    print(f"  {len(df):,} stories  |  {df['model_key'].nunique()} models")

    models  = sorted(df["model_key"].dropna().unique())
    cmap_fn = plt.cm.get_cmap("tab20", len(models))
    colors  = {m: cmap_fn(i) for i, m in enumerate(models)}

    fem_cols  = [("pmi_role_fem_raw",   "PMI\nRole"),
                 ("pmi_domain_fem_raw", "PMI\nDomain"),
                 ("pmi_trait_fem_raw",  "PMI\nTrait")]
    masc_cols = [("pmi_role_masc_raw",   "PMI\nRole"),
                 ("pmi_domain_masc_raw", "PMI\nDomain"),
                 ("pmi_trait_masc_raw",  "PMI\nTrait")]

    categories = [lbl for _, lbl in fem_cols]

    fem_vals  = {m: [df[df["model_key"] == m][c].mean() for c, _ in fem_cols]
                 for m in models}
    masc_vals = {m: [abs(df[df["model_key"] == m][c].mean()) for c, _ in masc_cols]
                 for m in models}

    draw_radar(
        fem_vals, categories, colors,
        "PMI Bias Radar — Feminine Lexicon\n(Daughter-Stereotyped Language)",
        fig_dir / "fig14_pmi_radar_daughter",
    )
    draw_radar(
        masc_vals, categories, colors,
        "PMI Bias Radar — Masculine Lexicon\n(Son-Stereotyped Language)",
        fig_dir / "fig14_pmi_radar_son",
    )


# ── Semantic radar charts (Fig 15) ────────────────────────────────────────────

def generate_sem_radars(fig_dir: Path) -> None:
    print("Loading semantic scores …")
    needed = [
        "model_key",
        "sem_feminine_traits", "sem_communal_role", "sem_family_domain",
        "sem_masculine_traits", "sem_agentic_role",  "sem_career_domain",
    ]
    df = pd.read_csv(SEM_CSV, usecols=needed)
    print(f"  {len(df):,} stories  |  {df['model_key'].nunique()} models")

    models  = sorted(df["model_key"].dropna().unique())
    cmap_fn = plt.cm.get_cmap("tab20", len(models))
    colors  = {m: cmap_fn(i) for i, m in enumerate(models)}

    fem_cols  = [("sem_feminine_traits", "Sem\nTrait"),
                 ("sem_communal_role",   "Sem\nRole"),
                 ("sem_family_domain",   "Sem\nDomain")]
    masc_cols = [("sem_masculine_traits", "Sem\nTrait"),
                 ("sem_agentic_role",     "Sem\nRole"),
                 ("sem_career_domain",    "Sem\nDomain")]

    categories = [lbl for _, lbl in fem_cols]

    fem_vals  = {m: [df[df["model_key"] == m][c].mean() for c, _ in fem_cols]
                 for m in models}
    masc_vals = {m: [df[df["model_key"] == m][c].mean() for c, _ in masc_cols]
                 for m in models}

    draw_radar(
        fem_vals, categories, colors,
        "Semantic Bias Radar — Feminine Prototypes\n(Daughter-Stereotyped Language)",
        fig_dir / "fig15_sem_radar_daughter",
    )
    draw_radar(
        masc_vals, categories, colors,
        "Semantic Bias Radar — Masculine Prototypes\n(Son-Stereotyped Language)",
        fig_dir / "fig15_sem_radar_son",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating PMI radar charts …")
    generate_pmi_radars(OUT_DIR)

    print("\nGenerating semantic radar charts …")
    generate_sem_radars(OUT_DIR)

    print(f"\nDone — all figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
