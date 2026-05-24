from __future__ import annotations

import argparse
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")



from gender_bias_analysis_new import (
    ALPHA, EPS, FIG_DPI, FIG_EXT, N_BOOT,
    PALETTE_REGION, REGION_ORDER,
    _cohens_d, _save,
    enrich, load_data,
)



DEFAULT_OUT = Path("Narratives3/cultural_bias")
TOKEN_RE    = re.compile(r"\b[a-z]+\b")
PER         = 1000
MFD_PATH    = Path(__file__).parent / "mfd2.dic"

# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="",           help="Path to clean CSV / JSONL")
    p.add_argument("--output-dir",   default=str(DEFAULT_OUT))
    p.add_argument("--n-bootstrap",  type=int, default=N_BOOT)
    p.add_argument("--skip-tfidf",   action="store_true",  help="Skip TF-IDF distinctiveness")
    return p.parse_args()

# MFD 2.0 loader 

def load_mfd(path: Path) -> dict[str, set[str]]:
    """
    Parse an MFD 2.0 LIWC-format .dic file.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = text.split("%")
   

    # Parse header: "1\tcare.virtue" etc. 
    cat_map: dict[str, str] = {}     # "1" → "care.virtue"
    for line in sections[1].splitlines():
        line = line.strip()
        if "\t" in line:
            num, name = line.split("\t", 1)
            cat_map[num.strip()] = name.strip()

    # Parse word entries 
    foundation_words: dict[str, set[str]] = {
        name: set() for name in cat_map.values()
    }
    for line in sections[2].splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        cols = line.split("\t")
        word = cols[0].strip().lower()
        if " " in word:          # skip multi-word phrases
            continue
        for cat_num in cols[1:]:
            cat_num = cat_num.strip()
            if cat_num in cat_map:
                foundation_words[cat_map[cat_num]].add(word)

    return foundation_words


# Cultural lexicons 

def _build_cultural_lexicons() -> dict[str, set[str]]:
    """
    Build cultural lexicons, preferring MFD 2.0 over hand-crafted lists.
    """
    # Liberty supplement: words absent from MFD 2.0 that belong
    # to the missing 6th foundation (Haidt 2012, Ch.8).
    _liberty_supplement: set[str] = {
        "freedom", "liberty", "autonomy", "independent", "independently",
        "independence", "individual", "individually", "privacy", "private",
        "self-reliant", "self-determination", "personal", "personally",
        "sovereign", "sovereignty", "rights", "free", "freely", "liberate",
        "liberated", "liberation", "empower", "empowered", "empowerment",
        "choose", "choice", "choices", "chosen", "initiative", "ambition",
        "aspire", "aspiration", "pursue", "unique", "uniquely", "uniqueness",
    }

    _tradition_fallback: set[str] = {
        "tradition", "custom", "ancestor", "heritage", "ceremony",
        "folklore", "myth", "legend", "ritual", "celebration",
        "festival", "ancient", "generations", "wisdom",
    }

    if MFD_PATH.exists():
        mfd = load_mfd(MFD_PATH)
        # individualism = MFD care+fairness (universal moral vocabulary)
        #               + Liberty supplement (self-orientation vocabulary)
        individualism = (
            mfd.get("care.virtue",    set()) |
            mfd.get("fairness.virtue", set()) |
            _liberty_supplement
        )
        lexicons = {
            "collectivism":  mfd.get("loyalty.virtue",  set()) | mfd.get("authority.virtue", set()),
            "individualism": individualism,
            "religion":      mfd.get("sanctity.virtue", set()),
            "tradition":     _tradition_fallback,
        }
        print(
            f"[MFD 2.0] lexicons loaded — "
            f"collectivism={len(lexicons['collectivism'])} "
            f"(loyalty.v ∪ authority.v)  |  "
            f"individualism={len(lexicons['individualism'])} "
            f"(care.v ∪ fairness.v ∪ liberty-supplement)  |  "
            f"religion={len(lexicons['religion'])} "
            f"(sanctity.v)  |  "
            f"tradition={len(lexicons['tradition'])} (hand-crafted)"
        )
        return lexicons

    # Fallback: original hand-crafted lists 
    print("[lexicons] mfd2.0.dic not found — using hand-crafted fallback lists")
    return {
        "collectivism": {
            "community", "together", "harmony", "duty", "loyalty", "sacrifice",
            "unity", "collective", "clan", "cooperation", "kinship",
            "obligation", "belonging", "bond", "solidarity", "communal",
            "village", "mutual", "tribe", "togetherness", "interdependence",
            "common", "fellowship",
        },
        "individualism": {
            "individual", "freedom", "independence", "personal", "ambition",
            "unique", "private", "autonomy", "pursue", "independently",
            "self-reliant", "choose", "aspire", "initiative",
        },
        "religion": {
            "pray", "prayer", "god", "faith", "bless", "divine",
            "sacred", "holy", "temple", "mosque", "church", "spirit",
            "worship", "heaven", "blessing", "spiritual", "miracle",
        },
        "tradition": _tradition_fallback,
    }


CULTURAL_LEXICONS: dict[str, set[str]] = _build_cultural_lexicons()



# GCI scores 

GCI_SCORES: dict[str, float] = {
    # Americas
    "United States": -1.18,
    "Canada":        -1.10,
    "Mexico":         0.09,
    "Brazil":        -0.06,
    "Argentina":     -0.37,
    "Colombia":       0.15,
    # MENA
    "United Arab Emirates": -0.14,
    "Saudi Arabia":          0.41,
    "Iran":                  0.14,
    "Egypt":                 0.79,
    "Turkey":                0.04,
    "Morocco":               0.96,
    # Sub-Saharan Africa
    "Nigeria":       0.70,
    "Kenya":         0.50,
    "Ethiopia":      0.74,
    "South Africa":  -0.10,
    "Ghana":          0.30,
    # South Asia
    "India":         0.25,
    "Sri Lanka":     0.75,
    "Pakistan":      0.70,
    # East / SE Asia
    "Japan":        -1.18,
    "China":        -0.13,
    "South Korea":  -0.74,
    "Indonesia":     0.67,
    "Thailand":     -0.14,
    "Vietnam":       0.12,
    "Philippines":   0.67,
    # Europe / Oceania
    "Russia":        -0.73,
    "Germany":       -1.35,
    "Greece":        -0.62,
    "Italy":         -0.73,
    "France":        -1.17,
    "Spain":         -0.90,
    "Poland":        -0.50,
    "Australia":     -1.17,
}


REGION_PALETTE = dict(zip(REGION_ORDER, sns.color_palette("tab10", len(REGION_ORDER))))

# Tokenization 

def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower()) if isinstance(text, str) else []

def _norm(count: int, total: int) -> float:
    return (count / total * PER) if total > 0 else 0.0

# Cultural lexicon scoring 

def score_cultural(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-story cultural lexicon counts, per-1000-token rates, and composite proxies."""
    out = df.copy()
    out["_toks"]         = out["story"].apply(_tokenize)
    out["token_count_c"] = out["_toks"].str.len()

    for name, lex in CULTURAL_LEXICONS.items():
        cnt = out["_toks"].apply(lambda t, l=lex: sum(1 for w in t if w in l))
        out[f"{name}_count"]       = cnt
        out[f"{name}_per{PER}"]    = out.apply(
            lambda r, n=name: _norm(r[f"{n}_count"], r["token_count_c"]), axis=1
        )

    # Composite proxies (−1 to +1)
    def idx(a, b):
        ra, rb = out[f"{a}_per{PER}"], out[f"{b}_per{PER}"]
        return (ra - rb) / (ra + rb + EPS)

    out["idv_proxy"] = idx("individualism", "collectivism")

    # Absolute cultural richness score (sum of religion + tradition, normalised)
    out["cultural_richness"] = (
        out[f"religion_per{PER}"] + out[f"tradition_per{PER}"]
    ) / 2.0

    out = out.drop(columns=["_toks"])
    return out


# PMI-weighted cultural scoring

# Condition split: stories from countries with GCI ≥ corpus median are labelled 1
# (more collectivistic).  PMI then encodes which words are over-represented in
# high-GCI (collectivistic) vs low-GCI (individualistic) stories.

def compute_cultural_pmi(
    df: pd.DataFrame,
    condition_col: str,
    min_count: int = 5,
) -> dict[str, float]:
    """
    Compute PMI(w, condition=1) for every token in the corpus.

    PMI(w, cond) = log[ P(w, cond) / (P(w) × P(cond)) ]

    Positive: word over-represented in condition=1 stories 
    Negative: word over-represented in condition=0 stories 

    Words with fewer than min_count total occurrences are excluded to reduce noise.
    """
    pos_counts: Counter = Counter()
    neg_counts: Counter = Counter()

    for _, row in df.iterrows():
        toks = _tokenize(row["story"])
        if row[condition_col] == 1:
            pos_counts.update(toks)
        else:
            neg_counts.update(toks)

    n_pos  = sum(pos_counts.values())
    n_neg  = sum(neg_counts.values())
    n      = n_pos + n_neg
    p_cond = n_pos / n
    _eps   = 1e-9

    pmi: dict[str, float] = {}
    for w in set(pos_counts) | set(neg_counts):
        c_total = pos_counts[w] + neg_counts[w]
        if c_total < min_count:
            continue
        p_w     = c_total / n
        p_w_pos = pos_counts[w] / n       # joint P(w, cond=1)
        if p_w_pos > 0:
            pmi[w] = math.log(p_w_pos / (p_w * p_cond))
        else:
            pmi[w] = math.log(1.0 / (n * p_w * p_cond + _eps))
    return pmi


def score_pmi_cultural(
    df: pd.DataFrame,
    pmi_collectivism: dict[str, float],
) -> pd.DataFrame:
    """
    PMI-weighted collectivism score for each story.

    pmi_collectivism_score = Σ PMI(w, high_GCI) for w ∈ story ∩ (individualism ∪ collectivism)
                             / n_tokens × 1000

    Collectivism words have positive PMI(w, high_GCI); individualism words
    have negative PMI(w, high_GCI).

    Positive pmi_collectivism_score: story linguistically resembles a high-GCI culture.
    Negative pmi_collectivism_score: story linguistically resembles a low-GCI / individualistic culture.
    """
    out      = df.copy()
    toks_col = out["story"].apply(_tokenize)

    collect_lex = CULTURAL_LEXICONS["individualism"] | CULTURAL_LEXICONS["collectivism"]

    def _pmi_score(tokens: list[str],
                   lex: set[str],
                   weights: dict[str, float]) -> float:
        n = max(len(tokens), 1)
        return sum(weights.get(w, 0.0) for w in tokens if w in lex) / n * PER

    out["pmi_collectivism_score"] = toks_col.apply(
        lambda t: _pmi_score(t, collect_lex, pmi_collectivism)
    )
    return out


# Country-level aggregation 

PROXIES = [
    "idv_proxy", "cultural_richness",
    f"religion_per{PER}", f"tradition_per{PER}",
    "pmi_collectivism_score",          # PMI-weighted collectivism (GCI-split)
]


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of cultural proxies per country, joined with GCI scores."""
    agg = (
        df.groupby("country")[PROXIES]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # Join GCI scores
    gci = pd.DataFrame(
        list(GCI_SCORES.items()), columns=["country", "gci_score"]
    )
    agg = agg.merge(gci, on="country", how="left")

    # Join region
    if "country_region" in df.columns:
        region_map = (df[["country", "country_region"]]
                      .drop_duplicates()
                      .set_index("country")["country_region"]
                      .to_dict())
        agg["region"] = agg["country"].map(region_map)

    return agg

# GCI correlation

def gci_correlation(
    summary: pd.DataFrame,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """
    Spearman ρ between each story-level proxy and country-level GCI score.
    Bootstrap 95% CI via Fisher z-transform.

    idv_proxy is expected to correlate negatively with GCI (IDV high = individualistic,
    GCI high = collectivistic — opposite directions).
    pmi_collectivism_score is expected to correlate positively with GCI.
    """
    pairs = [
        ("idv_proxy_mean",             "gci_score", "idv_proxy",             "GCI"),
        ("pmi_collectivism_score_mean","gci_score", "pmi_collectivism_score", "GCI (PMI)"),
    ]
    rows = []
    for proxy_col, gci_col, proxy_label, gci_label in pairs:
        sub = summary[[proxy_col, gci_col]].dropna()
        if len(sub) < 4:
            continue
        x = sub[proxy_col].values
        y = sub[gci_col].values

        rho, p = stats.spearmanr(x, y)

        rng    = np.random.default_rng(42)
        n      = len(x)
        z_boot = []
        for _ in range(n_boot):
            idx   = rng.integers(0, n, n)
            r_b, _ = stats.spearmanr(x[idx], y[idx])
            r_b = np.clip(r_b, -0.9999, 0.9999)
            z_boot.append(0.5 * np.log((1 + r_b) / (1 - r_b)))
        z_lo, z_hi = np.percentile(z_boot, [2.5, 97.5])
        ci_lo = float(np.tanh(z_lo))
        ci_hi = float(np.tanh(z_hi))

        rows.append({
            "proxy":       proxy_label,
            "external":    gci_label,
            "spearman_r":  round(float(rho), 4),
            "p_value":     round(float(p),   6),
            "ci_low":      round(ci_lo, 4),
            "ci_high":     round(ci_hi, 4),
            "n_countries": len(sub),
        })
    return pd.DataFrame(rows)

# Model homogeneity 

def model_homogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each country: std of idv_proxy across model_family values.
    Low std = all models portray the country similarly = cultural flattening.
    """
    rows = []
    for country, sub in df.groupby("country"):
        fam_means = sub.groupby("model_family")["idv_proxy"].mean()
        if len(fam_means) < 2:
            continue
        region = sub["country_region"].iloc[0] if "country_region" in sub.columns else "Unknown"
        rows.append({
            "country":          country,
            "region":           region,
            "idv_std":          round(float(fam_means.std()), 4),
            "idv_range":        round(float(fam_means.max() - fam_means.min()), 4),
            "n_model_families": int(len(fam_means)),
        })
    return pd.DataFrame(rows).sort_values("idv_std")

# TF-IDF distinctiveness 

def tfidf_distinctiveness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine distance of each country's mean TF-IDF vector from the global centroid.
    Low distance = culturally generic / flattened portrayal.
    Requires scikit-learn.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    stories = df["story"].fillna("").tolist()
    labels  = df["country"].tolist()

    print("  Fitting TF-IDF (this may take ~30 s)…")
    vec    = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X      = vec.fit_transform(stories)  # (n_stories, vocab)

    # Global centroid — convert to 2-D numpy array (sparse .mean returns np.matrix)
    global_centroid = np.asarray(X.mean(axis=0))  # (1, vocab)

    # Country centroids
    countries = sorted(set(labels))
    rows = []
    country_arr = np.array(labels)
    for country in countries:
        mask = country_arr == country
        if mask.sum() < 2:
            continue
        c_centroid  = np.asarray(X[mask].mean(axis=0))  # (1, vocab)
        sim         = float(cosine_similarity(c_centroid, global_centroid)[0, 0])
        region = df.loc[df["country"] == country, "country_region"].iloc[0] \
                 if "country_region" in df.columns else "Unknown"
        rows.append({
            "country":         country,
            "region":          region,
            "cosine_sim":      round(sim, 4),
            "distinctiveness": round(1 - sim, 4),  # higher = more distinctive
            "n_stories":       int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("distinctiveness", ascending=False)



def _scatter_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    proxy_col: str,
    ext_col: str,
    rho: float,
    p_val: float,
    ylabel: str,
    show_legend: bool = False,
) -> None:
    """Draw a single scatter panel (country points, OLS line, correlation box)."""
    sub = summary[[proxy_col, ext_col, "country", "region"]].dropna()
    if sub.empty:
        return

    for region in REGION_ORDER:
        mask = sub["region"] == region
        if not mask.any():
            continue
        ax.scatter(
            sub.loc[mask, ext_col],
            sub.loc[mask, proxy_col],
            color=REGION_PALETTE.get(region, "#999"),
            label=region, s=65, zorder=3, alpha=0.85,
        )

    for _, row in sub.iterrows():
        ax.annotate(
            row["country"],
            (row[ext_col], row[proxy_col]),
            fontsize=6.5, alpha=0.75,
            xytext=(3, 3), textcoords="offset points",
        )

    x_arr = sub[ext_col].values.astype(float)
    y_arr = sub[proxy_col].values.astype(float)
    m, b  = np.polyfit(x_arr, y_arr, 1)
    x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#555", linewidth=1.5,
            linestyle="--", zorder=2)

    sig_label = (
        "***" if p_val < 0.001 else
        "**"  if p_val < 0.01  else
        "*"   if p_val < 0.05  else
        "n.s."
    )
    ax.text(
        0.04, 0.96,
        f"Spearman ρ = {rho:+.3f} {sig_label}  (n={len(sub)})",
        transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)  # GCI mean = 0
    ax.set_ylabel(ylabel, fontsize=10)

    if show_legend:
        ax.legend(title="Region", fontsize=7.5, loc="lower right",
                  framealpha=0.85, ncol=2)


# Figure C1: GCI comparison (lexical IDV proxy vs PMI collectivism score) 

def fig_gci_comparison(
    summary: pd.DataFrame,
    gci_corr: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    Side-by-side scatter of lexical IDV proxy (left) and PMI collectivism score
    (right) both plotted against GCI (Pelham et al. 2022).

    Left panel  : raw lexical signed ratio (individualism − collectivism).
                  Expected direction: negative ρ with GCI (IDV ↑ = individualistic,
                  GCI ↑ = collectivistic).
    Right panel : PMI-weighted score trained on GCI median split.
                  Expected direction: positive ρ with GCI.

    X-axis reference line at GCI = 0 (global standardised mean).
    """
    def _get_rho(proxy_label, ext_label):
        row = gci_corr[
            (gci_corr["proxy"] == proxy_label) &
            (gci_corr["external"] == ext_label)
        ]
        if row.empty:
            return 0.0, 1.0
        return float(row["spearman_r"].iloc[0]), float(row["p_value"].iloc[0])

    rho_lex, p_lex = _get_rho("idv_proxy",             "GCI")
    rho_pmi, p_pmi = _get_rho("pmi_collectivism_score", "GCI (PMI)")

    xlabel = "GCI score (Pelham et al. 2022)\nhigher = more collectivistic"

    fig, (ax_lex, ax_pmi) = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

    _scatter_panel(ax_lex, summary, "idv_proxy_mean", "gci_score",
                   rho_lex, p_lex,
                   ylabel="Lexical IDV proxy\n(individ. − collect. signed ratio)",
                   show_legend=True)
    ax_lex.set_xlabel(xlabel, fontsize=10)
    ax_lex.set_title(
        "Lexical individualism proxy\n(signed word-count ratio, no corpus weighting)",
        fontsize=11, fontweight="bold",
    )

    _scatter_panel(ax_pmi, summary, "pmi_collectivism_score_mean", "gci_score",
                   rho_pmi, p_pmi,
                   ylabel="PMI collectivism score\n(Σ PMI(w, high-GCI) per 1000 tok.)",
                   show_legend=False)
    ax_pmi.set_xlabel(xlabel, fontsize=10)
    ax_pmi.set_title(
        "PMI-weighted collectivism score\n(corpus-driven word weights, GCI median split)",
        fontsize=11, fontweight="bold",
    )

    fig.suptitle(
        "Do LLM Stories Reflect Collectivism? Lexical proxy vs PMI score vs GCI\n"
        "One point per country — GCI: Pelham, Hardin, Murray, Shimizu & Vandello (2022)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_c1_gci_comparison", fig_dir)
    print(f"  Saved fig_c1_gci_comparison  "
          f"(ρ_lexical={rho_lex:+.3f}, ρ_pmi={rho_pmi:+.3f})")

# Figure C3: Cultural dimension heatmap 

def fig_cultural_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    """
    Heatmap: country × cultural dimension, sorted by region.
    Shows which regions score high on collectivism, religion, and tradition.
    Values are z-score normalised per column for comparability.
    """
    dims = ["idv_proxy_mean",
            f"religion_per{PER}_mean", f"tradition_per{PER}_mean",
            "pmi_collectivism_score_mean"]
    labels = {
        "idv_proxy_mean":                  "IDV proxy\n(individ.−collect.)",
        f"religion_per{PER}_mean":         "Religion\n(per 1000 tok.)",
        f"tradition_per{PER}_mean":        "Tradition\n(per 1000 tok.)",
        "pmi_collectivism_score_mean":     "PMI collect.\nscore",
    }

    # Sort countries by region then name
    countries_sorted: list[str] = []
    region_sizes:     list[int] = []
    for region in REGION_ORDER:
        cs = sorted(
            summary.loc[summary["region"] == region, "country"].tolist()
        )
        countries_sorted.extend(cs)
        region_sizes.append(len(cs))

    plot_df = (summary
               .set_index("country")
               .reindex(countries_sorted)[dims]
               .rename(columns=labels))

    # Z-score normalise each column
    plot_z = (plot_df - plot_df.mean()) / (plot_df.std() + EPS)

    n_c = len(countries_sorted)
    fig, ax = plt.subplots(figsize=(8, max(8, n_c * 0.42 + 2)))
    sns.heatmap(
        plot_z, cmap="RdBu_r", center=0, linewidths=0.25,
        ax=ax, mask=plot_z.isna(),
        cbar_kws={"label": "Z-score"},
        xticklabels=True, yticklabels=True,
    )

    # Region boundary lines + right-side labels
    cumulative = np.cumsum(region_sizes)
    for boundary in cumulative[:-1]:
        ax.axhline(boundary, color="black", linewidth=1.5, alpha=0.8)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    cum = 0
    for region, size in zip(REGION_ORDER, region_sizes):
        if size > 0:
            mid = (cum + size / 2) / n_c
            ax2.text(1.01, mid, region,
                     transform=ax2.transAxes,
                     va="center", fontsize=7.5, color="#333")
        cum += size

    ax.set_title("Cultural Vocabulary in LLM-Generated Stories by Country\n"
                 "(z-score normalised per dimension)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Cultural Dimension", fontsize=11)
    ax.set_ylabel("Country (grouped by region)", fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    _save(fig, "fig_c3_cultural_heatmap", fig_dir)
    print("  Saved fig_c3_cultural_heatmap")

# Figure C4: Model homogeneity 

def fig_homogeneity(hom: pd.DataFrame, fig_dir: Path) -> None:
    """
    Horizontal bar chart of within-country IDV-proxy std across model families,
    sorted ascending (least homogeneous at top, most at bottom).
    Countries with low std are culturally flattened across all models.
    """
    if hom.empty:
        return

    # Sort by std ascending so most-flattened countries are at bottom
    plot_df = hom.sort_values("idv_std", ascending=False).reset_index(drop=True)
    y      = np.arange(len(plot_df))
    colors = [REGION_PALETTE.get(r, "#999") for r in plot_df["region"]]

    fig, ax = plt.subplots(figsize=(9, max(6, len(plot_df) * 0.38 + 1.5)))
    ax.barh(y, plot_df["idv_std"], height=0.7, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["country"], fontsize=8)
    ax.set_xlabel(
        "Std of IDV proxy across model families\n(low = all models agree = cultural flattening)",
        fontsize=10,
    )
    ax.set_title(
        "Cultural Homogeneity Across Models per Country\n"
        "(how much do models disagree on cultural framing?)",
        fontsize=12, fontweight="bold",
    )

    # Legend for regions
    handles = [
        Patch(color=REGION_PALETTE[r], label=r)
        for r in REGION_ORDER if r in plot_df["region"].values
    ]
    ax.legend(handles=handles, title="Region", fontsize=8,
              loc="lower right", framealpha=0.85)

    _save(fig, "fig_c4_homogeneity", fig_dir)
    print("  Saved fig_c4_homogeneity")

# Main 

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    res_dir = out_dir / "results"
    fig_dir = out_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and score (lexical + PMI)
    raw    = load_data(args.input)
    df     = enrich(raw)
    scored = score_cultural(df)
    print(f"\nAnalysing {len(scored):,} stories across "
          f"{scored['country'].nunique()} countries, "
          f"{scored['model_family'].nunique()} model families")

    # 1b. PMI-weighted collectivism scoring (GCI median split)
    print("\n PMI-weighted collectivism scoring (GCI median split)")

    gci_vals = [GCI_SCORES.get(c) for c in scored["country"]]
    scored["gci_score"] = gci_vals
    known_gci = [v for v in GCI_SCORES.values()]
    gci_median = float(np.median(known_gci))

    scored["is_high_gci"] = (
        scored["gci_score"]
        .apply(lambda x: 1 if pd.notna(x) and x >= gci_median else 0)
        .astype(int)
    )
    print(f"  GCI median split: ≥{gci_median:.2f} = high-collectivism  "
          f"({scored['is_high_gci'].sum():,} high-GCI stories)")

    pmi_collect_weights = compute_cultural_pmi(scored, "is_high_gci")
    print(f"  PMI collectivism vocab: {len(pmi_collect_weights):,} tokens")

    scored = score_pmi_cultural(scored, pmi_collect_weights)
    scored.to_csv(res_dir / "story_level_cultural.csv", index=False)
    print(f"  Story-level scores saved ({len(scored):,} rows, "
          f"added: pmi_collectivism_score)")

    # 2. Country-level summary
    print("\n Country summary ")
    summary = country_summary(scored)
    summary.to_csv(res_dir / "country_summary.csv", index=False)
    print(summary[["country", "region", "idv_proxy_mean",
                   "pmi_collectivism_score_mean", "gci_score"]].round(3).to_string(index=False))

    # 3. GCI correlation
    print("\n GCI correlations ")
    gci_corr = gci_correlation(summary, n_boot=args.n_bootstrap)
    gci_corr.to_csv(res_dir / "gci_correlation.csv", index=False)
    print(gci_corr.to_string(index=False))

    # 4. Model homogeneity ")
    hom = model_homogeneity(scored)
    hom.to_csv(res_dir / "model_homogeneity.csv", index=False)
    print(f"  Most homogeneous (lowest IDV std): {hom.iloc[0]['country']} "
          f"({hom.iloc[0]['idv_std']:.4f})")
    print(f"  Most varied (highest IDV std):     {hom.iloc[-1]['country']} "
          f"({hom.iloc[-1]['idv_std']:.4f})")

    # 5. TF-IDF distinctiveness 
    if not args.skip_tfidf:
        print("\n TF-IDF distinctiveness ")
        tfidf = tfidf_distinctiveness(scored)
        tfidf.to_csv(res_dir / "tfidf_distinctiveness.csv", index=False)
        print(f"  Most distinctive  : {tfidf.iloc[0]['country']} "
              f"({tfidf.iloc[0]['distinctiveness']:.4f})")
        print(f"  Least distinctive : {tfidf.iloc[-1]['country']} "
              f"({tfidf.iloc[-1]['distinctiveness']:.4f})")
    else:
        print("\n TF-IDF skipped (--skip-tfidf) ")

    # 6. Figures
    print("\n Generating figures ")

    # Fig C1: lexical IDV proxy (left) vs PMI collectivism score (right) vs GCI
    fig_gci_comparison(summary, gci_corr, fig_dir)

    fig_cultural_heatmap(summary, fig_dir)
    fig_homogeneity(hom, fig_dir)

    # Pull correlation stats for summary printout
    def _get_corr(proxy, ext):
        row = gci_corr[(gci_corr["proxy"] == proxy) & (gci_corr["external"] == ext)]
        if row.empty:
            return 0.0, 1.0
        return float(row["spearman_r"].iloc[0]), float(row["p_value"].iloc[0])

    rho_lex, p_lex = _get_corr("idv_proxy",             "GCI")
    rho_pmi, p_pmi = _get_corr("pmi_collectivism_score", "GCI (PMI)")

    # 7. Summary
    print("CULTURAL BIAS ANALYSIS: COMPLETE")

    print(f"  Stories analysed : {len(scored):,}")
    print(f"  Countries        : {scored['country'].nunique()}")
    print()
    print(f"  {'Metric':<30}  {'GCI ρ':>8}  {'GCI p':>8}")
    print(f"  {'Lexical IDV proxy':<30}  {rho_lex:>+8.3f}  {p_lex:>8.4f}")
    print(f"  {'PMI collectivism score':<30}  {rho_pmi:>+8.3f}  {p_pmi:>8.4f}")
    if not hom.empty:
        print(f"  Most flattened   : {hom.iloc[0]['country']}")
    print(f"\n  Results : {res_dir}")
    print(f"  Figures  : {fig_dir}")


if __name__ == "__main__":
    main()
