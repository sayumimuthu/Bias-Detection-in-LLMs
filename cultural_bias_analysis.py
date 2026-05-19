from __future__ import annotations

import argparse
import re
import warnings
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

# CLI 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="",           help="Path to clean CSV / JSONL")
    p.add_argument("--output-dir",   default=str(DEFAULT_OUT))
    p.add_argument("--n-bootstrap",  type=int, default=N_BOOT)
    p.add_argument("--skip-tfidf",   action="store_true",  help="Skip TF-IDF distinctiveness")
    return p.parse_args()

# Cultural lexicons 
# Sources: Hofstede et al. (2010); Triandis (1995) Individualism & Collectivism;
#          Minkov (2011). Categories are mutually exclusive.

CULTURAL_LEXICONS: dict[str, set[str]] = {

    # Group orientation, interdependence, duty to collective
    # (High collectivism = low IDV)
    "collectivism": {
        "community", "together", "harmony", "duty", "loyalty", "sacrifice",
        "unity", "collective", "clan", "cooperation", "kinship",
        "obligation", "belonging", "bond", "solidarity", "communal",
        "village", "mutual", "tribe", "togetherness", "interdependence",
        "common", "fellowship",
    },

    # Self-orientation, personal freedom, private goals
    # (High individualism = high IDV)
    "individualism": {
        "individual", "freedom", "independence", "personal", "ambition",
        "unique", "private", "autonomy", "pursue", "independently",
        "self-reliant", "choose", "aspire", "initiative",
    },

    # Deference to power, hierarchy, authority figures
    # (High PDI)
    "authority": {
        "obey", "command", "rank", "hierarchy", "rule", "superior",
        "reverence", "submit", "obedience", "discipline", "elder",
        "lord", "authority", "master", "subordinate",
    },

    # Equality, questioning hierarchy, democratic norms
    # (Low PDI)
    "egalitarian": {
        "equal", "fair", "debate", "democratic", "challenge",
        "peer", "voice", "rights", "question", "negotiate",
    },

    # Presence of religious / spiritual framing
    "religion": {
        "pray", "prayer", "god", "faith", "bless", "divine",
        "sacred", "holy", "temple", "mosque", "church", "spirit",
        "worship", "heaven", "blessing", "spiritual", "miracle",
    },

    # Cultural heritage, ancestral wisdom, ceremonies
    "tradition": {
        "tradition", "custom", "ancestor", "heritage", "ceremony",
        "folklore", "myth", "legend", "ritual", "celebration",
        "festival", "ancient", "generations", "wisdom",
    },
}

# Hofstede scores for all 35 study countries:
# PDI = Power Distance Index (high = more hierarchy)
# IDV = Individualism (high = more individualistic; low = more collectivist)
# Source: Hofstede, Hofstede & Minkov (2010).
# Values marked (*) are estimates from Minkov (2011) / regional neighbours
# where the original survey did not include the country directly.

HOFSTEDE: dict[str, dict[str, int]] = {
    # Americas
    "United States":        {"PDI": 40, "IDV": 91},
    "Canada":               {"PDI": 39, "IDV": 80},
    "Mexico":               {"PDI": 81, "IDV": 30},
    "Brazil":               {"PDI": 69, "IDV": 38},
    "Argentina":            {"PDI": 49, "IDV": 46},
    "Colombia":             {"PDI": 67, "IDV": 13},
    # MENA
    "United Arab Emirates": {"PDI": 90, "IDV": 25},  # (*)
    "Saudi Arabia":         {"PDI": 95, "IDV": 25},  # (*)
    "Iran":                 {"PDI": 58, "IDV": 41},
    "Egypt":                {"PDI": 70, "IDV": 25},  # (*)
    "Turkey":               {"PDI": 66, "IDV": 37},
    "Morocco":              {"PDI": 70, "IDV": 46},
    # Sub-Saharan Africa
    "Nigeria":              {"PDI": 80, "IDV": 20},
    "Kenya":                {"PDI": 70, "IDV": 27},  # (*)
    "Ethiopia":             {"PDI": 64, "IDV": 20},  # (*)
    "South Africa":         {"PDI": 49, "IDV": 65},
    "Ghana":                {"PDI": 80, "IDV": 15},  # (*)
    # South Asia
    "India":                {"PDI": 77, "IDV": 48},
    "Sri Lanka":            {"PDI": 80, "IDV": 35},  # (*)
    "Pakistan":             {"PDI": 55, "IDV": 14},
    # East / SE Asia
    "Japan":                {"PDI": 54, "IDV": 46},
    "China":                {"PDI": 80, "IDV": 20},
    "South Korea":          {"PDI": 60, "IDV": 18},
    "Indonesia":            {"PDI": 78, "IDV": 14},
    "Thailand":             {"PDI": 64, "IDV": 20},
    "Vietnam":              {"PDI": 70, "IDV": 20},
    "Philippines":          {"PDI": 94, "IDV": 32},
    # Europe / Oceania
    "Russia":               {"PDI": 93, "IDV": 39},
    "Germany":              {"PDI": 35, "IDV": 67},
    "Greece":               {"PDI": 60, "IDV": 35},
    "Italy":                {"PDI": 50, "IDV": 76},
    "France":               {"PDI": 68, "IDV": 71},
    "Spain":                {"PDI": 57, "IDV": 51},
    "Poland":               {"PDI": 68, "IDV": 60},
    "Australia":            {"PDI": 36, "IDV": 90},
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
    out["pdi_proxy"] = idx("authority",     "egalitarian")

    # Absolute cultural richness score (sum of religion + tradition, normalised)
    out["cultural_richness"] = (
        out[f"religion_per{PER}"] + out[f"tradition_per{PER}"]
    ) / 2.0

    out = out.drop(columns=["_toks"])
    return out

# Country-level aggregation 

PROXIES = ["idv_proxy", "pdi_proxy", "cultural_richness",
           f"religion_per{PER}", f"tradition_per{PER}"]


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of cultural proxies per country, joined with Hofstede scores."""
    agg = (
        df.groupby("country")[PROXIES]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # Join Hofstede
    hof = pd.DataFrame(HOFSTEDE).T.reset_index()
    hof.columns = ["country", "hofstede_PDI", "hofstede_IDV"]
    agg = agg.merge(hof, on="country", how="left")

    # Join region
    if "country_region" in df.columns:
        region_map = (df[["country", "country_region"]]
                      .drop_duplicates()
                      .set_index("country")["country_region"]
                      .to_dict())
        agg["region"] = agg["country"].map(region_map)

    return agg

# Hofstede correlation 

def hofstede_correlation(
    summary: pd.DataFrame,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """
    For each (proxy, Hofstede dimension) pair:
      Spearman ρ, p-value, bootstrap 95 % CI (fisher z-transform).
    """
    pairs = [
        ("idv_proxy_mean", "hofstede_IDV", "idv_proxy",  "IDV"),
        ("pdi_proxy_mean", "hofstede_PDI", "pdi_proxy",  "PDI"),
    ]
    rows = []
    for proxy_col, hofstede_col, proxy_label, hof_label in pairs:
        sub = summary[[proxy_col, hofstede_col]].dropna()
        if len(sub) < 4:
            continue
        x = sub[proxy_col].values
        y = sub[hofstede_col].values

        rho, p = stats.spearmanr(x, y)

        # Bootstrap CI via Fisher z-transform
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
            "proxy":      proxy_label,
            "hofstede":   hof_label,
            "spearman_r": round(float(rho), 4),
            "p_value":    round(float(p),   6),
            "ci_low":     round(ci_lo, 4),
            "ci_high":    round(ci_hi, 4),
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

# Figure C1/C2: Hofstede scatter plots 

def fig_hofstede_scatter(
    summary: pd.DataFrame,
    proxy_col: str,
    hofstede_col: str,
    rho: float,
    p_val: float,
    fname: str,
    xlabel: str,
    ylabel: str,
    title: str,
    fig_dir: Path,
) -> None:
    """
    Scatter of lexical proxy vs Hofstede score, one point per country.
    Points coloured by region; country name labels; OLS regression line.
    """
    sub = summary[[proxy_col, hofstede_col, "country", "region"]].dropna()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    regions = sub["region"].unique()
    for region in REGION_ORDER:
        if region not in regions:
            continue
        mask = sub["region"] == region
        ax.scatter(
            sub.loc[mask, hofstede_col],
            sub.loc[mask, proxy_col],
            color=REGION_PALETTE.get(region, "#999"),
            label=region, s=70, zorder=3, alpha=0.85,
        )

    # Country labels (offset to reduce overlap)
    for _, row in sub.iterrows():
        ax.annotate(
            row["country"],
            (row[hofstede_col], row[proxy_col]),
            fontsize=6.5, alpha=0.75,
            xytext=(3, 3), textcoords="offset points",
        )

    # OLS regression line
    x_arr = sub[hofstede_col].values.astype(float)
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
        0.04, 0.95,
        f"Spearman ρ = {rho:+.3f} {sig_label}  (n={len(sub)})",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title="Region", fontsize=8, loc="lower right",
              framealpha=0.85, ncol=2)

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axvline(50, color="black", linewidth=0.6, linestyle=":", alpha=0.5)

    _save(fig, fname, fig_dir)
    print(f"  Saved {fname}")

# Figure C3: Cultural dimension heatmap 

def fig_cultural_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    """
    Heatmap: country × cultural dimension, sorted by region.
    Shows which regions score high on collectivism, authority, religion, tradition.
    Values are z-score normalised per column for comparability.
    """
    dims = ["idv_proxy_mean", "pdi_proxy_mean",
            f"religion_per{PER}_mean", f"tradition_per{PER}_mean"]
    labels = {
        "idv_proxy_mean":             "IDV proxy\n(individ.−collect.)",
        "pdi_proxy_mean":             "PDI proxy\n(auth.−egalit.)",
        f"religion_per{PER}_mean":    "Religion\n(per 1000 tok.)",
        f"tradition_per{PER}_mean":   "Tradition\n(per 1000 tok.)",
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

    # 1. Load and score 
    raw     = load_data(args.input)
    df      = enrich(raw)
    scored  = score_cultural(df)
    print(f"\nAnalysing {len(scored):,} stories across "
          f"{scored['country'].nunique()} countries, "
          f"{scored['model_family'].nunique()} model families")

    scored.to_csv(res_dir / "story_level_cultural.csv", index=False)
    print(f"  Story-level scores saved ({len(scored):,} rows)")

    # 2. Country-level summary 
    print("\n Country summary ")
    summary = country_summary(scored)
    summary.to_csv(res_dir / "country_summary.csv", index=False)
    print(summary[["country", "region", "idv_proxy_mean", "pdi_proxy_mean",
                   "hofstede_IDV", "hofstede_PDI"]].round(3).to_string(index=False))

    # 3. Hofstede correlation 
    print("\n Hofstede correlations ")
    hof_corr = hofstede_correlation(summary, n_boot=args.n_bootstrap)
    hof_corr.to_csv(res_dir / "hofstede_correlation.csv", index=False)
    print(hof_corr.to_string(index=False))

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

    # Extract correlation stats for annotation
    def _get_corr(proxy, dim):
        row = hof_corr[(hof_corr["proxy"] == proxy) & (hof_corr["hofstede"] == dim)]
        if row.empty:
            return 0.0, 1.0
        return float(row["spearman_r"].iloc[0]), float(row["p_value"].iloc[0])

    rho_idv, p_idv = _get_corr("idv_proxy", "IDV")
    rho_pdi, p_pdi = _get_corr("pdi_proxy", "PDI")

    fig_hofstede_scatter(
        summary,
        proxy_col="idv_proxy_mean", hofstede_col="hofstede_IDV",
        rho=rho_idv, p_val=p_idv,
        fname="fig_c1_hofstede_idv",
        xlabel="Hofstede IDV score  (higher = more individualistic culture)",
        ylabel="IDV proxy  (higher = more individualism vocabulary in stories)",
        title="Do LLM Stories Reflect Cultural Individualism?\n"
              "Lexical IDV proxy vs Hofstede IDV score per country",
        fig_dir=fig_dir,
    )

    fig_hofstede_scatter(
        summary,
        proxy_col="pdi_proxy_mean", hofstede_col="hofstede_PDI",
        rho=rho_pdi, p_val=p_pdi,
        fname="fig_c2_hofstede_pdi",
        xlabel="Hofstede PDI score  (higher = more hierarchical culture)",
        ylabel="PDI proxy  (higher = more authority vocabulary in stories)",
        title="Do LLM Stories Reflect Cultural Power Distance?\n"
              "Lexical PDI proxy vs Hofstede PDI score per country",
        fig_dir=fig_dir,
    )

    fig_cultural_heatmap(summary, fig_dir)
    fig_homogeneity(hom, fig_dir)

    # 7. Summary 

    print("CULTURAL BIAS ANALYSIS: COMPLETE")

    print(f"  Stories analysed   : {len(scored):,}")
    print(f"  Countries          : {scored['country'].nunique()}")
    print(f"  Hofstede IDV  ρ    : {rho_idv:+.3f}  (p={p_idv:.4f})")
    print(f"  Hofstede PDI  ρ    : {rho_pdi:+.3f}  (p={p_pdi:.4f})")
    if not hom.empty:
        print(f"  Most flattened     : {hom.iloc[0]['country']}")
    print(f"\n  Results : {res_dir}")
    print(f"  Figures  : {fig_dir}")


if __name__ == "__main__":
    main()
