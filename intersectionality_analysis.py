from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


from gender_bias_analysis_new import (
    ALPHA, EPS, FIG_DPI, FIG_EXT, N_BOOT,
    PALETTE_GENDER, PALETTE_ROLE, REGION_ORDER,
    _bootstrap_ci, _cliffs_delta, _cohens_d, _save,
    enrich, load_data, score_lexicon,
    compute_pmi_weights, score_pmi, PMI_METRICS, _DIM_INFO,
)


DEFAULT_OUT   = Path("Narratives3/intersectionality")
FOCAL_METRICS = [
    "stereotype_score",
    "trait_bias_index",
    "role_bias_index",
    "marker_bias_index",
]

# CLI 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="",           help="Path to clean CSV / JSONL")
    p.add_argument("--output-dir",   default=str(DEFAULT_OUT))
    p.add_argument("--n-bootstrap",  type=int, default=N_BOOT)
    return p.parse_args()

# gender gap + stats per group 

def _gap_table(
    df: pd.DataFrame,
    group_col: str,
    val_col: str = "stereotype_score",
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """
    For each level of group_col compute:
      gap = mean(daughter) − mean(son), 95 % bootstrap CI,
      Cohen's d, Cliff's delta, Welch t p-value, FDR correction.
    """
    rows = []
    for grp, sub in df.groupby(group_col):
        d = sub[sub["child_label"] == "daughter"][val_col].dropna().values
        s = sub[sub["child_label"] == "son"][val_col].dropna().values
        if len(d) < 2 or len(s) < 2:
            continue
        gap = float(d.mean() - s.mean())
        rng = np.random.default_rng(42)
        boots = [
            rng.choice(d, len(d)).mean() - rng.choice(s, len(s)).mean()
            for _ in range(n_boot)
        ]
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
        t, p = stats.ttest_ind(d, s, equal_var=False)
        rows.append({
            group_col:       grp,
            "n_daughter":    int(len(d)),
            "n_son":         int(len(s)),
            "daughter_mean": float(d.mean()),
            "son_mean":      float(s.mean()),
            "gap":           gap,
            "ci_low":        lo,
            "ci_high":       hi,
            "cohens_d":      _cohens_d(s, d),
            "cliffs_delta":  _cliffs_delta(s, d),
            "t_stat":        float(t),
            "p_value":       float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        _, p_fdr, _, _ = multipletests(out["p_value"], method="fdr_bh")
        out["p_fdr"]   = p_fdr
        out["sig_fdr"] = p_fdr < ALPHA
    return out

# Analysis 1: Interaction OLS 

def run_interaction_ols(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS: metric ~ is_daughter * C(country_region) + model_family
                + person_gender_role + log_wc

    Reference: North America (Treatment baseline).
    Extracts both the main effect (gender gap in North America) and the
    interaction terms (additional gap in each other region).
    """
    df = df.copy()
    df["log_wc"] = np.log1p(df["word_count"].fillna(0).clip(lower=1))
    required = ["is_daughter", "country_region", "model_family",
                "person_gender_role", "log_wc"]
    df = df.dropna(subset=required)

    available = [m for m in FOCAL_METRICS if m in df.columns]
    rows: list[dict] = []

    for metric in available:
        data = df[required + [metric]].dropna()
        if len(data) < 50:
            continue
        try:
            formula = (
                f"{metric} ~ "
                "is_daughter * C(country_region, Treatment('North America'))"
                " + C(model_family, Treatment('llama'))"
                " + C(person_gender_role, Treatment('neutral'))"
                " + log_wc"
            )
            fit = smf.ols(formula, data=data).fit()
            ci  = fit.conf_int()

            for term, coef in fit.params.items():
                is_main        = (term == "is_daughter")
                is_interaction = ("is_daughter:" in term and "country_region" in term)
                if not (is_main or is_interaction):
                    continue

                region = (
                    "North America (baseline)" if is_main
                    else term.split("[T.")[-1].rstrip("]")
                )
                rows.append({
                    "metric":      metric,
                    "region":      region,
                    "term":        term,
                    "coef":        float(coef),
                    "ci_low":      float(ci.loc[term, 0]),
                    "ci_high":     float(ci.loc[term, 1]),
                    "p_value":     float(fit.pvalues[term]),
                    "r_squared":   float(fit.rsquared),
                    "interaction": is_interaction,
                })
        except Exception as e:
            print(f"  OLS failed for {metric}: {e}")

    out = pd.DataFrame(rows)
    if not out.empty:
        _, p_fdr, _, _ = multipletests(out["p_value"], method="fdr_bh")
        out["p_fdr"]   = p_fdr
        out["sig_fdr"] = p_fdr < ALPHA
    return out

# Figure I1: Interaction forest plot 

def fig_interaction_forest(ols: pd.DataFrame, fig_dir: Path) -> None:
    """
    Faceted forest plot: one panel per focal metric.
    Shows is_daughter:region interaction coefficients (vs North America baseline).
    Positive = daughters receive a stronger stereotype in that region.
    """
    interactions = ols[ols["interaction"]].copy()
    if interactions.empty:
        print("  fig_i1 skipped: no interaction terms found")
        return

    metrics_present = [m for m in FOCAL_METRICS if m in interactions["metric"].unique()]
    n_panels        = len(metrics_present)
    regions_present = [r for r in REGION_ORDER if r in interactions["region"].values]

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, max(5, len(regions_present) * 0.65)),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_present):
        sub = (
            interactions[interactions["metric"] == metric]
            .set_index("region")
            .reindex(regions_present)
        )
        y      = np.arange(len(regions_present))
        coefs  = sub["coef"].fillna(0).values.astype(float)
        ci_lo  = sub["ci_low"].fillna(0).values.astype(float)
        ci_hi  = sub["ci_high"].fillna(0).values.astype(float)
        sigs   = sub["sig_fdr"].fillna(False).values

        colors = ["#7BC7E0" if c > 0 else "#D5A45B" for c in coefs]
        ax.barh(
            y, coefs,
            xerr=[coefs - ci_lo, ci_hi - coefs],
            height=0.6, color=colors,
            capsize=4, error_kw={"elinewidth": 1.2},
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(regions_present, fontsize=9)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Interaction coef. vs N. America", fontsize=8)

        for i, (sig, ch) in enumerate(zip(sigs, ci_hi)):
            if sig:
                ax.text(ch + abs(ch) * 0.08 + 0.001, i, "*",
                        va="center", fontsize=11, color="black")

    fig.suptitle(
        "Cultural Moderation: is_daughter × Region Interaction\n"
        "Positive = daughters more stereotyped vs North America baseline"
        "  (* = FDR p<0.05)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig_i1_interaction_forest", fig_dir)
    print("  Saved fig_i1_interaction_forest")

# Figure I2: Region × model_family heatmap 

def fig_gap_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Heatmap of gender gap (daughter − son on stereotype_score)
    per (region × model_family).  * = bootstrap 95 % CI excludes 0.
    """
    families = sorted(df["model_family"].dropna().unique())

    gap_mat = pd.DataFrame(np.nan, index=REGION_ORDER, columns=families)
    sig_mat = pd.DataFrame(False,  index=REGION_ORDER, columns=families)

    for region in REGION_ORDER:
        for fam in families:
            sub = df[(df["country_region"] == region) & (df["model_family"] == fam)]
            d   = sub[sub["child_label"] == "daughter"]["stereotype_score"].dropna().values
            s   = sub[sub["child_label"] == "son"]["stereotype_score"].dropna().values
            if len(d) < 3 or len(s) < 3:
                continue
            gap = float(d.mean() - s.mean())
            rng = np.random.default_rng(42)
            boots = [
                rng.choice(d, len(d)).mean() - rng.choice(s, len(s)).mean()
                for _ in range(500)
            ]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            gap_mat.loc[region, fam] = gap
            sig_mat.loc[region, fam] = bool(lo > 0 or hi < 0)

    # Build annotation matrix
    annot = pd.DataFrame("", index=REGION_ORDER, columns=families)
    for r in REGION_ORDER:
        for f in families:
            v = gap_mat.loc[r, f]
            if pd.notna(v):
                star = "*" if bool(sig_mat.loc[r, f]) else ""
                annot.loc[r, f] = f"{float(v):.2f}{star}"

    fig, ax = plt.subplots(
        figsize=(max(7, len(families) * 0.95 + 1), len(REGION_ORDER) * 0.7 + 1.8)
    )
    sns.heatmap(
        gap_mat.astype(float), annot=annot, fmt="s",
        cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, mask=gap_mat.isna(),
        cbar_kws={"label": "Stereotype gap (daughter − son)"},
    )
    ax.set_title(
        "Gender Bias Across Culture × Model Family\n(* = 95 % bootstrap CI excludes 0)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Model Family", fontsize=11)
    ax.set_ylabel("World Region", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    _save(fig, "fig_i2_gap_heatmap", fig_dir)
    print("  Saved fig_i2_gap_heatmap")

# Figure I3: Simple slopes (regional moderation) 

def fig_simple_slopes(
    region_gap: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    Horizontal bar chart of gender gap per region, sorted by magnitude.
    Annotated with Cohen's d. Primary moderation figure.
    """
    if region_gap.empty or "country_region" not in region_gap.columns:
        print("  fig_i3 skipped — empty region gap table")
        return

    plot_df = region_gap.sort_values("gap", ascending=True).reset_index(drop=True)
    y       = np.arange(len(plot_df))
    colors  = ["#7BC7E0" if g > 0 else "#D5A45B" for g in plot_df["gap"].fillna(0)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.65)))

    ax.barh(
        y, plot_df["gap"],
        xerr=[plot_df["gap"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["gap"]],
        height=0.6, color=colors,
        capsize=5, error_kw={"elinewidth": 1.5},
    )
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["country_region"].fillna(""), fontsize=10)
    ax.set_xlabel(
        "Stereotype gap  (daughter − son)\n(positive = daughters more stereotyped)",
        fontsize=10,
    )
    ax.set_title(
        "Cultural Moderation of Gender Bias — Simple Slopes per World Region\n"
        "(95 % bootstrap CI, FDR-corrected significance)",
        fontsize=12, fontweight="bold",
    )

    # Right-side annotation: Cohen's d + significance star
    xmax = ax.get_xlim()[1]
    ax.set_xlim(right=xmax * 1.35)
    for i, row in enumerate(plot_df.itertuples()):
        cd  = getattr(row, "cohens_d", np.nan)
        sig = getattr(row, "sig_fdr",  False)
        if pd.notna(cd):
            lbl = f"d = {cd:+.2f}{'*' if sig else ''}"
            ax.text(xmax * 1.05, i, lbl, va="center", fontsize=8.5, color="#444")

    ax.legend(
        handles=[
            Patch(color="#7BC7E0", label="Daughters more stereotyped"),
            Patch(color="#D5A45B", label="Sons more stereotyped"),
        ],
        loc="lower right", fontsize=9,
    )
    _save(fig, "fig_i3_simple_slopes", fig_dir)
    print("  Saved fig_i3_simple_slopes")

# Figure I4: Country × model_family heatmap 

def fig_country_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Fine-grained heatmap: gender gap at country granularity × model_family.
    Rows sorted by region so regional clusters are visible; thick lines mark
    region boundaries; region labels added on the right axis.
    """
    if "country" not in df.columns:
        return

    country_to_region = (
        df[["country", "country_region"]]
        .drop_duplicates()
        .set_index("country")["country_region"]
        .to_dict()
    )

    # Sort countries: by region order, then alphabetically within region
    countries_sorted: list[str] = []
    region_sizes:     list[int] = []
    for region in REGION_ORDER:
        cs = sorted(c for c, r in country_to_region.items() if r == region)
        countries_sorted.extend(cs)
        region_sizes.append(len(cs))

    families = sorted(df["model_family"].dropna().unique())
    n_c      = len(countries_sorted)

    data = pd.DataFrame(np.nan, index=countries_sorted, columns=families)
    for country in countries_sorted:
        for fam in families:
            sub = df[(df["country"] == country) & (df["model_family"] == fam)]
            d   = sub[sub["child_label"] == "daughter"]["stereotype_score"].dropna().values
            s   = sub[sub["child_label"] == "son"]["stereotype_score"].dropna().values
            if len(d) >= 2 and len(s) >= 2:
                data.loc[country, fam] = float(d.mean() - s.mean())

    fig, ax = plt.subplots(
        figsize=(max(6, len(families) * 0.95 + 1), max(8, n_c * 0.44 + 2))
    )
    sns.heatmap(
        data.astype(float), cmap="RdBu_r", center=0, linewidths=0.25,
        ax=ax, mask=data.isna(),
        cbar_kws={"label": "Stereotype gap (daughter − son)"},
        xticklabels=True, yticklabels=True,
    )

    # Region boundary lines
    cumulative = np.cumsum(region_sizes)
    for boundary in cumulative[:-1]:
        ax.axhline(boundary, color="black", linewidth=1.5, alpha=0.8)

    # Region labels on right
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

    ax.set_title("Country × Model Family: Gender Stereotype Gap",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Model Family", fontsize=11)
    ax.set_ylabel("Country (grouped by region)", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    _save(fig, "fig_i4_country_heatmap", fig_dir)
    print("  Saved fig_i4_country_heatmap")

# Analysis: Narrator × child gender (in-group leniency test) 

def run_narrator_child_interaction(
    df: pd.DataFrame,
    n_boot: int = N_BOOT,
) -> tuple[pd.DataFrame, dict]:
    """
    2×2 analysis: narrator gender role (female / male) × child gender.

    Returns:
    cell_stats : DataFrame with mean ± bootstrap CI for each of 4 cells
    ols_result : dict with interaction coefficient from OLS
    """
    import statsmodels.formula.api as smf

    sub = df[df["person_gender_role"].isin(["female", "male"])].copy()
    if len(sub) < 20:
        print("  Narrator×child skipped — not enough gendered-narrator stories")
        return pd.DataFrame(), {}

    sub["narrator_is_female"] = (sub["person_gender_role"] == "female").astype(int)
    sub["log_wc"] = np.log1p(sub["word_count"].fillna(0).clip(lower=1))

    # Cell statistics 
    rows: list[dict] = []
    for narrator_role in ("female", "male"):
        for child_label in ("daughter", "son"):
            cell = sub[
                (sub["person_gender_role"] == narrator_role) &
                (sub["child_label"] == child_label)
            ]
            vals = cell["stereotype_score"].dropna().values
            if len(vals) < 2:
                continue
            rng   = np.random.default_rng(42)
            boots = [rng.choice(vals, len(vals)).mean() for _ in range(n_boot)]
            rows.append({
                "narrator_role": narrator_role,
                "child_label":   child_label,
                "dyad_type": (
                    "same-gender"
                    if (narrator_role == "female" and child_label == "daughter") or
                       (narrator_role == "male"   and child_label == "son")
                    else "cross-gender"
                ),
                "mean":    float(vals.mean()),
                "ci_low":  float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "n":       int(len(vals)),
            })
    cell_stats = pd.DataFrame(rows)

    # OLS interaction test 
    ols_result: dict = {}
    required = ["stereotype_score", "is_daughter", "narrator_is_female",
                "model_family", "country_region", "log_wc"]
    sub_clean = sub[required].dropna()
    if len(sub_clean) >= 50:
        try:
            fit = smf.ols(
                "stereotype_score ~ is_daughter * narrator_is_female"
                " + C(model_family, Treatment('llama'))"
                " + C(country_region, Treatment('North America'))"
                " + log_wc",
                data=sub_clean,
            ).fit()
            ci       = fit.conf_int()
            int_term = "is_daughter:narrator_is_female"
            if int_term in fit.params:
                ols_result = {
                    "interaction_coef":    float(fit.params[int_term]),
                    "interaction_ci_low":  float(ci.loc[int_term, 0]),
                    "interaction_ci_high": float(ci.loc[int_term, 1]),
                    "interaction_p":       float(fit.pvalues[int_term]),
                    "r_squared":           float(fit.rsquared),
                }
        except Exception as e:
            print(f"  OLS narrator×child failed: {e}")

    return cell_stats, ols_result


def run_pmi_narrator_interaction(
    df: pd.DataFrame,
    n_boot: int = N_BOOT,
) -> tuple[pd.DataFrame, dict]:
    """
    2×2 narrator_gender × child_gender interaction on all three PMI dimensions.

    Replaces the legacy stereotype_score interaction with PMI-weighted scores
    so each bias axis (role / domain / trait) is tested independently.

    Returns:
    cell_stats : DataFrame —> one row per (metric × narrator_role × child_label)
                 with mean PMI score + 95 % bootstrap CI.
    ols_results : dict[metric -> {interaction_coef, CI, p, r^2}]
                  OLS: pmi_X ~ is_daughter × narrator_is_female + covariates.
    """
    sub = df[df["person_gender_role"].isin(["female", "male"])].copy()
    if len(sub) < 20:
        print("  PMI narrator×child skipped — not enough gendered-narrator stories")
        return pd.DataFrame(), {}

    sub["narrator_is_female"] = (sub["person_gender_role"] == "female").astype(int)
    sub["log_wc"] = np.log1p(sub["word_count"].fillna(0).clip(lower=1))

    rng  = np.random.default_rng(42)
    rows: list[dict] = []

    for metric in PMI_METRICS:
        if metric not in sub.columns:
            continue
        for narrator_role in ("female", "male"):
            for child_label in ("daughter", "son"):
                cell = sub[
                    (sub["person_gender_role"] == narrator_role) &
                    (sub["child_label"] == child_label)
                ]
                vals = cell[metric].dropna().values
                if len(vals) < 2:
                    continue
                boots = [rng.choice(vals, len(vals)).mean() for _ in range(n_boot)]
                rows.append({
                    "metric":        metric,
                    "narrator_role": narrator_role,
                    "child_label":   child_label,
                    "dyad_type": (
                        "same-gender"
                        if (narrator_role == "female" and child_label == "daughter") or
                           (narrator_role == "male"   and child_label == "son")
                        else "cross-gender"
                    ),
                    "mean":    float(vals.mean()),
                    "ci_low":  float(np.percentile(boots, 2.5)),
                    "ci_high": float(np.percentile(boots, 97.5)),
                    "n":       int(len(vals)),
                })

    cell_stats = pd.DataFrame(rows)

    # Per-metric OLS interaction test
    ols_results: dict = {}
    required_base = ["is_daughter", "narrator_is_female",
                     "model_family", "country_region", "log_wc"]
    for metric in PMI_METRICS:
        if metric not in sub.columns:
            continue
        sub_clean = sub[required_base + [metric]].dropna()
        if len(sub_clean) < 50:
            continue
        try:
            fit = smf.ols(
                f"{metric} ~ is_daughter * narrator_is_female"
                " + C(model_family, Treatment('llama'))"
                " + C(country_region, Treatment('North America'))"
                " + log_wc",
                data=sub_clean,
            ).fit()
            ci       = fit.conf_int()
            int_term = "is_daughter:narrator_is_female"
            if int_term in fit.params:
                ols_results[metric] = {
                    "interaction_coef":    float(fit.params[int_term]),
                    "interaction_ci_low":  float(ci.loc[int_term, 0]),
                    "interaction_ci_high": float(ci.loc[int_term, 1]),
                    "interaction_p":       float(fit.pvalues[int_term]),
                    "r_squared":           float(fit.rsquared),
                }
        except Exception as e:
            print(f"  OLS narrator×child (PMI) failed for {metric}: {e}")

    return cell_stats, ols_results


# Figure I5: Narrator × child gender 2×2 plot 

def fig_narrator_child_2x2(
    cell_stats: pd.DataFrame,
    ols_result: dict,
    fig_dir: Path,
) -> None:
    """
    Two-panel figure:
      Left: interaction line plot (x = child gender, lines = narrator role)
               Parallel lines → no interaction; crossing → in-group leniency.
      Right: stereotype gap (daughter − son) by narrator role with CI.
               Smaller gap for female narrator → in-group leniency confirmed.
    """
    if cell_stats.empty:
        print("  fig_i5 skipped — no cell stats")
        return

    palette = {"female": "#7BC7E0", "male": "#D5A45B"}
    child_order = ["daughter", "son"]
    x_pos = {c: i for i, c in enumerate(child_order)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Interaction line plot 
    for narrator_role, color in palette.items():
        sub = (cell_stats[cell_stats["narrator_role"] == narrator_role]
               .set_index("child_label")
               .reindex(child_order))
        x  = [x_pos[c] for c in child_order]
        y  = sub["mean"].values.astype(float)
        lo = sub["ci_low"].values.astype(float)
        hi = sub["ci_high"].values.astype(float)

        ax1.plot(x, y, "o-", color=color, linewidth=2.2, markersize=9,
                 label=f"{narrator_role.capitalize()} narrator", zorder=3)
        ax1.fill_between(x, lo, hi, color=color, alpha=0.12, zorder=2)
        ax1.errorbar(x, y, yerr=[y - lo, hi - y],
                     fmt="none", color=color, capsize=5, elinewidth=1.5)

        # Label same-gender dyads
        same_child = "daughter" if narrator_role == "female" else "son"
        xi = x_pos[same_child]
        yi = sub.loc[same_child, "mean"] if same_child in sub.index else np.nan
        if pd.notna(yi):
            ax1.annotate("same-gender\ndyad", (xi, yi),
                         xytext=(xi + 0.08, yi + 0.02),
                         fontsize=7.5, color=color, alpha=0.8)

    ax1.set_xticks(list(x_pos.values()))
    ax1.set_xticklabels(["Daughter stories", "Son stories"], fontsize=11)
    ax1.set_ylabel("Mean stereotype score", fontsize=11)
    ax1.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax1.set_title("Narrator × Child Gender Interaction\n(95 % bootstrap CI)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10, framealpha=0.85)

    if ols_result:
        coef = ols_result.get("interaction_coef", 0.0)
        p    = ols_result.get("interaction_p", 1.0)
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax1.text(0.04, 0.04,
                 f"OLS interaction coef = {coef:+.3f} {sig}",
                 transform=ax1.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Panel 2: Gender gap by narrator role 
    gap_rows = []
    for narrator_role, color in palette.items():
        sub = (cell_stats[cell_stats["narrator_role"] == narrator_role]
               .set_index("child_label"))
        if "daughter" in sub.index and "son" in sub.index:
            d = sub.loc["daughter"]
            s = sub.loc["son"]
            gap = float(d["mean"]) - float(s["mean"])
            # Propagate CI error in quadrature
            err = float(np.sqrt(
                ((d["mean"] - d["ci_low"]) ** 2 +
                 (s["mean"] - s["ci_low"]) ** 2)
            ))
            gap_rows.append({"narrator": narrator_role, "gap": gap,
                              "err": err, "color": color})

    if gap_rows:
        gdf = pd.DataFrame(gap_rows)
        ax2.bar(gdf["narrator"].str.capitalize(), gdf["gap"],
                color=gdf["color"].tolist(), width=0.45,
                yerr=gdf["err"], capsize=7, error_kw={"elinewidth": 1.5})
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_ylabel("Stereotype gap  (daughter − son)", fontsize=11)
        ax2.set_xlabel("Narrator gender role", fontsize=11)
        ax2.set_title(
            "Gender Bias Gap by Narrator Role\n"
            "(smaller gap = narrator moderates own-gender stereotyping?)",
            fontsize=11, fontweight="bold",
        )
        for i, row in gdf.iterrows():
            ax2.text(i, row["gap"] + row["err"] + 0.005,
                     f"{row['gap']:+.3f}", ha="center", fontsize=10,
                     fontweight="bold", color=row["color"])

    plt.tight_layout()
    _save(fig, "fig_i5_narrator_child_2x2", fig_dir)
    print("  Saved fig_i5_narrator_child_2x2")


# Figure I6: PMI narrator × child —> overall (avg across models) 

def fig_pmi_narrator_interaction(
    cell_stats: pd.DataFrame,
    ols_results: dict,
    fig_dir: Path,
) -> None:
    """
    3-panel interaction line plot (one panel per PMI dimension).

    X-axis  : child gender (daughter / son)
    Lines   : narrator gender role 
    Y-axis  : mean PMI score with shaded 95 % bootstrap CI

    Parallel lines : no moderating effect of narrator gender.
    One line steeper : that narrator role amplifies the daughter-son gap.
    OLS interaction coefficient annotated in each panel.

    This replaces the legacy stereotype_score version (I5) with three
    independent bias axes (role / domain / trait).
    """
    if cell_stats.empty:
        print("  fig_i6 skipped: no PMI narrator cell stats")
        return

    narrator_palette = {"female": PALETTE_ROLE["female"], "male": PALETTE_ROLE["male"]}
    child_order      = ["daughter", "son"]
    x_pos            = {c: i for i, c in enumerate(child_order)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, (col, _, _, dim_label) in zip(axes, _DIM_INFO):
        sub_dim = cell_stats[cell_stats["metric"] == col]
        if sub_dim.empty:
            ax.set_visible(False)
            continue

        for narrator_role, color in narrator_palette.items():
            sub = (
                sub_dim[sub_dim["narrator_role"] == narrator_role]
                .set_index("child_label")
                .reindex(child_order)
            )
            x  = [x_pos[c] for c in child_order]
            y  = sub["mean"].values.astype(float)
            lo = sub["ci_low"].values.astype(float)
            hi = sub["ci_high"].values.astype(float)

            ax.plot(x, y, "o-", color=color, linewidth=2.2, markersize=9,
                    label=f"{narrator_role.capitalize()} narrator", zorder=3)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, zorder=2)
            ax.errorbar(x, y, yerr=[y - lo, hi - y],
                        fmt="none", color=color, capsize=5, elinewidth=1.5)

            # Mark same-gender dyads
            same_child = "daughter" if narrator_role == "female" else "son"
            if same_child in sub.index:
                yi = float(sub.loc[same_child, "mean"])
                ax.annotate(
                    "same-gender",
                    (x_pos[same_child], yi),
                    xytext=(x_pos[same_child] + 0.07, yi + abs(yi) * 0.05 + 0.01),
                    fontsize=7.5, color=color, alpha=0.85,
                )

        # OLS annotation
        if col in ols_results:
            res = ols_results[col]
            coef = res["interaction_coef"]
            p    = res["interaction_p"]
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(
                0.04, 0.04,
                f"OLS interaction: {coef:+.3f} {sig}",
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
            )

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(["Daughter stories", "Son stories"], fontsize=10)
        ax.set_ylabel("Mean PMI score\n(+ = daughter-stereotyped)", fontsize=9)
        ax.set_title(dim_label, fontsize=11, fontweight="bold")
        if ax is axes[0]:
            ax.legend(fontsize=10, framealpha=0.85)

    fig.suptitle(
        "Narrator Gender × Child Gender Interaction : PMI Bias Scores (All Models)\n"
        "Parallel lines = no moderation; steeper line = narrator role amplifies bias"
        "  (shading = 95 % bootstrap CI)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_i6_pmi_narrator_interaction", fig_dir)
    print("  Saved fig_i6_pmi_narrator_interaction")


# Figure I7: PMI narrator × child —> per model 

def fig_pmi_narrator_by_model(
    df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    3-panel grouped horizontal bar chart (one per PMI dimension).

    For each model, two bars show the daughter−son PMI gap.

    Reveals which models have narrator-moderated bias: if one bar is
    systematically larger than the other, that model amplifies stereotyping
    depending on who tells the story.

    Models are sorted by the average gap across narrator roles.
    """
    if "person_gender_role" not in df.columns or "model_key" not in df.columns:
        print("  fig_i7 skipped — missing person_gender_role or model_key")
        return

    cols_present = [col for col, *_ in _DIM_INFO if col in df.columns]
    if not cols_present:
        print("  fig_i7 skipped — no PMI score columns found")
        return

    models = sorted(df["model_key"].unique())
    rng    = np.random.default_rng(42)

    narrator_styles = {
        "female": {"color": "#41C0AB", "label": "Female narrator"},
        "male":   {"color": "#AD702E", "label": "Male narrator"},
    }
    bar_h = 0.35

    fig, axes = plt.subplots(
        1, 3,
        figsize=(17, max(5, len(models) * 0.6 + 2)),
        sharey=True,
    )

    for ax, (col, _, _, dim_label) in zip(axes, _DIM_INFO):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        # Sort models by mean absolute gap across both narrator roles
        model_avg_gap: dict[str, float] = {}
        for model in models:
            gsub = df[df["model_key"] == model]
            gaps = []
            for narrator in ("female", "male"):
                d_v = gsub[(gsub["person_gender_role"] == narrator) &
                            (gsub["child_label"] == "daughter")][col].dropna().values
                s_v = gsub[(gsub["person_gender_role"] == narrator) &
                            (gsub["child_label"] == "son")][col].dropna().values
                if len(d_v) >= 2 and len(s_v) >= 2:
                    gaps.append(d_v.mean() - s_v.mean())
            model_avg_gap[model] = float(np.mean(gaps)) if gaps else 0.0

        sorted_models = sorted(models, key=lambda m: model_avg_gap.get(m, 0.0))
        y_base = np.arange(len(sorted_models))

        for j, (narrator, style) in enumerate(narrator_styles.items()):
            gaps, lo_errs, hi_errs = [], [], []
            for model in sorted_models:
                msub = df[(df["model_key"] == model) &
                          (df["person_gender_role"] == narrator)]
                d_v = msub[msub["child_label"] == "daughter"][col].dropna().values
                s_v = msub[msub["child_label"] == "son"][col].dropna().values
                if len(d_v) < 2 or len(s_v) < 2:
                    gaps.append(0.0); lo_errs.append(0.0); hi_errs.append(0.0)
                    continue
                gap   = float(d_v.mean() - s_v.mean())
                boots = [
                    rng.choice(d_v, len(d_v)).mean() - rng.choice(s_v, len(s_v)).mean()
                    for _ in range(N_BOOT)
                ]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                gaps.append(gap)
                lo_errs.append(float(gap - lo))
                hi_errs.append(float(hi - gap))

            y_off = (j - 0.5) * bar_h
            ax.barh(
                y_base + y_off, gaps,
                xerr=[lo_errs, hi_errs],
                height=bar_h,
                color=style["color"],
                label=style["label"],
                capsize=2,
                error_kw={"elinewidth": 0.8, "alpha": 0.6},
                alpha=0.85,
            )

        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_yticks(y_base)
        ax.set_yticklabels(sorted_models, fontsize=7.5)
        ax.set_xlabel("Daughter − Son mean PMI score\n(95 % bootstrap CI)", fontsize=9)
        ax.set_title(dim_label, fontsize=10, fontweight="bold")

    axes[0].legend(fontsize=9, framealpha=0.9, loc="lower right")
    fig.suptitle(
        "Narrator Gender × Child Gender Interaction per Model : PMI Bias Scores\n"
        "(positive = daughters receive more stereotyped language)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_i7_pmi_narrator_by_model", fig_dir)
    print("  Saved fig_i7_pmi_narrator_by_model")


# Main

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    res_dir = out_dir / "results"
    fig_dir = out_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load, enrich, score (legacy lexicon + PMI)
    raw    = load_data(args.input)
    df     = enrich(raw)
    scored = score_lexicon(df)
    print(f"\nAnalysing {len(scored):,} stories "
          f"({scored['child_label'].value_counts().to_dict()})")

    print("\n PMI-weighted scoring")
    pmi_weights = compute_pmi_weights(scored)
    pmi_scored  = score_pmi(scored, pmi_weights)
    print(f"  PMI weights: {len(pmi_weights):,} tokens  |  "
          f"PMI columns: {PMI_METRICS}")

    # 2. Interaction OLS 
    print("\n Interaction OLS ")
    ols = run_interaction_ols(scored)
    if not ols.empty:
        ols.to_csv(res_dir / "interaction_ols.csv", index=False)
        ss_int = ols[(ols["metric"] == "stereotype_score") & ols["interaction"]]
        if not ss_int.empty:
            print(ss_int[["region", "coef", "ci_low", "ci_high", "p_fdr", "sig_fdr"]]
                  .round(4).to_string(index=False))

    # 3. Regional moderation table 
    print("\n Regional moderation (stereotype_score) ")
    region_gap = _gap_table(scored, "country_region", n_boot=args.n_bootstrap)
    if not region_gap.empty:
        region_gap.to_csv(res_dir / "region_moderation.csv", index=False)
        print(region_gap[
            ["country_region", "gap", "ci_low", "ci_high",
             "cohens_d", "cliffs_delta", "p_fdr", "sig_fdr"]
        ].round(4).to_string(index=False))

    # 4. Gap tables (region×model, country×model) 
    print("\n Gap tables ")
    rm_rows: list[dict] = []
    for region in REGION_ORDER:
        for fam in scored["model_family"].dropna().unique():
            sub = scored[(scored["country_region"] == region) &
                         (scored["model_family"] == fam)]
            d   = sub[sub["child_label"] == "daughter"]["stereotype_score"].dropna().values
            s   = sub[sub["child_label"] == "son"]["stereotype_score"].dropna().values
            if len(d) >= 2 and len(s) >= 2:
                rm_rows.append({
                    "region":       region,
                    "model_family": fam,
                    "gap":          float(d.mean() - s.mean()),
                    "cohens_d":     _cohens_d(s, d),
                    "n_daughter":   int(len(d)),
                    "n_son":        int(len(s)),
                })
    gap_rm = pd.DataFrame(rm_rows)
    gap_rm.to_csv(res_dir / "gap_by_region_model.csv", index=False)

    cm_rows: list[dict] = []
    if "country" in scored.columns:
        for country in scored["country"].unique():
            region_of = scored.loc[scored["country"] == country, "country_region"].iloc[0]
            for fam in scored["model_family"].dropna().unique():
                sub = scored[(scored["country"] == country) &
                             (scored["model_family"] == fam)]
                d   = sub[sub["child_label"] == "daughter"]["stereotype_score"].dropna().values
                s   = sub[sub["child_label"] == "son"]["stereotype_score"].dropna().values
                if len(d) >= 2 and len(s) >= 2:
                    cm_rows.append({
                        "country":      country,
                        "region":       region_of,
                        "model_family": fam,
                        "gap":          float(d.mean() - s.mean()),
                        "cohens_d":     _cohens_d(s, d),
                    })
    gap_cm = pd.DataFrame(cm_rows)
    gap_cm.to_csv(res_dir / "gap_by_country_model.csv", index=False)
    print(f"  Region × model cells  : {len(gap_rm)}")
    print(f"  Country × model cells : {len(gap_cm)}")

    # 5. Narrator × child gender (legacy stereotype_score: in-group leniency)
    print("\n Narrator × Child gender (legacy stereotype_score) ")
    cell_stats, ols_narrator = run_narrator_child_interaction(scored, n_boot=args.n_bootstrap)
    if not cell_stats.empty:
        cell_stats.to_csv(res_dir / "narrator_child_cells.csv", index=False)
        print(cell_stats[["narrator_role", "child_label", "dyad_type",
                           "mean", "ci_low", "ci_high", "n"]].round(4).to_string(index=False))
    if ols_narrator:
        pd.DataFrame([ols_narrator]).to_csv(res_dir / "narrator_child_ols.csv", index=False)
        coef = ols_narrator.get("interaction_coef", 0)
        p    = ols_narrator.get("interaction_p", 1)
        print(f"  Interaction coef (is_daughter×narrator_is_female): {coef:+.4f}  p={p:.4f}")

    # 5b. Narrator × child gender (PMI: all three dimensions)
    print("\n Narrator × Child gender (PMI-weighted, three dimensions) ")
    pmi_cell_stats, pmi_ols_narrator = run_pmi_narrator_interaction(
        pmi_scored, n_boot=args.n_bootstrap
    )
    if not pmi_cell_stats.empty:
        pmi_cell_stats.to_csv(res_dir / "pmi_narrator_child_cells.csv", index=False)
        print(pmi_cell_stats[["metric", "narrator_role", "child_label", "dyad_type",
                               "mean", "ci_low", "ci_high", "n"]].round(4).to_string(index=False))
    if pmi_ols_narrator:
        ols_rows = [
            {"metric": m, **v} for m, v in pmi_ols_narrator.items()
        ]
        pd.DataFrame(ols_rows).to_csv(res_dir / "pmi_narrator_child_ols.csv", index=False)
        for metric, res in pmi_ols_narrator.items():
            coef = res["interaction_coef"]
            p    = res["interaction_p"]
            print(f"  {metric}: interaction coef = {coef:+.4f}  p = {p:.4f}")

    # 6. Figures
    print("\n Generating figures ")
    fig_interaction_forest(ols, fig_dir)
    fig_gap_heatmap(scored, fig_dir)
    fig_simple_slopes(region_gap, fig_dir)
    if "country" in scored.columns:
        fig_country_heatmap(scored, fig_dir)
    fig_narrator_child_2x2(cell_stats, ols_narrator, fig_dir)
    # PMI narrator interactions (figs I6, I7)
    fig_pmi_narrator_interaction(pmi_cell_stats, pmi_ols_narrator, fig_dir)
    fig_pmi_narrator_by_model(pmi_scored, fig_dir)

    # 7. Summary
    print("INTERSECTIONALITY ANALYSIS: COMPLETE")

    if not region_gap.empty:
        top    = region_gap.loc[region_gap["gap"].abs().idxmax(), "country_region"]
        bottom = region_gap.loc[region_gap["gap"].abs().idxmin(), "country_region"]
        n_sig  = int(region_gap["sig_fdr"].sum())
        print(f"  Largest  gender gap region : {top}")
        print(f"  Smallest gender gap region : {bottom}")
        print(f"  Regions sig at FDR<0.05    : {n_sig} / {len(region_gap)}")
    if not gap_cm.empty:
        worst = gap_cm.loc[gap_cm["gap"].abs().idxmax()]
        print(f"  Largest country×model gap  : {worst['country']} × {worst['model_family']}"
              f"  (gap={worst['gap']:+.3f})")
    print(f"\n  Results  : {res_dir}")
    print(f"  Figures  : {fig_dir}")


if __name__ == "__main__":
    main()


