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


DEFAULT_OUT = Path("Narratives3/intersectionality")

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

    available = [m for m in PMI_METRICS if m in df.columns]
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
    Faceted forest plot — one panel per PMI dimension.
    Shows is_daughter × region interaction coefficients (vs North America baseline).
    Positive = daughters receive a stronger stereotype in that region.
    """
    interactions = ols[ols["interaction"]].copy()
    if interactions.empty:
        print("  fig_i1 skipped — no interaction terms found")
        return

    metrics_present = [m for m in PMI_METRICS if m in interactions["metric"].unique()]
    n_panels        = len(metrics_present)
    regions_present = [r for r in REGION_ORDER if r in interactions["region"].values]

    # Map PMI metric name → readable label from _DIM_INFO
    dim_labels = {col: lbl.replace("\n", " ") for col, *_, lbl in _DIM_INFO}

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

        colors = ["#E07B8C" if c > 0 else "#5B9BD5" for c in coefs]
        ax.barh(
            y, coefs,
            xerr=[coefs - ci_lo, ci_hi - coefs],
            height=0.6, color=colors,
            capsize=4, error_kw={"elinewidth": 1.2},
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(regions_present, fontsize=9)
        ax.set_title(dim_labels.get(metric, metric), fontsize=10, fontweight="bold")
        ax.set_xlabel("Interaction coef. vs N. America", fontsize=8)

        for i, (sig, ch) in enumerate(zip(sigs, ci_hi)):
            if sig:
                ax.text(ch + abs(ch) * 0.08 + 0.001, i, "*",
                        va="center", fontsize=11, color="black")

    fig.suptitle(
        "Cultural Moderation: is_daughter × Region Interaction — PMI Bias Scores\n"
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
    3-panel heatmap of gender gap (daughter − son) per (region × model_family),
    one panel per PMI dimension.  * = bootstrap 95 % CI excludes 0.
    """
    families = sorted(df["model_family"].dropna().unique())
    rng      = np.random.default_rng(42)
    dim_labels = {col: lbl.replace("\n", " ") for col, *_, lbl in _DIM_INFO}

    fig, axes = plt.subplots(
        1, 3,
        figsize=(max(7, len(families) * 0.95 + 1) * 3, len(REGION_ORDER) * 0.7 + 2.5),
        sharey=True,
    )

    for ax, (col, _, _, dim_lbl) in zip(axes, _DIM_INFO):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        gap_mat = pd.DataFrame(np.nan,  index=REGION_ORDER, columns=families)
        sig_mat = pd.DataFrame(False,   index=REGION_ORDER, columns=families)

        for region in REGION_ORDER:
            for fam in families:
                sub = df[(df["country_region"] == region) & (df["model_family"] == fam)]
                d   = sub[sub["child_label"] == "daughter"][col].dropna().values
                s   = sub[sub["child_label"] == "son"][col].dropna().values
                if len(d) < 3 or len(s) < 3:
                    continue
                gap    = float(d.mean() - s.mean())
                boots  = [
                    rng.choice(d, len(d)).mean() - rng.choice(s, len(s)).mean()
                    for _ in range(500)
                ]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                gap_mat.loc[region, fam] = gap
                sig_mat.loc[region, fam] = bool(lo > 0 or hi < 0)

        annot = pd.DataFrame("", index=REGION_ORDER, columns=families)
        for r in REGION_ORDER:
            for f in families:
                v = gap_mat.loc[r, f]
                if pd.notna(v):
                    star = "*" if bool(sig_mat.loc[r, f]) else ""
                    annot.loc[r, f] = f"{float(v):.2f}{star}"

        sns.heatmap(
            gap_mat.astype(float), annot=annot, fmt="s",
            cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, mask=gap_mat.isna(),
            cbar_kws={"label": "PMI gap (daughter − son)"},
        )
        ax.set_title(dim_lbl, fontsize=11, fontweight="bold")
        ax.set_xlabel("Model Family", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("World Region", fontsize=10)
        plt.sca(ax)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

    fig.suptitle(
        "PMI Gender Bias Gap Across Culture × Model Family\n"
        "(* = 95 % bootstrap CI excludes 0)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_i2_gap_heatmap", fig_dir)
    print("  Saved fig_i2_gap_heatmap")

# Figure I3: Simple slopes (regional moderation) 

def fig_simple_slopes(
    region_gaps: dict[str, pd.DataFrame],
    fig_dir: Path,
) -> None:
    """
    3-panel horizontal bar chart — one panel per PMI dimension.
    Each panel shows the daughter−son gap per world region, sorted by magnitude,
    annotated with Cohen's d.
    """
    if not region_gaps:
        print("  fig_i3 skipped — empty region gap tables")
        return

    dim_labels = {col: lbl.replace("\n", " ") for col, *_, lbl in _DIM_INFO}
    present    = [(col, lbl) for col, *_, lbl in _DIM_INFO if col in region_gaps]
    if not present:
        return

    # Use region order from the first available gap table
    first_df   = next(iter(region_gaps.values()))
    n_regions  = len(first_df)

    fig, axes = plt.subplots(
        1, len(present),
        figsize=(10 * len(present), max(4, n_regions * 0.65)),
        sharey=True,
    )
    if len(present) == 1:
        axes = [axes]

    for ax, (col, dim_lbl) in zip(axes, present):
        gap_df  = region_gaps[col]
        plot_df = gap_df.sort_values("gap", ascending=True).reset_index(drop=True)
        y       = np.arange(len(plot_df))
        colors  = ["#E07B8C" if g > 0 else "#5B9BD5" for g in plot_df["gap"].fillna(0)]

        ax.barh(
            y, plot_df["gap"],
            xerr=[plot_df["gap"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["gap"]],
            height=0.6, color=colors,
            capsize=5, error_kw={"elinewidth": 1.5},
        )
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(plot_df["country_region"].fillna(""), fontsize=10)
        ax.set_xlabel(
            "PMI gap  (daughter − son)\n(positive = daughters more stereotyped)",
            fontsize=10,
        )
        ax.set_title(dim_lbl, fontsize=11, fontweight="bold")

        xmax = ax.get_xlim()[1]
        ax.set_xlim(right=xmax * 1.35)
        for i, row in enumerate(plot_df.itertuples()):
            cd  = getattr(row, "cohens_d", np.nan)
            sig = getattr(row, "sig_fdr",  False)
            if pd.notna(cd):
                ax.text(xmax * 1.05, i,
                        f"d = {cd:+.2f}{'*' if sig else ''}",
                        va="center", fontsize=8.5, color="#444")

    axes[0].legend(
        handles=[
            Patch(color="#E07B8C", label="Daughters more stereotyped"),
            Patch(color="#5B9BD5", label="Sons more stereotyped"),
        ],
        loc="lower right", fontsize=9,
    )
    fig.suptitle(
        "Cultural Moderation of PMI Gender Bias — Simple Slopes per World Region\n"
        "(95 % bootstrap CI, FDR-corrected significance)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig_i3_simple_slopes", fig_dir)
    print("  Saved fig_i3_simple_slopes")

# Figure I4: Country × model_family heatmap 

def fig_country_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Fine-grained heatmap: mean PMI gender gap (averaged across all three
    PMI dimensions) at country × model_family granularity.
    Rows sorted by region; thick lines mark region boundaries.
    """
    if "country" not in df.columns:
        return

    pmi_cols_present = [col for col, *_ in _DIM_INFO if col in df.columns]
    if not pmi_cols_present:
        print("  fig_i4 skipped — no PMI columns in dataframe")
        return

    country_to_region = (
        df[["country", "country_region"]]
        .drop_duplicates()
        .set_index("country")["country_region"]
        .to_dict()
    )

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
            gaps = []
            for col in pmi_cols_present:
                d = sub[sub["child_label"] == "daughter"][col].dropna().values
                s = sub[sub["child_label"] == "son"][col].dropna().values
                if len(d) >= 2 and len(s) >= 2:
                    gaps.append(float(d.mean() - s.mean()))
            if gaps:
                data.loc[country, fam] = float(np.mean(gaps))

    fig, ax = plt.subplots(
        figsize=(max(6, len(families) * 0.95 + 1), max(8, n_c * 0.44 + 2))
    )
    sns.heatmap(
        data.astype(float), cmap="RdBu_r", center=0, linewidths=0.25,
        ax=ax, mask=data.isna(),
        cbar_kws={"label": "Mean PMI gap (daughter − son)"},
        xticklabels=True, yticklabels=True,
    )

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

    ax.set_title(
        "Country × Model Family: Mean PMI Gender Bias Gap\n"
        "(averaged across role, domain, and trait dimensions)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Model Family", fontsize=11)
    ax.set_ylabel("Country (grouped by region)", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    _save(fig, "fig_i4_country_heatmap", fig_dir)
    print("  Saved fig_i4_country_heatmap")

# Analysis: Narrator × child gender (in-group leniency test) 

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

    Reveals which models have narrator-moderated bias

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

    # 1. Load, enrich, score (lexicon needed for word_count covariates + PMI)
    raw    = load_data(args.input)
    df     = enrich(raw)
    scored = score_lexicon(df)
    print(f"\nAnalysing {len(scored):,} stories "
          f"({scored['child_label'].value_counts().to_dict()})")

    print("\n PMI-weighted scoring")
    pmi_weights = compute_pmi_weights(scored)
    pmi_scored  = score_pmi(scored, pmi_weights)
    print(f"  PMI weights: {len(pmi_weights):,} tokens  |  PMI columns: {PMI_METRICS}")

    # 2. Interaction OLS (PMI metrics × region)
    print("\n Interaction OLS (PMI) ")
    ols = run_interaction_ols(pmi_scored)
    if not ols.empty:
        ols.to_csv(res_dir / "pmi_interaction_ols.csv", index=False)
        for metric in PMI_METRICS:
            sub_int = ols[(ols["metric"] == metric) & ols["interaction"]]
            if not sub_int.empty:
                print(f"\n  {metric}:")
                print(sub_int[["region", "coef", "ci_low", "ci_high", "p_fdr", "sig_fdr"]]
                      .round(4).to_string(index=False))

    # 3. Regional moderation — one gap table per PMI dimension
    print("\n Regional moderation (PMI) ")
    region_gaps: dict[str, pd.DataFrame] = {}
    for col, *_, dim_lbl in _DIM_INFO:
        if col not in pmi_scored.columns:
            continue
        gap_df = _gap_table(pmi_scored, "country_region", val_col=col,
                             n_boot=args.n_bootstrap)
        region_gaps[col] = gap_df
        if not gap_df.empty:
            gap_df.to_csv(res_dir / f"pmi_region_moderation_{col}.csv", index=False)
            print(f"\n  {col}:")
            print(gap_df[["country_region", "gap", "ci_low", "ci_high",
                           "cohens_d", "cliffs_delta", "p_fdr", "sig_fdr"]]
                  .round(4).to_string(index=False))

    # 4. Gap tables (region × model_family, country × model_family) using mean PMI
    print("\n Gap tables (mean PMI across dimensions) ")
    pmi_cols = [col for col, *_ in _DIM_INFO if col in pmi_scored.columns]

    rm_rows: list[dict] = []
    for region in REGION_ORDER:
        for fam in pmi_scored["model_family"].dropna().unique():
            sub  = pmi_scored[(pmi_scored["country_region"] == region) &
                              (pmi_scored["model_family"]   == fam)]
            gaps = []
            for col in pmi_cols:
                d = sub[sub["child_label"] == "daughter"][col].dropna().values
                s = sub[sub["child_label"] == "son"][col].dropna().values
                if len(d) >= 2 and len(s) >= 2:
                    gaps.append(float(d.mean() - s.mean()))
            if gaps:
                rm_rows.append({
                    "region":       region,
                    "model_family": fam,
                    "mean_pmi_gap": float(np.mean(gaps)),
                    "n_dimensions": len(gaps),
                })
    gap_rm = pd.DataFrame(rm_rows)
    gap_rm.to_csv(res_dir / "pmi_gap_by_region_model.csv", index=False)

    cm_rows: list[dict] = []
    if "country" in pmi_scored.columns:
        for country in pmi_scored["country"].unique():
            region_of = pmi_scored.loc[
                pmi_scored["country"] == country, "country_region"
            ].iloc[0]
            for fam in pmi_scored["model_family"].dropna().unique():
                sub  = pmi_scored[(pmi_scored["country"]      == country) &
                                  (pmi_scored["model_family"] == fam)]
                gaps = []
                for col in pmi_cols:
                    d = sub[sub["child_label"] == "daughter"][col].dropna().values
                    s = sub[sub["child_label"] == "son"][col].dropna().values
                    if len(d) >= 2 and len(s) >= 2:
                        gaps.append(float(d.mean() - s.mean()))
                if gaps:
                    cm_rows.append({
                        "country":      country,
                        "region":       region_of,
                        "model_family": fam,
                        "mean_pmi_gap": float(np.mean(gaps)),
                    })
    gap_cm = pd.DataFrame(cm_rows)
    gap_cm.to_csv(res_dir / "pmi_gap_by_country_model.csv", index=False)
    print(f"  Region × model cells  : {len(gap_rm)}")
    print(f"  Country × model cells : {len(gap_cm)}")

    # 5. Narrator × child gender (PMI — all three dimensions)
    print("\n Narrator × Child gender (PMI-weighted, three dimensions) ")
    pmi_cell_stats, pmi_ols_narrator = run_pmi_narrator_interaction(
        pmi_scored, n_boot=args.n_bootstrap
    )
    if not pmi_cell_stats.empty:
        pmi_cell_stats.to_csv(res_dir / "pmi_narrator_child_cells.csv", index=False)
        print(pmi_cell_stats[["metric", "narrator_role", "child_label", "dyad_type",
                               "mean", "ci_low", "ci_high", "n"]].round(4).to_string(index=False))
    if pmi_ols_narrator:
        ols_rows = [{"metric": m, **v} for m, v in pmi_ols_narrator.items()]
        pd.DataFrame(ols_rows).to_csv(res_dir / "pmi_narrator_child_ols.csv", index=False)
        for metric, res in pmi_ols_narrator.items():
            coef = res["interaction_coef"]
            p    = res["interaction_p"]
            print(f"  {metric}: interaction coef = {coef:+.4f}  p = {p:.4f}")

    # 6. Figures
    print("\n Generating figures ")
    fig_interaction_forest(ols, fig_dir)
    fig_gap_heatmap(pmi_scored, fig_dir)
    fig_simple_slopes(region_gaps, fig_dir)
    if "country" in pmi_scored.columns:
        fig_country_heatmap(pmi_scored, fig_dir)
    fig_pmi_narrator_interaction(pmi_cell_stats, pmi_ols_narrator, fig_dir)
    fig_pmi_narrator_by_model(pmi_scored, fig_dir)

    # 7. Summary
    print("INTERSECTIONALITY ANALYSIS: COMPLETE")

    first_gap = next(iter(region_gaps.values()), pd.DataFrame())
    if not first_gap.empty:
        top   = first_gap.loc[first_gap["gap"].abs().idxmax(), "country_region"]
        bot   = first_gap.loc[first_gap["gap"].abs().idxmin(), "country_region"]
        n_sig = int(first_gap["sig_fdr"].sum())
        print(f"  Largest  PMI gap region : {top}")
        print(f"  Smallest PMI gap region : {bot}")
        print(f"  Regions sig at FDR<0.05 : {n_sig} / {len(first_gap)}")
    if not gap_cm.empty:
        worst = gap_cm.loc[gap_cm["mean_pmi_gap"].abs().idxmax()]
        print(f"  Largest country×model gap : {worst['country']} × {worst['model_family']}"
              f"  (mean_pmi_gap={worst['mean_pmi_gap']:+.3f})")
    print(f"\n  Results  : {res_dir}")
    print(f"  Figures  : {fig_dir}")

if __name__ == "__main__":
    main()


