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
    PALETTE_GENDER, REGION_ORDER,
    _bootstrap_ci, _cliffs_delta, _cohens_d, _save,
    enrich, load_data, score_lexicon,
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
    Faceted forest plot — one panel per focal metric.
    Shows is_daughter:region interaction coefficients (vs North America baseline).
    Positive = daughters receive a stronger stereotype in that region.
    """
    interactions = ols[ols["interaction"]].copy()
    if interactions.empty:
        print("  fig_i1 skipped — no interaction terms found")
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
    colors  = ["#E07B8C" if g > 0 else "#5B9BD5" for g in plot_df["gap"].fillna(0)]

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
            Patch(color="#E07B8C", label="Daughters more stereotyped"),
            Patch(color="#5B9BD5", label="Sons more stereotyped"),
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

# Main 

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    res_dir = out_dir / "results"
    fig_dir = out_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load, enrich, score 
    raw    = load_data(args.input)
    df     = enrich(raw)
    scored = score_lexicon(df)
    print(f"\nAnalysing {len(scored):,} stories "
          f"({scored['child_label'].value_counts().to_dict()})")

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

    # 5. Figures 
    print("\n Generating figures ")
    fig_interaction_forest(ols, fig_dir)
    fig_gap_heatmap(scored, fig_dir)
    fig_simple_slopes(region_gap, fig_dir)
    if "country" in scored.columns:
        fig_country_heatmap(scored, fig_dir)

    # 6. Summary 
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
    print(f"\n  Results → {res_dir}")
    print(f"  Figures → {fig_dir}")


if __name__ == "__main__":
    main()

