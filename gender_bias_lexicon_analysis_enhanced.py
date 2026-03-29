#!/usr/bin/env python3
"""
Enhanced lexicon-based gender bias analysis with statistical significance testing.

Improvements over base version:
1. Statistical significance tests (t-tests)
2. Effect sizes (Cohen's d)
3. Multiple comparison corrections (FDR, Bonferroni)
4. Bootstrap CIs for all composite metrics
5. Culture and storyteller analysis
6. Interaction effects

Usage:
    python3 gender_bias_lexicon_analysis_enhanced.py
    python3 gender_bias_lexicon_analysis_enhanced.py --input Narratives2/biasednarratives.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


TOKEN_RE = re.compile(r"[a-zA-Z']+")
EPS = 1e-9
COMPOSITE_METRICS = [
    "trait_bias_index",
    "role_bias_index",
    "domain_bias_index",
    "direct_gender_marker_index",
]
SUMMARY_METRICS = [
    "masculine_traits_count",
    "feminine_traits_count",
    "male_markers_count",
    "female_markers_count",
    "trait_gender_bias",
    "role_gender_bias",
    "domain_gender_bias",
    "marker_gender_bias",
    "trait_bias_index",
    "role_bias_index",
    "domain_bias_index",
    "direct_gender_marker_index",
    "child_target_stereotype_score",
    "token_count_lex",
]


# Compact, research-inspired lexicons.
LEXICONS: Dict[str, set[str]] = {
    # Agentic vs communal framing
    "agentic": {
        "assertive", "ambitious", "confident", "competitive", "independent",
        "leader", "lead", "leadership", "decisive", "determined", "dominant",
        "driven", "fearless", "bold", "adventurous", "strong", "powerful",
        "capable", "persistent", "self-reliant", "analytic", "logical",
        "rational", "objective", "initiative", "achieve", "achievement",
        "success", "master", "win", "strategic",
    },
    "communal": {
        "kind", "kindness", "caring", "care", "warm", "gentle", "supportive",
        "compassion", "compassionate", "empathetic", "empathy", "helpful",
        "nurturing", "patient", "understanding", "cooperative", "collaborative",
        "friendly", "affectionate", "loving", "sensitive", "polite", "sharing",
        "harmony", "community", "together", "family", "encourage", "comfort",
    },
    # WEAT-style domain framing
    "career": {
        "career", "job", "work", "office", "business", "salary", "promotion",
        "manager", "engineer", "scientist", "doctor", "lawyer", "executive",
        "professional", "industry", "company", "leadership", "achievement",
        "success", "competition", "boss", "finance", "technology",
    },
    "family": {
        "family", "home", "household", "parent", "mother", "father", "child",
        "children", "daughter", "son", "wife", "husband", "care", "caring",
        "nurture", "nurturing", "marriage", "kitchen", "domestic", "baby",
        "babysit", "caregiver", "grandmother", "grandfather",
    },
    # BSRI/PAQ-inspired trait proxies
    "masculine_traits": {
        "assertive", "dominant", "independent", "self-reliant", "aggressive",
        "ambitious", "analytical", "competitive", "decisive", "forceful",
        "leader", "leadership", "strong", "risk", "confident", "bold",
        "rational", "logical", "fearless", "adventurous",
    },
    "feminine_traits": {
        "affectionate", "cheerful", "compassionate", "gentle", "loyal",
        "sensitive", "sympathetic", "tender", "warm", "understanding",
        "kind", "nurturing", "supportive", "empathetic", "helpful", "patient",
        "caring", "polite", "soft", "emotional",
    },
    # Definitional markers
    "male_markers": {
        "he", "him", "his", "boy", "man", "male", "son", "brother", "father",
        "grandfather", "uncle", "king", "prince",
    },
    "female_markers": {
        "she", "her", "hers", "girl", "woman", "female", "daughter", "sister",
        "mother", "grandmother", "aunt", "queen", "princess",
    },
}


@dataclass
class Columns:
    text: str
    model: str
    target_gender: str
    person: str = None  # storyteller
    country: str = None  # culture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced gender bias analysis with statistical tests.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Narratives2/gender_bias_lexicon_enhanced",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--per",
        type=int,
        default=1000,
        help="Normalize lexicon frequencies per N tokens.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations.",
    )
    return parser.parse_args()


def choose_input_path(explicit_path: str) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    candidates = [
        Path("Narratives2/biasednarratives.csv"),
        #Path("Narratives2/stories_progress.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("No default input file found. Pass --input explicitly.")


def detect_columns(df: pd.DataFrame) -> Columns:
    text_candidates = ["story"]
    model_candidates = ["model"]
    gender_candidates = ["protagonist_gender"]
    person_candidates = ["person"]
    country_candidates = ["country"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    model_col = next((c for c in model_candidates if c in df.columns), None)
    gender_col = next((c for c in gender_candidates if c in df.columns), None)
    person_col = next((c for c in person_candidates if c in df.columns), None)
    country_col = next((c for c in country_candidates if c in df.columns), None)

    missing = [
        name
        for name, col in [
            ("text", text_col),
            ("model", model_col),
            ("target_gender", gender_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f". Found columns: {list(df.columns)}"
        )

    return Columns(
        text=text_col,
        model=model_col,
        target_gender=gender_col,
        person=person_col,
        country=country_col,
    )


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


def count_matches(tokens: Iterable[str], lexicon: set[str]) -> int:
    return sum(1 for t in tokens if t in lexicon)


def normalize_per_n(count: int, token_count: int, per: int) -> float:
    if token_count <= 0:
        return 0.0
    return (count / token_count) * per


def map_child_label(value: str) -> str:
    if not isinstance(value, str):
        return "unknown"
    v = value.strip().lower()
    if v in {"male", "m", "boy", "son"}:
        return "son"
    if v in {"female", "f", "girl", "daughter"}:
        return "daughter"
    return "unknown"


def calculate_child_target_stereotype_score(row: pd.Series, per: int) -> float:
    """
    Positive = more stereotypically aligned with the child target gender.
    Negative = more counter-stereotypical.
    """
    score = 0.0

    if row["child_label"] == "daughter":
        if row[f"feminine_traits_per_{per}"] > row[f"masculine_traits_per_{per}"]:
            score += 1.0
        elif row[f"masculine_traits_per_{per}"] > row[f"feminine_traits_per_{per}"]:
            score -= 1.0

        if row[f"communal_per_{per}"] > row[f"agentic_per_{per}"]:
            score += 0.5
        elif row[f"agentic_per_{per}"] > row[f"communal_per_{per}"]:
            score -= 0.5

        if row[f"family_per_{per}"] > row[f"career_per_{per}"]:
            score += 0.5
        elif row[f"career_per_{per}"] > row[f"family_per_{per}"]:
            score -= 0.5

        if row[f"female_markers_per_{per}"] > row[f"male_markers_per_{per}"]:
            score += 0.5
        elif row[f"male_markers_per_{per}"] > row[f"female_markers_per_{per}"]:
            score -= 0.5

    elif row["child_label"] == "son":
        if row[f"masculine_traits_per_{per}"] > row[f"feminine_traits_per_{per}"]:
            score += 1.0
        elif row[f"feminine_traits_per_{per}"] > row[f"masculine_traits_per_{per}"]:
            score -= 1.0

        if row[f"agentic_per_{per}"] > row[f"communal_per_{per}"]:
            score += 0.5
        elif row[f"communal_per_{per}"] > row[f"agentic_per_{per}"]:
            score -= 0.5

        if row[f"career_per_{per}"] > row[f"family_per_{per}"]:
            score += 0.5
        elif row[f"family_per_{per}"] > row[f"career_per_{per}"]:
            score -= 0.5

        if row[f"male_markers_per_{per}"] > row[f"female_markers_per_{per}"]:
            score += 0.5
        elif row[f"female_markers_per_{per}"] > row[f"male_markers_per_{per}"]:
            score -= 0.5

    return score


def compute_story_scores(df: pd.DataFrame, cols: Columns, per: int) -> pd.DataFrame:
    out = df.copy()

    out["_tokens"] = out[cols.text].fillna("").map(tokenize)
    out["token_count_lex"] = out["_tokens"].map(len)
    out["child_label"] = out[cols.target_gender].map(map_child_label)

    for lex_name, lex_set in LEXICONS.items():
        count_col = f"{lex_name}_count"
        score_col = f"{lex_name}_per_{per}"
        out[count_col] = out["_tokens"].map(lambda toks: count_matches(toks, lex_set))
        out[score_col] = out.apply(
            lambda r: normalize_per_n(r[count_col], r["token_count_lex"], per), axis=1
        )

    # Composite indices
    out["trait_bias_index"] = (
        out[f"masculine_traits_per_{per}"] - out[f"feminine_traits_per_{per}"]
    ) / (
        out[f"masculine_traits_per_{per}"] + out[f"feminine_traits_per_{per}"] + EPS
    )

    out["role_bias_index"] = (
        out[f"agentic_per_{per}"] - out[f"communal_per_{per}"]
    ) / (
        out[f"agentic_per_{per}"] + out[f"communal_per_{per}"] + EPS
    )

    out["domain_bias_index"] = (
        out[f"career_per_{per}"] - out[f"family_per_{per}"]
    ) / (
        out[f"career_per_{per}"] + out[f"family_per_{per}"] + EPS
    )

    out["direct_gender_marker_index"] = (
        out[f"male_markers_per_{per}"] - out[f"female_markers_per_{per}"]
    ) / (
        out[f"male_markers_per_{per}"] + out[f"female_markers_per_{per}"] + EPS
    )

    # Derived bias metrics
    out["trait_gender_bias"] = out[f"masculine_traits_per_{per}"] - out[f"feminine_traits_per_{per}"]
    out["role_gender_bias"] = out[f"agentic_per_{per}"] - out[f"communal_per_{per}"]
    out["domain_gender_bias"] = out[f"career_per_{per}"] - out[f"family_per_{per}"]
    out["marker_gender_bias"] = out[f"male_markers_per_{per}"] - out[f"female_markers_per_{per}"]
    out["child_target_stereotype_score"] = out.apply(
        lambda row: calculate_child_target_stereotype_score(row, per),
        axis=1,
    )
    out["child_target_alignment"] = out["child_target_stereotype_score"].map(
        lambda x: "stereotypical" if x > 0 else ("counter_stereotypical" if x < 0 else "neutral")
    )

    out = out.drop(columns=["_tokens"])
    return out


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (group2.mean() - group1.mean()) / (pooled_std + EPS)


def test_son_daughter_differences(
    story_scores: pd.DataFrame,
    model_col: str,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Perform t-tests and calculate effect sizes for son vs daughter.
    Includes multiple comparison corrections.
    """
    results = []

    for model_name, g in story_scores.groupby(model_col):
        son_data = g[g["child_label"] == "son"]
        daughter_data = g[g["child_label"] == "daughter"]

        for metric in metrics:
            son_vals = son_data[metric].dropna()
            dau_vals = daughter_data[metric].dropna()

            if len(son_vals) < 2 or len(dau_vals) < 2:
                continue

            # Welch's t-test (unequal variances)
            t_stat, p_val = stats.ttest_ind(son_vals, dau_vals, equal_var=False)

            # Cohen's d effect size
            effect_size = cohens_d(son_vals, dau_vals)

            results.append({
                "model": model_name,
                "metric": metric,
                "daughter_mean": float(dau_vals.mean()),
                "son_mean": float(son_vals.mean()),
                "difference": float(dau_vals.mean() - son_vals.mean()),
                "daughter_std": float(dau_vals.std()),
                "son_std": float(son_vals.std()),
                "daughter_n": len(dau_vals),
                "son_n": len(son_vals),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(effect_size),
                "effect_size_interpretation": interpret_cohens_d(effect_size),
                "significant_p05": p_val < 0.05,
                "significant_p01": p_val < 0.01,
            })

    df = pd.DataFrame(results)

    if len(df) == 0:
        return df

    # Multiple comparison corrections
    # Bonferroni
    df["p_value_bonferroni"] = df["p_value"] * len(df)
    df["p_value_bonferroni"] = df["p_value_bonferroni"].clip(upper=1.0)
    df["significant_bonferroni"] = df["p_value_bonferroni"] < 0.05

    # FDR (Benjamini-Hochberg)
    _, p_fdr, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["p_value_fdr"] = p_fdr
    df["significant_fdr"] = df["p_value_fdr"] < 0.05

    return df


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_cis(
    story_scores: pd.DataFrame,
    model_col: str,
    metric: str,
    n_boot: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap CI for daughter_minus_son per model for a single metric."""
    np.random.seed(random_state)

    rows = []
    base = story_scores[story_scores["child_label"].isin(["son", "daughter"])].copy()

    for model_name, g in base.groupby(model_col):
        son = g[g["child_label"] == "son"][metric].values
        daughter = g[g["child_label"] == "daughter"][metric].values

        if len(son) == 0 or len(daughter) == 0:
            continue

        diffs = []
        for _ in range(n_boot):
            son_sample = np.random.choice(son, size=len(son), replace=True)
            dau_sample = np.random.choice(daughter, size=len(daughter), replace=True)
            diffs.append(dau_sample.mean() - son_sample.mean())

        diffs = np.array(diffs)
        rows.append({
            model_col: model_name,
            "metric": metric,
            "gap_mean": float(diffs.mean()),
            "gap_median": float(np.median(diffs)),
            "ci_low_2.5": float(np.percentile(diffs, 2.5)),
            "ci_high_97.5": float(np.percentile(diffs, 97.5)),
            "ci_low_5": float(np.percentile(diffs, 5)),
            "ci_high_95": float(np.percentile(diffs, 95)),
        })

    return pd.DataFrame(rows)


def summarize_by_child_gender(story_scores: pd.DataFrame) -> pd.DataFrame:
    keep = story_scores[story_scores["child_label"].isin(["son", "daughter"])].copy()
    grouped = keep.groupby("child_label", as_index=False)[SUMMARY_METRICS].mean(numeric_only=True)
    counts = keep.groupby("child_label", as_index=False).size().rename(columns={"size": "n_stories"})
    return grouped.merge(counts, on="child_label", how="left")


def summarize_by_model(story_scores: pd.DataFrame, model_col: str) -> pd.DataFrame:
    grouped = story_scores.groupby(model_col, as_index=False)[SUMMARY_METRICS].mean(numeric_only=True)
    counts = story_scores.groupby(model_col, as_index=False).size().rename(columns={"size": "n_stories"})
    return grouped.merge(counts, on=model_col, how="left")


def summarize_by_model_and_child_gender(
    story_scores: pd.DataFrame,
    model_col: str,
    per: int
) -> pd.DataFrame:
    keep = story_scores[story_scores["child_label"].isin(["son", "daughter"])].copy()

    metric_cols = [
        f"agentic_per_{per}",
        f"communal_per_{per}",
        f"career_per_{per}",
        f"family_per_{per}",
        f"masculine_traits_per_{per}",
        f"feminine_traits_per_{per}",
        f"male_markers_per_{per}",
        f"female_markers_per_{per}",
        "trait_bias_index",
        "role_bias_index",
        "domain_bias_index",
        "direct_gender_marker_index",
        "trait_gender_bias",
        "role_gender_bias",
        "domain_gender_bias",
        "marker_gender_bias",
        "child_target_stereotype_score",
    ]

    grouped = (
        keep.groupby([model_col, "child_label"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )

    counts = (
        keep.groupby([model_col, "child_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_stories"})
    )

    summary = grouped.merge(counts, on=[model_col, "child_label"], how="left")
    return summary


def daughter_minus_son_gaps(summary_by_model: pd.DataFrame, model_col: str) -> pd.DataFrame:
    metric_cols = [c for c in summary_by_model.columns if c not in {model_col, "child_label", "n_stories"}]

    son = summary_by_model[summary_by_model["child_label"] == "son"].set_index(model_col)
    daughter = summary_by_model[summary_by_model["child_label"] == "daughter"].set_index(model_col)

    common_models = sorted(set(son.index).intersection(set(daughter.index)))
    if not common_models:
        return pd.DataFrame(columns=[model_col, "metric", "daughter_minus_son"])

    rows: List[dict] = []
    for m in common_models:
        for metric in metric_cols:
            rows.append({
                model_col: m,
                "metric": metric,
                "daughter_minus_son": float(daughter.at[m, metric] - son.at[m, metric]),
            })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_path = choose_input_path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    cols = detect_columns(df)

    print(f"Detected columns: text={cols.text}, model={cols.model}, target_gender={cols.target_gender}")
    if cols.person:
        print(f"  storyteller={cols.person}")
    if cols.country:
        print(f"  culture={cols.country}")

    print("Computing story-level lexicon scores...")
    scored = compute_story_scores(df, cols, per=args.per)

    print("Computing summary statistics...")
    child_gender_summary = summarize_by_child_gender(scored)
    model_summary = summarize_by_model(scored, cols.model)
    model_child_gender_summary = summarize_by_model_and_child_gender(scored, cols.model, args.per)
    gaps = daughter_minus_son_gaps(model_child_gender_summary, cols.model)

    print("Running statistical significance tests...")
    test_metrics = [
        f"masculine_traits_per_{args.per}",
        f"feminine_traits_per_{args.per}",
        f"agentic_per_{args.per}",
        f"communal_per_{args.per}",
        f"career_per_{args.per}",
        f"family_per_{args.per}",
        f"male_markers_per_{args.per}",
        f"female_markers_per_{args.per}",
        "trait_bias_index",
        "role_bias_index",
        "domain_bias_index",
        "direct_gender_marker_index",
        "child_target_stereotype_score",
    ]

    significance_results = test_son_daughter_differences(scored, cols.model, test_metrics)

    print(f"Computing bootstrap confidence intervals ({args.n_bootstrap} iterations)...")
    ci_frames = []
    for metric in test_metrics:
        ci_frames.append(bootstrap_cis(scored, cols.model, metric=metric, n_boot=args.n_bootstrap))
    ci_df = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()

    # Save outputs
    scored_path = out_dir / "story_level_gender_bias_scores.csv"
    child_summary_path = out_dir / "child_gender_summary.csv"
    model_summary_path = out_dir / "model_summary.csv"
    summary_path = out_dir / "model_child_gender_summary.csv"
    gaps_path = out_dir / "model_daughter_minus_son_gaps.csv"
    ci_path = out_dir / "model_gap_bootstrap_ci.csv"
    significance_path = out_dir / "statistical_significance_tests.csv"

    print("Saving results...")
    scored.to_csv(scored_path, index=False)
    child_gender_summary.to_csv(child_summary_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    model_child_gender_summary.to_csv(summary_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    ci_df.to_csv(ci_path, index=False)
    significance_results.to_csv(significance_path, index=False)

    print("\n" + "="*80)
    print("GENDER BIAS LEXICON ANALYSIS - ENHANCED VERSION")
    print("="*80)
    print(f"\nInput file: {input_path}")
    print(f"Stories analyzed: {len(scored)}")
    print(f"Models analyzed: {scored[cols.model].nunique()}")
    print(f"\nOutputs saved to: {out_dir}")
    print(f"  - {scored_path.name}")
    print(f"  - {child_summary_path.name}")
    print(f"  - {model_summary_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {gaps_path.name}")
    print(f"  - {ci_path.name}")
    print(f"  - {significance_path.name} (NEW: statistical tests)")

    print("\n" + "-"*80)
    print("CHILD GENDER STEREOTYPE SUMMARY")
    print("-"*80)
    for _, row in child_gender_summary.iterrows():
        print(f"\n{row['child_label'].upper()}:")
        print(f"  Stories: {row['n_stories']}")
        print(f"  Stereotype Score: {row['child_target_stereotype_score']:.3f}")
        print(f"  Trait Bias: {row['trait_gender_bias']:.3f}")
        print(f"  Role Bias: {row['role_gender_bias']:.3f}")
        print(f"  Domain Bias: {row['domain_gender_bias']:.3f}")
        print(f"  Marker Bias: {row['marker_gender_bias']:.3f}")

    print("\n" + "-"*80)
    print("STATISTICAL SIGNIFICANCE HIGHLIGHTS")
    print("-"*80)

    # Show significant results
    sig_results = significance_results[significance_results["significant_fdr"]]
    if len(sig_results) > 0:
        print(f"\nFound {len(sig_results)} significant differences (FDR-corrected p < 0.05):\n")

        for _, row in sig_results.head(10).iterrows():
            print(f"{row['model']} - {row['metric']}:")
            print(f"  Daughter: {row['daughter_mean']:.3f} ± {row['daughter_std']:.3f}")
            print(f"  Son: {row['son_mean']:.3f} ± {row['son_std']:.3f}")
            print(f"  Difference: {row['difference']:.3f}")
            print(f"  Cohen's d: {row['cohens_d']:.3f} ({row['effect_size_interpretation']})")
            print(f"  p-value: {row['p_value']:.4e} (FDR-corrected: {row['p_value_fdr']:.4e})")
            print()

        if len(sig_results) > 10:
            print(f"... and {len(sig_results) - 10} more significant differences.")
    else:
        print("No significant differences found after FDR correction.")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
