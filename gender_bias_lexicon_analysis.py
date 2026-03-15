#!/usr/bin/env python3
"""
Lexicon-based gender bias analysis for story-generation outputs.

What this script does:
1) Loads a story dataset (CSV) from this project.
2) Scores each story against research-backed gender-related lexicon axes.
3) Compares scores for son vs daughter prompts per model.
4) Writes story-level and summary CSV outputs.

Lexicon axes (compact research-inspired sets):
- Agentic vs communal language (Gaucher et al.-style framing)
- Career vs family framing (WEAT-style domains)
- Masculine-coded vs feminine-coded trait words (BSRI/PAQ-inspired)

Notes:
- The BSRI/PAQ originals are psychometric scales, not full open NLP lexicons.
  This script uses compact proxy term sets inspired by that literature.
- You should report this as "lexicon-inspired" and include limitations.

Usage:
    python3 gender_bias_lexicon_analysis.py
    python3 gender_bias_lexicon_analysis.py --input Narratives/clean_stories_for_analysis.csv
    python3 gender_bias_lexicon_analysis.py --output-dir Narratives/visualizations/gender_bias
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


TOKEN_RE = re.compile(r"[a-zA-Z']+")
EPS = 1e-9


# Compact, research-inspired lexicons.
# Keep terms lowercased; scoring uses exact token match after normalization.
LEXICONS: Dict[str, set[str]] = {
    # Agentic vs communal framing (job-ad and gender language literature)
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
    # Definitional markers (for direct gendered references)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze son/daughter bias with lexicons.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input CSV path. If omitted, script auto-selects a known dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Narratives2/gender_bias_lexicon",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--per",
        type=int,
        default=1000,
        help="Normalize lexicon frequencies per N tokens (default: 1000).",
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
        Path("Narratives2/stories_progress.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No default input file found. Pass --input explicitly."
    )


def detect_columns(df: pd.DataFrame) -> Columns:
    text_candidates = ["story", "text", "content"]
    model_candidates = ["model", "provider", "llm"]
    gender_candidates = ["protagonist_gender", "child_gender", "target_gender", "gender"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    model_col = next((c for c in model_candidates if c in df.columns), None)
    gender_col = next((c for c in gender_candidates if c in df.columns), None)

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

    return Columns(text=text_col, model=model_col, target_gender=gender_col)


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

    # Composite indices in [-1, 1] style.
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

    # Positive means more male-coded direct mentions.
    out["direct_gender_marker_index"] = (
        out[f"male_markers_per_{per}"] - out[f"female_markers_per_{per}"]
    ) / (
        out[f"male_markers_per_{per}"] + out[f"female_markers_per_{per}"] + EPS
    )

    out = out.drop(columns=["_tokens"])
    return out


def summarize_by_model(story_scores: pd.DataFrame, model_col: str, per: int) -> pd.DataFrame:
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
            rows.append(
                {
                    model_col: m,
                    "metric": metric,
                    "daughter_minus_son": float(daughter.at[m, metric] - son.at[m, metric]),
                }
            )

    return pd.DataFrame(rows)


def bootstrap_cis(
    story_scores: pd.DataFrame,
    model_col: str,
    metric: str,
    n_boot: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap CI for daughter_minus_son per model for a single metric.
    """
    rng = pd.Series(range(n_boot)).sample(frac=1, random_state=random_state).tolist()
    # rng is only used to vary random_state reproducibly across iterations.

    rows = []
    base = story_scores[story_scores["child_label"].isin(["son", "daughter"])].copy()
    for model_name, g in base.groupby(model_col):
        son = g[g["child_label"] == "son"][metric]
        daughter = g[g["child_label"] == "daughter"][metric]
        if son.empty or daughter.empty:
            continue

        diffs = []
        for i in rng:
            son_s = son.sample(n=len(son), replace=True, random_state=i)
            dau_s = daughter.sample(n=len(daughter), replace=True, random_state=i + 10000)
            diffs.append(float(dau_s.mean() - son_s.mean()))

        diffs_s = pd.Series(diffs)
        rows.append(
            {
                model_col: model_name,
                "metric": metric,
                "gap_mean": float(diffs_s.mean()),
                "ci_low_2.5": float(diffs_s.quantile(0.025)),
                "ci_high_97.5": float(diffs_s.quantile(0.975)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_path = choose_input_path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    cols = detect_columns(df)

    scored = compute_story_scores(df, cols, per=args.per)

    model_summary = summarize_by_model(scored, cols.model, args.per)
    gaps = daughter_minus_son_gaps(model_summary, cols.model)

    # Optional bootstrap CIs for key composite metrics.
    ci_frames = []
    for metric in ["trait_bias_index", "role_bias_index", "domain_bias_index", "direct_gender_marker_index"]:
        ci_frames.append(bootstrap_cis(scored, cols.model, metric=metric, n_boot=500))
    ci_df = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()

    scored_path = out_dir / "story_level_gender_bias_scores.csv"
    summary_path = out_dir / "model_child_gender_summary.csv"
    gaps_path = out_dir / "model_daughter_minus_son_gaps.csv"
    ci_path = out_dir / "model_gap_bootstrap_ci.csv"

    scored.to_csv(scored_path, index=False)
    model_summary.to_csv(summary_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    ci_df.to_csv(ci_path, index=False)

    print("Gender Bias Lexicon Analysis Complete")
    print(f"Input file: {input_path}")
    print(f"Detected columns: text={cols.text}, model={cols.model}, target_gender={cols.target_gender}")
    print(f"Stories analyzed: {len(scored)}")
    print(f"Models analyzed: {scored[cols.model].nunique()}")
    print(f"Outputs:")
    print(f"- {scored_path}")
    print(f"- {summary_path}")
    print(f"- {gaps_path}")
    print(f"- {ci_path}")


if __name__ == "__main__":
    main()
