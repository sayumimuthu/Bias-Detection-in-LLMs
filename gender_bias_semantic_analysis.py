#!/usr/bin/env python3
"""
Semantic similarity-based gender bias analysis using sentence transformers.

This complements the lexicon-based analysis by using neural embeddings to:
- Capture paraphrases and synonyms
- Handle negation and context
- Detect implicit stereotypes
- Measure semantic similarity to stereotype concepts

Parallel to gender_bias_lexicon_analysis_enhanced.py but uses transformers
instead of word matching.

Usage:
    python3 gender_bias_semantic_analysis.py
    python3 gender_bias_semantic_analysis.py --input Narratives2/biasednarratives.csv
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests

# Suppress warnings
warnings.filterwarnings('ignore')

EPS = 1e-9

# Stereotype concept definitions
# Each concept defined by multiple phrases to capture full meaning
STEREOTYPE_CONCEPTS = {
    # Gender trait dimensions (inspired by BSRI/PAQ)
    'feminine_traits': [
        'affectionate, gentle, warm, and tender behavior',
        'caring, nurturing, and compassionate nature',
        'sensitive, understanding, and empathetic character',
        'kind, supportive, and considerate actions',
        'patient, loyal, and helpful personality'
    ],
    'masculine_traits': [
        'assertive, dominant, and forceful behavior',
        'independent, self-reliant, and confident nature',
        'strong, competitive, and ambitious character',
        'analytical, logical, and rational thinking',
        'bold, decisive, and fearless actions'
    ],

    # Role dimensions (agentic vs communal)
    'communal_roles': [
        'helping others and providing emotional support',
        'cooperation, collaboration, and working together',
        'building relationships and community connections',
        'showing kindness and caring for others',
        'sharing, harmony, and togetherness'
    ],
    'agentic_roles': [
        'taking initiative and showing leadership',
        'ambitious pursuit of achievement and success',
        'competitive striving and determination to win',
        'assertive decision-making and independence',
        'confident pursuit of goals and mastery'
    ],

    # Domain dimensions (career vs family)
    'career_domain': [
        'professional work and career achievement',
        'business, office, and workplace activities',
        'job performance and career advancement',
        'professional success and industry leadership',
        'work-related goals and accomplishments'
    ],
    'family_domain': [
        'family life and domestic activities',
        'home, household, and family responsibilities',
        'parenting and childcare activities',
        'family relationships and connections',
        'caring for family members and home life'
    ],

    # Overall stereotype alignment
    'feminine_stereotype': [
        'gentle, caring woman who nurtures and supports others emotionally',
        'kind, empathetic female character focused on relationships and helping',
        'warm, compassionate girl who cares for family and community',
        'sensitive, understanding female who puts others first',
        'patient, nurturing woman devoted to emotional care'
    ],
    'masculine_stereotype': [
        'strong, confident man who leads and takes charge',
        'competitive, ambitious male character pursuing success',
        'assertive, independent boy who solves problems logically',
        'bold, fearless male who takes risks and achieves goals',
        'rational, decisive man focused on winning and mastery'
    ],
}


@dataclass
class Columns:
    text: str
    model: str
    target_gender: str
    person: str = None
    country: str = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic similarity-based gender bias analysis"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Narratives2/gender_bias_semantic",
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations",
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


def map_child_label(value: str) -> str:
    if not isinstance(value, str):
        return "unknown"
    v = value.strip().lower()
    if v in {"male", "m", "boy", "son"}:
        return "son"
    if v in {"female", "f", "girl", "daughter"}:
        return "daughter"
    return "unknown"


def encode_stereotype_concepts(
    model: SentenceTransformer,
) -> Dict[str, np.ndarray]:
    """
    Encode all stereotype concepts as embeddings.
    Each concept is represented by the mean of its phrases.
    """
    print("Encoding stereotype concepts...")
    concept_embeddings = {}

    for concept_name, phrases in STEREOTYPE_CONCEPTS.items():
        # Encode all phrases for this concept
        phrase_embeddings = model.encode(phrases, show_progress_bar=False)
        # Use mean embedding as concept representation
        concept_embeddings[concept_name] = np.mean(phrase_embeddings, axis=0)
        print(f"  ✓ Encoded {concept_name} ({len(phrases)} phrases)")

    return concept_embeddings


def compute_semantic_similarity(
    story_text: str,
    story_embedding: np.ndarray,
    concept_embeddings: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute cosine similarity between story and all stereotype concepts.
    Returns similarity scores (0-1, higher = more similar).
    """
    similarities = {}

    for concept_name, concept_vec in concept_embeddings.items():
        sim = cosine_similarity([story_embedding], [concept_vec])[0][0]
        similarities[f'{concept_name}_similarity'] = float(sim)

    return similarities


def compute_story_scores(
    df: pd.DataFrame,
    cols: Columns,
    model: SentenceTransformer,
) -> pd.DataFrame:
    """
    Compute semantic similarity scores for all stories.
    """
    print("\nEncoding stories and computing similarities...")
    out = df.copy()
    out['child_label'] = out[cols.target_gender].map(map_child_label)

    # Encode concepts once
    concept_embeddings = encode_stereotype_concepts(model)

    # Encode all stories
    print(f"\nEncoding {len(out)} stories...")
    stories = out[cols.text].fillna("").tolist()
    story_embeddings = model.encode(
        stories,
        show_progress_bar=True,
        batch_size=32,
    )

    # Compute similarities for each story
    print("Computing semantic similarities...")
    all_similarities = []
    for i, (story_text, story_emb) in enumerate(zip(stories, story_embeddings)):
        sims = compute_semantic_similarity(story_text, story_emb, concept_embeddings)
        all_similarities.append(sims)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(stories)} stories")

    # Add similarities to dataframe
    sim_df = pd.DataFrame(all_similarities)
    out = pd.concat([out, sim_df], axis=1)

    # Compute composite indices (parallel to lexicon analysis)
    # Trait bias: masculine - feminine (normalized)
    out['trait_bias_index_semantic'] = (
        out['masculine_traits_similarity'] - out['feminine_traits_similarity']
    ) / (
        out['masculine_traits_similarity'] + out['feminine_traits_similarity'] + EPS
    )

    # Role bias: agentic - communal (normalized)
    out['role_bias_index_semantic'] = (
        out['agentic_roles_similarity'] - out['communal_roles_similarity']
    ) / (
        out['agentic_roles_similarity'] + out['communal_roles_similarity'] + EPS
    )

    # Domain bias: career - family (normalized)
    out['domain_bias_index_semantic'] = (
        out['career_domain_similarity'] - out['family_domain_similarity']
    ) / (
        out['career_domain_similarity'] + out['family_domain_similarity'] + EPS
    )

    # Overall stereotype alignment
    out['stereotype_alignment_index_semantic'] = (
        out['masculine_stereotype_similarity'] - out['feminine_stereotype_similarity']
    ) / (
        out['masculine_stereotype_similarity'] +
        out['feminine_stereotype_similarity'] + EPS
    )

    # Derived bias scores (not normalized, for comparison)
    out['trait_gender_bias_semantic'] = (
        out['masculine_traits_similarity'] - out['feminine_traits_similarity']
    )
    out['role_gender_bias_semantic'] = (
        out['agentic_roles_similarity'] - out['communal_roles_similarity']
    )
    out['domain_gender_bias_semantic'] = (
        out['career_domain_similarity'] - out['family_domain_similarity']
    )
    out['stereotype_gender_bias_semantic'] = (
        out['masculine_stereotype_similarity'] - out['feminine_stereotype_similarity']
    )

    # Child target stereotype score (semantic version)
    out['child_target_stereotype_score_semantic'] = out.apply(
        lambda row: calculate_child_target_stereotype_score_semantic(row),
        axis=1,
    )

    return out


def calculate_child_target_stereotype_score_semantic(row: pd.Series) -> float:
    """
    Semantic version of stereotype alignment score.
    Positive = aligns with traditional stereotypes for child gender.
    """
    score = 0.0

    if row['child_label'] == 'daughter':
        # Expect higher feminine similarity for daughters
        if row['feminine_traits_similarity'] > row['masculine_traits_similarity']:
            score += 1.0
        elif row['masculine_traits_similarity'] > row['feminine_traits_similarity']:
            score -= 1.0

        if row['communal_roles_similarity'] > row['agentic_roles_similarity']:
            score += 0.5
        elif row['agentic_roles_similarity'] > row['communal_roles_similarity']:
            score -= 0.5

        if row['family_domain_similarity'] > row['career_domain_similarity']:
            score += 0.5
        elif row['career_domain_similarity'] > row['family_domain_similarity']:
            score -= 0.5

        if row['feminine_stereotype_similarity'] > row['masculine_stereotype_similarity']:
            score += 0.5
        elif row['masculine_stereotype_similarity'] > row['feminine_stereotype_similarity']:
            score -= 0.5

    elif row['child_label'] == 'son':
        # Expect higher masculine similarity for sons
        if row['masculine_traits_similarity'] > row['feminine_traits_similarity']:
            score += 1.0
        elif row['feminine_traits_similarity'] > row['masculine_traits_similarity']:
            score -= 1.0

        if row['agentic_roles_similarity'] > row['communal_roles_similarity']:
            score += 0.5
        elif row['communal_roles_similarity'] > row['agentic_roles_similarity']:
            score -= 0.5

        if row['career_domain_similarity'] > row['family_domain_similarity']:
            score += 0.5
        elif row['family_domain_similarity'] > row['career_domain_similarity']:
            score -= 0.5

        if row['masculine_stereotype_similarity'] > row['feminine_stereotype_similarity']:
            score += 0.5
        elif row['feminine_stereotype_similarity'] > row['masculine_stereotype_similarity']:
            score -= 0.5

    return score


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group2.mean() - group1.mean()) / (pooled_std + EPS)


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


def test_son_daughter_differences(
    story_scores: pd.DataFrame,
    model_col: str,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Perform t-tests and calculate effect sizes for son vs daughter.
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

            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(son_vals, dau_vals, equal_var=False)
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
    df["p_value_bonferroni"] = df["p_value"] * len(df)
    df["p_value_bonferroni"] = df["p_value_bonferroni"].clip(upper=1.0)
    df["significant_bonferroni"] = df["p_value_bonferroni"] < 0.05

    # FDR correction
    _, p_fdr, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["p_value_fdr"] = p_fdr
    df["significant_fdr"] = df["p_value_fdr"] < 0.05

    return df


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


def summarize_by_child_gender(story_scores: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    keep = story_scores[story_scores["child_label"].isin(["son", "daughter"])].copy()
    grouped = keep.groupby("child_label", as_index=False)[metrics].mean(numeric_only=True)
    counts = keep.groupby("child_label", as_index=False).size().rename(columns={"size": "n_stories"})
    return grouped.merge(counts, on="child_label", how="left")


def summarize_by_model(story_scores: pd.DataFrame, model_col: str, metrics: List[str]) -> pd.DataFrame:
    grouped = story_scores.groupby(model_col, as_index=False)[metrics].mean(numeric_only=True)
    counts = story_scores.groupby(model_col, as_index=False).size().rename(columns={"size": "n_stories"})
    return grouped.merge(counts, on=model_col, how="left")


def summarize_by_model_and_child_gender(
    story_scores: pd.DataFrame,
    model_col: str,
    metrics: List[str]
) -> pd.DataFrame:
    keep = story_scores[story_scores["child_label"].isin(["son", "daughter"])].copy()
    grouped = keep.groupby([model_col, "child_label"], as_index=False)[metrics].mean(numeric_only=True)
    counts = keep.groupby([model_col, "child_label"], as_index=False).size().rename(columns={"size": "n_stories"})
    return grouped.merge(counts, on=[model_col, "child_label"], how="left")


def daughter_minus_son_gaps(summary_by_model: pd.DataFrame, model_col: str, metrics: List[str]) -> pd.DataFrame:
    son = summary_by_model[summary_by_model["child_label"] == "son"].set_index(model_col)
    daughter = summary_by_model[summary_by_model["child_label"] == "daughter"].set_index(model_col)

    common_models = sorted(set(son.index).intersection(set(daughter.index)))
    if not common_models:
        return pd.DataFrame(columns=[model_col, "metric", "daughter_minus_son"])

    rows = []
    for m in common_models:
        for metric in metrics:
            if metric in son.columns and metric in daughter.columns:
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

    print("="*80)
    print("SEMANTIC SIMILARITY-BASED GENDER BIAS ANALYSIS")
    print("="*80)
    print(f"\nLoading sentence transformer model: {args.model_name}")

    # Load sentence transformer model
    model = SentenceTransformer(args.model_name)
    print(f"Model loaded successfully!")

    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    cols = detect_columns(df)

    print(f"Detected columns: text={cols.text}, model={cols.model}, target_gender={cols.target_gender}")
    if cols.person:
        print(f"  storyteller={cols.person}")
    if cols.country:
        print(f"  culture={cols.country}")

    # Compute semantic similarity scores
    scored = compute_story_scores(df, cols, model)

    # Define metrics to analyze
    similarity_metrics = [
        'feminine_traits_similarity',
        'masculine_traits_similarity',
        'communal_roles_similarity',
        'agentic_roles_similarity',
        'career_domain_similarity',
        'family_domain_similarity',
        'feminine_stereotype_similarity',
        'masculine_stereotype_similarity',
        'trait_bias_index_semantic',
        'role_bias_index_semantic',
        'domain_bias_index_semantic',
        'stereotype_alignment_index_semantic',
        'trait_gender_bias_semantic',
        'role_gender_bias_semantic',
        'domain_gender_bias_semantic',
        'stereotype_gender_bias_semantic',
        'child_target_stereotype_score_semantic',
    ]

    print("\nComputing summary statistics...")
    child_gender_summary = summarize_by_child_gender(scored, similarity_metrics)
    model_summary = summarize_by_model(scored, cols.model, similarity_metrics)
    model_child_gender_summary = summarize_by_model_and_child_gender(
        scored, cols.model, similarity_metrics
    )

    # Compute gaps
    gaps = daughter_minus_son_gaps(
        model_child_gender_summary,
        cols.model,
        similarity_metrics
    )

    print("Running statistical significance tests...")
    significance_results = test_son_daughter_differences(
        scored, cols.model, similarity_metrics
    )

    print(f"Computing bootstrap confidence intervals ({args.n_bootstrap} iterations)...")
    ci_frames = []
    for metric in similarity_metrics:
        ci_frames.append(bootstrap_cis(scored, cols.model, metric=metric, n_boot=args.n_bootstrap))
    ci_df = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()

    # Save outputs
    print("\nSaving results...")
    scored.to_csv(out_dir / "story_level_semantic_scores.csv", index=False)
    child_gender_summary.to_csv(out_dir / "child_gender_summary.csv", index=False)
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    model_child_gender_summary.to_csv(out_dir / "model_child_gender_summary.csv", index=False)
    gaps.to_csv(out_dir / "model_daughter_minus_son_gaps.csv", index=False)
    ci_df.to_csv(out_dir / "model_gap_bootstrap_ci.csv", index=False)
    significance_results.to_csv(out_dir / "statistical_significance_tests.csv", index=False)

    print("\n" + "="*80)
    print("SEMANTIC SIMILARITY ANALYSIS - COMPLETE")
    print("="*80)
    print(f"\nInput file: {input_path}")
    print(f"Stories analyzed: {len(scored)}")
    print(f"Models analyzed: {scored[cols.model].nunique()}")
    print(f"Sentence transformer: {args.model_name}")
    print(f"\nOutputs saved to: {out_dir}")
    print(f"  - story_level_semantic_scores.csv")
    print(f"  - child_gender_summary.csv")
    print(f"  - model_summary.csv")
    print(f"  - model_child_gender_summary.csv")
    print(f"  - model_daughter_minus_son_gaps.csv")
    print(f"  - model_gap_bootstrap_ci.csv")
    print(f"  - statistical_significance_tests.csv")

    print("\n" + "-"*80)
    print("CHILD GENDER SEMANTIC SIMILARITY SUMMARY")
    print("-"*80)
    for _, row in child_gender_summary.iterrows():
        print(f"\n{row['child_label'].upper()}:")
        print(f"  Stories: {row['n_stories']}")
        print(f"  Feminine Traits Similarity: {row['feminine_traits_similarity']:.3f}")
        print(f"  Masculine Traits Similarity: {row['masculine_traits_similarity']:.3f}")
        print(f"  Communal Roles Similarity: {row['communal_roles_similarity']:.3f}")
        print(f"  Agentic Roles Similarity: {row['agentic_roles_similarity']:.3f}")
        print(f"  Stereotype Score: {row['child_target_stereotype_score_semantic']:.3f}")

    print("\n" + "-"*80)
    print("STATISTICAL SIGNIFICANCE HIGHLIGHTS")
    print("-"*80)

    sig_results = significance_results[significance_results["significant_fdr"]]
    if len(sig_results) > 0:
        print(f"\nFound {len(sig_results)} significant differences (FDR-corrected p < 0.05):\n")

        for _, row in sig_results.head(10).iterrows():
            print(f"{row['model']} - {row['metric']}:")
            print(f"  Daughter: {row['daughter_mean']:.4f} ± {row['daughter_std']:.4f}")
            print(f"  Son: {row['son_mean']:.4f} ± {row['son_std']:.4f}")
            print(f"  Difference: {row['difference']:.4f}")
            print(f"  Cohen's d: {row['cohens_d']:.3f} ({row['effect_size_interpretation']})")
            print(f"  p-value: {row['p_value']:.4e} (FDR: {row['p_value_fdr']:.4e})")
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
