from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

from gender_bias_analysis_new import (
    ALPHA, EPS, FIG_DPI, FIG_EXT, LEXICONS, N_BOOT, PER, TOKEN_RE,
    REGION_ORDER, PALETTE_GENDER,
    _cohens_d, _save,
    enrich, load_data, score_lexicon,
)
from cultural_bias_analysis import CULTURAL_LEXICONS, score_cultural



DEFAULT_OUT        = Path("Narratives3/mitigation")
OLLAMA_HOST        = "http://127.0.0.1:11434"
CHAT_URL           = f"{OLLAMA_HOST}/api/chat"
DEFAULT_MODEL      = "llama3.1:8b"

# Which lexicon categories signal a stereotype (not culture)
BIAS_LEXICON_KEYS  = {
    "feminine_traits", "masculine_traits",
    "communal", "agentic",
    "family",
}
# Which lexicon categories are cultural anchors to preserve
CULTURAL_LEX_KEYS  = set(CULTURAL_LEXICONS.keys())   # from cultural_bias_analysis

BALANCED_VOCAB = {
    "daughter": [
        "curious", "creative", "resourceful", "thoughtful", "determined",
        "clever", "imaginative", "enthusiastic", "observant", "adaptable",
    ],
    "son": [
        "curious", "creative", "resourceful", "thoughtful", "determined",
        "clever", "imaginative", "enthusiastic", "observant", "adaptable",
    ],
}

MIN_IMPROVEMENT    = 0.20   # accept if |stereotype_score| drops by ≥ 20 %
MAX_ITER           = 3      # maximum regeneration attempts per story
OLLAMA_TIMEOUT     = 90
SLEEP_BETWEEN      = 0.5    # seconds between API calls (rate limiting)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",            default="",                 help="CSV / JSONL data path")
    p.add_argument("--output-dir",       default=str(DEFAULT_OUT))
    p.add_argument("--ollama-model",     default=DEFAULT_MODEL)
    p.add_argument("--threshold-sigma",  type=float, default=1.0,
                   help="Flag stories with stereotype_score > mean + N*std (per child gender)")
    p.add_argument("--max-stories",      type=int,   default=0,
                   help="Limit number of stories to process (0 = no limit)")
    p.add_argument("--min-improvement",  type=float, default=MIN_IMPROVEMENT)
    p.add_argument("--max-iter",         type=int,   default=MAX_ITER)
    p.add_argument("--skip-semantic",    action="store_true")
    p.add_argument("--dry-run",          action="store_true",
                   help="Score and flag only; no LLM regeneration calls")
    return p.parse_args()

# 1: Detection 

def compute_thresholds(scored: pd.DataFrame, sigma: float) -> dict[str, float]:
    """
    Per child-gender group: flag threshold = mean + sigma * std of stereotype_score.
    Positive stereotype_score = overly stereotyped for that gender.
    """
    thresholds: dict[str, float] = {}
    for child_label in ("daughter", "son"):
        vals = scored.loc[scored["child_label"] == child_label, "stereotype_score"].dropna()
        thresholds[child_label] = float(vals.mean() + sigma * vals.std())
    return thresholds


def flag_biased_stories(
    scored: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Add 'biased' boolean column: True if story's stereotype_score exceeds
    the per-gender threshold (i.e., it is more stereotyped than expected).
    """
    df = scored.copy()
    df["bias_threshold"] = df["child_label"].map(thresholds)
    df["biased"]         = df["stereotype_score"] > df["bias_threshold"]
    return df

# 2: Attribution 

def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower()) if isinstance(text, str) else []


def attribute_story(story: str) -> tuple[list[str], list[str]]:
    """
    Split lexicon-matched tokens in `story` into:
      bias_targets     — stereotype/trait/communal/agentic tokens (reduce these)
      cultural_anchors — cultural vocabulary tokens (preserve these)

    Returns deduplicated lists, preserving original case where possible.
    """
    tokens = _tokenize(story)

    # Bias-category token set
    bias_lex: set[str] = set()
    for key in BIAS_LEXICON_KEYS:
        if key in LEXICONS:
            bias_lex |= LEXICONS[key]

    # Cultural-anchor token set
    cultural_lex: set[str] = set()
    for key in CULTURAL_LEX_KEYS:
        cultural_lex |= CULTURAL_LEXICONS[key]

    # Additionally, cultural_anchors take priority if a word appears in both
    bias_targets_raw     = [t for t in tokens if t in bias_lex and t not in cultural_lex]
    cultural_anchors_raw = [t for t in tokens if t in cultural_lex]

    # Deduplicate while preserving order of first occurrence
    seen: set[str] = set()
    bias_targets: list[str] = []
    for t in bias_targets_raw:
        if t not in seen:
            bias_targets.append(t)
            seen.add(t)

    seen = set()
    cultural_anchors: list[str] = []
    for t in cultural_anchors_raw:
        if t not in seen:
            cultural_anchors.append(t)
            seen.add(t)

    return bias_targets, cultural_anchors

# 3: Adaptive Regeneration 

CORRECTION_PROMPT_TEMPLATE = """\
You are a children's story editor specialising in gender-fair narratives.

ORIGINAL STORY:
{story}

YOUR TASK:
Rewrite this story for a {child_label} protagonist.
The story must remain the same length (approximately {word_count} words),
keep the same plot events, setting, and cultural context, but reduce
gender stereotyping in how the protagonist is characterised.

GUIDANCE:
1. Replace or Vary these gender-stereotyped descriptors (use them less, or
   substitute with more gender-neutral alternatives):
   {bias_targets}

2. Preserve these culturally specific words — they reflect authentic cultural
   context and MUST remain in the story:
   {cultural_anchors}

3. Suggested neutral alternatives you may incorporate:
   {balanced_vocab}

4. Do not change: the protagonist's name or gender, the core plot, the setting,
   the cultural references, or the storyteller's voice.

5. Output only the rewritten story.
"""


def build_correction_prompt(
    story: str,
    child_label: str,
    bias_targets: list[str],
    cultural_anchors: list[str],
) -> str:
    word_count = len(story.split())
    bt_str  = ", ".join(bias_targets[:20]) if bias_targets else "none identified"
    ca_str  = ", ".join(cultural_anchors[:15]) if cultural_anchors else "none identified"
    bv_str  = ", ".join(BALANCED_VOCAB.get(child_label, BALANCED_VOCAB["daughter"])[:8])
    return CORRECTION_PROMPT_TEMPLATE.format(
        story=story,
        child_label=child_label,
        word_count=word_count,
        bias_targets=bt_str,
        cultural_anchors=ca_str,
        balanced_vocab=bv_str,
    )


def call_ollama(prompt: str, model: str, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    """Call local Ollama chat endpoint; return generated text or None on failure."""
    payload = {
        "model":  model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7, "top_p": 0.9},
    }
    try:
        r = requests.post(CHAT_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"    [Ollama error] {e}")
        return None


def score_single(text: str, child_label: str) -> dict:
    """Score one story using the lexicon scorer. Returns a dict of bias metrics."""
    row = pd.DataFrame([{"story": text, "child_label": child_label,
                          "word_count": len(text.split())}])
    # Minimal enrichment: is_daughter flag
    row["is_daughter"] = int(child_label == "daughter")
    scored = score_lexicon(row)
    return scored.iloc[0].to_dict()


def star_correct(
    story: str,
    child_label: str,
    original_score: float,
    ollama_model: str,
    min_improvement: float = MIN_IMPROVEMENT,
    max_iter: int = MAX_ITER,
) -> tuple[str, dict, int]:
    """
    Run the STAR algorithm on a single biased story.

    Returns
    -------
    best_story  : corrected story text (or original if no improvement)
    best_scores : dict of lexicon scores for the returned story
    n_iters     : number of regeneration attempts made
    """
    # 2: Attribution
    bias_targets, cultural_anchors = attribute_story(story)

    best_story  = story
    best_scores = score_single(story, child_label)
    best_abs    = abs(original_score)
    n_iters     = 0

    for attempt in range(max_iter):
        n_iters = attempt + 1
        prompt  = build_correction_prompt(story, child_label, bias_targets, cultural_anchors)
        corrected = call_ollama(prompt, ollama_model)
        if not corrected or len(corrected.split()) < 30:
            continue

        new_scores = score_single(corrected, child_label)
        new_abs    = abs(float(new_scores.get("stereotype_score", 0)))

        reduction = (best_abs - new_abs) / (best_abs + EPS)
        if reduction >= min_improvement:
            best_story  = corrected
            best_scores = new_scores
            best_abs    = new_abs
            break  # good enough; stop early

        # Slight update even if below threshold — keep best so far
        if new_abs < best_abs:
            best_story  = corrected
            best_scores = new_scores
            best_abs    = new_abs

        time.sleep(SLEEP_BETWEEN)

    return best_story, best_scores, n_iters

# Semantic similarity 

def load_embedder():
    """Load sentence transformer model if available."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def semantic_similarity(text_a: str, text_b: str, embedder) -> float:
    """Cosine similarity between two stories' sentence embeddings."""
    if embedder is None:
        return float("nan")
    from numpy.linalg import norm
    ea, eb = embedder.encode([text_a, text_b])
    return float(np.dot(ea, eb) / (norm(ea) * norm(eb) + EPS))

# Figures 

def fig_score_shift(results: pd.DataFrame, fig_dir: Path) -> None:
    """
    Scatter plot: before vs after stereotype_score for corrected stories.
    Identity line marks no change; points below = reduction in absolute bias.
    """
    sub = results[results["accepted"]].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [PALETTE_GENDER.get(c, "#888") for c in sub["child_label"]]

    scatter = ax.scatter(
        sub["score_before"], sub["score_after"],
        c=colors, alpha=0.6, s=40, edgecolors="none",
    )
    lim = max(abs(sub["score_before"]).max(), abs(sub["score_after"]).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "--", color="#888", linewidth=1, label="No change")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Stereotype score: BEFORE STAR", fontsize=11)
    ax.set_ylabel("Stereotype score: AFTER STAR",  fontsize=11)
    ax.set_title("STAR Bias Correction: Before vs. After\n"
                 "(points below diagonal = bias reduced)",
                 fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    handles = [Patch(color=PALETTE_GENDER[c], label=c.capitalize())
               for c in ("daughter", "son") if c in sub["child_label"].values]
    ax.legend(handles=handles, fontsize=10, framealpha=0.85)

    # Reduction stats annotation
    mean_red = sub["reduction_pct"].mean()
    n_accepted = len(sub)
    ax.text(0.04, 0.97,
            f"Mean reduction: {mean_red:.1f}%\nn = {n_accepted} stories accepted",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    _save(fig, "fig_m1_score_shift", fig_dir)
    print("  Saved fig_m1_score_shift")


def fig_reduction_by_model(results: pd.DataFrame, fig_dir: Path) -> None:
    """
    Bar chart: mean percentage bias reduction by model family.
    Compares STAR effectiveness across different model architectures.
    """
    sub = results[results["accepted"] & results["model_family"].notna()].copy()
    if sub.empty:
        return

    agg = (sub.groupby("model_family")["reduction_pct"]
             .agg(["mean", "std", "count"])
             .reset_index()
             .sort_values("mean", ascending=False))

    fig, ax = plt.subplots(figsize=(max(6, len(agg) * 0.9 + 1.5), 5))
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["mean"], yerr=agg["std"], capsize=5,
                  color=sns.color_palette("tab10", len(agg)),
                  error_kw={"elinewidth": 1.5}, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["model_family"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean % reduction in |stereotype_score|", fontsize=11)
    ax.set_title("STAR Effectiveness by Model Family\n(mean ± std reduction in bias score)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + row["std"] + 0.5,
                f"n={int(row['count'])}", ha="center", fontsize=8)

    plt.tight_layout()
    _save(fig, "fig_m2_reduction_by_model", fig_dir)
    print("  Saved fig_m2_reduction_by_model")


def fig_cultural_preservation(results: pd.DataFrame, fig_dir: Path) -> None:
    """
    Scatter: cultural_richness before vs after.
    Points near identity line = cultural vocabulary preserved.
    """
    sub = results[results["accepted"] &
                  results["cultural_richness_before"].notna() &
                  results["cultural_richness_after"].notna()].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub["cultural_richness_before"], sub["cultural_richness_after"],
               alpha=0.6, s=35, color="#5B7FA0", edgecolors="none")
    lim = max(sub["cultural_richness_before"].max(),
              sub["cultural_richness_after"].max()) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="#888", linewidth=1.2, label="No change")
    ax.set_xlabel("Cultural richness: BEFORE", fontsize=11)
    ax.set_ylabel("Cultural richness: AFTER",  fontsize=11)
    ax.set_title("Cultural Vocabulary Preservation After STAR\n"
                 "(points near diagonal = culture preserved)",
                 fontsize=12, fontweight="bold")

    corr, pval = stats.pearsonr(sub["cultural_richness_before"],
                                 sub["cultural_richness_after"])
    ax.text(0.04, 0.97, f"r = {corr:.3f}  (p={pval:.3f})",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.legend(fontsize=9, framealpha=0.85)
    plt.tight_layout()
    _save(fig, "fig_m3_cultural_preservation", fig_dir)
    print("  Saved fig_m3_cultural_preservation")


def fig_semantic_similarity(results: pd.DataFrame, fig_dir: Path) -> None:
    """
    Distribution of semantic similarity between original and corrected stories.
    High similarity = narrative content preserved after debiasing.
    """
    sub = results[results["accepted"] & results["semantic_sim"].notna()].copy()
    if sub.empty or sub["semantic_sim"].isna().all():
        print("  fig_m4 skipped — no semantic similarity scores")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sub["semantic_sim"], bins=20, color="#78C47A", edgecolor="white",
            linewidth=0.5)
    ax.axvline(sub["semantic_sim"].mean(), color="darkgreen", linewidth=2,
               linestyle="--", label=f"Mean = {sub['semantic_sim'].mean():.3f}")
    ax.axvline(0.85, color="red", linewidth=1.5, linestyle=":",
               label="Quality threshold (0.85)")
    ax.set_xlabel("Cosine similarity (original vs. corrected story)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Narrative Content Preservation After STAR\n"
                 "(higher = more content preserved)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    pct_above = (sub["semantic_sim"] >= 0.85).mean() * 100
    ax.text(0.02, 0.95, f"{pct_above:.1f}% above 0.85 threshold",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.tight_layout()
    _save(fig, "fig_m4_semantic_similarity", fig_dir)
    print("  Saved fig_m4_semantic_similarity")

# Summary table 

def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate reduction statistics for the paper's results section.
    Rows: overall + per child_gender + per model_family.
    """
    rows: list[dict] = []

    def _row(label: str, sub: pd.DataFrame) -> dict:
        accepted = sub[sub["accepted"]]
        return {
            "group":              label,
            "n_flagged":          len(sub),
            "n_accepted":         len(accepted),
            "acceptance_rate":    round(len(accepted) / max(len(sub), 1), 3),
            "mean_score_before":  round(sub["score_before"].mean(), 4),
            "mean_score_after":   round(accepted["score_after"].mean(), 4) if len(accepted) else float("nan"),
            "mean_reduction_pct": round(accepted["reduction_pct"].mean(), 2) if len(accepted) else float("nan"),
            "pct_above_sim085":   round((accepted["semantic_sim"] >= 0.85).mean() * 100, 1)
                                  if "semantic_sim" in accepted.columns and not accepted["semantic_sim"].isna().all()
                                  else float("nan"),
        }

    rows.append(_row("Overall", results))
    for cl in ("daughter", "son"):
        rows.append(_row(f"child_gender={cl}", results[results["child_label"] == cl]))
    if "model_family" in results.columns:
        for fam in results["model_family"].dropna().unique():
            rows.append(_row(f"model={fam}", results[results["model_family"] == fam]))
    if "country_region" in results.columns:
        for reg in REGION_ORDER:
            sub = results[results.get("country_region", pd.Series()) == reg]
            if len(sub) > 0:
                rows.append(_row(f"region={reg}", sub))

    return pd.DataFrame(rows)

# Main 

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    res_dir = out_dir / "results"
    fig_dir = out_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load, enrich, score 
    print("\nLoading and scoring stories…")
    raw    = load_data(args.input)
    df     = enrich(raw)
    scored = score_lexicon(df)
    # Add cultural richness scores
    scored = score_cultural(scored)

    # Drop failed/empty stories
    scored = scored[
        scored["story"].str.strip().ne("") &
        ~scored["story"].str.contains(r"\[FAILED\]", na=False, regex=True)
    ]
    print(f"  {len(scored):,} stories loaded")

    # 1: Detection 
    thresholds = compute_thresholds(scored, sigma=args.threshold_sigma)
    flagged    = flag_biased_stories(scored, thresholds)

    n_flagged = flagged["biased"].sum()
    print(f"\n Stage 1: Detection (σ={args.threshold_sigma}) ")
    print(f"  Daughter threshold : {thresholds['daughter']:+.4f}")
    print(f"  Son threshold      : {thresholds['son']:+.4f}")
    print(f"  Flagged stories    : {n_flagged:,} / {len(flagged):,} "
          f"({n_flagged / len(flagged) * 100:.1f}%)")

    # Work only on flagged stories
    to_process = flagged[flagged["biased"]].copy()
    if args.max_stories > 0:
        to_process = to_process.head(args.max_stories)
        print(f"  Processing limited to {args.max_stories} stories (--max-stories)")

    if args.dry_run:
        print("\n DRY RUN: skipping LLM regeneration ")
        flagged.to_csv(res_dir / "flagged_stories.csv", index=False)
        print(f"  Flagged stories saved: {res_dir}/flagged_stories.csv")
        return

    # Load optional embedding model 
    embedder = None if args.skip_semantic else load_embedder()
    if embedder is None and not args.skip_semantic:
        print("  (sentence-transformers not available; semantic similarity will be NaN)")

    # 2 & 3: Attribution + Regeneration 
    print(f"\n Stages 2 & 3: Attribute + Regenerate ({len(to_process)} stories) ")
    result_rows: list[dict] = []

    for i, (idx, row) in enumerate(to_process.iterrows(), 1):
        story       = str(row["story"])
        child_label = str(row["child_label"])
        score_before = float(row["stereotype_score"])
        cultural_before = float(row.get("cultural_richness", float("nan")))

        print(f"  [{i:4d}/{len(to_process)}] {row.get('model_key', '?'):25s} "
              f"| {child_label:8s} | score_before={score_before:+.3f}", end="  ", flush=True)

        corrected_story, new_scores, n_iters = star_correct(
            story        = story,
            child_label  = child_label,
            original_score = score_before,
            ollama_model = args.ollama_model,
            min_improvement = args.min_improvement,
            max_iter     = args.max_iter,
        )

        score_after      = float(new_scores.get("stereotype_score", score_before))
        cultural_after   = float(new_scores.get("cultural_richness", float("nan")))
        reduction_abs    = abs(score_before) - abs(score_after)
        reduction_pct    = reduction_abs / (abs(score_before) + EPS) * 100
        accepted         = reduction_pct >= args.min_improvement * 100

        sem_sim = semantic_similarity(story, corrected_story, embedder)

        print(f"→ score_after={score_after:+.3f}  Δ={reduction_pct:+.1f}%  "
              f"{'✓ ACCEPTED' if accepted else '✗ reverted'}")

        result_rows.append({
            # Identifiers
            "story_id":              row.get("id", idx),
            "model_key":             row.get("model_key", ""),
            "model_family":          row.get("model_family", ""),
            "model_params":          row.get("model_params", ""),
            "country":               row.get("country", ""),
            "country_region":        row.get("country_region", ""),
            "person":                row.get("person", ""),
            "child_label":           child_label,
            # Bias scores
            "score_before":          round(score_before, 4),
            "score_after":           round(score_after, 4),
            "reduction_pct":         round(reduction_pct, 2),
            "accepted":              accepted,
            "n_iters":               n_iters,
            # Additional bias metrics
            "trait_bias_before":     round(float(row.get("trait_bias_index", float("nan"))), 4),
            "trait_bias_after":      round(float(new_scores.get("trait_bias_index", float("nan"))), 4),
            "role_bias_before":      round(float(row.get("role_bias_index", float("nan"))), 4),
            "role_bias_after":       round(float(new_scores.get("role_bias_index", float("nan"))), 4),
            "marker_bias_before":    round(float(row.get("marker_bias_index", float("nan"))), 4),
            "marker_bias_after":     round(float(new_scores.get("marker_bias_index", float("nan"))), 4),
            # Cultural preservation
            "cultural_richness_before": round(cultural_before, 4),
            "cultural_richness_after":  round(cultural_after, 4),
            # Semantic similarity
            "semantic_sim":          round(float(sem_sim), 4) if not np.isnan(sem_sim) else float("nan"),
            # Text
            "story_original":        story,
            "story_corrected":       corrected_story if accepted else story,
        })

        time.sleep(SLEEP_BETWEEN)

    results = pd.DataFrame(result_rows)
    results.to_csv(res_dir / "star_results.csv", index=False)
    print(f"\n  Results saved: {res_dir}/star_results.csv")

    # Summary 
    summary = build_summary(results)
    summary.to_csv(res_dir / "star_summary.csv", index=False)
    print(f"  Summary saved: {res_dir}/star_summary.csv")

    # Figures 
    print("\n Generating figures ")
    fig_score_shift(results, fig_dir)
    fig_reduction_by_model(results, fig_dir)
    fig_cultural_preservation(results, fig_dir)
    fig_semantic_similarity(results, fig_dir)

    # Final report 
    accepted     = results[results["accepted"]]
    n_accepted   = len(accepted)
    n_total      = len(results)
    mean_red     = accepted["reduction_pct"].mean() if n_accepted > 0 else 0.0
    mean_sim     = accepted["semantic_sim"].mean()  if n_accepted > 0 else float("nan")

    print("\nSTAR MITIGATION: COMPLETE")
    print(f"  Stories processed   : {n_total}")
    print(f"  Stories accepted    : {n_accepted} ({n_accepted / max(n_total, 1) * 100:.1f}%)")
    print(f"  Mean bias reduction : {mean_red:.1f}% in |stereotype_score|")
    if not np.isnan(mean_sim):
        pct_q = (accepted["semantic_sim"] >= 0.85).mean() * 100
        print(f"  Semantic similarity : {mean_sim:.3f} mean  "
              f"({pct_q:.1f}% ≥ 0.85 quality threshold)")
    if not accepted.empty and "cultural_richness_before" in accepted.columns:
        delta_c = (accepted["cultural_richness_after"] -
                   accepted["cultural_richness_before"]).mean()
        print(f"  Cultural richness Δ : {delta_c:+.4f} (≈0 = preserved)")
    print(f"\n  Results : {res_dir}")
    print(f"  Figures : {fig_dir}")
    print()
    overall_row = summary[summary["group"] == "Overall"]
    if not overall_row.empty:
        print(overall_row.to_string(index=False))


if __name__ == "__main__":
    main()

