"""
lexical diversity and restrictiveness.

Computes, per (model, country, narrator_role, recipient_gender_condition,
axis), how many DISTINCT words are used, not just the F/M pole totals --
addresses the concern that two cells with identical Y_F/Y_M counts could
still differ hugely in whether that's 3 repeated words or 30 distinct ones.

NOTE on the workplan's H formula: it defines H_{n,c,k} as entropy NORMALIZED
by log|L_k| (stated to lie in [0,1]), but then separately asks to report
"effective vocabulary size exp(H)". Those two don't compose -- exp() of a
[0,1]-bounded value is bounded in [1, e~2.72], which isn't a vocabulary-size
estimate. This script computes both, correctly:

  - H_raw: raw Shannon entropy (nats) over the distinct words matched in
    the cell -> effective_vocab_size = exp(H_raw) is the actual "effective
    number of equally-common words" (a standard Hill-number diversity
    measure).
  - H_normalized: H_raw / log(n_unique_words) -- Pielou's evenness, bounded
    in [0,1], "are the words used roughly equally often, or dominated by a
    couple of repeats". Normalized against the words OBSERVED in this cell,
    not the full lexicon -- normalizing against the full lexicon (100s of
    words via the wildcard stems) would make nearly every cell's score
    collapse toward 0, which isn't informative.
  - lexicon_coverage: n_unique_words / (distinct words for this axis
    realized ANYWHERE in the corpus) -- "how much of the vocabulary this
    axis actually uses across the whole dataset does this one cell touch".

Diversity is computed over `lemma` (not raw token) so inflectional variants
("help"/"helped"/"helping") count as one word, not three. Negated matches
are excluded, consistent with how Y_F/Y_M were built.

Usage:
    python3 diversity_entropy.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MATCHES = "../Narratives3/lexicon_matches_long.csv"
DEFAULT_OUTPUT = "../Narratives3/diversity_entropy.csv"

CELL_KEYS = ["model_key", "country", "narrator_role", "recipient_gender_condition", "axis"]


def compute_cell_diversity(group: pd.DataFrame) -> pd.Series:
    counts = group["lemma"].value_counts()
    n_unique = len(counts)
    probs = counts / counts.sum()

    h_raw = -(probs * np.log(probs)).sum()
    effective_vocab_size = np.exp(h_raw)
    h_normalized = h_raw / np.log(n_unique) if n_unique > 1 else np.nan
    simpson_concentration = (probs ** 2).sum()  # higher = more concentrated

    return pd.Series({
        "n_unique_words": n_unique,
        "n_matched_tokens": len(group),
        "H_raw": h_raw,
        "effective_vocab_size": effective_vocab_size,
        "H_normalized": h_normalized,
        "simpson_concentration": simpson_concentration,
        "gini_simpson_diversity": 1 - simpson_concentration,
    })


def compute_axis_vocab_size(matches: pd.DataFrame) -> dict[str, int]:
    """Corpus-realized vocabulary per axis -- the denominator for
    lexicon_coverage. Not the theoretical wildcard-lexicon size (unbounded),
    but how many distinct lemmas this axis actually matched anywhere in the
    13k-story corpus."""
    return matches.groupby("axis")["lemma"].nunique().to_dict()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.matches} ...")
    matches = pd.read_csv(args.matches)
    matches = matches.loc[~matches["negated"]]

    axis_vocab_size = compute_axis_vocab_size(matches)
    print("\nCorpus-realized vocabulary size per axis:")
    for axis, size in axis_vocab_size.items():
        print(f"  {axis}: {size} distinct lemmas")

    print("\nComputing per-cell diversity (this groups ~380k matched-token "
          "rows into cells, may take a moment)...")
    result = (
        matches.groupby(CELL_KEYS)[["lemma"]]
        .apply(compute_cell_diversity)
        .reset_index()
    )
    result["lexicon_coverage"] = result["n_unique_words"] / result["axis"].map(axis_vocab_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("DIVERSITY / ENTROPY: SUMMARY")
    print(f"Rows (model x country x narrator x recipient x axis): {len(result)}")
    print("\nBy axis:")
    print(result.groupby("axis")[
        ["n_unique_words", "H_normalized", "effective_vocab_size", "lexicon_coverage"]
    ].mean().to_string())

    # daughter-son entropy gap, pivoted the same way as tensor_indices.py
    gap_wide = result.pivot_table(
        index=["model_key", "country", "narrator_role", "axis"],
        columns="recipient_gender_condition",
        values="H_normalized",
    ).reset_index()
    gap_wide["daughter_son_entropy_gap"] = gap_wide["male"] - gap_wide["female"]
    gap_path = args.output.parent / "diversity_entropy_gap.csv"
    gap_wide.to_csv(gap_path, index=False)
    print(f"\nDaughter-son entropy gap (H_son - H_daughter) by axis:")
    print(gap_wide.groupby("axis")["daughter_son_entropy_gap"].describe().to_string())
    print(f"\nSaved -> {args.output}")
    print(f"Saved -> {gap_path}")


if __name__ == "__main__":
    main()
