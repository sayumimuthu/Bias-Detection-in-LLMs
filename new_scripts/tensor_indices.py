"""
descriptive tensor indices.

Reads Narratives3/stereotype_tensor.csv and computes, per
(model, country, narrator_role, axis):

  - O_daughter, O_son: baseline-corrected masculine-to-feminine log-odds
    orientation for each recipient condition.
  - RSA (Recipient Stereotype Alignment): O_son - O_daughter, the primary
    quantity. RSA > 0 = traditional alignment (sons get more masculine-coded
    content), RSA < 0 = counter-stereotypical, RSA ~ 0 = no child-pole
    association.
  - Q: bounded version of RSA via tanh(RSA/2), in [-1, 1], for cross-model
    visualization.
  - MI, JSD: non-directional adaptation magnitude -- "does the model adapt
    content to recipient gender at all", independent of whether that
    adaptation is stereotype-congruent (that's what RSA/Q answer).

Baseline: estimated from the corpus's OWN pooled Y_F/Y_M totals per axis
(the comparison corpus doubles as its own background, in the Fightin' Words
tradition) rather than requiring a separate "unspecified" control-condition
generation run. Note RSA and Q are algebraically baseline-invariant (the
baseline term cancels in the son-vs-daughter contrast) -- only the
standalone O_daughter/O_son values depend on it, and only for descriptive
interpretability of "is this cell more/less masculine-coded than the
corpus's own overall rate".

Usage:
    python3 tensor_indices.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = "../Narratives3/stereotype_tensor.csv"
DEFAULT_OUTPUT = "../Narratives3/tensor_indices.csv"

ALPHA = 0.5  # Jeffreys-type smoothing
CELL_KEYS = ["model_key", "country", "narrator_role", "axis"]
SPARSE_THRESHOLD = 3  # n_stories below this (on EITHER recipient side) -> low_confidence


def compute_axis_baseline(tensor: pd.DataFrame) -> dict[str, float]:
    """O_baseline_k = pooled log-odds of M vs F across the WHOLE corpus for
    axis k. Corrects for unequal lexicon sizes / base-rate asymmetry between
    poles, without needing an external reference corpus.
    """
    totals = tensor.groupby("axis")[["Y_F", "Y_M"]].sum()
    baseline = np.log((totals["Y_M"] + ALPHA) / (totals["Y_F"] + ALPHA))
    return baseline.to_dict()


def pivot_by_recipient(tensor: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, country, narrator_role, axis), with
    daughter/son counts and exposure/n_stories side by side as columns.
    """
    wide = tensor.pivot_table(
        index=CELL_KEYS,
        columns="recipient_gender_condition",
        values=["Y_F", "Y_M", "T", "n_stories", "exposure_E"],
    )
    wide.columns = [f"{val}_{cond}" for val, cond in wide.columns]
    wide = wide.reset_index()

    # recipient_gender_condition is "female"/"male" (daughter/son prompt
    # condition) -- rename to the human-readable labels used elsewhere.
    rename = {}
    for val in ("Y_F", "Y_M", "T", "n_stories", "exposure_E"):
        rename[f"{val}_female"] = f"{val}_daughter"
        rename[f"{val}_male"] = f"{val}_son"
    return wide.rename(columns=rename)


def compute_orientation_and_rsa(wide: pd.DataFrame, baseline: dict[str, float]) -> pd.DataFrame:
    wide = wide.copy()
    wide["axis_baseline"] = wide["axis"].map(baseline)

    for cond in ("daughter", "son"):
        raw_log_odds = np.log(
            (wide[f"Y_M_{cond}"] + ALPHA) / (wide[f"Y_F_{cond}"] + ALPHA)
        )
        wide[f"O_{cond}"] = raw_log_odds - wide["axis_baseline"]

    wide["RSA"] = wide["O_son"] - wide["O_daughter"]
    wide["Q"] = np.tanh(wide["RSA"] / 2)
    return wide


def compute_adaptation_magnitude(wide: pd.DataFrame) -> pd.DataFrame:
    """Non-directional adaptation magnitude from the 2x2 (recipient x pole)
    contingency table per cell: mutual information I(C;P) and Jensen-Shannon
    divergence between the daughter and son pole distributions. Both use
    log base 2 (bits), the conventional unit for binary-variable MI/JSD, so
    both are bounded in [0, 1].

    Cells are Jeffreys-smoothed the same way as RSA/O, so a cell with zero
    matches on one side doesn't produce an undefined probability.
    """
    wide = wide.copy()

    d_f = wide["Y_F_daughter"] + ALPHA
    d_m = wide["Y_M_daughter"] + ALPHA
    s_f = wide["Y_F_son"] + ALPHA
    s_m = wide["Y_M_son"] + ALPHA
    total = d_f + d_m + s_f + s_m

    # joint distribution q(c, p)
    q_d_f, q_d_m, q_s_f, q_s_m = d_f / total, d_m / total, s_f / total, s_m / total
    # marginals
    q_d, q_s = q_d_f + q_d_m, q_s_f + q_s_m
    q_f, q_m = q_d_f + q_s_f, q_d_m + q_s_m

    def _mi_term(q_cp, q_c, q_p):
        return q_cp * np.log2(q_cp / (q_c * q_p))

    mi = (
        _mi_term(q_d_f, q_d, q_f) + _mi_term(q_d_m, q_d, q_m)
        + _mi_term(q_s_f, q_s, q_f) + _mi_term(q_s_m, q_s, q_m)
    )
    wide["MI"] = mi.clip(lower=0)  # MI is >= 0 by definition; floating-point
                                    # rounding can produce ~-1e-17 otherwise

    # JSD between the two Bernoulli(P = masculine) distributions, one per
    # recipient condition: p_d = P(M | daughter), p_s = P(M | son)
    p_d = d_m / (d_f + d_m)
    p_s = s_m / (s_f + s_m)
    m = (p_d + p_s) / 2

    def _kl_bernoulli(p, q):
        return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))

    wide["JSD"] = 0.5 * _kl_bernoulli(p_d, m) + 0.5 * _kl_bernoulli(p_s, m)
    return wide


def finalize(wide: pd.DataFrame) -> pd.DataFrame:
    wide["low_confidence"] = (
        (wide["n_stories_daughter"].fillna(0) < SPARSE_THRESHOLD)
        | (wide["n_stories_son"].fillna(0) < SPARSE_THRESHOLD)
    )
    cols = CELL_KEYS + [
        "exposure_E_daughter", "exposure_E_son",
        "O_daughter", "O_son", "RSA", "Q", "MI", "JSD",
        "Y_F_daughter", "Y_M_daughter", "Y_F_son", "Y_M_son",
        "n_stories_daughter", "n_stories_son", "low_confidence",
    ]
    return wide[cols]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.input} ...")
    tensor = pd.read_csv(args.input)

    baseline = compute_axis_baseline(tensor)
    print("\nAxis baselines (pooled corpus-wide M/F log-odds):")
    for axis, val in baseline.items():
        print(f"  {axis}: {val:+.4f}")

    wide = pivot_by_recipient(tensor)
    wide = compute_orientation_and_rsa(wide, baseline)
    wide = compute_adaptation_magnitude(wide)
    result = finalize(wide)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("TENSOR INDICES: SUMMARY")
    print(f"Rows (model x country x narrator x axis): {len(result)}")
    print(f"Low-confidence rows: {result['low_confidence'].sum()} "
          f"({result['low_confidence'].mean():.1%})")
    print("\nRSA / Q distribution by axis:")
    print(result.groupby("axis")[["RSA", "Q"]].describe().to_string())
    print("\nMI / JSD distribution by axis:")
    print(result.groupby("axis")[["MI", "JSD"]].describe().to_string())
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
