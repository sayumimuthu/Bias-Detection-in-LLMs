"""
model-version deltas.

For every model-lineage family with >=2 members (llama3, qwen25, gemma,
mistral, gpt, claude -- from MODEL_VERSION_FAMILIES in
build_master_dataset.py; gptoss has only one member and is skipped), computes
the pairwise difference in Recipient Stereotype Alignment between every pair
of models in the family, per axis:

    delta_RSA_{a,b,k} = RSA_{b,k} - RSA_{a,k}
    delta_Q_{a,b,k}   = Q_{b,k}   - Q_{a,k},   Q = tanh(RSA/2)

Both deltas are computed from PAIRED posterior draws (same chain/draw index,
both models are random effects within the same axis fit) rather than by
differencing the already-summarized means -- this is the statistically
correct way to get an uncertainty interval on a difference between two
correlated quantities from the same posterior, and is why this script re-opens
the .nc idata files rather than reading only model_rsa_report_{axis}.csv.

Pair ordering (a = "smaller/earlier", b = "larger/later") uses
master_dataset.csv's model_size_rank when both sides have one. Two families
-- claude and gpt -- have NO parseable parameter count (API models; providers
don't disclose size), so model_size_rank is NaN for both their members; for
those, pair order falls back to alphabetical by model_key, flagged in the
`ordering_basis` column. Separately, build_master_dataset.py already
documents that even a same-family, size-ordered pair is not guaranteed to be
a clean "old version -> new version" comparison (llama3.1-8b vs
llama3.2-1b/3b differ in generation as well as size, despite llama3.2 being
smaller) -- this script does not attempt to resolve that; `model_a`/`model_b`
name the exact models compared so a reader can judge each pair on its own
terms rather than trusting a single size-based "old/new" label.

Multiple-comparison handling mirrors bootstrap_inference.py exactly (same
posterior-tail-probability p-value, same Kruschke ROPE+HDI verdict on the RSA
scale, same BH-FDR step, now run across all family-pair x axis comparisons)
so the two reports are read the same way.

Usage:
    python3 model_version_deltas.py
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from build_master_dataset import MODEL_VERSION_FAMILIES
from bootstrap_inference import bh_fdr, extract_rsa_draws, posterior_two_sided_p, rope_verdict

DEFAULT_MODEL_DIR = "../Narratives3/models"
DEFAULT_MASTER = "../Narratives3/master_dataset.csv"
DEFAULT_TENSOR_INDICES = "../Narratives3/tensor_indices.csv"
DEFAULT_OUTPUT = "../Narratives3/models/model_version_deltas.csv"

AXES = ["role", "domain", "trait"]


def ordered_pairs(family_models: list[str], size_rank: dict[str, float]) -> list[tuple[str, str, str]]:
    """Every pair within a family, as (model_a, model_b, ordering_basis).
    model_a is the smaller/lower-rank side when both have a size_rank;
    otherwise pairs fall back to alphabetical order, flagged accordingly.
    """
    pairs = []
    for m1, m2 in combinations(sorted(family_models), 2):
        r1, r2 = size_rank.get(m1), size_rank.get(m2)
        if pd.notna(r1) and pd.notna(r2):
            a, b = (m1, m2) if r1 <= r2 else (m2, m1)
            basis = "param_count_rank"
        else:
            a, b = sorted((m1, m2))
            basis = "alphabetical (no param-count data)"
        pairs.append((a, b, basis))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--tensor-indices", type=Path, default=DEFAULT_TENSOR_INDICES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--rope", type=float, default=0.1,
                    help="Same ROPE half-width as bootstrap_inference.py, applied "
                         "here to the DELTA (|delta_RSA|<rope = 'no meaningful "
                         "version difference'), not the raw RSA.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.master} ...")
    master = pd.read_csv(args.master)
    size_rank = (
        master[["model_key", "model_size_rank"]].drop_duplicates()
        .set_index("model_key")["model_size_rank"].to_dict()
    )

    print(f"Loading {args.tensor_indices} ...")
    tensor_indices = pd.read_csv(args.tensor_indices)

    rows = []
    for axis in AXES:
        idata_path = args.model_dir / f"orientation_model_{axis}_idata.nc"
        print(f"Loading {idata_path} ...")
        idata = az.from_netcdf(str(idata_path))

        axis_tensor = tensor_indices[tensor_indices["axis"] == axis]
        model_keys = sorted(axis_tensor["model_key"].unique())
        draws_by_model = extract_rsa_draws(idata, model_keys)

        for family, family_models in MODEL_VERSION_FAMILIES.items():
            present = [m for m in family_models if m in draws_by_model]
            if len(present) < 2:
                continue  # gptoss, or a family missing from this axis's fit
            for model_a, model_b, basis in ordered_pairs(present, size_rank):
                draws_a = draws_by_model[model_a]
                draws_b = draws_by_model[model_b]

                delta_rsa = draws_b - draws_a
                q_a = np.tanh(draws_a / 2)
                q_b = np.tanh(draws_b / 2)
                delta_q = q_b - q_a

                hdi_low, hdi_high = az.hdi(delta_rsa, hdi_prob=0.95)
                rows.append({
                    "family": family,
                    "axis": axis,
                    "model_a": model_a,
                    "model_b": model_b,
                    "ordering_basis": basis,
                    "delta_RSA_mean": float(delta_rsa.mean()),
                    "delta_RSA_hdi_low": float(hdi_low),
                    "delta_RSA_hdi_high": float(hdi_high),
                    "delta_Q_mean": float(delta_q.mean()),
                    "delta_Q_hdi_low": float(np.percentile(delta_q, 2.5)),
                    "delta_Q_hdi_high": float(np.percentile(delta_q, 97.5)),
                    "p_value_posterior": posterior_two_sided_p(delta_rsa),
                    "rope_verdict": rope_verdict(hdi_low, hdi_high, args.rope),
                })

    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_fdr(result["p_value_posterior"].values)
    result["significant_after_fdr"] = result["q_value_bh"] < 0.05
    result = result.sort_values(["axis", "family", "model_a", "model_b"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("MODEL-VERSION DELTAS: pairwise within-family RSA/Q comparisons")
    for axis in AXES:
        sub = result[result["axis"] == axis]
        if sub.empty:
            continue
        print(f"\naxis={axis}")
        print(sub[[
            "family", "model_a", "model_b", "delta_RSA_mean", "delta_Q_mean",
            "rope_verdict", "q_value_bh", "significant_after_fdr",
        ]].to_string(index=False))

    print(f"Total pairwise comparisons: {len(result)}")
    print(f"Significant after BH-FDR (q < 0.05): {result['significant_after_fdr'].sum()}")
    print("\nVerdict counts:")
    print(result["rope_verdict"].value_counts().to_string())

    alpha_fallback_pairs = (
        result.loc[result["ordering_basis"] != "param_count_rank", ["family", "model_a", "model_b"]]
        .drop_duplicates()
    )
    if not alpha_fallback_pairs.empty:
        print(f"\n{len(alpha_fallback_pairs)} pair(s) (each computed on all {len(AXES)} axes) "
              "ordered alphabetically -- no param-count data available (claude/gpt API "
              "models), not by size:")
        print(alpha_fallback_pairs.to_string(index=False))

    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
