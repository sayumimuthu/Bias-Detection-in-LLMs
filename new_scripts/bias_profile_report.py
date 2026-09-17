"""
final per-model bias profile report.

Assembles every quantity computed so far in the pipeline into one table with
one row per (model, axis): exposure, RSA/Q with 95% HDI, non-directional
adaptation magnitude (MI/JSD), narrator moderation (NMA), lexical
diversity/restrictiveness, and cross-country consistency (ICC). This is the
"report card" table the paper's forest plots and per-model discussion are
built from -- everything else in new_scripts/ produces one piece of this;
this script only aggregates already-computed outputs, it does not fit or
recompute anything statistical from raw data.

Column sources (five separate files, all already produced upstream):
  - exposure_*        <- exposure_model_{axis}_idata.nc   (hierarchical_exposure_model.py)
  - RSA_*, Q_*         <- orientation_model_{axis}_idata.nc (hierarchical_orientation_model.py)
  - MI_mean, JSD_mean  <- tensor_indices.csv                (tensor_indices.py)
  - NMA_mean           <- factorial_decomposition.csv       (factorial_decomposition.py)
  - diversity_*        <- diversity_entropy.csv             (diversity_entropy.py)
  - RSA_country_icc    <- tensor_indices.csv (computed here, see below)

Two things are deliberately descriptive (simple corpus means), not
hierarchical, because no hierarchical version of them was built upstream:
MI/JSD and diversity are aggregated as an unweighted mean of the per-cell
values in tensor_indices.csv / diversity_entropy.csv across that model's
country x narrator_role cells -- the same treatment tensor_indices.py itself
already gives these quantities in its own printed summary. NMA is similarly
an unweighted mean of factorial_decomposition.csv's per-country NMA, and
inherits that script's restriction to the Father/Mother narrator sub-design
(the other 10 narrator roles aren't part of the clean 2x2x2 factorial and
have no NMA equivalent computed anywhere in the pipeline).

Cross-country consistency uses a one-way random-effects intraclass
correlation, ICC(1) (Shrout & Fleiss 1979): for a given (model, axis), treat
country as the grouping factor over the ~12 narrator-role RSA observations
per country from tensor_indices.csv, and partition variance into
between-country and within-country (residual) components. A HIGH ICC means
this model's recipient-stereotype alignment depends heavily on which country
the story is set in (inconsistent across countries); a LOW ICC means it's
similar everywhere the model is asked to write. This is a fast, closed-form
computation over already-aggregated cell-level RSA -- it does not refit the
hierarchical model's own country random effect, which pools information
differently (across models too, not just within one).

Usage:
    python3 bias_profile_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from bootstrap_inference import extract_rsa_draws

DEFAULT_MODEL_DIR = "../Narratives3/models"
DEFAULT_TENSOR_INDICES = "../Narratives3/tensor_indices.csv"
DEFAULT_FACTORIAL = "../Narratives3/factorial_decomposition.csv"
DEFAULT_DIVERSITY = "../Narratives3/diversity_entropy.csv"
DEFAULT_OUTPUT = "../Narratives3/models/bias_profile_report.csv"

AXES = ["role", "domain", "trait"]


def extract_exposure_draws(idata, model_keys: list[str]) -> dict[str, np.ndarray]:
    """Full posterior draws of the model_key random INTERCEPT from the
    exposure model (log-rate deviation from the population baseline -- there
    is no recipient-condition slope on model_key in this model, unlike the
    orientation model, so this is a plain group-effect lookup, not a
    slope-plus-population-term sum)."""
    posterior = idata.posterior
    var_candidates = [v for v in posterior.data_vars if v == "1|model_key"]
    if not var_candidates:
        raise KeyError(f"'1|model_key' not found. Available: {list(posterior.data_vars)}")
    effect = posterior[var_candidates[0]]

    model_key_dim = None
    for dim in effect.dims:
        if dim in ("chain", "draw"):
            continue
        if dim in effect.coords and set(effect.coords[dim].values.tolist()) == set(model_keys):
            model_key_dim = dim
            break
    if model_key_dim is None:
        raise KeyError(f"Could not match a coordinate dimension of '1|model_key' to model_keys.")

    return {mk: effect.sel({model_key_dim: mk}).values.flatten() for mk in model_keys}


def one_way_icc(values_by_group: list[np.ndarray]) -> float:
    """ICC(1) -- Shrout & Fleiss (1979) one-way random-effects intraclass
    correlation: fraction of total variance attributable to between-group
    differences. Handles unbalanced group sizes (countries don't all have
    exactly 12 narrator-role rows if any are missing)."""
    values_by_group = [v[~np.isnan(v)] for v in values_by_group if len(v) > 0]
    n_groups = len(values_by_group)
    if n_groups < 2:
        return np.nan
    group_sizes = np.array([len(v) for v in values_by_group])
    group_means = np.array([v.mean() for v in values_by_group])
    n_total = group_sizes.sum()
    grand_mean = np.concatenate(values_by_group).mean()

    ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)
    ms_between = ss_between / (n_groups - 1)

    ss_within = sum(((v - m) ** 2).sum() for v, m in zip(values_by_group, group_means))
    df_within = n_total - n_groups
    if df_within <= 0:
        return np.nan
    ms_within = ss_within / df_within

    k_bar = (n_total - (group_sizes ** 2).sum() / n_total) / (n_groups - 1)  # avg group size, unbalanced-corrected
    denom = ms_between + (k_bar - 1) * ms_within
    if denom == 0:
        return np.nan
    return float((ms_between - ms_within) / denom)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--tensor-indices", type=Path, default=DEFAULT_TENSOR_INDICES)
    p.add_argument("--factorial", type=Path, default=DEFAULT_FACTORIAL)
    p.add_argument("--diversity", type=Path, default=DEFAULT_DIVERSITY)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.tensor_indices} ...")
    tensor_indices = pd.read_csv(args.tensor_indices)
    print(f"Loading {args.factorial} ...")
    factorial = pd.read_csv(args.factorial)
    print(f"Loading {args.diversity} ...")
    diversity = pd.read_csv(args.diversity)

    # NMA per (model, axis): mean of NCP*8 across countries (Father/Mother design only)
    factorial = factorial.copy()
    factorial["NMA"] = factorial["NCP"] * 8
    nma_by_model_axis = factorial.groupby(["model_key", "axis"])["NMA"].mean()

    # diversity per (model, axis): mean across country x narrator_role x recipient cells
    diversity_by_model_axis = diversity.groupby(["model_key", "axis"]).agg(
        diversity_H_normalized_mean=("H_normalized", "mean"),
        diversity_effective_vocab_mean=("effective_vocab_size", "mean"),
        lexicon_coverage_mean=("lexicon_coverage", "mean"),
    )

    rows = []
    for axis in AXES:
        axis_tensor = tensor_indices[tensor_indices["axis"] == axis]
        model_keys = sorted(axis_tensor["model_key"].unique())

        exposure_idata = az.from_netcdf(str(args.model_dir / f"exposure_model_{axis}_idata.nc"))
        orientation_idata = az.from_netcdf(str(args.model_dir / f"orientation_model_{axis}_idata.nc"))
        print(f"Loaded exposure + orientation idata for axis={axis}")

        exposure_draws = extract_exposure_draws(exposure_idata, model_keys)
        rsa_draws = extract_rsa_draws(orientation_idata, model_keys)

        for model_key in model_keys:
            exp_d = exposure_draws[model_key]
            exp_hdi_low, exp_hdi_high = az.hdi(exp_d, hdi_prob=0.95)

            rsa_d = rsa_draws[model_key]
            rsa_hdi_low, rsa_hdi_high = az.hdi(rsa_d, hdi_prob=0.95)
            q_d = np.tanh(rsa_d / 2)
            q_hdi_low, q_hdi_high = np.percentile(q_d, [2.5, 97.5])

            sub = axis_tensor[axis_tensor["model_key"] == model_key]
            mi_mean = float(sub["MI"].mean())
            jsd_mean = float(sub["JSD"].mean())
            country_groups = [g["RSA"].values for _, g in sub.groupby("country")]
            icc = one_way_icc(country_groups)

            nma_mean = nma_by_model_axis.get((model_key, axis), np.nan)
            div_row = (
                diversity_by_model_axis.loc[(model_key, axis)]
                if (model_key, axis) in diversity_by_model_axis.index
                else pd.Series({"diversity_H_normalized_mean": np.nan,
                                 "diversity_effective_vocab_mean": np.nan,
                                 "lexicon_coverage_mean": np.nan})
            )

            rows.append({
                "model_key": model_key,
                "axis": axis,
                "exposure_log_rate_effect_mean": float(exp_d.mean()),
                "exposure_log_rate_hdi_low": float(exp_hdi_low),
                "exposure_log_rate_hdi_high": float(exp_hdi_high),
                "RSA_mean": float(rsa_d.mean()),
                "RSA_hdi_low": float(rsa_hdi_low),
                "RSA_hdi_high": float(rsa_hdi_high),
                "Q_mean": float(q_d.mean()),
                "Q_hdi_low": float(q_hdi_low),
                "Q_hdi_high": float(q_hdi_high),
                "MI_mean": mi_mean,
                "JSD_mean": jsd_mean,
                "NMA_mean_father_mother_only": nma_mean,
                "diversity_H_normalized_mean": div_row["diversity_H_normalized_mean"],
                "diversity_effective_vocab_mean": div_row["diversity_effective_vocab_mean"],
                "lexicon_coverage_mean": div_row["lexicon_coverage_mean"],
                "RSA_country_icc": icc,
            })

    result = pd.DataFrame(rows).sort_values(["axis", "RSA_mean"], ascending=[True, False])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("BIAS PROFILE REPORT: per (model, axis)")
    for axis in AXES:
        sub = result[result["axis"] == axis]
        print(f"\naxis={axis}")
        print(sub[[
            "model_key", "RSA_mean", "Q_mean", "MI_mean", "JSD_mean",
            "NMA_mean_father_mother_only", "diversity_H_normalized_mean", "RSA_country_icc",
        ]].round(3).to_string(index=False))

    print("\n" + "-" * 70)
    print("Cross-country ICC summary by axis (higher = more country-dependent behavior):")
    print(result.groupby("axis")["RSA_country_icc"].describe()[["mean", "min", "max"]].to_string())
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
