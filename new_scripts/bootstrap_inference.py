"""
cross-axis aggregation, a non-parametric cross-check,
multiple-comparison correction, and practical-equivalence reporting for the
per-model RSA estimates produced by hierarchical_orientation_model.py.

Two independent sources of evidence are combined here, deliberately kept
separate rather than substituted for one another:

  1. The Bayesian posterior for theta_C + b_C,m (already fit, per axis, in
     orientation_model_{axis}_idata.nc) -- the primary estimate, with
     country/narrator_role/model already partially pooled as crossed random
     effects. This script does not refit anything; it re-reads the saved
     InferenceData.
  2. A block bootstrap over the CELL-LEVEL descriptive RSA in
     tensor_indices.csv -- a fast, model-free, frequentist cross-check that
     the Bayesian point estimate isn't an artifact of the hierarchical
     model's particular parametric assumptions (Beta-Binomial family, logit
     link, Normal group-level priors).


BH-FDR and the ROPE-based practical-equivalence verdict are both computed
from the POSTERIOR (source 1 above), not the bootstrap. The two-sided
"posterior p-value" used for BH-FDR is the standard
p ~= 2 * min(P(RSA>0), P(RSA<0)) heuristic (Makowski et al. 2019), which lets
a conventional Benjamini-Hochberg correction run across all 48
(model x axis) comparisons even though the underlying model is Bayesian. The
practical-equivalence verdict instead follows Kruschke's ROPE+HDI decision
rule directly on the posterior (not a bare probability cutoff): a model's
95% HDI entirely inside +-ROPE is "equivalent" (no meaningful stereotyping),
entirely outside is "nonzero", and a straddling HDI is "undecided". Default
ROPE is |RSA| < 0.1, chosen so it corresponds to |Q| < 0.05 under
tensor_indices.py's Q = tanh(RSA/2) bounded transform (2*arctanh(0.05) ~= 0.1).

Usage:
    python3 bootstrap_inference.py
    python3 bootstrap_inference.py --n-boot 2000 --rope 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

DEFAULT_MODEL_DIR = "../Narratives3/models"
DEFAULT_TENSOR_INDICES = "../Narratives3/tensor_indices.csv"
DEFAULT_OUTPUT = "../Narratives3/models/rsa_inference_report.csv"

AXES = ["role", "domain", "trait"]


def extract_rsa_draws(idata, model_keys: list[str]) -> dict[str, np.ndarray]:
    """Full posterior draw array (chains flattened) of theta_C + b_C,m per
    model, keyed by model_key. Same variable-discovery logic as
    hierarchical_orientation_model.py's extract_per_model_rsa (including the
    "match on coordinate VALUES, not dimension position" check -- the random
    slope's coords also include a size-1 recipient_gender_condition[male]
    contrast dimension, which position-based guessing latches onto instead
    of the real 16-level model_key dimension), but returns every draw
    instead of collapsing to a mean/HDI -- needed for the posterior
    tail-probability p-value and the ROPE verdict below.
    """
    posterior = idata.posterior
    theta_c = posterior["recipient_gender_condition"]
    slope_var_candidates = [
        v for v in posterior.data_vars
        if "model_key" in v and "recipient_gender_condition" in v and not v.endswith("_sigma")
    ]
    if not slope_var_candidates:
        raise KeyError(
            "Could not find the per-model recipient_gender_condition random "
            f"slope in the posterior. Available variables: {list(posterior.data_vars)}"
        )
    b_c_m = posterior[slope_var_candidates[0]]

    model_key_dim = None
    for dim in b_c_m.dims:
        if dim in ("chain", "draw"):
            continue
        coord_values = set(b_c_m.coords[dim].values.tolist()) if dim in b_c_m.coords else set()
        if coord_values == set(model_keys):
            model_key_dim = dim
            break
    if model_key_dim is None:
        raise KeyError(
            f"'{slope_var_candidates[0]}' has dims {b_c_m.dims}, but none of their "
            f"coordinate values match the model_key list."
        )

    return {
        model_key: (theta_c + b_c_m.sel({model_key_dim: model_key})).values.flatten()
        for model_key in model_keys
    }


def posterior_two_sided_p(draws: np.ndarray) -> float:
    """p ~= 2 * min(P(RSA>0), P(RSA<0)) -- the posterior-tail-probability
    analogue of a two-sided p-value (Makowski et al. 2019), used only so a
    conventional BH-FDR step can run across models/axes."""
    p_pos = float((draws > 0).mean())
    p_neg = float((draws < 0).mean())
    return 2 * min(p_pos, p_neg)


def rope_verdict(hdi_low: float, hdi_high: float, rope: float) -> str:
    """Kruschke's ROPE+HDI decision rule, applied to the 95% posterior HDI."""
    if hdi_low > -rope and hdi_high < rope:
        return "equivalent"
    if hdi_low > rope or hdi_high < -rope:
        return "nonzero"
    return "undecided"


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR correction. Returns q-values in the
    same order as the input."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    q_raw = ranked * n / (np.arange(n) + 1)
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_monotone, 0, 1)
    return q


def block_bootstrap_rsa(
    country_means: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Resample COUNTRIES with replacement from their per-country mean RSA
    (narrator_role already averaged out within country beforehand). Returns
    (point_estimate, ci_low, ci_high) over n_boot resamples.
    """
    if len(country_means) == 0:
        return (np.nan, np.nan, np.nan)
    point = float(country_means.mean())
    n_countries = len(country_means)
    idx = rng.integers(0, n_countries, size=(n_boot, n_countries))
    boot_means = country_means[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return (point, float(ci_low), float(ci_high))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--tensor-indices", type=Path, default=DEFAULT_TENSOR_INDICES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--rope", type=float, default=0.1,
                    help="Practical-equivalence half-width on RSA (default 0.1, "
                         "matching |Q|<0.05 under Q=tanh(RSA/2)).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

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

        for model_key, draws in draws_by_model.items():
            mean = float(draws.mean())
            hdi_low, hdi_high = az.hdi(draws, hdi_prob=0.95)
            p_value = posterior_two_sided_p(draws)
            verdict = rope_verdict(hdi_low, hdi_high, args.rope)
            rope_mass = float(((draws > -args.rope) & (draws < args.rope)).mean())

            sub = axis_tensor[axis_tensor["model_key"] == model_key]
            country_means = sub.groupby("country")["RSA"].mean().values
            boot_point, boot_lo, boot_hi = block_bootstrap_rsa(country_means, args.n_boot, rng)

            rows.append({
                "model_key": model_key,
                "axis": axis,
                "RSA_posterior_mean": mean,
                "RSA_hdi_low": float(hdi_low),
                "RSA_hdi_high": float(hdi_high),
                "p_value_posterior": p_value,
                "rope_mass": rope_mass,
                "rope_verdict": verdict,
                "RSA_bootstrap_point": boot_point,
                "RSA_bootstrap_ci_low": boot_lo,
                "RSA_bootstrap_ci_high": boot_hi,
                "n_countries_bootstrapped": len(country_means),
            })

    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_fdr(result["p_value_posterior"].values)
    result["significant_after_fdr"] = result["q_value_bh"] < 0.05

    # cross-check: does the model-free bootstrap CI agree in sign/exclusion
    # with the Bayesian HDI, or does it disagree with the headline result?
    hdi_excludes_zero = (result["RSA_hdi_low"] > 0) | (result["RSA_hdi_high"] < 0)
    boot_excludes_zero = (result["RSA_bootstrap_ci_low"] > 0) | (result["RSA_bootstrap_ci_high"] < 0)
    result["hdi_vs_bootstrap_agree"] = hdi_excludes_zero == boot_excludes_zero

    result = result.sort_values(["axis", "RSA_posterior_mean"], ascending=[True, False])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("RSA INFERENCE REPORT: BH-FDR + ROPE + bootstrap cross-check")
    for axis in AXES:
        sub = result[result["axis"] == axis]
        print(f"\naxis={axis}")
        print(sub[[
            "model_key", "RSA_posterior_mean", "q_value_bh", "rope_verdict",
            "RSA_bootstrap_point", "hdi_vs_bootstrap_agree",
        ]].to_string(index=False))

    print("Verdict counts by axis (of 16 models each):")
    print(result.groupby("axis")["rope_verdict"].value_counts().to_string())
    print("\nSignificant after BH-FDR (q < 0.05), by axis:")
    print(result.groupby("axis")["significant_after_fdr"].sum().to_string())
    print(
        "\nNote: on the sparser axes (domain, trait) the bootstrap POINT estimate "
        "typically runs well below the posterior mean -- this is expected, not a "
        "bug. tensor_indices.csv's per-cell RSA is Jeffreys-smoothed (alpha=0.5) "
        "toward 0 whenever a cell's counts are tiny, which is most cells on these "
        "low-exposure axes; averaging many such attenuated cells (this script) "
        "is a much more conservative estimator than the Beta-Binomial hierarchical "
        "model, which pools information across cells multiplicatively in logit "
        "space rather than shrinking each cell's contrast independently before "
        "averaging. The meaningful cross-check is SIGN/ZERO-EXCLUSION agreement "
        "(next line), not point-estimate magnitude."
    )
    n_disagree = (~result["hdi_vs_bootstrap_agree"]).sum()
    print(f"\nHDI vs. bootstrap sign/exclusion disagreements: {n_disagree} / {len(result)}")
    if n_disagree:
        print(result.loc[~result["hdi_vs_bootstrap_agree"], [
            "model_key", "axis", "RSA_posterior_mean", "RSA_hdi_low", "RSA_hdi_high",
            "RSA_bootstrap_point", "RSA_bootstrap_ci_low", "RSA_bootstrap_ci_high",
        ]].to_string(index=False))

    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
