"""
orientation model conditional on exposure
This is the model whose output IS the primary
result: a hierarchical, uncertainty-quantified, partially-pooled
estimate of Recipient Stereotype Alignment per model, replacing the noisy
single-story RSA computed in tensor_indices.py.

    Y_i,M,k | H_i,k ~ BetaBinomial(H_i,k, pi_i,k, rho_k)
    logit pi_i,k = theta0 + thetaN*N_i + thetaC*C_i + thetaNC*N_i*C_i
                   + u_country^(o) + u_narrator_role^(o) + u_model^(o)
    theta_C,m = theta_C + b_C,m     (random slope, per workplan Section 7
                                      "Model-specific effects")

theta_C IS the population-level Recipient Stereotype Alignment: the average
log-odds shift toward masculine-pole coding when recipient=son vs=daughter,
across the whole corpus. b_C,m is how much a specific model's alignment
departs from that average -- so theta_C + b_C,m, per model, is the
headline deliverable: "each model's own recipient stereotype alignment,
posterior mean + 95% HDI, with partial pooling so a model seen in few
countries doesn't get an unstably extreme estimate."

Only stories with H_i,k > 0 (at least one axis-k match) are included --
BetaBinomial(n=0, ...) carries no information about orientation, only about
exposure (which is done in Part A).

Practical adjustment (same as Part A): narrator_gender is a 3-level
categorical (male/female/unspecified), and country/narrator_role/model are
crossed random effects over individual stories, not per-cell replicates.

Usage:
    python3 hierarchical_orientation_model.py --axis role --draws 500 --tune 500  # smoke test
    python3 hierarchical_orientation_model.py --axis role  # full run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

DEFAULT_INPUT = "../Narratives3/story_level_dataset.csv"
DEFAULT_OUTPUT_DIR = "../Narratives3/models"

FORMULA = (
    "p(Y_M, H_exposure) ~ narrator_gender * recipient_gender_condition "
    "+ (1|country) + (1|narrator_role) "
    "+ (1 + recipient_gender_condition|model_key)"
)


def fit_orientation_model(
    df: pd.DataFrame, draws: int, tune: int, chains: int, seed: int, target_accept: float
):
    # bambi's beta-binomial family name -- if this raises an unknown-family
    # error, check bmb.families.univariate for the exact string this bambi
    # version uses (possibly "betabinomial" without the underscore).
    model = bmb.Model(FORMULA, df, family="beta_binomial")
    print(model)
    idata = model.fit(
        draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=True,
        target_accept=target_accept,
    )
    return model, idata


def extract_per_model_rsa(idata, model_keys: list[str]) -> pd.DataFrame:
    """theta_C (population-level) + b_C,m (per-model random slope) posterior
    -> one row per model with mean/HDI. This is the headline 'model report
    card' the whole hierarchical exercise is for.
    """
    posterior = idata.posterior
    theta_c = posterior["recipient_gender_condition"]  # population-level fixed effect
    # bambi names the random-slope term something like
    # "1 + recipient_gender_condition|model_key" -- the exact variable name
    # Exclude "_sigma" variables (the group-level standard deviation, e.g.
    # "recipient_gender_condition|model_key_sigma") -- that also matches
    # both substrings but isn't the per-model slope values themselves.
    slope_var_candidates = [
        v for v in posterior.data_vars
        if "model_key" in v and "recipient_gender_condition" in v and not v.endswith("_sigma")
    ]
    if not slope_var_candidates:
        raise KeyError(
            "Could not find the per-model recipient_gender_condition random "
            f"slope in the posterior. Available variables: {list(posterior.data_vars)}"
        )
    slope_var = slope_var_candidates[0]
    b_c_m = posterior[slope_var]

    # Find which dimension actually indexes over model_key by checking
    # COORDINATE VALUES, not assumed dimension order/position -- b_c_m has
    # an extra size-1 dimension for the recipient_gender_condition[male]
    # contrast level, and which dim comes "last" isn't something to guess.
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
            f"'{slope_var}' has dims {b_c_m.dims}, but none of their coordinate "
            f"values match the model_key list. Coords available: "
            f"{ {d: list(b_c_m.coords[d].values) for d in b_c_m.dims if d in b_c_m.coords} }"
        )

    rows = []
    for model_key in model_keys:
        model_rsa = theta_c + b_c_m.sel({model_key_dim: model_key})
        hdi = az.hdi(model_rsa.values.flatten(), hdi_prob=0.95)
        rows.append({
            "model_key": model_key,
            "RSA_hierarchical_mean": float(model_rsa.mean()),
            "RSA_hierarchical_hdi_low": float(hdi[0]),
            "RSA_hierarchical_hdi_high": float(hdi[1]),
        })
    return pd.DataFrame(rows).sort_values("RSA_hierarchical_mean", ascending=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--axis", choices=["role", "domain", "trait"], required=True)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-accept", type=float, default=0.95,
                    help="Starting at 0.95 by default, not 0.9 -- Part A's "
                         "model_key group needed this to converge, and this "
                         "model has an even more complex model_key random "
                         "slope on top of the intercept.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df[(df["axis"] == args.axis) & (df["H_exposure"] > 0)].copy()
    print(f"Fitting orientation model for axis='{args.axis}' on {len(df)} stories "
          f"with H_exposure > 0 (draws={args.draws}, tune={args.tune}, "
          f"chains={args.chains}, target_accept={args.target_accept}) ...")

    model, idata = fit_orientation_model(
        df, args.draws, args.tune, args.chains, args.seed, args.target_accept
    )

    summary = az.summary(idata)
    print(f"POSTERIOR SUMMARY -- axis={args.axis}")
    print(summary.to_string())

    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()
    print(f"\nmax r_hat: {max_rhat:.4f}  (want < 1.01)")
    print(f"min ess_bulk: {min_ess:.0f}  (want > ~400)")
    if max_rhat > 1.01:
        print("WARNING: convergence looks shaky -- do not trust these estimates "
              "yet, needs more tuning or a higher --target-accept.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"orientation_model_{args.axis}_summary.csv"
    idata_path = args.output_dir / f"orientation_model_{args.axis}_idata.nc"
    summary.to_csv(summary_path)
    idata.to_netcdf(str(idata_path))
    print(f"\nSaved posterior summary -> {summary_path}")
    print(f"Saved full InferenceData -> {idata_path}")

    print("\nExtracting per-model hierarchical RSA (theta_C + b_C,m) ...")
    model_keys = sorted(df["model_key"].unique())
    try:
        rsa_report = extract_per_model_rsa(idata, model_keys)
        rsa_path = args.output_dir / f"model_rsa_report_{args.axis}.csv"
        rsa_report.to_csv(rsa_path, index=False)
        print(f"PER-MODEL HIERARCHICAL RSA -- axis={args.axis}")
        print(rsa_report.to_string(index=False))
        print(f"\nSaved -> {rsa_path}")
    except KeyError as e:
        print(f"\nCould not auto-extract per-model RSA: {e}")
        print("The full posterior is still saved in the .nc file -- this can "
              "be extracted manually once the correct variable name is known.")


if __name__ == "__main__":
    main()
