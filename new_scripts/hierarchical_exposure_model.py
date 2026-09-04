"""
stereotype-language exposure model 

    H_i,k ~ NegBinomial(mu_i,k, phi_k)
    log mu_i,k = log T_i + eta0 + etaN*N_i + etaC*C_i + etaNC*N_i*C_i
                 + u_country + u_narrator_role + u_model

Fit once per axis (role/domain/trait) on story_level_dataset.csv.

Practical adjustments vs. the literal workplan spec:
  - N_i (narrator) is a 3-level categorical (male/female/unspecified) fixed
    effect here, not a +-1 code -- the full 12-role dataset includes
    Teacher/Family Doctor/Cousin, which have no inherent gender. bambi's
    formula interface handles this as a categorical automatically.
  - country/narrator_role/model are CROSSED random intercepts over the
    individual story as the unit of observation (not per-cell replicate
    variance -- there is none, every cell has exactly 1 story). This is
    still a standard, valid multilevel design; what's lost is the ability
    to separate "true behavior in this exact cell" from "one random draw's
    noise" -- phi_k (the NegBinomial dispersion) will therefore reflect a
    mix of genuine cross-context heterogeneity AND generation stochasticity
    that a replicated design could have separated. 

Usage:
    python3 hierarchical_exposure_model.py --axis role
    python3 hierarchical_exposure_model.py --axis role --draws 500 --tune 500  # quick test
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
    "H_exposure ~ narrator_gender * recipient_gender_condition "
    "+ offset(log_T) + (1|country) + (1|narrator_role) + (1|model_key)"
)


def fit_exposure_model(
    df: pd.DataFrame, draws: int, tune: int, chains: int, seed: int, target_accept: float
):
    model = bmb.Model(FORMULA, df, family="negativebinomial")
    print(model)
    idata = model.fit(
        draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=True,
        target_accept=target_accept,
    )
    return model, idata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--axis", choices=["role", "domain", "trait"], required=True)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-accept", type=float, default=0.9,
                    help="Raise toward 0.95-0.99 if r_hat/ess look bad on the "
                         "model_key group effects specifically -- that's the "
                         "group with the largest between-group variance here "
                         "and the one most prone to funnel geometry.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df[df["axis"] == args.axis].copy()
    df["log_T"] = np.log(df["T"].clip(lower=1))  # token_count should never be
                                                   # 0 for a compliant story, but
                                                   # clip defensively rather than
                                                   # let log(0) = -inf slip through
    print(f"Fitting exposure model for axis='{args.axis}' on {len(df)} stories "
          f"(draws={args.draws}, tune={args.tune}, chains={args.chains}, "
          f"target_accept={args.target_accept}) ...")

    model, idata = fit_exposure_model(
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
        print("WARNING: convergence looks shaky (r_hat > 1.01) -- do not trust these "
              "estimates yet, needs more tuning/draws or a reparameterization.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"exposure_model_{args.axis}_summary.csv"
    idata_path = args.output_dir / f"exposure_model_{args.axis}_idata.nc"
    summary.to_csv(summary_path)
    idata.to_netcdf(str(idata_path))
    print(f"\nSaved posterior summary -> {summary_path}")
    print(f"Saved full InferenceData -> {idata_path}")


if __name__ == "__main__":
    main()
