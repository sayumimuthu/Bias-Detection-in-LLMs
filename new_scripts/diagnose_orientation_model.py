"""
Convergence diagnostics for a fitted hierarchical_orientation_model.py run.

The role-axis fit came back with max r_hat = 1.0100 -- right at the
"want < 1.01" threshold printed by the fitting script -- concentrated in the
model_key group-level terms (1|model_key, recipient_gender_condition|model_key
and their _sigma companions). Zero divergences were reported, which suggests
a mild non-centered-parameterization funnel rather than a broken model, but
that's a guess until someone looks at the actual chains. This script:

  1. Lists every parameter above an r_hat threshold, worst first.
  2. Rank plots for the flagged model_key terms (a good rank plot for all
     four chains should be roughly uniform; spikes/gaps mean a chain is
     stuck in a different part of the posterior than the others).
  3. Pair plots of each *_sigma against its matching *_offset -- bambi's
     non-centered parameterization couples these, and a banana/funnel shape
     here is the classic signature of the r_hat problem this run has.
  4. A forest plot of the per-model RSA report (model_rsa_report_<axis>.csv)
     so the headline result can be eyeballed without re-deriving it.


Usage:
    python3 diagnose_orientation_model.py --axis role
    python3 diagnose_orientation_model.py --axis role --rhat-threshold 1.005
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL_DIR = "../Narratives3/models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", choices=["role", "domain", "trait"], required=True)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--rhat-threshold", type=float, default=1.005,
                    help="Flag any parameter at or above this r_hat (default 1.005, "
                         "tighter than the fitting script's 1.01 pass/fail line, so "
                         "this catches the terms actually dragging the max up).")
    p.add_argument("--top-n", type=int, default=15,
                    help="How many worst-r_hat parameters to print/plot (default 15).")
    return p.parse_args()


def find_offset_sigma_pairs(var_names: list[str]) -> list[tuple[str, str]]:
    """Match bambi's non-centered '<term>_sigma' / '<term>_offset' variable
    pairs -- these are exactly the pair-plot axes that reveal a funnel.
    """
    pairs = []
    for name in var_names:
        if name.endswith("_sigma"):
            base = name[: -len("_sigma")]
            offset_name = base + "_offset"
            if offset_name in var_names:
                pairs.append((name, offset_name))
    return pairs


def main() -> None:
    args = parse_args()
    idata_path = args.model_dir / f"orientation_model_{args.axis}_idata.nc"
    rsa_path = args.model_dir / f"model_rsa_report_{args.axis}.csv"
    out_dir = args.model_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {idata_path} ...")
    idata = az.from_netcdf(str(idata_path))

    n_divergent = int(idata.sample_stats["diverging"].sum())
    print(f"Divergences: {n_divergent}")

    summary = az.summary(idata)
    flagged = summary[summary["r_hat"] >= args.rhat_threshold].sort_values("r_hat", ascending=False)
    print(f"\nParameters with r_hat >= {args.rhat_threshold} "
          f"({len(flagged)} of {len(summary)} total):")
    print(flagged.head(args.top_n).to_string())

    if flagged.empty:
        print("\nNothing above threshold -- convergence looks clean, no plots needed.")
    else:
        # Rank plots for the worst offenders, grouped by their base variable
        # name (posterior.data_vars, not the per-coordinate summary rows).
        flagged_vars = []
        for var in idata.posterior.data_vars:
            if any(idx.startswith(var) for idx in flagged.index[: args.top_n]):
                flagged_vars.append(var)
        flagged_vars = list(dict.fromkeys(flagged_vars))  # de-dup, keep order
        print(f"\nRank-plotting: {flagged_vars}")
        for var in flagged_vars:
            az.plot_rank(idata, var_names=[var])
            fig_path = out_dir / f"rank_{args.axis}_{var.replace('|', '_').replace(' ', '')}.png"
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close("all")
            print(f"  saved -> {fig_path}")

        # Pair plots for sigma/offset funnels among the flagged variables.
        pairs = find_offset_sigma_pairs(flagged_vars)
        for sigma_var, offset_var in pairs:
            az.plot_pair(
                idata, var_names=[sigma_var, offset_var],
                divergences=True, kind="scatter",
            )
            fig_path = out_dir / f"pair_{args.axis}_{sigma_var.replace('|', '_')}.png"
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close("all")
            print(f"  saved -> {fig_path}")
        if not pairs:
            print("  (no matching *_sigma/*_offset pairs found among flagged vars "
                  "-- check the printed flagged-vars list above and pair manually "
                  "with az.plot_pair if needed)")

    if rsa_path.exists():
        rsa = pd.read_csv(rsa_path).sort_values("RSA_hierarchical_mean")
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(rsa) + 1))
        y = range(len(rsa))
        ax.errorbar(
            rsa["RSA_hierarchical_mean"], y,
            xerr=[
                rsa["RSA_hierarchical_mean"] - rsa["RSA_hierarchical_hdi_low"],
                rsa["RSA_hierarchical_hdi_high"] - rsa["RSA_hierarchical_mean"],
            ],
            fmt="o", capsize=3,
        )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(list(y))
        ax.set_yticklabels(rsa["model_key"])
        ax.set_xlabel("RSA_hierarchical (mean, 95% HDI)")
        ax.set_title(f"Per-model RSA -- axis={args.axis}")
        fig.tight_layout()
        forest_path = out_dir / f"rsa_forest_{args.axis}.png"
        fig.savefig(forest_path, dpi=120)
        plt.close(fig)
        print(f"\nSaved RSA forest plot -> {forest_path}")
    else:
        print(f"\n{rsa_path} not found -- skipping RSA forest plot.")


if __name__ == "__main__":
    main()
