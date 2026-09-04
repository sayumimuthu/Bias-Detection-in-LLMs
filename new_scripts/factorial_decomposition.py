"""
full 2x2x2 factorial decomposition,
plus narrator moderation NMA as a direct byproduct.

Uses ONLY the Father/Mother narrator rows -- this is the workplan's
"focused analysis" (narrator n in {father, mother}). The other
10 narrator roles (Teacher, Nanny, Cousin, ...) are not part of this clean
2x2x2 design; they're handled in the hierarchical model as a 12-level
random effect instead.

For each (model, country, axis), the 8 cells are:
    narrator N in {Father=+1, Mother=-1}
    recipient C in {son=+1, daughter=-1}
    pole P in {M=+1, F=-1}
y_{N,C,P} = log(Y_{N,C,P} + alpha) -- log-smoothed cell count.

The 8 orthogonal effects are computed by the standard 2^3 factorial
contrast: effect_T = (1/8) * sum_over_8_cells( y_cell * product of the
+-1 codes for the factors in T ), for T in {1, N, C, P, NC, NP, CP, NCP}.
This is a saturated decomposition (8 cells, 8 orthogonal contrasts) -- no
residual term, it's an exact re-expression of the 8 log-counts.

Sign convention chosen so CP and NCP line up with tensor_indices.py's RSA:
    CP   = (RSA_father + RSA_mother) / 8   (recipient stereotyping, pooled)
    NCP  = (RSA_father - RSA_mother) / 8   = NMA / 8  (narrator moderation)
This script asserts that relationship numerically against tensor_indices.csv
as a correctness check, not just states it.

Usage:
    python3 factorial_decomposition.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TENSOR = "../Narratives3/stereotype_tensor.csv"
DEFAULT_INDICES = "../Narratives3/tensor_indices.csv"  # for the validation check
DEFAULT_OUTPUT = "../Narratives3/factorial_decomposition.csv"

ALPHA = 0.5
GROUP_KEYS = ["model_key", "country", "axis"]
SPARSE_THRESHOLD = 3


def build_cell_values(tensor: pd.DataFrame) -> dict[tuple[int, int, int], pd.Series]:
    """Pivot Father/Mother x daughter/son x F/M-pole into the 8 named cell
    series, indexed by (model_key, country, axis) so they align for the
    per-group vectorized effect computation below.
    """
    sub = tensor[tensor["narrator_role"].isin(["Father", "Mother"])].copy()
    sub["y_F"] = np.log(sub["Y_F"] + ALPHA)
    sub["y_M"] = np.log(sub["Y_M"] + ALPHA)

    wide = sub.pivot_table(
        index=GROUP_KEYS,
        columns=["narrator_role", "recipient_gender_condition"],
        values=["y_F", "y_M"],
    )

    # (N, C, P) codes: Father=+1/Mother=-1, son=+1/daughter=-1, M=+1/F=-1
    cells = {
        (+1, -1, -1): wide[("y_F", "Father", "female")],   # Father, daughter, F
        (+1, -1, +1): wide[("y_M", "Father", "female")],   # Father, daughter, M
        (+1, +1, -1): wide[("y_F", "Father", "male")],     # Father, son, F
        (+1, +1, +1): wide[("y_M", "Father", "male")],     # Father, son, M
        (-1, -1, -1): wide[("y_F", "Mother", "female")],   # Mother, daughter, F
        (-1, -1, +1): wide[("y_M", "Mother", "female")],   # Mother, daughter, M
        (-1, +1, -1): wide[("y_F", "Mother", "male")],     # Mother, son, F
        (-1, +1, +1): wide[("y_M", "Mother", "male")],     # Mother, son, M
    }
    return cells


def compute_effects(cells: dict[tuple[int, int, int], pd.Series]) -> pd.DataFrame:
    def effect(select_signs) -> pd.Series:
        total = None
        for (n, c, p), y in cells.items():
            sign = select_signs(n, c, p)
            term = sign * y
            total = term if total is None else total + term
        return total / 8

    effects = pd.DataFrame({
        "intercept_1": effect(lambda n, c, p: 1),
        "N": effect(lambda n, c, p: n),
        "C": effect(lambda n, c, p: c),
        "P": effect(lambda n, c, p: p),
        "NC": effect(lambda n, c, p: n * c),
        "NP": effect(lambda n, c, p: n * p),
        "CP": effect(lambda n, c, p: c * p),
        "NCP": effect(lambda n, c, p: n * c * p),
    })
    return effects.reset_index()


def validate_against_rsa(effects: pd.DataFrame, indices_path: Path) -> None:
    """CP should equal (RSA_father + RSA_mother)/8 and NCP should equal
    (RSA_father - RSA_mother)/8 = NMA/8, derived analytically from the
    definitions (the baseline term in O cancels out of both). Check this
    numerically rather than trusting the derivation blindly.
    """
    indices = pd.read_csv(indices_path)
    fm = indices[indices["narrator_role"].isin(["Father", "Mother"])]
    rsa_wide = fm.pivot_table(index=GROUP_KEYS, columns="narrator_role", values="RSA").reset_index()
    rsa_wide["CP_expected"] = (rsa_wide["Father"] + rsa_wide["Mother"]) / 8
    rsa_wide["NCP_expected"] = (rsa_wide["Father"] - rsa_wide["Mother"]) / 8

    merged = effects.merge(rsa_wide, on=GROUP_KEYS, how="inner")
    cp_diff = (merged["CP"] - merged["CP_expected"]).abs()
    ncp_diff = (merged["NCP"] - merged["NCP_expected"]).abs()

    print("VALIDATION: CP/NCP against tensor_indices.csv RSA")
    print(f"max |CP - (RSA_father+RSA_mother)/8|:  {cp_diff.max():.2e}")
    print(f"max |NCP - (RSA_father-RSA_mother)/8|: {ncp_diff.max():.2e}")
    if cp_diff.max() < 1e-8 and ncp_diff.max() < 1e-8:
        print("PASS: factorial decomposition is consistent with tensor_indices.py's RSA.")
    else:
        print("MISMATCH: something is inconsistent between the two scripts, investigate.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    p.add_argument("--indices", type=Path, default=DEFAULT_INDICES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.tensor} ...")
    tensor = pd.read_csv(args.tensor)

    cells = build_cell_values(tensor)
    effects = compute_effects(cells)

    validate_against_rsa(effects, args.indices)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.output, index=False)

    print("FACTORIAL DECOMPOSITION: SUMMARY")
    print(f"Rows (model x country x axis): {len(effects)}")
    print("\nEffect distributions by axis:")
    print(effects.groupby("axis")[["N", "C", "P", "NC", "NP", "CP", "NCP"]].mean().to_string())
    print(f"\nNMA (narrator moderation) = NCP * 8:")
    nma = effects.copy()
    nma["NMA"] = nma["NCP"] * 8
    print(nma.groupby("axis")["NMA"].describe().to_string())
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
