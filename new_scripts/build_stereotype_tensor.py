#!/usr/bin/env python3
"""
Assemble the stereotype tensor.

Combines the master dataset and the negation/attribution-aware
lexicon matches into the single long-format table that every
downstream statistic reads from:

    Y_{n,c,p}^{(m,k,r)}  — matched-token counts by
        narrator_role (n, 12-level) / narrator_gender,
        recipient_gender_condition (c),
        pole (p in {F, M}),
        for model (m), axis (k in {role, domain, trait}), country (r)

plus T_{n,c} — total generated-token counts per (model, country, narrator,
recipient) cell, used as the exposure denominator.

Negation rule: a lexicon match flagged
`negated=True` by lexicon_match_v2.py is DROPPED from its pole's count
before aggregation. This is the conservative choice — it does not assume a
negated feminine-coded word ("not timid") is evidence of masculine-coded
content, it just removes it from the feminine count. The drop-rate is
reported below so it's auditable, not silent.

Usage:
    python3 build_stereotype_tensor.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd



DEFAULT_MASTER = "../Narratives3/master_dataset.csv"
DEFAULT_MATCHES = "../Narratives3/lexicon_matches_long.csv"
DEFAULT_OUTPUT = "../Narratives3/stereotype_tensor.csv"

CELL_KEYS = ["model_key", "country", "narrator_role", "recipient_gender_condition"]
AXES = ["role", "domain", "trait"]
SPARSE_THRESHOLD = 3  # n_stories below this -> low_confidence flag


def report_negation_drop(matches: pd.DataFrame) -> pd.DataFrame:
    print("NEGATION DROP REPORT")
    drop_report = (
        matches.groupby("axis")["negated"]
        .agg(n_total="count", n_negated="sum")
        .assign(pct_negated=lambda d: d["n_negated"] / d["n_total"])
    )
    print(drop_report.to_string())
    return drop_report


def build_Y(matches: pd.DataFrame) -> pd.DataFrame:
    non_negated = matches.loc[~matches["negated"]]
    y_long = (
        non_negated.groupby(CELL_KEYS + ["axis", "pole"])
        .size()
        .reset_index(name="Y")
    )
    y_wide = y_long.pivot_table(
        index=CELL_KEYS + ["axis"], columns="pole", values="Y", fill_value=0
    ).reset_index()
    y_wide = y_wide.rename(columns={"F": "Y_F", "M": "Y_M"})
    for col in ("Y_F", "Y_M"):
        if col not in y_wide.columns:
            y_wide[col] = 0
    return y_wide[CELL_KEYS + ["axis", "Y_F", "Y_M"]]


def build_complete_grid(master: pd.DataFrame, y_wide: pd.DataFrame) -> pd.DataFrame:
    """Ensure every (cell x axis) combination that actually occurs in the
    corpus is present, with zero-filled Y_F/Y_M where no matches occurred —
    a cell with zero stereotype-lexicon hits is a real, meaningful data
    point (low exposure), not a missing one.
    """
    base_cells = master[CELL_KEYS].drop_duplicates()
    grid = base_cells.merge(pd.DataFrame({"axis": AXES}), how="cross")
    full = grid.merge(y_wide, on=CELL_KEYS + ["axis"], how="left")
    full[["Y_F", "Y_M"]] = full[["Y_F", "Y_M"]].fillna(0).astype(int)
    return full


def build_exposure_denominator(master: pd.DataFrame) -> pd.DataFrame:
    return (
        master.groupby(CELL_KEYS)
        .agg(T=("token_count", "sum"), n_stories=("id", "count"))
        .reset_index()
    )


def attach_cell_metadata(tensor: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    narrator_gender_lookup = master[["narrator_role", "narrator_gender"]].drop_duplicates()
    model_family_lookup = master[
        ["model_key", "model_family_version_group", "model_size_rank"]
    ].drop_duplicates()
    tensor = tensor.merge(narrator_gender_lookup, on="narrator_role", how="left")
    tensor = tensor.merge(model_family_lookup, on="model_key", how="left")
    return tensor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    print(f"Loading master dataset from {args.master} ...")
    master = pd.read_csv(args.master)
    print(f"Loading lexicon matches from {args.matches} ...")
    matches = pd.read_csv(args.matches)

    report_negation_drop(matches)

    y_wide = build_Y(matches)
    full_grid = build_complete_grid(master, y_wide)
    denom = build_exposure_denominator(master)

    tensor = full_grid.merge(denom, on=CELL_KEYS, how="left")
    tensor = attach_cell_metadata(tensor, master)

    tensor["exposure_E"] = 1000 * (tensor["Y_F"] + tensor["Y_M"]) / tensor["T"].replace(0, pd.NA)
    tensor["low_confidence"] = tensor["n_stories"] < SPARSE_THRESHOLD

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor.to_csv(args.output, index=False)

    print("STEREOTYPE TENSOR: SUMMARY")
    print(f"Rows (cell x axis): {len(tensor)}")
    print(f"Distinct cells (model x country x narrator x recipient): "
          f"{tensor[CELL_KEYS].drop_duplicates().shape[0]}")
    print(f"Low-confidence cells (< {SPARSE_THRESHOLD} stories): "
          f"{tensor['low_confidence'].sum()} ({tensor['low_confidence'].mean():.1%})")
    print("\nTotal Y_F / Y_M by axis:")
    print(tensor.groupby("axis")[["Y_F", "Y_M"]].sum().to_string())
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
