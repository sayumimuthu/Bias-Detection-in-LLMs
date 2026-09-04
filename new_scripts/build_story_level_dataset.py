"""
Story-level dataset for the hierarchical model.

This is deliberately NOT the stereotype_tensor.csv aggregation. That tensor
collapses to one row per (model, country, narrator_role, recipient, axis)
cell - which is exactly the grain where we found n=1 story per cell
everywhere. The hierarchical model is specified at
the STORY level (H_i,k, Y_i,M,k for story i), with country/narrator_role/
model as CROSSED RANDOM EFFECTS providing the pooling instead of per-cell
replication. So this script joins master_dataset.csv (one row per story)
with lexicon_matches_long.csv (one row per matched token) to get, per
(story, axis): total matches (exposure) and the masculine-pole count.

Practical adjustment from the workplan's literal N in {father,mother}:
narrator_gender here is a 3-level categorical (male / female / unspecified)
rather than a +-1 code, since the full dataset's 12 narrator roles include
Teacher/Family Doctor/Cousin with no inherent gender. The model script
handles this as a categorical fixed effect, not a manual +-1 recoding.

Usage:
    python3 build_story_level_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_MASTER = "../Narratives3/master_dataset.csv"
DEFAULT_MATCHES = "../Narratives3/lexicon_matches_long.csv"
DEFAULT_OUTPUT = "../Narratives3/story_level_dataset.csv"

AXES = ["role", "domain", "trait"]


def build_story_axis_counts(matches: pd.DataFrame) -> pd.DataFrame:
    non_negated = matches.loc[~matches["negated"]]
    counts = (
        non_negated.groupby(["id", "axis", "pole"])
        .size()
        .reset_index(name="Y")
    )
    wide = counts.pivot_table(index=["id", "axis"], columns="pole", values="Y", fill_value=0)
    wide = wide.reset_index().rename(columns={"F": "Y_F", "M": "Y_M"})
    for col in ("Y_F", "Y_M"):
        if col not in wide.columns:
            wide[col] = 0
    return wide[["id", "axis", "Y_F", "Y_M"]]


def build_complete_story_axis_grid(master: pd.DataFrame, story_axis: pd.DataFrame) -> pd.DataFrame:
    """Every story x axis combination, zero-filled where a story had no
    matches at all for that axis -- a real data point (H_i,k = 0), not
    missing."""
    grid = master[["id"]].merge(pd.DataFrame({"axis": AXES}), how="cross")
    full = grid.merge(story_axis, on=["id", "axis"], how="left")
    full[["Y_F", "Y_M"]] = full[["Y_F", "Y_M"]].fillna(0).astype(int)
    return full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    print(f"Loading {args.master} ...")
    master = pd.read_csv(args.master)
    print(f"Loading {args.matches} ...")
    matches = pd.read_csv(args.matches)

    story_axis = build_story_axis_counts(matches)
    full = build_complete_story_axis_grid(master, story_axis)

    meta_cols = [
        "id", "token_count", "country", "narrator_role", "narrator_gender",
        "recipient_gender_condition", "recipient_gender_label",
        "model_key", "model_family_version_group",
    ]
    result = full.merge(master[meta_cols], on="id", how="left")
    result = result.rename(columns={"token_count": "T"})
    result["H_exposure"] = result["Y_F"] + result["Y_M"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("STORY-LEVEL DATASET: SUMMARY")
    print(f"Rows (story x axis): {len(result)}")
    print(f"Distinct stories: {result['id'].nunique()}")
    print("\nH_exposure (Y_F + Y_M) distribution by axis:")
    print(result.groupby("axis")["H_exposure"].describe().to_string())
    print("\nnarrator_gender levels:")
    print(result["narrator_gender"].value_counts().to_string())
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
