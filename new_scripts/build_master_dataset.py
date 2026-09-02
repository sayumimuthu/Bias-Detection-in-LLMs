"""
Build the master dataset for the recipient-conditioned stereotype
adaptation.

Usage:
    python3 build_master_dataset.py
    python3 build_master_dataset.py --input Narratives3/clean_stories_for_analysis.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "../Narratives3/clean_stories_for_analysis.csv"
DEFAULT_OUTPUT = "../Narratives3/master_dataset.csv"

# Narrator role -> narrator gender

NARRATOR_GENDER_MAP = {
    "Father": "male",
    "Grandfather": "male",
    "Older Brother": "male",
    "Uncle": "male",
    "Mother": "female",
    "Grandmother": "female",
    "Nanny": "female",
    "Older Sister": "female",
    "Aunt": "female",
    "Teacher": "unspecified",
    "Family Doctor": "unspecified",
    "Cousin": "unspecified",
}


# Recipient-gender-condition labels 

RECIPIENT_LABEL_MAP = {"female": "daughter", "male": "son"}

MODEL_VERSION_FAMILIES = {
    "llama3": [
        "ollama-llama32-1b",
        "ollama-llama32-3b",
        "ollama-llama31-8b",
        "ollama-llama3-70b",
    ],
    "qwen25": ["ollama-qwen25-3b", "ollama-qwen25-7b"],
    "gemma": ["ollama-gemma2-2b", "ollama-gemma3-12b", "ollama-gemma3-27b"],
    "mistral": ["ollama-mistral-7b", "ollama-mistral-nemo"],
    "gptoss": ["ollama-gptoss-20b"],
    "gpt": ["openai-gpt4o", "openai-gpt41"],
    "claude": ["anthropic-haiku45", "anthropic-sonnet46"],
}
MODEL_KEY_TO_FAMILY_GROUP = {
    mk: family for family, keys in MODEL_VERSION_FAMILIES.items() for mk in keys
}


def parse_param_count(value: object) -> float:
    """'8B' -> 8.0, '20B' -> 20.0, unparsable -> NaN."""
    match = re.search(r"([\d.]+)\s*[bB]", str(value))
    return float(match.group(1)) if match else float("nan")


def build_master_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_person = set(df["person"].unique()) - set(NARRATOR_GENDER_MAP)
    if missing_person:
        raise ValueError(
            f"Unmapped narrator role(s) found: {missing_person}. "
            "Add them to NARRATOR_GENDER_MAP before proceeding."
        )

    # fix the conflated column 
    df = df.rename(columns={"protagonist_gender": "recipient_gender_condition"})
    df["recipient_gender_label"] = df["recipient_gender_condition"].map(RECIPIENT_LABEL_MAP)
    if df["recipient_gender_label"].isna().any():
        bad = df.loc[df["recipient_gender_label"].isna(), "recipient_gender_condition"].unique()
        raise ValueError(f"Unmapped recipient_gender_condition value(s): {bad}")

    # narrator gender 
    df["narrator_role"] = df["person"]
    df["narrator_gender"] = df["narrator_role"].map(NARRATOR_GENDER_MAP)

    # model version family 
    df["model_family_version_group"] = df["model_key"].map(MODEL_KEY_TO_FAMILY_GROUP)
    unmapped_models = df.loc[df["model_family_version_group"].isna(), "model_key"].unique()
    if len(unmapped_models):
        print(
            f"WARNING: {len(unmapped_models)} model_key value(s) not in "
            f"MODEL_VERSION_FAMILIES, left ungrouped: {list(unmapped_models)}"
        )
    df["model_param_count_b"] = df["model_params"].apply(parse_param_count)
    df["model_size_rank"] = (
        df.groupby("model_family_version_group")["model_param_count_b"]
        .rank(method="dense")
    )

    keep_cols = [
        "id", "story", "word_count", "word_count_compliant", "token_count",
        "country",
        "narrator_role", "narrator_gender",
        "recipient_gender_condition", "recipient_gender_label",
        "model", "model_key", "model_family", "model_params",
        "model_family_version_group", "model_param_count_b", "model_size_rank",
        "generated_at", "cleaned_at",
    ]
    return df[keep_cols]


def print_validation_summary(df: pd.DataFrame) -> None:
    print("MASTER DATASET — VALIDATION SUMMARY")
    print(f"Rows: {len(df)}")
    print(f"Models: {df['model_key'].nunique()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Narrator roles: {df['narrator_role'].nunique()}")
    print("\nNarrator gender counts:")
    print(df["narrator_gender"].value_counts().to_string())
    print("\nRecipient gender condition counts:")
    print(df["recipient_gender_label"].value_counts().to_string())
    print("\nNarrator role x narrator gender (check : each role must map "
          "to exactly one gender):")
    print(pd.crosstab(df["narrator_role"], df["narrator_gender"]).to_string())
    print("\nModel family version groups:")
    print(df.groupby("model_family_version_group")["model_key"].unique().to_string())
    empty_cells = (
        df.groupby(["model_key", "country", "narrator_role", "recipient_gender_label"])
        .size()
        .reset_index(name="n")
    )
    print(f"\nDistinct (model x country x narrator x recipient) cells: {len(empty_cells)}")
    print(f"Cells with < 3 stories: {(empty_cells['n'] < 3).sum()} "
          f"({(empty_cells['n'] < 3).mean():.1%}) — expect sparsity, this "
          "drives the need for partial pooling in the Paper 1 hierarchical model.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)

    master = build_master_dataset(df)
    print_validation_summary(master)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)
    print(f"\nSaved master dataset -> {args.output} ({len(master)} rows)")


if __name__ == "__main__":
    main()
