# Prepares a clean DataFrame from Narratives3 for bias analysis.
# Filters bad stories, tokenizes with spaCy, and saves a clean CSV.

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

import subprocess
import nltk
from nltk.tokenize import word_tokenize
import spacy

try:
    word_tokenize("test")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)

def _load_spacy():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("en_core_web_sm not found — downloading now...")
        subprocess.run(
            ["python", "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
        )
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])

nlp = _load_spacy()

# Paths

INPUT_DIR  = Path("Narratives3")
INPUT_JSONL = INPUT_DIR / "biasednarratives.jsonl"
INPUT_CSV   = INPUT_DIR / "biasednarratives.csv"
OUTPUT_CSV  = INPUT_DIR / "clean_stories_for_analysis.csv"

# Filter thresholds 

MIN_WORD_COUNT = 20      
FAILED_MARKERS = {"[FAILED]", "[GENERATION FAILED]", "[UNKNOWN PROVIDER]"}

# Load 

print("Loading stories...")

if INPUT_JSONL.exists():
    df = pd.read_json(INPUT_JSONL, lines=True)
elif INPUT_CSV.exists():
    df = pd.read_csv(INPUT_CSV)
else:
    raise FileNotFoundError(f"No input file found in {INPUT_DIR}. ")

print(f"Total rows loaded : {len(df):,}")

# Filter 

# 1. Explicit failure markers
mask_failed = (
    df["story"].isin(FAILED_MARKERS) |
    df["story"].str.contains(r"\[FAILED\]", na=False, regex=True)
)

# 2. Zero / near-zero word count 
mask_short = df["word_count"] < MIN_WORD_COUNT

# 3. NaN or blank
mask_empty = df["story"].isna() | (df["story"].str.strip() == "")

removed = mask_failed | mask_short | mask_empty
df_clean = df[~removed].copy()

print(f"\nFiltering summary:")
print(f"  Explicit failures (FAILED marker) : {mask_failed.sum():>5}")
print(f"  Too short (< {MIN_WORD_COUNT} words)            : {mask_short.sum():>5}")
print(f"  Empty / NaN                       : {mask_empty.sum():>5}")
print(f"  Total removed                     : {removed.sum():>5}")
print(f"  Rows kept                         : {len(df_clean):,}")

# Report word-count compliance breakdown on the kept stories
if "word_count_compliant" in df_clean.columns:
    n_compliant = df_clean["word_count_compliant"].sum()
    n_noncompliant = (~df_clean["word_count_compliant"]).sum()
    print(f"\nOf kept stories:")
    print(f"  Within 150-200 words (compliant) : {n_compliant:,}")
    print(f"  Outside range (best attempt kept): {n_noncompliant:,}")

# Tokenize 

def tokenize(text: str) -> list[str]:
    """Lemmatise, remove stop words and punctuation."""
    if not isinstance(text, str) or not text.strip():
        return []
    doc = nlp(text.lower())
    return [
        t.lemma_ for t in doc
        if not t.is_punct and not t.is_stop and len(t.lemma_) > 1
    ]

print("\nTokenizing stories ...")
df_clean["tokens"]      = df_clean["story"].apply(tokenize)
df_clean["token_count"] = df_clean["tokens"].apply(len)
df_clean["cleaned_at"]  = datetime.now(timezone.utc).isoformat()

# Select columns

KEEP = [
    "id",
    "person",
    "country",
    "protagonist_gender",
    "model",
    "model_key",
    "model_family",
    "model_params",
    "model_year",
    "story",
    "word_count",
    "word_count_compliant",
    "n_retries",
    "token_count",
    "tokens",
    "generated_at",
    "cleaned_at",
]

available = [c for c in KEEP if c in df_clean.columns]
missing   = [c for c in KEEP if c not in df_clean.columns]
if missing:
    print(f"\nColumns not found : {missing}")

df_final = df_clean[available].copy()

# Save 

df_final.to_csv(OUTPUT_CSV, index=False)

print(f"\nClean dataset saved  : {OUTPUT_CSV}")
print(f"Rows ready for analysis: {len(df_final):,}")
print(f"\nWord count distribution of clean stories:")
print(df_final["word_count"].describe().round(1).to_string())
if "model_family" in df_final.columns:
    print(f"\nStories per model family:")
    print(df_final["model_family"].value_counts().to_string())

