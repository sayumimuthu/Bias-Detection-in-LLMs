"""
Negation- and attribution-aware lexicon matching.

This is the step that fixes the "Maya was not timid" problem: a plain
bag-of-words / PMI match counts `timid` as feminine-coded regardless of
polarity or who it describes. This script runs each story through spaCy's
dependency parser and, for every lexicon-axis word match, records:

  - negated: is the match inside a negated scope? ("was not timid",
    "never helped", "wasn't brave")
  - subject_text: the surface text of the nearest clausal subject, as a
    COARSE proxy for who the trait/action is attached to.


Runs on the RAW `story` text, not the precomputed `tokens` column in
clean_stories_for_analysis.csv — that column already strips stopwords
(including negation cues like "not"), which is exactly the information this
step needs to recover.

Usage:
    python3 lexicon_match_v2.py --sample 50        # quick smoke test
    python3 lexicon_match_v2.py                    # full run (~13k stories)
    python3 lexicon_match_v2.py --resume            # continue an interrupted run

Requires:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc, Token



DEFAULT_INPUT = "../Narratives3/master_dataset.csv"
DEFAULT_LOOKUP = "../Narratives3/lexicon_axis_lookup.json"
DEFAULT_OUTPUT = "../Narratives3/lexicon_matches_long.csv"

NEGATION_CUES = {"not", "never", "no", "hardly", "rarely", "n't", "without", "barely"}
NEGATION_WINDOW = 3  # tokens to look back for a lexical negation cue


class LexiconLookup:
    """Exact + prefix (wildcard-stem) matcher, mirroring `_build_lex_matcher`
    in gender_bias_analysis_new.py: patterns ending in "*" in the source
    lexicon become prefixes here. Matching is on raw lowercased token text,
    not lemma — the source lexicon's wildcards were hand-designed as a
    substitute for lemmatization, so lemmatizing on top would double up.
    """

    def __init__(self, exact: dict[str, list[tuple[str, str]]], prefixes: list[tuple[str, str, str]]):
        self.exact = exact
        # longest prefix first is irrelevant for correctness here (we want
        # every matching axis/pole, not just one), kept sorted only for
        # readability when debugging.
        self.prefixes = sorted(prefixes, key=lambda t: len(t[0]), reverse=True)

    def match(self, word: str) -> list[tuple[str, str]]:
        pairs = list(self.exact.get(word, []))
        for prefix, axis_name, pole in self.prefixes:
            if word.startswith(prefix):
                pairs.append((axis_name, pole))
        return pairs


def load_lookup(path: Path) -> LexiconLookup:
    with open(path) as f:
        raw = json.load(f)
    exact = {word: [tuple(pair) for pair in pairs] for word, pairs in raw["exact"].items()}
    prefixes = [tuple(entry) for entry in raw["prefix"]]
    return LexiconLookup(exact, prefixes)


def is_negated(token: Token) -> bool:
    """Dependency-parse negation check + a lexical fallback window.

    Handles the common copular case ("X was not Y") where `neg` attaches to
    the auxiliary/copula rather than the adjective itself, plus direct verb
    negation ("did not help"), plus a fallback for parses the dependency
    check misses.

    A `neg` child is only trusted if it PRECEDES the matched token
    (child.i < token.i). English clause negation almost always precedes what
    it negates ("not brave", "did not help"); a `neg` that comes after the
    token in the sentence is negating something later, not this token. This
    matters for contrastive coordination like "strong, not weak" — spaCy's
    small model sometimes attaches `not` as a `neg` child of the FIRST
    conjunct ("strong") when it semantically negates the second ("weak").
    Requiring precedence rules that out.
    """
    # negation attached directly to this token (verb negation)
    if any(child.dep_ == "neg" and child.i < token.i for child in token.children):
        return True
    # negation attached to this token's head (copular/adjectival negation)
    if token.head is not token and any(
        child.dep_ == "neg" and child.i < token.i for child in token.head.children
    ):
        return True
    # lexical fallback: a negation cue within a short preceding window,
    # not crossing a sentence boundary or clause-separating punctuation
    sent_start = token.sent.start
    window_start = max(sent_start, token.i - NEGATION_WINDOW)
    for prior in token.doc[window_start:token.i]:
        if prior.is_punct and prior.text in {".", "!", "?", ";"}:
            break
        if prior.lower_ in NEGATION_CUES or prior.lower_.endswith("n't"):
            return True
    return False


def find_subject(token: Token) -> str | None:
    """Walk up the dependency tree to find the nearest clausal subject."""
    node = token
    for _ in range(6):
        for child in node.head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child.text
        if node.head is node:
            break
        node = node.head
    return None


def match_story(doc: Doc, lookup: LexiconLookup) -> list[dict]:
    records = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        pairs = lookup.match(token.lower_)
        if not pairs:
            continue
        negated = is_negated(token)
        subject_text = find_subject(token)
        for axis_name, pole in pairs:
            records.append({
                "token": token.text,
                "lemma": token.lemma_.lower(),
                "axis": axis_name,
                "pole": pole,
                "negated": negated,
                "subject_text": subject_text,
                "sent_index": token.sent.start,
            })
    return records


def run(
    df: pd.DataFrame,
    lookup: LexiconLookup,
    nlp: spacy.Language,
    output_path: Path,
    already_done_ids: set[str],
    batch_size: int,
    n_process: int,
) -> None:
    todo = df[~df["id"].isin(already_done_ids)]
    if todo.empty:
        print("Nothing to do, all stories already matched.")
        return

    write_header = not (output_path.exists() and already_done_ids)
    mode = "a" if output_path.exists() and already_done_ids else "w"

    ids = todo["id"].tolist()
    texts = todo["story"].fillna("").tolist()
    meta_cols = ["model_key", "country", "narrator_role", "recipient_gender_condition"]
    meta = todo[meta_cols].to_dict("records")

    buffer: list[dict] = []
    n_processed = 0
    with open(output_path, mode) as f:
        for story_id, doc, meta_row in zip(
            ids, nlp.pipe(texts, batch_size=batch_size, n_process=n_process), meta
        ):
            for record in match_story(doc, lookup):
                record["id"] = story_id
                record.update(meta_row)
                buffer.append(record)

            n_processed += 1
            if n_processed % 200 == 0 or n_processed == len(ids):
                print(f"  ...{n_processed}/{len(ids)} stories parsed, "
                      f"{len(buffer)} matches buffered")
                if buffer:
                    pd.DataFrame(buffer).to_csv(f, header=write_header, index=False)
                    write_header = False
                    buffer = []
                    f.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sample", type=int, default=None,
                    help="Only process the first N stories.")
    p.add_argument("--resume", action="store_true",
                    help="Skip story ids already present in --output.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-process", type=int, default=1,
                    help="spaCy multiprocessing workers. >1 speeds up the full "
                         "13k-story run but disable if you hit pickling issues.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    print(f"Loading lookup from {args.lookup} ...")
    lookup = load_lookup(args.lookup)

    print(f"Loading dataset from {args.input} ...")
    df = pd.read_csv(args.input)
    if args.sample:
        df = df.head(args.sample)
        print(f"--sample given: restricting to first {args.sample} stories.")

    already_done_ids: set[str] = set()
    if args.resume and args.output.exists():
        already_done_ids = set(pd.read_csv(args.output, usecols=["id"])["id"].unique())
        print(f"--resume given: {len(already_done_ids)} story ids already matched, skipping them.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(
        df, lookup, nlp, args.output,
        already_done_ids=already_done_ids,
        batch_size=args.batch_size,
        n_process=args.n_process,
    )

    result = pd.read_csv(args.output)
    print("MATCH SUMMARY")
    print(f"Total matches: {len(result)}")
    print(f"Negated matches (will be dropped before tensor build): "
          f"{result['negated'].sum()} ({result['negated'].mean():.1%})")
    print(result.groupby("axis")["pole"].value_counts().to_string())
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
