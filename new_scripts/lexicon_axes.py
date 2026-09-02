"""
Freeze the lexicon-to-axis mapping used by the stereotype tensor.

Sources LEXICONS from gender_bias_analysis_new.py — your final prior gender
bias analysis (Pietraszkiewicz et al. 2019 agentic/communal dictionary, WEAT
career/family sets, Bem 1974 BSRI trait adjectives), so the tensor work builds on the
same word lists your existing results already used.

Maps the lexicon sets onto the 3 tensor axes the workplan calls for:

    axis    pole=M (masculine-coded)   pole=F (feminine-coded)   source
    ------  -------------------------  ------------------------  ------
    role    agentic                    communal                  Pietraszkiewicz et al. 2019
    domain  career                     family                    WEAT gender-career/family
    trait   masculine_traits           feminine_traits           Bem 1974 BSRI

`male_markers` / `female_markers` (literal kinship/pronoun words) are kept
out of the 3 stereotype axes for the same reason as before: they overlap
with the prompt condition (recipient = daughter/son) and the narrator role
itself, so matching them would be circular. Kept as a `direct_marker`
diagnostic only.


Usage:
    python3 lexicon_axes.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from gender_bias_analysis_new import LEXICONS  # noqa: E402

OUTPUT_PATH = Path("../Narratives3/lexicon_axis_lookup.json")

AXIS_DEFINITIONS = {
    "role": {"M": "agentic", "F": "communal"},
    "domain": {"M": "career", "F": "family"},
    "trait": {"M": "masculine_traits", "F": "feminine_traits"},
}
DIRECT_MARKER_DEFINITION = {"M": "male_markers", "F": "female_markers"}


def build_lookup() -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Returns (exact_lookup, prefix_entries).

    exact_lookup: word -> list of (axis_name, pole) pairs.
    prefix_entries: list of (prefix, axis_name, pole) for "*"-suffixed
    patterns — kept as a list, not a dict, since prefixes must all be tested
    against each token rather than looked up by exact key.
    """
    exact_lookup: dict[str, list[tuple[str, str]]] = defaultdict(list)
    prefix_entries: list[tuple[str, str, str]] = []

    def register(lexicon_name: str, axis_name: str, pole: str) -> None:
        for pattern in LEXICONS[lexicon_name]:
            if pattern.endswith("*"):
                prefix_entries.append((pattern[:-1], axis_name, pole))
            else:
                exact_lookup[pattern].append((axis_name, pole))

    for axis_name, poles in AXIS_DEFINITIONS.items():
        for pole, lexicon_name in poles.items():
            register(lexicon_name, axis_name, pole)
    for pole, lexicon_name in DIRECT_MARKER_DEFINITION.items():
        register(lexicon_name, "direct_marker", pole)

    return dict(exact_lookup), prefix_entries


def check_contradictions(
    exact_lookup: dict[str, list[tuple[str, str]]],
    prefix_entries: list[tuple[str, str, str]],
) -> None:
    """Flag words/prefixes coded both M and F within the same axis."""
    contradictions = []
    for word, pairs in exact_lookup.items():
        by_axis: dict[str, set[str]] = defaultdict(set)
        for axis_name, pole in pairs:
            by_axis[axis_name].add(pole)
        for axis_name, poles in by_axis.items():
            if len(poles) > 1:
                contradictions.append((word, axis_name, poles))

    by_axis_prefix: dict[tuple[str, str], set[str]] = defaultdict(set)
    for prefix, axis_name, pole in prefix_entries:
        by_axis_prefix[(prefix, axis_name)].add(pole)
    for (prefix, axis_name), poles in by_axis_prefix.items():
        if len(poles) > 1:
            contradictions.append((prefix + "*", axis_name, poles))

    if contradictions:
        print(f"\nWARNING: {len(contradictions)} pattern(s) coded both M and F "
              "within the same axis:")
        for pattern, axis_name, poles in contradictions:
            print(f"  '{pattern}' in axis '{axis_name}': {poles}")
    else:
        print("\nNo within-axis M/F contradictions found.")


def print_coverage_report(
    exact_lookup: dict[str, list[tuple[str, str]]],
    prefix_entries: list[tuple[str, str, str]],
) -> None:
    axis_pole_counts: dict[tuple[str, str], int] = defaultdict(int)
    for pairs in exact_lookup.values():
        for axis_name, pole in pairs:
            axis_pole_counts[(axis_name, pole)] += 1
    for _, axis_name, pole in prefix_entries:
        axis_pole_counts[(axis_name, pole)] += 1

    print("LEXICON AXIS COVERAGE (exact words + wildcard stems)")
    print(f"Distinct exact words: {len(exact_lookup)}")
    print(f"Wildcard stems: {len(prefix_entries)}")
    for (axis_name, pole), count in sorted(axis_pole_counts.items()):
        print(f"  {axis_name:>14} / {pole}: {count} patterns")


def main() -> None:
    exact_lookup, prefix_entries = build_lookup()
    check_contradictions(exact_lookup, prefix_entries)
    print_coverage_report(exact_lookup, prefix_entries)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exact": exact_lookup,
        "prefix": [[prefix, axis, pole] for prefix, axis, pole in prefix_entries],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved lexicon axis lookup: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
