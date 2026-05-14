"""
fix_checkpoint.py — one-time repair for a truncated checkpoint.json.

Run from the project root on the ada server:
    python fix_checkpoint.py
"""

import json
from pathlib import Path

CHECKPOINT  = Path("Narratives3/checkpoint.json")
BACKUP      = Path("Narratives3/checkpoint.json.bak")
OUT_JSONL   = Path("Narratives3/checkpoint.jsonl")

# ── Read raw text ──────────────────────────────────────────────────────────────

raw = CHECKPOINT.read_text(encoding="utf-8", errors="replace")
print(f"File size : {len(raw):,} chars")

BACKUP.write_text(raw, encoding="utf-8")
print(f"Backup    : {BACKUP}")

# ── Incremental parse using raw_decode ────────────────────────────────────────
# raw_decode(s, idx) parses exactly one JSON value starting at position idx and
# returns (value, end_position). Walking the array this way stops at the first
# unparseable record instead of requiring the entire candidate to be valid.

decoder = json.JSONDecoder()
records: list[dict] = []

try:
    start = raw.index("[") + 1
except ValueError:
    print("ERROR: file does not contain '[' — not a JSON array.")
    raise SystemExit(1)

pos = start
while pos < len(raw):
    # Skip whitespace and commas between records
    while pos < len(raw) and raw[pos] in " \n\r\t,":
        pos += 1

    if pos >= len(raw) or raw[pos] == "]":
        break  # clean end of array

    try:
        obj, end = decoder.raw_decode(raw, pos)
        records.append(obj)
        pos = end
    except json.JSONDecodeError as exc:
        print(f"Corruption found at char {pos}, after record {len(records)}: {exc}")
        break

print(f"Recovered : {len(records):,} records")

if not records:
    print("\nERROR: No records recovered. Try restoring from stories_progress.csv instead.")
    raise SystemExit(1)

# ── Write repaired checkpoint.json ────────────────────────────────────────────

CHECKPOINT.write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"Repaired  : {CHECKPOINT}  ({CHECKPOINT.stat().st_size:,} bytes)")

# ── Also write checkpoint.jsonl so narratives_new.py can resume directly ──────

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"JSONL      : {OUT_JSONL}  ({OUT_JSONL.stat().st_size:,} bytes)")

print("\nDone. Run:  python narratives_new.py")
print("It will detect checkpoint.jsonl and resume from the recovered records.")

