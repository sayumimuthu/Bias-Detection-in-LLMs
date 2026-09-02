#!/usr/bin/env python3
"""
Strip unresolved git merge-conflict markers out of files, keeping the
"ours" (HEAD) side. In this repo, "theirs" is the old LFS-pointer stub
from the LFS-migration commit, so "ours" is the real data.
Always run --dry-run first and eyeball the .fixed previews / theirs-side
preview before --apply.
"""
import argparse
import shutil
from pathlib import Path

START, MID, END = "<<<<<<< ", "=======", ">>>>>>> "

def find_conflict_blocks(lines):
    blocks, i, n = [], 0, len(lines) 
    while i < n:
        if lines[i].startswith(START):
            start, mid, end = i, None, None
            j = i + 1
            while j < n:
                if lines[j].rstrip("\n") == MID:
                    mid = j
                elif lines[j].startswith(END) and mid is not None:
                    end = j
                    break
                j += 1
            if mid is None or end is None:
                raise ValueError(f"unbalanced markers starting at line {start+1}")
            blocks.append((start, mid, end))
            i = end + 1
        else:
            i += 1
    return blocks

def process(path: Path, apply: bool):
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    try:
        blocks = find_conflict_blocks(lines)
    except ValueError as e:
        print(f"[SKIP-ERROR] {path}: {e}  <-- needs manual review")
        return
    if not blocks:
        return
    if len(blocks) > 1:
        print(f"[MANUAL REVIEW] {path}: {len(blocks)} conflict blocks, skipping auto-fix")
        return

    start, mid, end = blocks[0]
    ours, theirs = lines[start+1:mid], lines[mid+1:end]
    fixed = lines[:start] + ours + lines[end+1:]

    theirs_preview = "".join(theirs).strip().replace("\n", " | ")[:150]
    print(f"[{path}] conflict lines {start+1}-{end+1}  "
          f"(ours={len(ours)} lines, theirs={len(theirs)} lines)")
    print(f"    theirs-side preview: {theirs_preview}")
    if "git-lfs.github.com" not in theirs_preview:
        print(f"    !! theirs-side does NOT look like an LFS pointer -- REVIEW BEFORE APPLYING !!")

    out_path = path.with_name(path.name + ".fixed")
    out_path.write_text("".join(fixed))
    print(f"    -> wrote {out_path} ({len(fixed)} lines)")

    if apply:
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        shutil.move(str(out_path), str(path))
        print(f"    -> applied. original backed up to {backup}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    for line in Path(args.list).read_text().splitlines():
        p = Path(line.strip())
        if p.exists():
            process(p, apply=args.apply)
        elif line.strip():
            print(f"[MISSING] {p}")

if __name__ == "__main__":
    main()
