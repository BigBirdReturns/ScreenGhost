#!/usr/bin/env python3
"""Score pick_action against a UI-grounding dataset. No device required.

Grounding is the load-bearing skill of the whole architecture: given an
instruction and a screenshot, does the model tap inside the right element?
Public benchmarks (ScreenSpot, RICO-derived sets, Android in the Wild
episodes) ship exactly that as (image, instruction, bounding-box) triples,
so accuracy is measurable today with nothing plugged in.

Manifest format — one JSON object per line:
    {"image": "screens/0001.png", "instruction": "turn on Dark Mode", "bbox": [x1, y1, x2, y2]}

Image paths are resolved relative to the manifest file. A row scores a hit
when the chosen tap lands inside bbox.

Usage:
    python evals/grounding_eval.py manifest.jsonl
    python evals/grounding_eval.py manifest.jsonl --limit 50 --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="JSONL manifest of (image, instruction, bbox) rows")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N rows")
    parser.add_argument("--out", type=Path, default=None, help="write per-row results as JSONL")
    args = parser.parse_args()

    from PIL import Image

    from screenghost import pick_action

    rows = load_manifest(args.manifest, args.limit)
    if not rows:
        print("manifest is empty", file=sys.stderr)
        return 1

    base = args.manifest.parent
    results = []
    hits = 0
    misses_no_tap = 0

    for i, row in enumerate(rows):
        img = Image.open(base / row["image"]).convert("RGB")
        x1, y1, x2, y2 = row["bbox"]
        try:
            action = pick_action(img, row["instruction"])
        except ValueError as e:
            results.append({**row, "hit": False, "error": str(e)})
            print(f"[{i + 1}/{len(rows)}] ✗ unparseable model response")
            continue

        if action.action_type != "tap":
            misses_no_tap += 1
            results.append({**row, "hit": False, "chose": action.action_type})
            print(f"[{i + 1}/{len(rows)}] ✗ chose {action.action_type}, expected tap")
            continue

        x, y = action.params["x"], action.params["y"]
        hit = x1 <= x <= x2 and y1 <= y <= y2
        hits += hit
        results.append({**row, "hit": hit, "tap": [x, y]})
        mark = "✓" if hit else "✗"
        print(f"[{i + 1}/{len(rows)}] {mark} tap=({x},{y}) target={row['bbox']}  {row['instruction'][:60]}")

    total = len(rows)
    print(f"\ngrounding accuracy: {hits}/{total} = {hits / total:.1%}"
          + (f"  (non-tap answers: {misses_no_tap})" if misses_no_tap else ""))

    if args.out:
        with args.out.open("w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"per-row results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
