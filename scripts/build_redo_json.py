#!/usr/bin/env python3
"""build_redo_json.py - assemble the take map for a stitched preview from whatever exists on disk.

Preference per shot: <SHOT>_FINAL.mp4 (role-map pipeline) > <SHOT>_MERGED_v3.mp4 (dual, corrected) >
swap/*.mp4 (single pass). Shots whose mask gate was too small stay None so the official frame is used.

  python3 build_redo_json.py --out /tmp/redo.json [--from S19] [--to S38]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

A = "/home/leomok/projects/video-regen/apt-family"


def pick(shot: str) -> str | None:
    d = f"{A}/generations/ext/{shot}"
    for pat in (f"{d}/{shot}_FIXED2.mp4", f"{d}/{shot}_FIXED.mp4", f"{d}/{shot}_FINAL.mp4",
                f"{d}/{shot}_MERGED_v3.mp4", f"{d}/swap/*.mp4"):
        hits = sorted(glob.glob(pat))
        if hits:
            return os.path.relpath(hits[0], A)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="frm", default="S13")
    ap.add_argument("--to", default="S38")
    a = ap.parse_args()

    segs = json.load(open("/tmp/segs.json"))
    frm, to = int(a.frm[1:]), int(a.to[1:])
    rows = []
    for s in segs:
        n = int(s["id"][1:])
        if n < frm or n > to:
            continue
        take = pick(s["id"])
        rows.append([s["a"], s["b"], take])
        print(f'{s["id"]:>4} {s["a"]:>5}-{s["b"]:<5} {take or "official"}')
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {sum(1 for r in rows if r[2])}/{len(rows)} shots have takes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
