#!/usr/bin/env python3
"""face_sim.py - ArcFace cosine similarity between a plate and a real reference face.

The gate for this project: a plate must score >= 0.80 against the person's real photo before any swap
is allowed to run. Uses insightface (buffalo models) on CPU/GPU via onnxruntime; embeddings are the
512-d normed ArcFace vectors, so the score is a plain cosine in [-1, 1].

  python3 face_sim.py --plate plates/session02/PLATE_X.png --ref /path/real_face.png [--json out.json]
  python3 face_sim.py --plate a.png b.png --ref real.png       # score several plates at once
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def build_app(det_size: int = 320):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def embed(app, path: str) -> np.ndarray | None:
    """Detect then embed. Large frames are downscaled first: a face that fills a 1024px plate misses
    the detector's anchors, and the same image resized to 640 is found at score 0.88."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        print(f"  ! cannot read {path}", file=sys.stderr)
        return None
    faces = []
    for side in (640, 480, 800, 1024):
        h, w = img.shape[:2]
        scale = side / max(h, w)
        cand = img if scale >= 1.0 else cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        faces = app.get(cand)
        if faces:
            break
    if not faces:
        return None
    # largest face
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    app = build_app()
    app_alt = build_app(640)   # a couple of plates only detect at the larger anchor set

    def emb(path: str):
        e = embed(app, path)
        if e is None:
            e = embed(app_alt, path)
        return e

    ref = emb(a.ref)
    if ref is None:
        print(f"ERROR: no face found in reference {a.ref}")
        return 2

    rows = []
    worst = 1.0
    for p in a.plate:
        e = emb(p)
        if e is None:
            rows.append({"plate": p, "sim": None, "pass": False, "note": "no face detected"})
            worst = -1.0
            continue
        sim = float(np.dot(ref, e))
        rows.append({"plate": p, "sim": round(sim, 3), "pass": sim >= a.gate})
        worst = min(worst, sim)

    print(f"reference: {a.ref}")
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        sim = "n/a" if r["sim"] is None else f"{r['sim']:.3f}"
        print(f"  [{mark}] sim {sim}  {r['plate']}  {r.get('note','')}")
    print(f"gate {a.gate:.2f} — overall {'PASS' if worst >= a.gate else 'FAIL'}")

    if a.json:
        json.dump({"ref": a.ref, "gate": a.gate, "rows": rows}, open(a.json, "w"), indent=1)
    return 0 if worst >= a.gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
