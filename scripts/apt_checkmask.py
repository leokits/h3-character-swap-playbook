#!/usr/bin/env python3
"""apt_checkmask.py - live SAM checkmask on :8190 for one apt-family shot, using a cloned pack body.

Gate rule (HANDOFF §1): a live checkmask on :8190 must show the intended subject solid at OPEN before any
DiT is posted. This strips the generation nodes from a proven pack body, swaps in the new plate and SAM
wording, posts once, and saves the mask overlays plus per-frame mask stats.

Usage
  python3 apt_checkmask.py --pack generations/shot_S003_lips_CU/pack_openjutsu_carrie_seed20261149/_post_body_seed20261149.json \
      --plate plates/session02/PLATE_S003_carrie_lips_noglasses_v1.png \
      --tag S003_redo_v1 --sam-text "a close-up of a person's lips and mouth" --bbox 280,70,840,560 \
      --out generations/shot_S003_lips_CU/checkmask_redo_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

API = "http://127.0.0.1:8190"
COMFY = Path("/home/leomok/comfyui/ComfyUI-h3-041")
INPUT = COMFY / "input"
OUT = COMFY / "output"
PROJ = Path("/home/leomok/projects/video-regen/apt-family")

# generation-only nodes in the apt 1pp pack
DIT = {"800", "803", "805", "806", "807", "808", "809", "850"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="proven pack post-body json")
    ap.add_argument("--plate", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sam-text", default="")
    ap.add_argument("--bbox", default="")
    ap.add_argument("--expand", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="", help="optional Video1 replacement")
    ap.add_argument("--duration", type=float, default=0.0)
    a = ap.parse_args()

    body = json.load(open(a.pack))
    p = body["prompt"] if "prompt" in body else body

    # stage the plate into the :8190 input dir (LoadImage only sees input/)
    plate_src = Path(a.plate)
    staged = INPUT / f"apt_{a.tag}_plate.png"
    shutil.copy2(plate_src, staged)
    rel = staged.name
    if a.source:
        srcname = f"apt_{a.tag}_source.mp4"
        shutil.copy2(a.source, INPUT / srcname)
        if "30" in p:
            p["30"]["inputs"]["file"] = srcname
        if a.duration and "trim" in p:
            p["trim"]["inputs"]["duration"] = a.duration
            if "31" in p and "duration_seconds" in p["31"]["inputs"]:
                p["31"]["inputs"]["duration_seconds"] = a.duration
    if "9" in p:
        p["9"]["inputs"]["image"] = rel
    if "anchor_image" in p:
        p["anchor_image"]["inputs"]["image"] = rel
    if a.sam_text and "101" in p:
        p["101"]["inputs"]["text"] = a.sam_text
    if a.bbox and "103c" in p:
        x, y, w, h = (int(v) for v in a.bbox.split(","))
        p["103c"]["inputs"] = {"x": x, "y": y, "width": w, "height": h}
    if a.expand and "110" in p:
        p["110"]["inputs"]["expansion_px"] = a.expand

    for nid in DIT:
        p.pop(nid, None)
    # mask previews -> saved files
    p["201"] = {"class_type": "MaskToImage", "inputs": {"mask": ["107", 0]}}
    p["202"] = {"class_type": "MaskToImage", "inputs": {"mask": ["110", 0]}}
    p["200"] = {"class_type": "ImageFromBatch", "inputs": {"batch_index": 0, "length": 1, "image": ["31", 0]}}
    p["203"] = {"class_type": "ImageFromBatch", "inputs": {"batch_index": 0, "length": 1, "image": ["201", 0]}}
    p["204"] = {"class_type": "ImageFromBatch", "inputs": {"batch_index": 0, "length": 1, "image": ["202", 0]}}
    p["205"] = {"class_type": "ImageCompositeMasked",
                "inputs": {"destination": ["200", 0], "source": ["204", 0], "x": 0, "y": 0,
                           "resize_source": False, "mask": ["204", 0]}}
    for nid, src, suf in (("300", "200", "raw_f0"), ("301", "203", "mask_raw_f0"),
                          ("302", "204", "mask_expanded_f0"), ("303", "205", "mask_over_f0")):
        p[nid] = {"class_type": "SaveImage",
                  "inputs": {"images": [src, 0], "filename_prefix": f"cm_{a.tag}_{suf}"}}
    body_out = PROJ / a.out / "_checkmask_post_body.json"
    body_out.parent.mkdir(parents=True, exist_ok=True)
    body_out.write_text(json.dumps({"prompt": p, "client_id": "whiskers-apt-cm"}, indent=2))

    q = json.load(urllib.request.urlopen(f"{API}/queue", timeout=30))
    if q.get("queue_running") or q.get("queue_pending"):
        raise SystemExit(f"ABORT: :8190 busy {q}")
    req = urllib.request.Request(f"{API}/prompt",
                                 data=json.dumps({"prompt": p, "client_id": "whiskers-apt-cm"}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=120))["prompt_id"]
    print("POSTED", pid, "| tag", a.tag)
    t0, hist = time.time(), None
    while time.time() - t0 < 1500:
        h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=60))
        if pid in h:
            hist = h[pid]
            break
        time.sleep(8)
        print(f"  …{int(time.time()-t0)}s", flush=True)
    if hist is None:
        print("TIMEOUT")
        return 1
    st = hist.get("status", {})
    print("status", st.get("status_str"))
    for m in st.get("messages", []):
        if m[0] == "execution_error":
            print("  ERR", m[1].get("node_type"), (m[1].get("exception_message") or "")[:300])
    dest = PROJ / a.out
    dest.mkdir(parents=True, exist_ok=True)
    for node in (hist.get("outputs") or {}).values():
        for it in node.get("images", []) or []:
            src = OUT / it.get("subfolder", "") / it["filename"]
            if src.is_file():
                shutil.copy2(src, dest / src.name)
    stats = []
    for f in sorted(dest.glob("cm_*mask_raw_f0*.png")):
        m = np.asarray(Image.open(f).convert("L")) > 127
        if m.any():
            ys, xs = np.nonzero(m)
            stats.append({"file": f.name, "nz": int(m.sum()), "frac": round(float(m.mean()), 4),
                          "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    if stats:
        (dest / "mask_stats.json").write_text(json.dumps(stats, indent=1))
        print("mask:", stats[0])
    print("collected →", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
