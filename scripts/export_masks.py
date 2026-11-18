#!/usr/bin/env python3
"""export_masks.py - tracking-only run on :8190 that saves EVERY frame of the expanded body mask.

No DiT / no generation nodes: this is the cheap SAM pass (seconds, not 13 minutes). The exported
mask frames let a take be composited back onto the source with a feathered edge, which removes the
hard oval boundary a hard mask leaves behind.

Usage
  python3 export_masks.py --pack <pack post-body json> --tag S004 --out generations/shot_S004_cutout/masks \
      --sam-text "a person's head" --bbox 540,140,460,500 --expand 48
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8190"
COMFY = Path("/home/leomok/comfyui/ComfyUI-h3-041")
INPUT, OUT = COMFY / "input", COMFY / "output"
PROJ = Path("/home/leomok/projects/video-regen/apt-family")

# every node that would run heavy generation
DIT = {"800", "801", "803", "805", "806", "807", "808", "809", "850"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam-text", default="")
    ap.add_argument("--bbox", default="")
    ap.add_argument("--expand", type=int, default=48)
    ap.add_argument("--source", default="", help="optional video to stage as Video1")
    ap.add_argument("--plate", default="", help="optional image for the reference node")
    ap.add_argument("--duration", type=float, default=0.0, help="trim duration in seconds")
    a = ap.parse_args()

    body = json.load(open(a.pack))
    p = body["prompt"] if "prompt" in body else body

    if a.source:
        staged = INPUT / f"apt_{a.tag}_source.mp4"
        shutil.copy2(a.source, staged)
        if "30" in p:
            p["30"]["inputs"]["file"] = staged.name
    if a.plate:
        staged = INPUT / f"apt_{a.tag}_plate.png"
        shutil.copy2(a.plate, staged)
        for nid in ("9", "anchor_image"):
            if nid in p:
                p[nid]["inputs"]["image"] = staged.name
    if a.duration and "trim" in p:
        p["trim"]["inputs"]["duration"] = a.duration
        for nid in ("31",):
            if nid in p and "duration_seconds" in p[nid]["inputs"]:
                p[nid]["inputs"]["duration_seconds"] = a.duration
    if a.sam_text and "101" in p:
        p["101"]["inputs"]["text"] = a.sam_text
    if a.bbox and "103c" in p:
        x, y, w, h = (int(v) for v in a.bbox.split(","))
        p["103c"]["inputs"] = {"x": x, "y": y, "width": w, "height": h}
    if "110" in p:
        p["110"]["inputs"]["expansion_px"] = a.expand

    for nid in DIT:
        p.pop(nid, None)

    # save the expanded mask for every frame (MASK -> IMAGE first, SaveImage only takes IMAGE)
    p["399"] = {"class_type": "MaskToImage", "inputs": {"mask": ["110", 0]}}
    p["400"] = {"class_type": "SaveImage",
                "inputs": {"images": ["399", 0], "filename_prefix": f"mask_{a.tag}_all"}}

    q = json.load(urllib.request.urlopen(f"{API}/queue", timeout=30))
    if q.get("queue_running") or q.get("queue_pending"):
        raise SystemExit(f"ABORT: :8190 busy {q}")
    req = urllib.request.Request(f"{API}/prompt",
                                 data=json.dumps({"prompt": p, "client_id": "whiskers-mask-export"}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=120))["prompt_id"]
    print("POSTED", pid, "| tag", a.tag, flush=True)

    t0, hist = time.time(), None
    while time.time() - t0 < 900:
        h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=60))
        if pid in h:
            hist = h[pid]
            break
        time.sleep(5)
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
    n = 0
    for node in (hist.get("outputs") or {}).values():
        for it in node.get("images", []) or []:
            src = OUT / it.get("subfolder", "") / it["filename"]
            if src.is_file():
                shutil.copy2(src, dest / src.name)
                n += 1
    print(f"collected {n} mask frames into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
