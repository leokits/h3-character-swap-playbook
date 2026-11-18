#!/usr/bin/env python3
"""apt_run_swap.py - POST the real Openjutsu 1pp pass for an apt-family shot on :8190.

Clones a proven pack post-body, swaps in the shot's reference plate + anchor plate + prompt + seed,
keeps every generation node, posts ONE job (never dual-queue) and collects the RAW take.

Gate discipline: only call this after `apt_checkmask.py` has shown the intended subject solid at OPEN.

Usage
  python3 apt_run_swap.py --pack <pack _post_body.json> --ref <open plate> --anchor <closed plate> \
      --prompt-file <dense prompt.txt> --tag S003_redo --seed 20261183 --window 39 --history 18 --expand 96 \
      --out generations/shot_S003_lips_CU/swap_redo
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
INPUT = COMFY / "input"
OUT = COMFY / "output"
PROJ = Path("/home/leomok/projects/video-regen/apt-family")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ref", required=True, help="identity/reference plate (node 9)")
    ap.add_argument("--anchor", required=True, help="first-frame anchor plate")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--window", default="39")
    ap.add_argument("--history", default="18")
    ap.add_argument("--expand", type=int, default=96)
    ap.add_argument("--sam-text", default="")
    ap.add_argument("--bbox", default="")
    ap.add_argument("--pos", default="", help='positive point prompt "x:y;x2:y2" (SAM3_Detect 103)')
    ap.add_argument("--neg", default="", help='negative point prompt "x:y"')
    ap.add_argument("--select", default="", choices=["", "largest", "leftmost", "rightmost"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="", help="optional Video1 replacement; duration is re-derived")
    ap.add_argument("--duration", type=float, default=0.0, help="trim seconds (default: keep pack value)")
    ap.add_argument("--resolution", type=int, default=0,
                    help="override H3CharacterSwapPrepareVideo.resolution (768 default; 1080 gives sharper faces)")
    ap.add_argument("--no-anchor", action="store_true",
                    help="disable use_anchor so the take keeps Video1's own background")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    body = json.load(open(a.pack))
    p = body["prompt"] if "prompt" in body else body
    text = Path(a.prompt_file).read_text().strip()

    refname = f"apt_{a.tag}_reference.png"
    ancname = f"apt_{a.tag}_anchor.png"
    shutil.copy2(a.ref, INPUT / refname)
    shutil.copy2(a.anchor, INPUT / ancname)
    p["9"]["inputs"]["image"] = refname
    p["anchor_image"]["inputs"]["image"] = ancname
    if a.source:
        srcname = f"apt_{a.tag}_source.mp4"
        shutil.copy2(a.source, INPUT / srcname)
        p["30"]["inputs"]["file"] = srcname
        if a.duration and "trim" in p:
            p["trim"]["inputs"]["duration"] = a.duration
            if "31" in p and "duration_seconds" in p["31"]["inputs"]:
                p["31"]["inputs"]["duration_seconds"] = a.duration
    if a.resolution and "31" in p and "resolution" in p["31"]["inputs"]:
        p["31"]["inputs"]["resolution"] = a.resolution
    p["801"]["inputs"]["prompt"] = text
    if a.no_anchor and "803" in p:
        p["803"]["inputs"]["use_anchor"] = False
    p["801"]["inputs"]["seed"] = a.seed
    p["805"]["inputs"]["window_frames"] = a.window
    p["805"]["inputs"]["history_frames"] = a.history
    if "110" in p:
        p["110"]["inputs"]["expansion_px"] = a.expand
    if a.sam_text and "101" in p:
        p["101"]["inputs"]["text"] = a.sam_text
    if a.bbox and "103c" in p:
        x, y, w, h = (int(v) for v in a.bbox.split(","))
        p["103c"]["inputs"] = {"x": x, "y": y, "width": w, "height": h}
    # Click-to-select: SAM3_Detect honours positive/negative POINT prompts, which is the only way to
    # name a small background person. Text and bbox both lose to the largest subject.
    if a.pos and "103" in p:
        pts = [{"x": int(x), "y": int(y)} for x, y in (q.split(":") for q in a.pos.split(";"))]
        p["103"]["inputs"]["positive_coords"] = json.dumps(pts)
        p["103"]["inputs"].pop("bboxes", None)
        p["103"]["inputs"].pop("conditioning", None)
    if a.neg and "103" in p:
        npts = [{"x": int(x), "y": int(y)} for x, y in (q.split(":") for q in a.neg.split(";"))]
        p["103"]["inputs"]["negative_coords"] = json.dumps(npts)
    if a.select and "104" in p:
        p["104"]["inputs"]["selection"] = a.select
    prefix = f"apt_{a.tag}_seed{a.seed}"
    for nid in ("807", "809"):
        if nid in p:
            p[nid]["inputs"]["filename_prefix"] = prefix

    dest = PROJ / a.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "_post_body.json").write_text(json.dumps({"prompt": p, "client_id": "whiskers-apt-swap"}, indent=2))
    print(f"body → {dest/'_post_body.json'} | seed {a.seed} | window {a.window}/{a.history} | expand {a.expand}")
    print(f"ref {refname} | anchor {ancname} | prompt {len(text)} chars")
    if not a.go:
        print("dry run — not posting")
        return 0

    q = json.load(urllib.request.urlopen(f"{API}/queue", timeout=30))
    if q.get("queue_running") or q.get("queue_pending"):
        raise SystemExit(f"ABORT: :8190 busy {q}")
    req = urllib.request.Request(f"{API}/prompt",
                                 data=json.dumps({"prompt": p, "client_id": "whiskers-apt-swap"}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=120))["prompt_id"]
    print("POSTED", pid)
    t0, hist = time.time(), None
    while time.time() - t0 < 3600:
        h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=60))
        if pid in h:
            hist = h[pid]
            break
        time.sleep(12)
        print(f"  …{int(time.time()-t0)}s", flush=True)
    if hist is None:
        print("TIMEOUT")
        return 1
    st = hist.get("status", {})
    print("status", st.get("status_str"))
    for m in st.get("messages", []):
        if m[0] == "execution_error":
            print("  ERR", m[1].get("node_type"), (m[1].get("exception_message") or "")[:300])
    got = 0
    # SaveVideo reports under "images" on some builds and "videos"/"gifs" on others - scan every
    # key, and fall back to globbing the output dir by the run's filename prefix.
    for node in (hist.get("outputs") or {}).values():
        for key in ("videos", "gifs", "images"):
            for it in node.get(key, []) or []:
                name = it.get("filename", "")
                if not name.lower().endswith((".mp4", ".webm", ".mov", ".mkv")):
                    continue
                src = OUT / it.get("subfolder", "") / name
                if src.is_file():
                    shutil.copy2(src, dest / src.name)
                    print("collected →", dest / src.name)
                    got += 1
    if not got:
        for src in sorted(OUT.glob(f"{prefix}*.mp4"), key=lambda p: p.stat().st_mtime):
            shutil.copy2(src, dest / src.name)
            print("collected (by prefix) →", dest / src.name)
            got += 1
    if not got:
        print("NO VIDEO OUTPUT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
