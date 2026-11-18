#!/usr/bin/env python3
"""h3_image_plate.py - MiniMax-H3-Image plate generator for the GB10 pipelines.

MiniMax-H3-Image (open weights) does text-to-image *and* multi-reference editing with one checkpoint,
6 steps, CFG 1, euler/simple. In the remake pipelines it replaces the cloud (Grok) plate step: give it
the official OPEN frame as <Picture 1> plus identity refs, describe the wardrobe, and it returns a
pose-matched plate for that shot.

Runs against the H3-Image instance on :8191 (ComfyUI v0.45.2). The video fork on :8190 is untouched.

Model files (shared models dir):
  diffusion_models/minimax_h3_image_pruned_w4a8.safetensors   12.5 GB
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors  15 GB  (shared with the video pipeline)
  vae/minimax_h3_image_vae_fp16.safetensors                   1.2 GB

Usage
  python3 h3_image_plate.py --prompt "..." --out plates/S003.png --size 2048x1152 \
      --ref official_open.png --ref carrie_mummy.png --seed 20261183
  python3 h3_image_plate.py --prompt "..." --node ph3 --steps 6       # pruned P-H3 path
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8191"
COMFY = Path("/home/leomok/comfyui/ComfyUI-h3-image-git")
OUT = COMFY / "output"
INPUT = COMFY / "input"

UNET = "minimax_h3_image_pruned_int8_convrot.safetensors"  # w4a8 needs comfy-kitchen >0.2.41
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_image_vae_fp16.safetensors"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))


def wait(pid: str, timeout: int = 1800, label: str = "") -> dict | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=60))
        except Exception:
            h = {}
        if pid in h:
            return h[pid]
        time.sleep(6)
        print(f"  …{int(time.time() - t0)}s {label}", flush=True)
    return None


def stage_refs(paths: list[str], tag: str) -> list[str]:
    """Copy references into the :8191 input dir and return their relative names."""
    d = INPUT / tag
    d.mkdir(parents=True, exist_ok=True)
    names = []
    for i, p in enumerate(paths, 1):
        src = Path(p)
        if not src.is_file():
            raise SystemExit(f"missing reference: {p}")
        dst = d / f"ref_{i}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        names.append(f"{tag}/{dst.name}")
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", action="append", default=[], help="reference picture, repeatable (image_1 = edit source)")
    ap.add_argument("--size", default="2048x1152", help="WxH, multiples of 32 (0x0 = follow image_1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--node", choices=["mini", "ph3"], default="mini")
    ap.add_argument("--unet", default=UNET)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    a = ap.parse_args()

    w, h = (int(v) for v in a.size.lower().split("x"))
    tag = "h3img_" + Path(a.out).stem
    refs = stage_refs(a.ref, tag) if a.ref else []

    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }
    cond_inputs = {"clip": ["2", 0], "prompt": a.prompt, "width": w, "height": h, "vae": ["3", 0]}
    for i, name in enumerate(refs):
        g[str(10 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        cond_inputs[f"images.image_{i}"] = [str(10 + i), 0]
    g["4"] = {"class_type": "PH3ImageConditioning" if a.node == "ph3" else "MiniMaxH3ImageConditioning",
              "inputs": cond_inputs}

    model_src = ["1", 0]
    if a.node == "ph3":
        g["5"] = {"class_type": "ModelSamplingMiniMaxH3",
                  "inputs": {"model": ["1", 0], "shift_video": 1.0, "shift_audio": 1.0}}
        model_src = ["5", 0]

    g["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
    g["7"] = {"class_type": "KSampler",
              "inputs": {"model": model_src, "positive": ["4", 0], "negative": ["6", 0],
                         "latent_image": ["4", 1], "seed": a.seed, "steps": a.steps, "cfg": 1.0,
                         "sampler_name": a.sampler, "scheduler": a.scheduler, "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0],
                                                    "filename_prefix": f"h3img_{Path(a.out).stem}"}}

    q = json.load(urllib.request.urlopen(f"{API}/queue", timeout=30))
    if q.get("queue_running") or q.get("queue_pending"):
        raise SystemExit(f"ABORT: :8191 queue busy {q}")
    pid = post("/prompt", {"client_id": "whiskers-h3img", "prompt": g})["prompt_id"]
    print(f"POSTED {pid} | {a.node} | {w}x{h} | steps {a.steps} | refs {len(refs)}")
    hist = wait(pid, label=Path(a.out).stem)
    if hist is None:
        print("TIMEOUT")
        return 1
    st = hist.get("status", {})
    print("status", st.get("status_str"))
    for m in st.get("messages", []):
        if m[0] == "execution_error":
            print("  ERR", m[1].get("node_type"), (m[1].get("exception_message") or "")[:300])
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    made = []
    for node in (hist.get("outputs") or {}).values():
        for it in node.get("images", []) or []:
            src = OUT / it.get("subfolder", "") / it["filename"]
            if src.is_file():
                made.append(src)
    if not made:
        print("NO OUTPUT")
        return 1
    shutil.copy2(made[-1], dest)
    print("OUT", dest, f"({made[-1].stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
