#!/usr/bin/env python3
"""merge_two_takes.py - two single-person swaps into one frame, each through its own tracked mask.

For shots where two characters share the frame and each was swapped in a separate 1pp run: keep the
untouched source everywhere, drop take A inside mask A and take B inside mask B. Masks are feathered and
normalised so they never add past 1.0.

Usage
  python3 merge_two_takes.py --source src.mp4 --take-a a.mp4 --masks-a dirA --take-b b.mp4 --masks-b dirB \
      --out merged.mp4 [--feather 12]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

FFMPEG = "/home/leomok/bin/ffmpeg"
FFPROBE = "/home/leomok/bin/ffprobe"


def probe(path: str) -> tuple[int, int, float]:
    out = subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def decode(path: str, w: int, h: int) -> np.ndarray:
    raw = subprocess.check_output([FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def load_mask(path: Path, w: int, h: int, feather: float) -> np.ndarray:
    m = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    m = m.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m).astype(np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--take-a", required=True)
    ap.add_argument("--masks-a", required=True)
    ap.add_argument("--take-b", required=True)
    ap.add_argument("--masks-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feather", type=float, default=12.0)
    ap.add_argument("--crf", type=int, default=12)
    a = ap.parse_args()

    w, h, fps = probe(a.source)
    src = decode(a.source, w, h)
    ta = decode(a.take_a, w, h)
    tb = decode(a.take_b, w, h)
    n = min(len(src), len(ta), len(tb))
    fa = sorted(Path(a.masks_a).glob("*_all_*.png"))
    fb = sorted(Path(a.masks_b).glob("*_all_*.png"))
    if not fa or not fb:
        raise SystemExit("missing mask frames")

    out = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        ma = load_mask(fa[min(i, len(fa) - 1)], w, h, a.feather)
        mb = load_mask(fb[min(i, len(fb) - 1)], w, h, a.feather)
        total = ma + mb
        # where both masks claim a pixel, split proportionally; where neither does, keep the source
        sa = np.divide(ma, np.maximum(total, 1e-6))
        sb = np.divide(mb, np.maximum(total, 1e-6))
        cover = np.clip(total, 0, 1)[..., None]
        a_part = ta[i] * sa[..., None]
        b_part = tb[i] * sb[..., None]
        mixed = a_part + b_part
        out[i] = np.clip(src[i] * (1 - cover) + mixed * cover, 0, 255).astype(np.uint8)

    silent = Path(a.out).with_suffix(".silent.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", str(silent)], input=out.tobytes(), check=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(silent), "-i", a.source,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy",
                    "-shortest", a.out], check=True)
    silent.unlink(missing_ok=True)
    print(f"merged {n} frames → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
