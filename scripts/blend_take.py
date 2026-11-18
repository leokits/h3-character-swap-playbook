#!/usr/bin/env python3
"""blend_take.py - put a take back over its source with a FEATHERED mask.

A hard SAM mask leaves a visible oval boundary: the generated interior has different grain,
sharpness and colour from the surrounding source, so the edge reads as a ring. Blurring the mask
alpha over a wide radius (default 28 px at 1344x736) dissolves the boundary while keeping the
generated performance in the middle.

Optionally applies the global pink overlay at the end - pink is applied to the WHOLE frame, never
per region, so no ring can come from the grade step.

Usage
  python3 blend_take.py --take take.mp4 --source Video1.mp4 --masks <dir> --out blended.mp4 \
      [--feather 28] [--pink 0.18]
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
PINK = np.array([255, 60, 140], dtype=np.float32)


def decode(path: str, w: int, h: int) -> np.ndarray:
    raw = subprocess.check_output([FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def size_of(path: str) -> tuple[int, int]:
    out = subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def fps_of(path: str) -> float:
    out = subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=r_frame_rate", "-of", "json", path])
    num, den = json.loads(out)["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feather", type=float, default=28.0)
    ap.add_argument("--pink", type=float, default=0.0)
    ap.add_argument("--auto-mask", action="store_true",
                    help="derive the alpha from |take-source| instead of the SAM masks (finds the real seam)")
    ap.add_argument("--diff-thresh", type=float, default=12.0)
    ap.add_argument("--protect-white", type=float, default=0.0,
                    help="keep source pixels at least this bright (0-255) inside the mask, e.g. white "
                         "handwritten lettering over the shot: 210")
    ap.add_argument("--crf", type=int, default=12)
    a = ap.parse_args()

    w, h = size_of(a.source)
    take = decode(a.take, w, h)
    src = decode(a.source, w, h)
    n = min(len(take), len(src))
    take, src = take[:n], src[:n]

    mask_files = sorted(Path(a.masks).glob("*mask_expanded_f0*.png")) or \
                 sorted(Path(a.masks).glob("*_all_*.png"))
    if not mask_files:
        raise SystemExit(f"no mask frames in {a.masks}")
    masks = [np.asarray(Image.open(p).convert("L")) for p in mask_files[:n]]
    if len(masks) < n:                      # a static mask is better than no mask
        masks = masks + [masks[-1]] * (n - len(masks))

    out = np.empty(take.shape, dtype=np.uint8)   # uint8: rawvideo expects bytes, not float32
    cover = []
    for i in range(n):
        if a.auto_mask:
            # alpha straight from the real difference, thresholded then heavily blurred: this hides
            # the seam wherever the take actually departs from the source, whatever mask the pack used.
            d = np.abs(take[i] - src[i]).mean(2)
            binm = (d > a.diff_thresh).astype(np.uint8) * 255
            m = Image.fromarray(binm).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(a.feather))
        else:
            m = Image.fromarray(masks[i].astype(np.uint8)).resize((w, h), Image.BILINEAR)
            m = m.filter(ImageFilter.GaussianBlur(a.feather))
        alpha = (np.asarray(m).astype(np.float32) / 255.0)[..., None]
        if a.protect_white > 0:
            # handwritten lettering sits on top of the plate: never let the take paint over it
            bright = (src[i].min(axis=2) >= a.protect_white).astype(np.float32)
            bright = np.asarray(Image.fromarray((bright * 255).astype(np.uint8))
                                .filter(ImageFilter.MaxFilter(5))
                                .filter(ImageFilter.GaussianBlur(2))).astype(np.float32) / 255.0
            alpha = alpha * (1 - bright[..., None])
        cover.append(float(alpha.mean()))
        frame = src[i] * (1 - alpha) + take[i] * alpha
        if a.pink > 0:
            frame = frame * (1 - a.pink) + PINK * a.pink
        out[i] = np.clip(frame, 0, 255).astype(np.uint8)

    fps = fps_of(a.source)
    silent = Path(a.out).with_suffix(".silent.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", str(silent)], input=out.tobytes(), check=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(silent), "-i", a.source,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy",
                    "-shortest", a.out], check=True)
    silent.unlink(missing_ok=True)

    print(f"frames {n} @ {fps:g}fps  {w}x{h}  feather {a.feather:g}px  pink {a.pink:g}")
    print(f"mean mask coverage {np.mean(cover)*100:.1f}% (blurred alpha)")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
