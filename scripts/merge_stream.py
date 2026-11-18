#!/usr/bin/env python3
"""merge_stream.py - memory-light version of merge_two_takes.

The original kept three float32 1080p clips in RAM (~1.6 GB each) and died inside a piped subshell.
This one keeps uint8 frames, blends in place, and streams every frame straight into ffmpeg.

  python3 merge_stream.py --source src.mp4 --take-a a.mp4 --masks-a dirA --take-b b.mp4 --masks-b dirB \
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


def decoder(path: str, w: int, h: int):
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE)
    nbytes = w * h * 3
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.stdout.close()
    proc.wait()


def load_mask(path: Path, w: int, h: int, feather: float) -> np.ndarray:
    m = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    if feather:
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
    fa = sorted(Path(a.masks_a).glob("*_all_*.png"))
    fb = sorted(Path(a.masks_b).glob("*_all_*.png"))
    if not fa or not fb:
        print("missing masks")
        return 1

    silent = Path(a.out + ".silent.mp4")
    enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
                            "-pix_fmt", "yuv420p", str(silent)], stdin=subprocess.PIPE)

    n = 0
    for i, (src, ta, tb) in enumerate(zip(decoder(a.source, w, h),
                                          decoder(a.take_a, w, h),
                                          decoder(a.take_b, w, h))):
        ma = load_mask(fa[min(i, len(fa) - 1)], w, h, a.feather)
        mb = load_mask(fb[min(i, len(fb) - 1)], w, h, a.feather)
        tot = ma + mb
        sa = ma / np.maximum(tot, 1e-6)
        sb = mb / np.maximum(tot, 1e-6)
        cover = np.clip(tot, 0, 1)[..., None]
        mixed = ta.astype(np.float32) * sa[..., None] + tb.astype(np.float32) * sb[..., None]
        frame = src.astype(np.float32) * (1 - cover) + mixed * cover
        enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()

    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(silent), "-i", a.source,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy",
                    "-shortest", a.out], check=True)
    silent.unlink(missing_ok=True)
    print(f"merged {n} frames → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
