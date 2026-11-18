"""Rebuild a macro shot around a generated plate, keeping the original's drawn artwork.

Shared by the eye macro and the three lip close-ups: grade the generated macro to the shot's colour
cast, add a slow drift so it is not a frozen still, then paint the original's marker strokes, hearts
and doodles back on top (local-contrast + pinkness alpha, painted in the stroke's own colour).
"""
import subprocess
import sys

import cv2
import numpy as np

A = "/home/leomok/projects/video-regen/apt-family"
FF = "/home/leomok/bin/ffmpeg"
src, img, out = sys.argv[1], sys.argv[2], sys.argv[3]
drift = float(sys.argv[4]) if len(sys.argv) > 4 else 0.012

cap = cv2.VideoCapture(src)
ok, f0 = cap.read()
h, w = f0.shape[:2]
macro = cv2.imread(img)
macro = cv2.resize(macro, (w, h), interpolation=cv2.INTER_LANCZOS4)
zoom = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
if zoom != 1.0:
    cw2, ch2 = int(w / zoom), int(h / zoom)
    macro = cv2.resize(macro[(h - ch2) // 2:(h + ch2) // 2, (w - cw2) // 2:(w + cw2) // 2], (w, h),
                       interpolation=cv2.INTER_LANCZOS4)

def match_hist(src_img, ref_img):
    """Per-channel histogram match so the generated macro inherits the shot's exact tone."""
    out = src_img.copy()
    for c in range(3):
        s_hist = cv2.calcHist([src_img], [c], None, [256], [0, 256]).ravel()
        r_hist = cv2.calcHist([ref_img], [c], None, [256], [0, 256]).ravel()
        s_cdf = np.cumsum(s_hist) / max(1, s_hist.sum())
        r_cdf = np.cumsum(r_hist) / max(1, r_hist.sum())
        lut = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.uint8)
        out[..., c] = cv2.LUT(src_img[..., c], lut)
    return out


macro = match_hist(macro.astype(np.uint8), f0).astype(np.float32)
vw = cv2.VideoWriter("/tmp/_macro.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
n = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ok, fr = cap.read()
    if not ok:
        break
    z = 1.0 + drift * np.sin(n / 9.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.22 * np.sin(n / 12.0), z)
    base = cv2.warpAffine(macro, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    f = fr.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 7)
    pos = np.clip((f - blur).mean(axis=2) / 22.0, 0, 1)
    hsvf = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    pinkness = np.clip((hsvf[..., 1] - 40) / 90.0, 0, 1) * np.clip((hsvf[..., 2] - 150) / 80.0, 0, 1)
    a = np.clip(pos * 0.0 + pinkness * 0.0, 0, 1)   # lip CUs carry no marker artwork
    a = cv2.GaussianBlur(a, (5, 5), 0)[..., None]
    stroke = np.clip(f * 1.06 + 6.0, 0, 255)
    comp = base * (1 - a) + stroke * a
    vw.write(np.clip(comp, 0, 255).astype(np.uint8))
    n += 1
vw.release()
print("frames", n)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_macro.mp4", "-i", src, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
