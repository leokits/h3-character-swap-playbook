"""Paste a swapped crop back through its own mask, histogram-matched.

Same job as crop_paste.py but the paste is graded per frame with a histogram match against the host
region instead of a single luminance gain, which is what left the bright rectangle in the earlier
attempt, and the mask (feathered) is the only alpha so the crop's edges never show.
"""
import glob
import subprocess
import sys

import cv2
import numpy as np

A = "/home/leomok/projects/video-regen/apt-family"
FF = "/home/leomok/bin/ffmpeg"
shot, box, role, base, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
cx, cy, cw, ch = [int(v) for v in box.split(",")]

take = sorted(glob.glob(f"{A}/generations/ext/{shot}/crop_take_{role}/*.mp4"))[0]
masks = sorted(glob.glob(f"{A}/generations/ext/{shot}/crop_mask_{role}/*_all_*.png"))
print("take", take, "masks", len(masks))


def match_hist(a, b):
    o = a.copy()
    for c in range(3):
        h1 = cv2.calcHist([a], [c], None, [256], [0, 256]).ravel()
        h2 = cv2.calcHist([b], [c], None, [256], [0, 256]).ravel()
        c1 = np.cumsum(h1) / max(1, h1.sum())
        c2 = np.cumsum(h2) / max(1, h2.sum())
        o[..., c] = cv2.LUT(a[..., c], np.interp(c1, c2, np.arange(256)).astype(np.uint8))
    return o


cb = cv2.VideoCapture(base)
ct = cv2.VideoCapture(take)
fps = cb.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cb.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cb.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter("/tmp/_cp2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
while True:
    ok, fr = cb.read()
    if not ok:
        break
    okt, tf = ct.read()
    if not okt or i >= len(masks):
        vw.write(fr)
        i += 1
        continue
    tf = cv2.resize(tf, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    mk = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
    if mk is None:
        vw.write(fr)
        i += 1
        continue
    mk = cv2.resize(mk, (cw, ch), interpolation=cv2.INTER_LINEAR)
    mk = cv2.GaussianBlur(mk, (17, 17), 0)
    roi = fr[cy:cy + ch, cx:cx + cw]
    # grade the take to the host only where it will be used
    sel = mk > 60
    if sel.sum() > 200:
        tf = match_hist(tf, roi)
    al = (mk.astype(np.float32) / 255.0)[..., None]
    fr[cy:cy + ch, cx:cx + cw] = (tf.astype(np.float32) * al + roi.astype(np.float32) * (1 - al)).astype(np.uint8)
    vw.write(fr)
    i += 1
cb.release()
ct.release()
vw.release()
print("frames", i)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_cp2.mp4", "-i", base, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
