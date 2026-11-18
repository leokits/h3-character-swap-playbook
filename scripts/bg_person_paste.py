"""Paste a full-body cut-out into a tracked person's mask, frame by frame.

For a distant background dancer the H3 swap cannot draw a replacement (mask is ~4 % of frame), so the
replacement is composited: the tracked mask's bbox positions and scales the plate, the source's own
softness is matched with a blur, and the result is pasted through the feathered mask.
"""
import glob
import subprocess
import sys

import cv2
import numpy as np

A = "/home/leomok/projects/video-regen/apt-family"
FF = "/home/leomok/bin/ffmpeg"
base_clip, plate_path, mask_glob, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
out_start = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # first output frame to touch
masks = sorted(glob.glob(mask_glob))[:]
print("masks", len(masks))

P = cv2.imread(plate_path, cv2.IMREAD_UNCHANGED)
if P.shape[2] == 4:
    P = cv2.cvtColor(P, cv2.COLOR_BGRA2BGR)
# key the plate's pale studio backdrop
hsvP = cv2.cvtColor(P, cv2.COLOR_BGR2HSV)
hP_, sP_, vP_ = hsvP[..., 0], hsvP[..., 1], hsvP[..., 2]
green = ((hP_ > 35) & (hP_ < 90) & (sP_ > 80) & (vP_ > 60))
green_frac = float(green.mean())
print("green fraction", round(green_frac, 3))
if green_frac > 0.15:
    # chroma-key plate: alpha is simply "not green"
    pa = (~green * 255).astype(np.uint8)
    pa = cv2.morphologyEx(pa, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    pa = cv2.erode(pa, np.ones((3, 3), np.uint8))
else:
    pale = ((sP_ < 70) & (vP_ > 140)).astype(np.uint8)
    dark = (vP_ < 25).astype(np.uint8)
    bgish = ((pale | dark) * 255).astype(np.uint8)
    ff = bgish.copy()
    hP, wP = ff.shape
    m2 = np.zeros((hP + 2, wP + 2), np.uint8)
    for pt in [(0, 0), (wP - 1, 0), (0, hP - 1), (wP - 1, hP - 1),
               (wP // 2, 0), (wP // 2, hP - 1), (0, hP // 2), (wP - 1, hP // 2)]:
        if ff[pt[1], pt[0]] > 0:
            cv2.floodFill(ff, m2, pt, 128)
    pa = (~(ff == 128) * 255).astype(np.uint8)
pa = cv2.GaussianBlur(pa, (5, 5), 0)
ys, xs = np.where(pa > 60)
px0, px1, py0, py1 = xs.min(), xs.max(), ys.min(), ys.max()
plate = P[py0:py1 + 1, px0:px1 + 1]
palpha = pa[py0:py1 + 1, px0:px1 + 1]
print("plate body", plate.shape)

cap = cv2.VideoCapture(base_clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter("/tmp/_bgpaste.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    mi = i if i < len(masks) else len(masks) - 1
    m = cv2.imread(masks[mi], cv2.IMREAD_GRAYSCALE)
    if m is None:
        vw.write(fr)
        i += 1
        continue
    cov = float((m > 60).mean()) * 100
    ys2, xs2 = np.where(m > 60)
    # a small distant dancer never covers more than a few percent; a bigger blob means the detector
    # grabbed the backdrop or the foreground, so keep the frame untouched
    if len(xs2) < 400 or cov > 9.0:
        vw.write(fr)
        i += 1
        continue
    bx0, bx1, by0, by1 = xs2.min(), xs2.max(), ys2.min(), ys2.max()
    bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
    target_h = int(bh * 1.30)
    scale = target_h / plate.shape[0]
    nw, nh = max(8, int(plate.shape[1] * scale)), target_h
    pl = cv2.resize(plate, (nw, nh), interpolation=cv2.INTER_AREA)
    al = cv2.resize(palpha, (nw, nh), interpolation=cv2.INTER_AREA)
    cx = (bx0 + bx1) // 2
    x0 = cx - nw // 2
    y1 = by1 + 1
    y0 = y1 - nh
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x0 + nw)
    if y1c <= y0c or x1c <= x0c:
        vw.write(fr)
        i += 1
        continue
    sub = pl[y0c - y0:y0c - y0 + (y1c - y0c), x0c - x0:x0c - x0 + (x1c - x0c)]
    suba = al[y0c - y0:y0c - y0 + (y1c - y0c), x0c - x0:x0c - x0 + (x1c - x0c)]
    a = (suba.astype(np.float32) / 255.0)[..., None]
    reg = fr[y0c:y1c, x0c:x1c].astype(np.float32)
    # match the shot's softness and tone
    subf = sub.astype(np.float32)
    subf = cv2.GaussianBlur(subf, (0, 0), 1.1)
    # match brightness only: scaling contrast against a now-empty pink backdrop washes the figure out
    plate_mean = float(subf.mean())
    reg_mean = float(reg.mean())
    if plate_mean > 1:
        gain = max(0.75, min(1.35, reg_mean / plate_mean))
        subf = np.clip(subf * gain, 0, 255)
    fr[y0c:y1c, x0c:x1c] = np.clip(subf * a + reg * (1 - a), 0, 255).astype(np.uint8)
    vw.write(fr)
    i += 1
vw.release()
print("frames", i)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_bgpaste.mp4", "-i", base_clip, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
