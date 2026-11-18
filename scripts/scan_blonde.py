"""Scan a preview for frames that still contain the ORIGINAL blonde performer (Rosé).

Carrie has dark hair and Leo has short dark hair, so any frame with a healthy share of bright
low-saturation pixels inside the subject area is almost certainly Rosé that nobody swapped.
Reports contiguous runs so they can be mapped straight back onto shot ids.
"""
import subprocess
import sys

import cv2
import numpy as np

W, H = 384, 216
src = sys.argv[1]
clip = sys.argv[2] if len(sys.argv) > 2 else ""

cmd = ["ffmpeg", "-v", "error", "-i", src, "-vf", f"scale={W}:{H}", "-f", "rawvideo",
       "-pix_fmt", "bgr24", "-"]
if clip:
    a, b = [int(v) for v in clip.split(",")]
    cmd = ["ffmpeg", "-v", "error", "-i", src, "-vf", f"select='between(n\\,{a}\\,{b})',scale={W}:{H}",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 8)
n = 0
hits = []
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3:
        break
    fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    # subject area: middle band, skipping the outer frame
    roi = fr[int(H * 0.12):int(H * 0.92), int(W * 0.10):int(W * 0.90)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blonde = ((hsv[..., 1] < 80) & (hsv[..., 2] > 150)).mean()
    hits.append((n, float(blonde)))
    n += 1
p.stdout.close()
p.wait()

TH = 0.12
runs = []
cur = None
for i, v in hits:
    if v >= TH:
        if cur is None:
            cur = [i, i, v]
        else:
            cur[1] = i
            cur[2] = max(cur[2], v)
    else:
        if cur:
            runs.append(cur)
            cur = None
if cur:
    runs.append(cur)

print(f"frames={n} threshold={TH} suspicious_runs={len(runs)}")
for a, b, v in runs:
    print(f"  preview {a}-{b} ({b-a+1}f) = {a/24:.2f}s-{b/24:.2f}s  max_blonde={v:.2f}  MV {a+283}-{b+283}")
