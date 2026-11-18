"""Rebuild a lip close-up from REAL Carrie mouth frames.

The video model's swap cannot operate on a macro - it paints a whole person inside the mouth open-
ing - and a still macro with a drift looks dead. So the shot is rebuilt from genuine singing frames:
each frame of the target takes the matching mouth crop from a Carrie close-up, warped to fill the
frame and histogram-matched to the shot's pink cast, so the lips actually form words.
"""
import glob
import subprocess
import sys

import cv2
import numpy as np

A = "/home/leomok/projects/video-regen/apt-family"
FF = "/home/leomok/bin/ffmpeg"
src, out = sys.argv[1], sys.argv[2]
donor_clip = sys.argv[3] if len(sys.argv) > 3 else f"{A}/generations/ext/S48/S48_CLEAN.mp4"

fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def match_hist(a, b):
    o = a.copy()
    for c in range(3):
        h1 = cv2.calcHist([a], [c], None, [256], [0, 256]).ravel()
        h2 = cv2.calcHist([b], [c], None, [256], [0, 256]).ravel()
        c1 = np.cumsum(h1) / max(1, h1.sum())
        c2 = np.cumsum(h2) / max(1, h2.sum())
        o[..., c] = cv2.LUT(a[..., c], np.interp(c1, c2, np.arange(256)).astype(np.uint8))
    return o


# ---- harvest Carrie mouth crops from the donor clip ------------------------------------
mouths = []
cap = cv2.VideoCapture(donor_clip)
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % 2 == 0:
        faces = fc.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 1.1, 5)
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda t: t[2] * t[3])
            mx, my = int(fx + 0.22 * fw), int(fy + 0.60 * fh)
            mw, mh = int(0.56 * fw), int(0.34 * fh)
            crop = f[max(0, my):my + mh, max(0, mx):mx + mw]
            if crop.size and crop.shape[0] > 40:
                # reject false faces: keep only crops that are mostly skin/lips at this light
                ch = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hh, ss, vv = ch[..., 0], ch[..., 1], ch[..., 2]
                skin = (((hh < 20) | (hh > 160)) & (ss > 40) & (vv > 60)).mean()
                wild = ((ss > 200) & (vv > 200)).mean()
                if skin > 0.55 and wild < 0.02:
                    mouths.append(crop)
    i += 1
cap.release()
print("donor mouths", len(mouths))
if not mouths:
    raise SystemExit("no donor mouths")

cap = cv2.VideoCapture(src)
ok, f0 = cap.read()
h, w = f0.shape[:2]
vw = cv2.VideoWriter("/tmp/_lipreal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
n = 0
j = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ok, fr = cap.read()
    if not ok:
        break
    donor = mouths[j % len(mouths)] if len(mouths) > 1 else mouths[0]
    j += 1
    patch = cv2.resize(donor, (w, h), interpolation=cv2.INTER_LANCZOS4)
    patch = match_hist(patch, fr)
    vw.write(patch)
    n += 1
vw.release()
print("frames", n)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_lipreal.mp4", "-i", src, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
