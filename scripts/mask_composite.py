"""Composite a replacement-person clip onto a base clip through the tracked person's mask.

The replacement clip was built from the SOURCE footage, so its foreground still carries the original
performer; only the tracked person's area may be taken from it. The mask is dilated and feathered so
the paste has no hard edge, and the patch is brightness-matched to its surroundings.
"""
import glob
import subprocess
import sys

import cv2
import numpy as np

FF = "/home/leomok/bin/ffmpeg"
base_clip, patch_clip, mask_glob, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
base_start = int(sys.argv[5])
dil = int(sys.argv[6]) if len(sys.argv) > 6 else 17
masks = sorted(glob.glob(mask_glob))
print("masks", len(masks))

cb = cv2.VideoCapture(base_clip)
cp = cv2.VideoCapture(patch_clip)
fps = cb.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cb.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cb.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter("/tmp/_maskcomp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
j = 0
done = 0
last_ok = None
while True:
    ok, fr = cb.read()
    if not ok:
        break
    if i >= base_start:
        ok2, pk = cp.read()
        if ok2 and j < len(masks):
            if pk.shape[1] != w or pk.shape[0] != h:
                pk = cv2.resize(pk, (w, h), interpolation=cv2.INTER_LANCZOS4)
            m = cv2.imread(masks[j], cv2.IMREAD_GRAYSCALE)
            j += 1
            good = m is not None and (m > 60).mean() * 100 < 12.0
            if not good and last_ok is not None:
                m = last_ok          # detector grabbed the foreground on this frame: reuse the track
                good = True
            if m is not None and good:
                last_ok = m
                a = cv2.dilate((m > 60).astype(np.uint8) * 255, np.ones((dil, dil), np.uint8))
                a = cv2.GaussianBlur(a, (45, 45), 0).astype(np.float32) / 255.0
                al = a[..., None]
                sub = pk.astype(np.float32)
                reg = fr.astype(np.float32)
                pm = float(sub[a > 0.2].mean()) if (a > 0.2).sum() > 100 else float(sub.mean())
                rm = float(reg[a > 0.2].mean()) if (a > 0.2).sum() > 100 else float(reg.mean())
                if pm > 1:
                    sub = np.clip(sub * max(0.8, min(1.2, rm / pm)), 0, 255)
                fr = np.clip(sub * al + reg * (1 - al), 0, 255).astype(np.uint8)
                done += 1
    vw.write(fr)
    i += 1
cb.release()
cp.release()
vw.release()
print("frames", i, "composited", done)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_maskcomp.mp4", "-i", base_clip, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
