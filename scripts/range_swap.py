"""Replace a frame range in a video with frames from another clip.

Used to graft a re-rendered shot back into the master without touching the frames on either side (a
title card lives immediately after this range and must survive).
"""
import subprocess
import sys

import cv2

FF = "/home/leomok/bin/ffmpeg"
base, patch, out = sys.argv[1], sys.argv[2], sys.argv[3]
b0, b1, p0 = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

cb = cv2.VideoCapture(base)
cp = cv2.VideoCapture(patch)
fps = cb.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cb.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cb.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter("/tmp/_rangeswap.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
used = 0
while True:
    ok, fr = cb.read()
    if not ok:
        break
    if b0 <= i <= b1:
        ok2, pk = cp.read()
        if ok2:
            if pk.shape[1] != w or pk.shape[0] != h:
                pk = cv2.resize(pk, (w, h), interpolation=cv2.INTER_LANCZOS4)
            fr = pk
            used += 1
    vw.write(fr)
    i += 1
cb.release()
cp.release()
vw.release()
print("frames", i, "replaced", used)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_rangeswap.mp4", "-i", base, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
