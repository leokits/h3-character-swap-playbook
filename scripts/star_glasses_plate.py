"""Lay a generated star-glasses plate over each face in a shot.

Chroma-keyed asset (hot pink frame, teal lens) scaled to the face width and centred on the eye line,
so the mangled props the swap left behind are covered by a clean, correct pair. No inpainting: filling
the old props leaves grey smudges that read worse than the mess itself.
"""
import subprocess
import sys

import cv2
import numpy as np

FF = "/home/leomok/bin/ffmpeg"
clip, asset, out = sys.argv[1], sys.argv[2], sys.argv[3]
f0, f1 = int(sys.argv[4]), int(sys.argv[5])
width_ratio = float(sys.argv[6]) if len(sys.argv) > 6 else 1.05

A = cv2.imread(asset)
hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 90) & (hsv[..., 1] > 80) & (hsv[..., 2] > 60))
alpha = (~green).astype(np.uint8)
alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
ys, xs = np.where(alpha > 0)
if len(xs) == 0:
    raise SystemExit("empty asset alpha")
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
glasses = A[y0:y1 + 1, x0:x1 + 1]
galpha = alpha[y0:y1 + 1, x0:x1 + 1]
print("asset", glasses.shape)

fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter("/tmp/_stargl.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
drawn = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if f0 <= i <= f1:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = fc.detectMultiScale(g, 1.08, 5, minSize=(130, 130))
        faces = sorted(faces, key=lambda t: t[2] * t[3], reverse=True)[:2]
        for (fx, fy, fw, fh) in faces:
            tw = int(fw * width_ratio)
            th = int(tw * glasses.shape[0] / glasses.shape[1])
            gl = cv2.resize(glasses, (tw, th), interpolation=cv2.INTER_AREA)
            ga = cv2.resize(galpha, (tw, th), interpolation=cv2.INTER_AREA)
            px = fx + fw // 2 - tw // 2
            py = int(fy + fh * 0.40) - th // 2
            if px < 0 or py < 0 or px + tw > w or py + th > h:
                continue
            roi = fr[py:py + th, px:px + tw].astype(np.float32)
            a = (ga.astype(np.float32) / 255.0)[..., None]
            fr[py:py + th, px:px + tw] = np.clip(gl.astype(np.float32) * a + roi * (1 - a),
                                                 0, 255).astype(np.uint8)
            drawn += 1
    vw.write(fr)
    i += 1
cap.release()
vw.release()
print("frames", i, "faces", drawn)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_stargl.mp4", "-i", clip, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
