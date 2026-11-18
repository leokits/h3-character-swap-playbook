"""Build a "wearing the props" plate from a HIGH-similarity identity plate.

Cutting plates out of a generated group scene loses identity (the faces come back generic). The fix is
to start from the identity plates that score best against the real people and draw the prop onto them:
the swap then carries identity and wardrobe/props at the same time.
"""
import sys

import cv2
import numpy as np

ident, asset, out = sys.argv[1], sys.argv[2], sys.argv[3]
scale_f = float(sys.argv[4]) if len(sys.argv) > 4 else 1.15

P = cv2.imread(ident)
A = cv2.imread(asset)
hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 78) & (hsv[..., 1] > 80) & (hsv[..., 2] > 60))
alpha = cv2.morphologyEx(((~green) * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
ys, xs = np.where(alpha > 0)
glasses = A[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
galpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

g = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY)
fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
faces = fc.detectMultiScale(g, 1.08, 5)
if len(faces) == 0:
    raise SystemExit("no face in identity plate")
fx, fy, fw, fh = max(faces, key=lambda t: t[2] * t[3])
eyes = ec.detectMultiScale(g[fy:fy + fh, fx:fx + fw], 1.08, 4,
                           minSize=(int(fw * 0.10), int(fh * 0.10)))
eyes = sorted([(x + fx, y + fy, ew, eh) for (x, y, ew, eh) in eyes if y < fh * 0.6], key=lambda t: t[0])
if len(eyes) >= 2:
    eye_cx = (eyes[0][0] + eyes[0][2] / 2 + eyes[-1][0] + eyes[-1][2] / 2) / 2
    eye_cy = (eyes[0][1] + eyes[0][3] / 2 + eyes[-1][1] + eyes[-1][3] / 2) / 2
    span = abs(eyes[-1][0] - eyes[0][0]) + eyes[-1][2]
else:
    eye_cx, eye_cy, span = fx + fw / 2, fy + fh * 0.42, fw * 0.62
print("face", (fx, fy, fw, fh), "eye centre", (round(eye_cx), round(eye_cy)), "span", round(span))

tw = int(fw * 1.02 * scale_f)   # glasses span roughly the face width
th = int(tw * glasses.shape[0] / glasses.shape[1])
gl = cv2.resize(glasses, (tw, th), interpolation=cv2.INTER_AREA)
ga = cv2.resize(galpha, (tw, th), interpolation=cv2.INTER_AREA)
px, py = int(eye_cx - tw / 2), int(eye_cy - th / 2)
print("paste box", px, py, tw, th, "canvas", P.shape[1], P.shape[0], "asset bbox", glasses.shape)
px = max(0, min(px, P.shape[1] - tw))
py = max(0, min(py, P.shape[0] - th))
if tw <= 0 or th <= 0 or py + th > P.shape[0] or px + tw > P.shape[1]:
    raise SystemExit("paste box out of canvas")
roi = P[py:py + th, px:px + tw].astype(np.float32)
a = (ga.astype(np.float32) / 255.0)[..., None]
P[py:py + th, px:px + tw] = np.clip(gl.astype(np.float32) * a + roi * (1 - a), 0, 255).astype(np.uint8)
cv2.imwrite(out, P)
print("wrote", out, P.shape)
