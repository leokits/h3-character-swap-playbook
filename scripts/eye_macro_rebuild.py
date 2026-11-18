"""Rebuild the eye-macro shot: generated Carrie eyes underneath, original artwork on top.

The macro is 1080p of nothing but eyes, so any transplant is out-resolved by the source. Instead a
Carrie macro is generated at the same size, graded to the shot's pink cast, given a slow drift so it
is not a dead still, and the original pink marker artwork is composited over it frame by frame so the
hand-drawn animation in the MV survives exactly.
"""
import glob
import subprocess

import cv2
import numpy as np

A = "/home/leomok/projects/video-regen/apt-family"
src = sorted(glob.glob(f"{A}/generations/ext/S44/Video1_S44_*.mp4"))[0]
macro = cv2.imread(f"{A}/generations/ext/S44/carrie_eyes_macro.png")
out = f"{A}/generations/ext/S44/S44_EYEMACRO.mp4"

cap = cv2.VideoCapture(src)
ok, f0 = cap.read()
h, w = f0.shape[:2]
macro = cv2.resize(macro, (w, h), interpolation=cv2.INTER_LANCZOS4)
# the MV macro is tighter on the eyes than a full-face macro, so push in and re-crop
z = 1.0
cw, ch2 = int(w / z), int(h / z)
macro = cv2.resize(macro[(h - ch2) // 2:(h + ch2) // 2, (w - cw) // 2:(w + cw) // 2], (w, h),
                   interpolation=cv2.INTER_LANCZOS4)

# grade the macro to the shot's colour cast, measured on the source's skin
mhsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
skin = ((mhsv[..., 0] < 12) | (mhsv[..., 0] > 168)) & (mhsv[..., 1] > 40)
for c in range(3):
    src_mean = f0[..., c][skin].mean() if skin.sum() > 100 else f0[..., c].mean()
    src_std = f0[..., c][skin].std() if skin.sum() > 100 else f0[..., c].std()
    mac_mean, mac_std = macro[..., c].mean(), macro[..., c].std() + 1e-6
    macro[..., c] = np.clip((macro[..., c] - mac_mean) * ((src_std + 1e-6) / mac_std) * 0.32 + src_mean, 0, 255)

macro = macro.astype(np.float32)
vw = cv2.VideoWriter("/tmp/_eyemacro.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
n = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ok, fr = cap.read()
    if not ok:
        break
    # slow drift so the regenerated still breathes with the shot
    z = 1.0 + 0.012 * np.sin(n / 9.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.25 * np.sin(n / 12.0), z)
    base = cv2.warpAffine(macro, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    # transfer the original's drawn detail as a ratio (frame / its own blur) so every marker
    # stroke and heart survives on top of Carrie's eyes, whatever its exact colour
    f = fr.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 7)
    # the marker strokes are brighter and pinker than the skin: build an alpha from the positive
    # local contrast, then PAINT the stroke colour on instead of embossing it
    pos = np.clip((f - blur).mean(axis=2) / 22.0, 0, 1)
    hsvf = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    pinkness = np.clip((hsvf[..., 1] - 40) / 90.0, 0, 1) * np.clip((hsvf[..., 2] - 150) / 80.0, 0, 1)
    a = np.clip(pos * 0.75 + pinkness * 0.55, 0, 1)
    a = cv2.GaussianBlur(a, (5, 5), 0)[..., None]
    stroke = np.clip(f * 1.18 + 18.0, 0, 255)     # the original stroke colour, slightly lifted
    comp = base * (1 - a) + stroke * a
    vw.write(np.clip(comp, 0, 255).astype(np.uint8))
    n += 1
vw.release()
print("frames", n)
subprocess.run(["/home/leomok/bin/ffmpeg", "-v", "error", "-y", "-i", "/tmp/_eyemacro.mp4", "-i", src,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
