"""Copy one rectangular region from a take onto a base clip, feathered.

Used when a take is right only in the area where a small background person was regenerated: the rest
of the take still carries the source performer, so a whole-frame replacement would undo the swap.
"""
import subprocess
import sys

import cv2
import numpy as np

FF = "/home/leomok/bin/ffmpeg"
base_clip, take_clip, out = sys.argv[1], sys.argv[2], sys.argv[3]
x0, y0, x1, y1 = [int(v) for v in sys.argv[4].split(",")]
base_start = int(sys.argv[5])          # frame index in base that aligns with take frame 0

cb = cv2.VideoCapture(base_clip)
ct = cv2.VideoCapture(take_clip)
fps = cb.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cb.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cb.get(cv2.CAP_PROP_FRAME_HEIGHT))
feather = 25
mask = np.zeros((y1 - y0, x1 - x0), np.float32)
mask[:, :] = 1.0
mask[:feather, :] *= np.linspace(0, 1, feather)[:, None]
mask[-feather:, :] *= np.linspace(1, 0, feather)[:, None]
mask[:, :feather] *= np.linspace(0, 1, feather)[None, :]
mask[:, -feather:] *= np.linspace(1, 0, feather)[None, :]
al = mask[..., None]

vw = cv2.VideoWriter("/tmp/_boxpaste.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
i = 0
pasted = 0
while True:
    ok, fr = cb.read()
    if not ok:
        break
    if i >= base_start:
        ok2, tk = ct.read()
        if ok2:
            if tk.shape[1] != w or tk.shape[0] != h:
                tk = cv2.resize(tk, (w, h), interpolation=cv2.INTER_LANCZOS4)
            reg = fr[y0:y1, x0:x1].astype(np.float32)
            sub = tk[y0:y1, x0:x1].astype(np.float32)
            fr[y0:y1, x0:x1] = np.clip(sub * al + reg * (1 - al), 0, 255).astype(np.uint8)
            pasted += 1
    vw.write(fr)
    i += 1
cb.release()
ct.release()
vw.release()
print("frames", i, "pasted", pasted)
subprocess.run([FF, "-v", "error", "-y", "-i", "/tmp/_boxpaste.mp4", "-i", base_clip, "-map", "0:v:0",
                "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=False)
print("wrote", out)
