"""Per-frame SAM3 detection used as a poor-man's tracker.

SAM3_VideoTrack lost the subject after frame 0 on a small, moving background figure, so tracking is
done the slow but reliable way: detect on frame i with points taken from frame i-1's mask centroid,
carry the centroid forward, and write one mask PNG per frame. Those masks drive either an inpaint
(remove the person) or a cut-out paste (replace them).
"""
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import urllib.request

sys.path.insert(0, "/home/leomok/projects/video-regen/scripts")
from comfy_run import API, run  # noqa: E402

video, out_dir, seed_pts = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(subprocess.run(["/home/leomok/bin/ffprobe", "-v", "error", "-count_frames",
                        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                        "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip())
os.makedirs(out_dir, exist_ok=True)
pts = [{"x": int(a), "y": int(b)} for a, b in (q.split(":") for q in seed_pts.split(";"))]
neg = []      # keep no negatives: the figure is small and the backdrop is not a person
print("frames", n, "seed", pts)

for i in range(n):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
        "2": {"class_type": "VHS_LoadVideoPath",
              "inputs": {"video": video, "force_rate": 24, "custom_width": 0, "custom_height": 0,
                         "frame_load_cap": 1, "skip_first_frames": i, "select_every_nth": 1}},
        "3": {"class_type": "SAM3_Detect",
              "inputs": {"model": ["1", 0], "image": ["2", 0], "threshold": 0.30,
                         "refine_iterations": 2, "individual_masks": True,
                         "positive_coords": json.dumps(pts)}},
        "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0],
                                                    "filename_prefix": f"pf_{i:03d}"}},
    }
    h = run(wf, wait=120)
    if not h:
        print("frame", i, "no result")
        continue
    imgs = h.get("outputs", {}).get("5", {}).get("images", [])
    if not imgs:
        print("frame", i, "no image")
        continue
    src = f"/home/leomok/comfyui/ComfyUI-h3-041/output/{imgs[0]['filename']}"
    m = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue
    cv2.imwrite(f"{out_dir}/m_{i:03d}.png", m)
    ys, xs = np.where(m > 60)
    if len(xs) > 300:
        cx, cy = int(xs.mean()), int(ys.mean())
        hgt = int(ys.max() - ys.min())
        pts = [{"x": cx, "y": cy}, {"x": cx, "y": cy + max(30, hgt // 3)}]
        print(f"frame {i}: mask {len(xs)} px centroid ({cx},{cy}) h={hgt}")
    else:
        print(f"frame {i}: LOST ({len(xs)} px) — keeping previous points")
print("done")
