"""Export the SAM3 point-seeded track as PNG masks (npz + per-frame png).

The swap regenerates only inside the tracked mask, so to composite a take back onto a different base
we need that mask separately. Seeds with the same point prompts used for the take, tracks the whole
clip, and writes both the npz (pack format) and per-frame PNGs (for cv2 compositing).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/leomok/projects/video-regen/scripts")
from comfy_run import API, run  # noqa: E402

import subprocess
video, pos, neg, prefix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
nframes = int(subprocess.run(["/home/leomok/bin/ffprobe", "-v", "error", "-count_frames",
                              "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                              "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip())
print("frames", nframes)
pos_pts = [{"x": int(a), "y": int(b)} for a, b in (q.split(":") for q in pos.split(";"))]
neg_pts = [{"x": int(a), "y": int(b)} for a, b in (q.split(":") for q in neg.split(";"))] if neg else []

wf = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
    "2": {"class_type": "VHS_LoadVideoPath",
          "inputs": {"video": video, "force_rate": 24, "custom_width": 0, "custom_height": 0,
                     "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1}},
    "3": {"class_type": "SAM3_Detect",
          "inputs": {"model": ["1", 0], "image": ["2", 0], "threshold": 0.35,
                     "refine_iterations": 2, "individual_masks": True,
                     "positive_coords": json.dumps(pos_pts),
                     "negative_coords": json.dumps(neg_pts)}},
    "4": {"class_type": "H3CharacterSwapSelectSubject",
          "inputs": {"masks": ["3", 0], "selection": "index", "index": 0}},
    "5": {"class_type": "SAM3_VideoTrack",
          "inputs": {"images": ["2", 0], "model": ["1", 0], "detection_threshold": 0.35,
                     "max_objects": 1, "detect_interval": 1, "initial_mask": ["4", 0]}},
    "6": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["5", 0], "object_indices": "0"}},
    "7": {"class_type": "H3CharacterSwapSaveMask",
          "inputs": {"masks": ["6", 0], "filename_prefix": prefix, "frames": nframes}},
    "8": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix + "_png"}},
}
h = run(wf, wait=300)
print(json.dumps(h.get("outputs", {}) if h else {}, indent=1)[:600])
