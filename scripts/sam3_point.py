"""Point-prompt SAM3 probe: isolate one specific person by clicking, not by describing."""
import json
import sys

sys.path.insert(0, "/home/leomok/projects/video-regen/scripts")
from comfy_run import run

video, frame_i, pos, neg, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
pos_pts = [{"x": int(a), "y": int(b)} for a, b in
           (pair.split(":") for pair in pos.split(";"))]
neg_pts = [{"x": int(a), "y": int(b)} for a, b in
           (pair.split(":") for pair in neg.split(";"))]

wf = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
    "2": {"class_type": "VHS_LoadVideoPath",
          "inputs": {"video": video, "force_rate": 24, "custom_width": 0, "custom_height": 0,
                     "frame_load_cap": 1, "skip_first_frames": frame_i, "select_every_nth": 1}},
    "3": {"class_type": "SAM3_Detect",
          "inputs": {"model": ["1", 0], "image": ["2", 0], "threshold": 0.35,
                     "refine_iterations": 2, "individual_masks": True,
                     "positive_coords": json.dumps(pos_pts),
                     "negative_coords": json.dumps(neg_pts)}},
    "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
    "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": prefix}},
}
run(wf)
