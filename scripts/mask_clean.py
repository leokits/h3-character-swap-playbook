"""Clean a per-frame mask track: keep only the plausible connected component.

When the detector briefly grabs the backdrop or the foreground, coverage jumps and the mask is
useless. Rather than dropping the frame, keep the single connected component that is closest to the
previous frame's centroid and within a plausible size, so the track stays continuous.
"""
import glob
import os
import sys

import cv2
import numpy as np

src_glob, out_dir = sys.argv[1], sys.argv[2]
max_cov = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0
os.makedirs(out_dir, exist_ok=True)
files = sorted(glob.glob(src_glob))
prev_c = None
prev_area = None
for i, f in enumerate(files):
    m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue
    bw = (m > 60).astype(np.uint8)
    cov = bw.mean() * 100
    if cov > max_cov or cov < 0.05:
        n, lab, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
        best = None
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            if area < 300 or area / bw.size * 100 > max_cov:
                continue
            c = cents[k]
            score = 0.0 if prev_c is None else float(np.hypot(c[0] - prev_c[0], c[1] - prev_c[1]))
            if prev_area:
                score += abs(area - prev_area) / max(1.0, prev_area) * 40
            if best is None or score < best[0]:
                best = (score, k, area, c)
        if best is not None:
            bw = (lab == best[1]).astype(np.uint8)
            prev_c, prev_area = best[3], best[2]
        else:
            print(f"{i}: no plausible component (cov {cov:.1f}%)")
            bw = bw * 0
    else:
        ys, xs = np.where(bw > 0)
        prev_c = (float(xs.mean()), float(ys.mean()))
        prev_area = int(len(xs))
    cv2.imwrite(f"{out_dir}/m_{i:03d}.png", (bw * 255).astype(np.uint8))
    print(f"{i}: cov {cov:.1f}% -> {(bw.mean()*100):.1f}%")
print("done", len(files))
