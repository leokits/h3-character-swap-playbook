"""Paste an ANIMATED cut-out head into every cell of a head-grid shot (v5).

Earlier versions pasted a still plate, which kills the performance: the MV's cut-out heads are
animated (mouth moving, head bobbing). The donors here are clips lifted from the already-swapped
cut-out-head shots (0:04 Leo, 0:05 Carrie), so the grid inherits real motion and lip movement.

Alignment: per frame, detect the face in the target cell and in the donor frame, then warp the
donor so its face lands on the cell's face. Alpha comes from the target cell's own silhouette when
it can be computed, otherwise a feathered ellipse fitted to the cell box.

Usage:
  grid_composite5.py --shot S65 --donor generations/donor_carrie_head.mp4
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_probe import app, heads_in_frame

A = "/home/leomok/projects/video-regen/apt-family"
FF = "/home/leomok/bin/ffmpeg"


def faces(bgr):
    out = []
    for f in app().get(bgr):
        x1, y1, x2, y2 = f.bbox
        if min(x2 - x1, y2 - y1) < 18:
            continue
        out.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
    out.sort(key=lambda t: (t[1] // 150, t[0]))
    return out


def load_donor(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--donor2", default=None,
                    help="second animated donor (male) for a MIXED grid; cell role decided by brightness")
    ap.add_argument("--inset", type=float, default=0.04)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clip = glob.glob(f"{A}/generations/ext/{args.shot}/Video1_{args.shot}_*.mp4")[0]
    out = args.out or f"{A}/generations/ext/{args.shot}/{args.shot}_GRID5.mp4"
    donor_frames = load_donor(args.donor)
    donor2_frames = load_donor(args.donor2) if args.donor2 else None
    # donor face boxes, cached per donor frame
    def fill_grid(fl, shape):
        """A 3x3 cut-out grid often loses a head to haar; rebuild the missing boxes from the
        median row/col of the ones we did find, so no cell keeps the original performer."""
        if len(fl) == 9 or len(fl) < 4:
            return fl
        h, w = shape[:2]
        xs = sorted(t[0] for t in fl)
        ys = sorted(t[1] for t in fl)
        mw = int(np.median([t[2] for t in fl]))
        mh = int(np.median([t[3] for t in fl]))
        cols = [int(np.median(xs[:3])), int(np.median(xs[len(xs)//2-1:len(xs)//2+2])), int(np.median(xs[-3:]))]
        rows = sorted(set([int(np.median(ys[:3])), int(np.median(ys[len(ys)//2-1:len(ys)//2+2])), int(np.median(ys[-3:]))]))
        out = []
        for ry in rows:
            for cx in cols:
                near = [t for t in fl if abs(t[0]-cx) < mw*0.9 and abs(t[1]-ry) < mh*0.9]
                out.append(near[0] if near else (cx, ry, mw, mh))
        return out

    donor_faces = [[tuple(int(v) for v in t) for t in faces(f)] for f in donor_frames]
    donor2_faces = [[tuple(int(v) for v in t) for t in faces(f)] for f in donor2_frames] if donor2_frames else None
    print(f"donor frames={len(donor_frames)}")

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = "/tmp/_grid5.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    last_good = None
    last_frame = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tail_hold = 8
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if total and n >= total - tail_hold and last_frame is not None:
            # the grid shrinks away at the very end of the shot and the detector (rightly) stops
            # finding heads there; hold the last composited frame instead of flashing the originals
            vw.write(last_frame)
            n += 1
            continue
        src_snapshot = fr.copy()
        # pick a donor frame that actually HAS a detected face; some donor frames come back empty and
        # then nothing is pasted at all, which is how the original heads reappear for a few frames
        valid = [k for k in range(len(donor_frames)) if donor_faces[k]]
        if not valid:
            vw.write(fr)
            n += 1
            continue
        dn = valid[n % len(valid)]
        dfr = donor_frames[dn]
        dfs = donor_faces[dn]
        if dfs:
            det = fill_grid(faces(fr), fr.shape)
            if len(det) < 7 and last_frame is not None:
                # the grid shrinks out of detector range at the end of the shot: hold the last
                # composited frame rather than letting the original heads flash back for 6 frames
                vw.write(last_frame)
                n += 1
                continue
            if len(det) >= 4:
                last_good = det
            for tf in det:
                tx, ty, tw, th = [int(v) for v in tf]
                # mixed grid: bright (blonde) cell -> primary donor, dark (cap) cell -> donor2
                use2 = False
                if donor2_frames:
                    # Calibrated on S71: Bruno (black cap + shades) head-box mean V = 111-140,
                    # Rosé (blonde curls) = 185-191, so V < 160 separates them cleanly.
                    hb = src_snapshot[ty:ty + th, tx:tx + tw]
                    if hb.size:
                        hv = float(cv2.cvtColor(hb, cv2.COLOR_BGR2HSV)[..., 2].mean())
                        use2 = hv < 160
                    else:
                        use2 = False
                else:
                    use2 = False
                if use2:
                    dfr2 = donor2_frames[dn % len(donor2_frames)]
                    dfs2 = donor2_faces[dn % len(donor2_faces)]
                    if not dfs2:
                        continue
                    dface = max(dfs2, key=lambda t: t[2] * t[3])
                    warp_src = dfr2
                else:
                    dface = max(dfs, key=lambda t: t[2] * t[3])
                    warp_src = dfr
                sc = tw / dface[2]
                dcx, dcy = dface[0] + dface[2] / 2, dface[1] + dface[3] / 2
                tcx, tcy = tx + tw / 2, ty + th / 2
                M = np.array([[sc, 0, tcx - sc * dcx], [0, sc, tcy - sc * dcy]], dtype=np.float32)
                warped = cv2.warpAffine(warp_src, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_REPLICATE)
                # the donor is a cut-out head on its own flat backdrop; key that backdrop out so the
                # wide ellipse cannot paste a solid colour block under every chin
                dh, dw = warp_src.shape[:2]
                corner = warp_src[2:12, 2:12].reshape(-1, 3).mean(axis=0)
                dkey = (np.abs(warp_src.astype(np.float32) - corner).max(axis=2) > 26).astype(np.uint8) * 255
                dkey = cv2.morphologyEx(dkey, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                dkey = cv2.erode(dkey, np.ones((3, 3), np.uint8))
                dkey_w = cv2.warpAffine(dkey, M, (w, h), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                # ellipse alpha sized to the cell's head, centred slightly above the face
                # keep the ellipse on the head: a 2x-wide ellipse drags the donor's own backdrop in as
                # a magenta block under every chin
                ew, eh = int(tw * 2.0), int(th * 2.2)
                mask = np.zeros((h, w), np.uint8)
                cv2.ellipse(mask, (int(tcx), int(tcy - th * 0.30)),
                            (max(1, int(ew * (0.5 - args.inset))), max(1, int(eh * (0.5 - args.inset)))),
                            0, 0, 360, 255, -1)
                mask = cv2.bitwise_and(mask, dkey_w)
                mask = cv2.GaussianBlur(mask, (13, 13), 0)
                al = (mask.astype(np.float32) / 255.0)[..., None]
                fr = (warped.astype(np.float32) * al + fr.astype(np.float32) * (1 - al)).astype(np.uint8)
        vw.write(fr)
        last_frame = fr
        n += 1
    cap.release()
    vw.release()
    print("frames", n)

    subprocess.run([FF, "-v", "error", "-y", "-i", tmp, "-i", clip,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", out], check=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
