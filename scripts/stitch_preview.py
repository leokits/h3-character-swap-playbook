#!/usr/bin/env python3
"""stitch_preview.py - assemble every finished apt-family shot into one standing preview.

Each segment is either a take file or the untouched official footage. Takes are trimmed or padded to
the exact frame count the official cut has, so the preview stays frame-accurate against the MV. The
audio is taken from the MV for the whole range, so the joins never drift.

Edit SEGMENTS / A / B and run:
  python3 stitch_preview.py --out generations/preview/PREVIEW_STANDING_<date>.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

FFMPEG = "/home/leomok/bin/ffmpeg"
FFPROBE = "/home/leomok/bin/ffprobe"
PROJ = Path("/home/leomok/projects/video-regen/apt-family")
SRC = PROJ / "source/APT_official_MV.mp4"
W, H, FPS = 1920, 1080, 24

# (start, end, path or None for official footage) — end exclusive, official MV frame numbers
SEGMENTS: list[tuple[int, int, str | None]] = [
    (283, 313, None),                                                             # S003 passthrough
    (313, 351, "generations/shot_S004_cutout/swap_v1/S004_v1_WITHAUDIO.mp4"),     # S004 Carrie
    (351, 382, "generations/shot_S005_cutout/swap_A_v4/S005a_v4_WITHAUDIO.mp4"),  # S005a Leo
    (382, 425, "generations/shot_S005B_cutout/swap_v1/S005b_HEADONSRC.mp4"),      # S005b Carrie
    (425, 445, None),                                                             # S005c passthrough
    (445, 477, None),                                                             # S006a/b passthrough
    (477, 533, "generations/shot_S006/S006c_MERGED.mp4"),                         # S006c both
    (533, 623, "generations/shot_S007/S007_FINAL.mp4"),                           # S007 Carrie
    (623, 689, "generations/shot_S008/S008_MERGED.mp4"),                          # S008 both
    (689, 700, "generations/shot_S009/swap_singer/apt_S009_singer_seed20261300_00001_.mp4"),
    (700, 749, "generations/shot_S010/S010_FINAL.mp4"),                           # S010 Carrie + lyric
    (749, 775, "generations/shot_S011/S011_MERGED.mp4"),                          # S011 both
    (775, 852, None),   # S13
    (852, 933, None),   # S14
    (933, 1008, None),  # S15
    (1008, 1044, None), # S16
    (1044, 1082, None), # S17 lips CU passthrough
    (1082, 1121, None), # S18
    (1121, 1140, None), # S19
    (1140, 1155, None), # S20
    (1155, 1164, None), # S21
    (1164, 1239, None), # S22
    (1239, 1278, None), # S23
    (1278, 1319, None), # S24
    (1319, 1331, None), # S25
    (1331, 1367, None), # S26
    (1367, 1388, None), # S27
    (1388, 1403, None), # S28 passthrough
    (1403, 1424, None), # S29
    (1424, 1449, None), # S30
    (1449, 1518, None), # S31
    (1518, 1539, None), # S32
    (1539, 1614, None), # S33 passthrough (mask 1.6%)
    (1614, 1630, None), # S34
    (1630, 1683, None), # S35
    (1683, 1697, None), # S36
    (1697, 1779, None), # S37
    (1779, 1853, None), # S38 passthrough (mask 3.3%)
    (1853, 1932, None), # S39
    (1932, 1971, None), # S40
    (1971, 2052, None), # S41
    (2052, 2127, None), # S42 passthrough
    (2127, 2170, None), # S43
    (2170, 2203, None), # S44 passthrough
    (2203, 2242, None), # S45 passthrough
    (2242, 2389, None), # S46 passthrough
    (2389, 2453, None), # S47
    (2453, 2502, None), # S48
    (2502, 2529, None), # S49 passthrough
    (2529, 2681, None), # S50
    (2681, 2737, None), # S51
    (2737, 2783, None), # S52
    (2783, 2836, None), # S53 passthrough
    (2836, 2937, None), # S54
    (2937, 2988, None), # S55 passthrough
    (2988, 3091, None), # S56
    (3091, 3165, None), # S57 passthrough
    (3165, 3240, None), # S58
    (3240, 3277, None), # S59
    (3277, 3323, None), # S60
    (3323, 3361, None), # S61
    (3361, 3373, None), # S62
    (3373, 3438, None), # S63
    (3438, 3476, None), # S64
    (3476, 3517, None), # S65
    (3517, 3593, None), # S66
    (3593, 3632, None), # S67
    (3632, 3667, None), # S68
    (3667, 3710, None), # S69
    (3710, 3787, None), # S70
    (3787, 3829, None), # S71
    (3829, 3865, None), # S72
    (3865, 3902, None), # S73
    (3902, 3945, None), # S74
    (3945, 3990, None), # S75
    (3990, 4098, None), # S76
    (4098, 4157, None), # S77
]


FACEFIX = {
    (313, 351): "generations/facefix/S004_ff.mp4",
    (351, 382): "generations/facefix/S005a_ff.mp4",
    (382, 425): "generations/facefix/S005b_ff.mp4",
    (477, 533): "generations/facefix/S006c_ff.mp4",
    (533, 623): "generations/facefix/S007_ff.mp4",
    (623, 689): "generations/facefix/S008_ff.mp4",
    (689, 700): "generations/facefix/S009_ff.mp4",
    (700, 749): "generations/facefix/S010_ff.mp4",
    (749, 775): "generations/facefix/S011_ff.mp4",
}


REDO = {
    (313, 351): "generations/redo/S004b/apt_S004b_seed20261700_00001_.mp4",
    (351, 382): "generations/redo/S005ab2/apt_S005ab2_seed20262050_00001_.mp4",
    (382, 425): "generations/redo/S005bb/apt_S005bb_seed20261700_00001_.mp4",
    (445, 477): "generations/shot_S006ab_rebuild/S006ab_MERGED.mp4",
    (477, 533): "generations/redo/S006cb_v3.mp4",
    (533, 623): "generations/redo/S007b2/apt_S007b2_seed20262010_00001_.mp4",
    (623, 689): "generations/redo/S008b_v3.mp4",
    (689, 700): "generations/redo/S009b/apt_S009b_seed20261700_00001_.mp4",
    (700, 749): "generations/redo/S010b2/apt_S010b2_seed20262010_00001_.mp4",
    (749, 775): "generations/redo/S011b_v3.mp4",
    (775, 852): "generations/ext/S13/S13_MERGED_v3.mp4",
    (852, 933): "generations/ext/S14/S14_MERGED_v3.mp4",
    (933, 1008): "generations/ext/S15/S15_MERGED_v3.mp4",
    (1008, 1044): "generations/ext/S16/S16_MERGED_v3.mp4",
    (1082, 1121): "generations/ext/S18/swap/apt_S18_lead_seed202632018_00001_.mp4",
    (1121, 1140): "generations/ext/S19/swap/apt_S19_lead_seed202632019_00001_.mp4",
    (1140, 1155): "generations/ext/S20/swap/apt_S20_lead_seed202632020_00001_.mp4",
    (1155, 1164): "generations/ext/S21/swap/apt_S21_lead_seed202632021_00001_.mp4",
    (1164, 1239): "generations/ext/S22/swap/apt_S22_lead_seed202632022_00001_.mp4",
    (1239, 1278): "generations/ext/S23/swap/apt_S23_lead_seed202632023_00001_.mp4",
    (1278, 1319): "generations/ext/S24/swap/apt_S24_lead_seed202632024_00001_.mp4",
    (1319, 1331): "generations/ext/S25/swap/apt_S25_lead_seed202632025_00001_.mp4",
    (1331, 1367): "generations/ext/S26/swap/apt_S26_lead_seed202632026_00001_.mp4",
    (1367, 1388): "generations/ext/S27/swap/apt_S27_lead_seed202632027_00001_.mp4",
    (1388, 1403): "generations/ext/S28/swap/apt_S28_lead_seed202632028_00001_.mp4",
    (1403, 1424): "generations/ext/S29/swap/apt_S29_lead_seed202632029_00001_.mp4",
    (1424, 1449): "generations/ext/S30/swap/apt_S30_lead_seed202632030_00001_.mp4",
    (1449, 1518): "generations/ext/S31/swap/apt_S31_lead_seed202632031_00001_.mp4",
    (1518, 1539): "generations/ext/S32/swap/apt_S32_lead_seed202632032_00001_.mp4",
    (1614, 1630): "generations/ext/S34/swap/apt_S34_lead_seed202632034_00001_.mp4",
    (1630, 1683): "generations/ext/S35/swap/apt_S35_lead_seed202632035_00001_.mp4",
    (1683, 1697): "generations/ext/S36/swap/apt_S36_lead_seed202632036_00001_.mp4",
    (1697, 1779): "generations/ext/S37/swap/apt_S37_lead_seed202632037_00001_.mp4",
}


def probe_frames(path: str) -> int:
    out = subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
                                   "-show_entries", "stream=nb_read_frames", "-of", "json", path])
    return int(json.loads(out)["streams"][0]["nb_read_frames"])


def frames_of(path: str) -> list[Path]:
    """Decode a clip to PNGs at the preview size."""
    tmp = Path(tempfile.mkdtemp(prefix="stitch_"))
    subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={W}:{H},setsar=1",
                    str(tmp / "f%05d.png")], check=True)
    return sorted(tmp.glob("f*.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--facefix", action="store_true", help="use the FaceFusion takes where available")
    ap.add_argument("--redo", action="store_true", help="use the plan-A redo takes where available")
    ap.add_argument("--redo-json", default="", help="JSON file: [[start, end, path|null], ...] overrides REDO")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="stitchwork_"))
    written = 0
    report = []
    global REDO
    if a.redo_json:
        import json as _json
        REDO = {(int(s0), int(s1)): (p or None) for s0, s1, p in _json.load(open(a.redo_json))}
    for start, end, rel in SEGMENTS:
        if a.redo and (start, end) in REDO:
            rel = REDO[(start, end)]
        if a.facefix and (start, end) in FACEFIX:
            rel = FACEFIX[(start, end)]
        need = end - start
        if rel is None:
            tmp = work / f"src_{start}.mp4"
            subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(SRC),
                            "-vf", f"select='between(n\\,{start}\\,{end - 1})',setpts=N/{FPS}/TB,scale={W}:{H},setsar=1",
                            "-fps_mode", "cfr", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "16", "-pix_fmt", "yuv420p", str(tmp)], check=True)
            frames = frames_of(str(tmp))
        else:
            path = PROJ / rel
            if not path.is_file():
                raise SystemExit(f"missing take: {path}")
            frames = frames_of(str(path))

        # pad by repeating the last frame, or trim, so every segment matches the official cut
        if len(frames) < need:
            pad = need - len(frames)
            src = frames[-1]
            im = Image.open(src)
            for k in range(pad):
                dst = work / f"pad_{start}_{k}.png"
                im.save(dst)
                frames.append(dst)
        frames = frames[:need]
        for i, f in enumerate(frames):
            dst = work / f"out_{written:05d}.png"
            Path(f).replace(dst)
            written += 1
        report.append((start, end, need, rel or "official"))

    out_path = PROJ / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    silent = work / "silent.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-framerate", str(FPS),
                    "-i", str(work / "out_%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                    "-pix_fmt", "yuv420p", str(silent)], check=True)

    first, last = SEGMENTS[0][0], SEGMENTS[-1][1]
    start_t, dur_t = first / FPS, (last - first) / FPS
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(silent), "-i", str(SRC),
                    "-filter_complex", f"[1:a]atrim=start={start_t:.6f}:duration={dur_t:.6f},asetpts=PTS-STARTPTS[a]",
                    "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(out_path)], check=True)

    print(f"segments: {len(report)}  frames written: {written}  ({written / FPS:.2f}s)")
    for s, e, n, rel in report:
        print(f"  [{s},{e}) {n:3d}f  {rel}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
