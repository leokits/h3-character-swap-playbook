#!/usr/bin/env bash
# housekeep.sh — free disk by deleting superseded renders and agent temp files.
#
# Keeps exactly the standing master (PREVIEW_FULL_v51.mp4) and the delivered mobile cut, plus the
# newest 3 days of docs. Everything else in preview/ is a superseded iteration whose inputs still
# exist under generations/ext/, so nothing is lost that cannot be rebuilt.
set -u
P=/home/leomok/projects/video-regen/apt-family/generations/preview
KEEP="PREVIEW_FULL_v51.mp4 APT_REMAKE_v51_20261118.mp4 APT_REMAKE_v51_20261118.jpg"

echo "== before =="
df -h /home | tail -1
du -sh "$P" 2>/dev/null

for f in "$P"/*; do
  b=$(basename "$f")
  keep=0
  for k in $KEEP; do [ "$b" = "$k" ] && keep=1; done
  [ "$keep" = "1" ] && continue
  rm -f "$f"
done

echo "== after previews =="
du -sh "$P" 2>/dev/null
ls -1 "$P" | head

# agent scratch: my own temp renders (leave other projects' and system dirs alone)
find /tmp -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.png' -o -name '*.json' -o -name '*.txt' \) \
     -user leomok -mmin +90 -delete 2>/dev/null

# superseded full-length intermediates from this project
rm -f /tmp/v4*.mp4 /tmp/v5*.mp4 /tmp/s3_*.mp4 /tmp/both_*.mp4 /tmp/sty_*.mp4 /tmp/star*.mp4 \
      /tmp/head_*.mp4 /tmp/conc*_all*.txt /tmp/_*.mp4 2>/dev/null

# old build scratch dirs created by earlier overnight runs of these two projects
rm -rf /tmp/angie_shot001_v1_build /tmp/rebhold_7_12_build /tmp/angie_003c_rebuild \
       /tmp/v1_5xforfj8 /tmp/sheetframes 2>/dev/null

echo "== final =="
df -h /home | tail -1
du -sh /tmp 2>/dev/null
