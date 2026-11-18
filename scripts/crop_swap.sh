#!/usr/bin/env bash
# crop_swap.sh — swap ONE figure inside a cropped window, then paste it back.
#
# For silhouette shots SAM ignores the bbox entirely and always locks the largest/rightmost
# figure, so a targeted pass is impossible on the full frame. Cropping the wanted figure into its
# own clip makes it the only subject; paste back through the crop's own mask afterwards.
#
#   bash crop_swap.sh S53 200,350,520,620 F 2027540
set +e
A=/home/leomok/projects/video-regen/apt-family
S=/home/leomok/projects/video-regen/scripts
PY=/home/leomok/miniconda3/bin/python3
FF=/home/leomok/bin/ffmpeg
PACK=$A/generations/shot_S003_lips_CU/_pack_redo_v1_base.json
LOG=$A/generations/crop_swap.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

ID=$1; CROP=$2; ROLE=$3; SEED=$4
IFS=',' read -r CX CY CW CH <<< "$CROP"
CLIP=$(ls $A/generations/ext/$ID/Video1_${ID}_*.mp4 | head -1)
N=$(/home/leomok/bin/ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$CLIP")
DUR=$($PY -c "print(f'{$N/24:.6f}')")
say "$ID crop=$CROP role=$ROLE frames=$N"

# upscale the crop ~1.6x so the swap has pixels to work with
SW=$($PY -c "print(int($CW*1.6)//2*2)")
SH=$($PY -c "print(int($CH*1.6)//2*2)")
$FF -v error -y -i "$CLIP" -vf "crop=$CW:$CH:$CX:$CY,scale=$SW:$SH" -r 24 \
  -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p -an $A/generations/ext/$ID/crop_$ROLE.mp4

if [ "$ROLE" = "F" ]; then
  REF=$A/plates/session02/GROK_CARRIE_v1.png; PROMPT=$A/prompts/CLEAN_female.txt; TXT="the woman"
else
  REF=$A/plates/session02/GROK_LEO_v4.png; PROMPT=$A/prompts/CLEAN_male.txt; TXT="the man"
fi
say "swap cropped figure"
$PY "$S/apt_run_swap.py" --pack "$PACK" --source $A/generations/ext/$ID/crop_$ROLE.mp4 \
  --duration "$DUR" --ref "$REF" --anchor "$REF" --prompt-file "$PROMPT" \
  --tag "${ID}crop${ROLE}" --seed "$SEED" --window 56 --history 18 --expand 30 \
  --sam-text "$TXT" --bbox 0,0,$SW,$SH --no-anchor --out $A/generations/ext/$ID/crop_take_$ROLE --go 2>&1 | tail -1 | tee -a "$LOG"
say "mask"
$PY "$S/export_masks.py" --pack "$PACK" --tag "${ID}crop${ROLE}mk" --source $A/generations/ext/$ID/crop_$ROLE.mp4 \
  --duration "$DUR" --sam-text "$TXT" --bbox 0,0,$SW,$SH --expand 30 \
  --out $A/generations/ext/$ID/crop_mask_$ROLE 2>&1 | tail -1 | tee -a "$LOG"
say "BLOCK DONE"
