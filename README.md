# APT MV — 素人重製（MiniMax H3 / Openjutsu character swap）

將官方 MV 逐個鏡頭換成屋企人（Leo + Carrie），保留原片剪接、機位、表演同音軌。
呢個 repo 係**可複用嘅工作流程 + 踩過嘅坑**，唔包含任何媒體素材。

> 用咗：ComfyUI 0.41.2（video fork，`:8190`）＋ MiniMax H3 bf16 ＋ SAM 3.1
> ＋ `comfyui-h3-oneclick-swap` / `h3_character_swap` ＋ H3-Image（`:8191`）＋ Grok（cursor-agent）

---

## 1. 核心概念

一段片有 N 個 shot。每個 shot 都係同一條流水線：

```
plate（人物參考圖） → mask gate（SAM 確認揀中人） → swap（H3 生成）
→ take gate（數字驗證）→ 必要時合成上原片背景 → stitch 成 master
```

**鐵律**

1. **Mask gate 未 PASS 唔准開 DiT。** SAM 揀錯人 = 白燒 20 分鐘 GPU。
2. **每個 take 都要有數字**（per-frame MAD），mean < 5 等於乜都冇做。
3. **角色映射以髮色／衣物為準，唔用性別分類器**（彩色燈光下會跳）。
4. **服裝由我哋決定，唔跟原片**（寫入 plate 同 prompt）。
5. **大特寫（眼／嘴）唔可以用 swap**，要用合成或重新生成。

---

## 2. 環境

| 服務 | 用途 | systemd unit |
|---|---|---|
| `:8190` | ComfyUI 0.41.2 — H3 影片 swap / SAM3 | `comfy-8190.service` |
| `:8191` | MiniMax-H3-Image — 生成 plate | `comfy-8191-image.service` |

兩種服務都係 **systemd user unit**。長時間批次一定要：

```bash
systemd-run --user --unit=<name> --collect bash -lc 'bash script.sh'
```

（agent background task 會收 SIGTERM，實測三次都死在中途。）

**關鍵參數**（官方配方）：8 steps、CFG 1、`er_sde`、window 124、history 18、
mask expand 64（我哋大部份 shot 用 **expand 0**，見第 6 節）。

`window_frames` 只接受 **17k+5** 網格：22 / 39 / 56 / 73 / 90 / 107 / 124 / 141 / 158…

---

## 3. 每日流程（per shot）

```bash
# 1) 檢查 SAM 揀邊個（唔燒 DiT，1–2 分鐘）
python3 scripts/apt_checkmask.py --pack pack.json --source clip.mp4 \
        --duration 2.0 --sam-text "the woman" --expand 0 \
        --plate plates/CARRIE.png --out out/checkmask

# 2) swap
python3 scripts/apt_run_swap.py --pack pack.json --source clip.mp4 --duration 2.0 \
        --ref plates/CARRIE.png --anchor plates/CARRIE.png \
        --prompt-file prompts/CLEAN_female.txt --tag S042 \
        --seed 2042001 --window 56 --history 18 --expand 0 \
        --sam-text "the woman" --no-anchor --out out/take_F --go

# 3) 雙人鏡頭 = 兩次單人 pass + 用 mask 合併
python3 scripts/export_masks.py --pack pack.json --tag mk --source clip.mp4 \
        --duration 2.0 --sam-text "the woman" --expand 0 --out out/mask_F
python3 scripts/merge_stream.py --source clip.mp4 \
        --take-a out/take_F/x.mp4 --masks-a out/mask_F \
        --take-b out/take_M/y.mp4 --masks-b out/mask_M \
        --out out/MERGED.mp4 --feather 10
```

---

## 4. Plate 生成（最影響成果嘅一步）

```bash
# H3-Image（本地）：由身份照生成造型 plate
python3 scripts/h3_image_plate.py --prompt "<wardrobe + pose>" \
        --ref plates/IDENTITY.png --out plates/S042.png --size 1920x1080 --seed 2042001
```

* **Identity 一定要用 Grok 生成**（`cursor-agent` → GenerateImage，以真人相做 reference）：
  ArcFace 相似度 **0.84–0.96**；H3-Image 自己生只有 **0.50–0.62**。
* **道具／配件要用 chroma key**：叫 Grok 生「綠幕」版本，key 綠色 → alpha 準。
  用「淺色背景」plate 去 key 會連白恤衫一齊刪走。
* **plate 要 match shot 嘅內容比例**：大特寫要 composition-matched plate；
  全身／中景用 portrait；剪影用 cut-out head。

---

## 5. 角色映射（唔可以用猜）

* 原片 **金色長髮 = Rosé（女）**、**黑 cap／黑超 = Bruno（男）**。
* 我哋嘅 cast：**Carrie = 黑色泡泡袖連身裙 + 細圓金屬框眼鏡**、
  **Leo = 深藍西裝 + 白恤衫 + 透明眼鏡，無帽無黑超**。
* **同一 shot 可以前後換人**（先女後男）。開頭／中間／尾各抽一格睇清楚。

---

## 6. 踩過嘅坑（重點）

| 坑 | 症狀 | 解法 |
|---|---|---|
| `expand 40/64` | 背景污塊、鼓棍變啡色 smear | **expand 0** 貼身 mask |
| prompt 大寫標題 | `ENDCAST_V1 REPLACE-PERSON` 被畫成**字幕**燒入畫面 | 用簡短細楷 prompt（`CLEAN_*.txt`） |
| text / bbox 揀錯人 | 永遠揀最大人物；兩個唔同 bbox 出**同一個 mask** | SAM3 **point prompt**：`--pos "x:y;x2:y2" --neg "x:y"` |
| mask 100% 重疊 | 雙人 pass 合併出**重影** | 先比對 `inter/B.sum()`；重疊就改單 pass |
| 剪影 shot | SAM 無視 bbox | `crop_swap.sh` + `crop_paste2.py`（裁細→換→貼回） |
| take 中間有窿 | 前後換咗、中間幾格仲係原片 | 逐格 MAD 對原片，搵 <12 嘅連續段，補跑 + splice |
| 大頭 grid 尾段 | 最後 6–8 格彈返原片 | 偵測唔到就**沿用上一格完成畫面** |
| 眼特寫 | 叫模型「換人」→ 佢畫個半身像落眼位 | 生成 Carrie 眼 macro + 重建原片筆觸 |
| 嘴特寫 | 靜態唔動＝假；真人嘴貼上＝lip-sync 錯 | **passthrough**，唔好做 |
| 音質越嚟越差 | 每次改片都 re-encode AAC | 最後一步 `-c:a copy` 用**原片音軌** |
| 裙款接縫唔同 | 相鄰 take 各自「發明」服裝 | 用**相鄰 shot 已收貨嘅畫面**做 plate |

---

## 7. 檔案結構

```
scripts/            # 全部工具（見下）
prompts/            # CLEAN_female.txt / CLEAN_male.txt（簡短、無大寫標題）
plates/             # 人物參考圖（唔入 repo）
generations/ext/    # 每個 shot 嘅 take / mask / merged
generations/preview/# master
NEXT_STEPS.md       # 逐日進度 + 失敗路線記錄
```

主要腳本：

| 腳本 | 用途 |
|---|---|
| `apt_checkmask.py` | mask gate（唔燒 DiT） |
| `apt_run_swap.py` | 1pp swap（支援 `--pos/--neg` point prompt） |
| `export_masks.py` / `merge_stream.py` | 逐格 mask / 兩 take 合併 |
| `crop_swap.sh` / `crop_paste2.py` | 剪影、多人、背景人物 |
| `grid_composite5.py` | 大頭 grid（animated donor + 逐格分類） |
| `eye_macro_rebuild.py` / `macro_rebuild.py` | 大特寫重建 |
| `range_swap.py` / `box_paste.py` | 局部替換入 master |
| `stitch_preview.py` / `build_redo_json.py` | 砌 master |
| `face_sim.py` | ArcFace 相似度（gate 0.75） |
| `housekeep.sh` | 清磁碟 |

---

## 8. 交付檢查清單

- [ ] `ffprobe` 格數 = 原片格數
- [ ] 音軌 = 原片 stream（`-c:a copy`，128 kbps）
- [ ] 逐個 user 報過嘅時間點抽格對比（用戶係用 playback timestamp 報 bug，
      要換算：`frame = t × 24`）
- [ ] 全片掃一次仲有冇原片角色（`scan_blonde.py`，注意粉紅／白色假陽性）
- [ ] 接縫位（每個 splice 前後 2 格）冇跳

---

MIT。媒體素材、人物相片、model weights 全部唔會入 repo。
