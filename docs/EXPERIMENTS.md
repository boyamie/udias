# Experiment Log & Paper-Value Tracker

Tracks measured/verified values for the Overleaf paper
("Dual-Spectrum Image Alignment Dataset for Maritime Ship Detection")
and maps each `\needsdata{...}` placeholder → its source → status.

Status legend: ✅ measured/verified · ⏳ pending experiment · 🙋 needs author input · ✏️ paper-text fix

Last updated: 2026-07-25 (school Windows PC, torch_env).

---

## Environment
- Machine: Windows 11, RTX 4060 (8 GB), driver 591.86
- Env: conda `torch_env` (Python 3.11.14), torch 2.5.1 + CUDA 12.1 (`cuda avail: True`)
- Added packages: pycocotools 2.0.11, ensemble-boxes 1.0.9
- Self-tests: `scripts/08` PASS, `scripts/11` PASS
- Config: all data paths → sibling `../data` (= `maritime/data`). Run all scripts from `code/`.

## ⚠️ ENCODING GOTCHA (must apply to every script run on this machine)
This is a Korean-locale Windows box (default `open()` codec = cp949). 13 scripts do
`yaml.safe_load(open(sys.argv[1]))` with NO encoding, so they crash reading the UTF-8
config (em-dash byte 0xe2): `UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2`.
FIX: run every script with **`PYTHONUTF8=1`** (Python UTF-8 mode → open() defaults to UTF-8).
  e.g. `PYTHONUTF8=1 python scripts/NN_....py config/default.yaml`
(Durable fix would be to add `encoding="utf-8"` to the 13 open() calls — offered, not yet done.)
Note: `PYTHONIOENCODING=utf-8` alone is NOT enough — it only affects stdout/stderr, not open().

## Source video inventory (measured from files, all 44)
IR = Hanwha QUANTUM RED, RGB = iPhone 14.

**IR (22 clips) — uniform:** 640×480, 30.1 fps, H.264/MP4. Duration 36.8–517.9 s, total 4918.5 s.

**RGB (22 clips) — varies:**
- Resolution: 1920×1080 ×5 (day-RGB-1..5), 3840×2160 ×17 (day-RGB-6..9 + all night)
- fps: 60 ×5, 30 ×11, and 24.0/25.3/25.6/27.1/27.1/28.8 ×6 (all night — iPhone variable frame rate in low light)
- Codec: HEVC ×21, H.264 ×1 (day-RGB-1). Duration 33.2–520.8 s, total 4848.1 s.

→ Justifies time-based (0.5 s) sampling over fixed frame stride: the two modalities and even
  RGB sessions among themselves run at different / variable fps.

## Extraction run
- Command: `python preprocessing/video_to_frames.py config/default.yaml --sec 0.5` (from `code/`)
- Sampling: elapsed-time grid, stride = round(fps × 0.5). IR→15, RGB→30 (60fps) / 15 (30fps) / 12–14 (night).
- Output: `../data/frames/{rgb,ir}/{Day,Night}_NN_00000.jpg`
- Status: ⏳ running (day done, night in progress). Final pair count N pending.

---

## `\needsdata` placeholder tracker

### §1 Table 1 (comparison)
| placeholder | value | status | source |
|---|---|---|---|
| Ours "Size" = N pairs | — | ⏳ | build_pairs after extraction (manifest count) |

### §3.1 Thermal Camera (IR)
| placeholder | value | status | source |
|---|---|---|---|
| IR width×height | 640×480 | ✅ | file probe |
| IR fps | ~30 (30.1) | ✅ | file probe |
| container/codec | MP4 / H.264 (8-bit) | ✅ | file probe |
| LWIR spectral band | "not publicly specified" | 🙋 | Hanwha does not document; author confirm |

### §3.2 Visible Camera (RGB)
| placeholder | value | status | source |
|---|---|---|---|
| RGB resolution | 1080p (5) + 4K/2160p (17) | ✅ | file probe (varies) |
| RGB fps | 60 / 30 / variable 24–28.8 (night) | ✅ | file probe (varies) |
| HDR / stabilization settings | — | 🙋 | iPhone capture setting; not in file header |

### §3.3 Acquisition Geometry & Sessions
| placeholder | value | status | source |
|---|---|---|---|
| platform / mounting / baseline | — | 🙋 | author |
| distance range to vessels | — | 🙋 | author |
| # daytime sessions | 9 day videos (confirm=sessions?) | 🙋 | author |
| # nighttime sessions | 13 night videos (confirm?) | 🙋 | author |
| location description | — | 🙋 | author |
| dates / season | — | 🙋 | author |
| scene types | open water / nearshore / harbor? | 🙋 | author (also per-video scene_type tags) |

### §3.4 Synchronization
| placeholder | value | status | source |
|---|---|---|---|
| sync method | no hardware trigger; method? | 🙋 | author |
| residual temporal offset bound | — | 🙋 | author + eyeball check on Day_04 pairs |
| **"fixed stride of ten frames"** | → **time-based 0.5 s sampling** | ✏️ | pipeline changed; text must be updated |

### §6 Ethics
| placeholder | value | status | source |
|---|---|---|---|
| vessel identifiability policy | blurred / retained | 🙋 | author policy decision |

### §4.3 / §5 Benchmark result tables (pages 4–6, 8)
| group | status | source |
|---|---|---|
| Dataset stats (counts, per-scene, ship-size dist.) | ⏳ | scripts/01,02,14 |
| Alignment metrics (IoU vs landmarks, inliers, reproj, success rate) | ⏳ | scripts/01,10 |
| IAA (kappa / F1) | ⏳ | scripts/13 |
| Detection mAP per baseline (RGB / IR / early / late / middle) | ⏳ | scripts/04,05,06,07 |
| Robustness (drop_rgb, drop_ir, ir_contrast40, ir_noise03) | ⏳ | scripts/09 |
(Exact placeholder text to be captured from the PDF when filling.)

---

## Paper-text issues found (not just number fills)
1. ✅ DONE §3.4 "fixed stride of ten frames" → rewritten to "fixed time interval of 0.5 s ...".
   Note: the top-of-file comment "Frame stride confirmed 10 from the extraction code" AND the
   §3.4 source comments (lines ~170-172, "FRAME_INTERVAL=10") are still stale — remove/update.
2. ✅ DONE §3.2 RGB resolution/fps now reports the measured distribution, not a single value.

## Overleaf edits applied (2026-07-25, via Chrome find-replace, verified in compiled PDF)
- §3.1 IR resolution → `$640\times480$`
- §3.1 IR fps → `approximately $30$~fps`
- §3.1 IR codec → `an MP4/H.264 stream`
- §3.2 RGB resolution → `$1920\times1080$ or $3840\times2160$ (17 of the 22 clips at the latter)`
- §3.2 RGB fps → `a frame rate of $25$--$60$~fps (predominantly $30$ or $60$~fps, ... variable $24$--$29$~fps)`
- §3.4 stride sentence → time-based 0.5 s wording

## CORRECTION (2026-07-26, home): the edits above were NOT actually persisted
The 2026-07-25 Overleaf "dual" project download (md5-identical to the local pre-edit
main.tex) showed NONE of the §3.1/§3.2/§3.4 edits above — they were lost / not saved.
Re-applied for real on 2026-07-26 to the local `main_revised2.tex` AND re-uploaded to
Overleaf (recompiled clean, Errors 0). Now genuinely in both:
- §3.1 IR res/fps/codec = 640×480 / ~30 fps / MP4-H.264   ✅ (LWIR band + §3.2 HDR still 🙋)
- §3.2 RGB res/fps = 1080p/4K (17@4K) / 24–60 fps          ✅
- §3.4 fixed frame-stride → fixed 0.5 s time interval       ✅
- stale stride comments (header + §3.4) refreshed           ✅
Remaining red placeholders in §3.1-3.4 are all 🙋 author-only (see tracker above).
