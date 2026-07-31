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

## Overnight sweep architecture (2026-07-28 evening, 12h unattended window)
Three OOM crashes diagnosed to ONE root cause: **16 GB system RAM exhaustion** (workers=8
prefetching 4K frames; cv2 CPU alloc fail → disguised CUDA "OOM with VRAM free"), plus a
second hard limit found: **C: had only 9 GB free** (full-res exports would need 100+ GB).
Fixes, in order:
1. `train.workers: 2` in config; scripts/04 passes it.
2. `fusion.export_max_side: 960` — export downscales training copies (labels are normalized
   → invariant; originals untouched; imgsz=640 unaffected). Disk ~19 GB total, decode 24× faster.
3. TIFF LZW compression for stack4; export made idempotent (skip existing files).
4. scripts/04 restructured: export all 5 datasets first, then **seed-OUTER loop**
   (seed0 across all baselines first → best Table-2 row coverage in a 12h window);
   completed runs auto-skipped via results.csv row count (restart-safe).
5. `_run_sweep.py`: **detached runner** (Start-Process, survives session restarts — earlier
   session restart killed an in-flight sweep), keep-awake via SetThreadExecutionState
   (no system settings changed), OOM batch downgrade 8→6→4.
6. `_watch_sweep.py`: read-only live viewer for VS Code terminal (safe to close anytime).
7. W&B: enabled (ultralytics settings), account `boyamie`, runs appear per-baseline.
Old 4K exports + smoke outputs deleted (~22 GB reclaimed → 30.9 GB free).

## Sweep crash post-mortem (2026-07-29, diagnosed REMOTELY from home via W&B output.log)
Run logs are fully readable from home: W&B team entity `boyamie-pusan-national-university`,
projects `udias-data-runs-<baseline>` (prefix changed to `udias-<baseline>` mid-sweep);
GraphQL `files { directUrl }` → fetch `output.log`. Timeline (KST): rgb_only 21:44→03:12
✅ mAP50 0.509 · ir_only 03:12→08:42 ✅ mAP50 0.021 · early_stack4 08:42 ❌ 39 s ·
rgb_only RERUN 08:48→14:20 ✅ 0.480 · ir_only 14:20 ❌ 15:44 · ir_only retry 16:04→ 🟢.
Two DISTINCT crash causes:
1. **early_stack4 (run s6bwohit)**: `torch.OutOfMemoryError: CUDA out of memory. Tried to
   allocate 14.00 MiB. GPU 0 total 8 GiB, 2.89 GiB free` — OOM-with-VRAM-free at training
   start, right after dataset scan. Everything before it was healthy: 4-ch model built
   (`Conv [4,64,3,2]`), 643/649 pretrained weights transferred, AMP checks passed, and the
   aligned-only dataset scanned **3,812 train images ≈ 58.4% of 6,554** — the XoFTR
   alignment gate is filtering exactly as designed. Fix: set
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and start stack4 at batch 4–6.
2. **ir_only 2nd attempt (run ktiphfk0)**: `OSError: [Errno 28] No space left on device`
   while saving last.pt — **C: filled up AGAIN** during the sweep (checkpoints+plots
   ~80 MB/run, local `wandb/` cache). The in-flight retry risks the same death; disk
   cleanup is the first job on returning to this machine.
3. **Completion-check bug (why rgb_only ran twice, wasting 5.5 h)**: the auto-skip in
   scripts/04 counts results.csv rows against `train.epochs` (100); a run that early-stops
   writes fewer rows and is judged incomplete after a runner restart, so a finished
   baseline re-trains from scratch. Fix: also accept EarlyStopping (e.g. best.pt present
   AND trainer logged early stop) as complete.

## noalign-skip diagnosis + scripts/04 hardening (2026-07-31, from home)
**Why early_stack4_noalign never ran in the sweep**: the completion check found
`data/runs/early_stack4_noalign/seed{0,1,2}/results.csv` with >=100 rows — leftovers of a
pre-W&B (<=07-27) training on the OLD exports/alignment. Those numbers are NOT comparable
to the current sweep (960px exports, XoFTR-realigned manifest) and must be re-run.
scripts/04 changes (pushed):
1. `--force name1,name2|all` — retrain named baselines ignoring completion.
2. Preflight table printed at start: every (name, seed) with its skip reason
   (DONE marker / results.csv rows+mtime / missing) — no more silent skips.
3. `DONE` marker written after each successful model.train() return (covers
   early-stopped runs) — fixes the rgb_only double-run bug; legacy row-count
   check kept for marker-less completed runs.
**RUNBOOK (school PC, in order):**
```
git pull
python scripts/04_train_baselines.py config/default.yaml --force early_stack4_noalign   # ~3.4h x3 seeds
python scripts/06_train_middle_fusion.py config/default.yaml                            # middle fusion x3 seeds
python scripts/05_eval_benchmark.py config/default.yaml                                 # late fusion needs NO training - eval only (uses rgb/ir best.pt)
```
Before step 1, sanity-check the stale runs the preflight prints (mtime should be <=07-27).

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

### §3.3 Acquisition Geometry & Sessions (author answers 2026-07-26)
| placeholder | value | status | source |
|---|---|---|---|
| platform / mounting / baseline | **hand-held aboard a vessel under way**, two devices held side by side, baseline ~10–20 cm | ✅ | author |
| distance range to vessels | **not instrumented** — paper now declines to quote a range and points to the ship-size distribution instead | ✅ | author |
| # daytime / nighttime | **9 day / 13 night source videos** (written as "source videos", not "sessions") | ✅ | file inventory |
| location description | GPS coordinates — author will disclose to that granularity | 🙋 | author (needs the actual coordinates) |
| dates / season | **single capture day 2023-02-17 (winter); day 14:17–15:32 KST, night 21:03–21:49 KST** (from QuickTime `mvhd` creation_time, all 44 files; file mtimes were useless copy-dates 2026-05-13) | ✅ | container metadata probe (school PC, 2026-07-28) |
| scene types | **open water / nearshore / bridge** — "bridge" added as a new `scene_type` category (tall foreground structure ⇒ worst-case parallax, ties to §4.2) | ✅ | author |

### §3.4 Synchronization (author answers 2026-07-26)
| placeholder | value | status | source |
|---|---|---|---|
| sync method | **manual simultaneous recording start** — no hardware trigger, no common clock, no post-hoc offset estimation | ✅ | author |
| residual temporal offset bound | **~1 s, i.e. LARGER than the 0.5 s sampling interval** — stated plainly as the least-controlled part of acquisition | ⚠️ | author estimate; still needs an empirical tightening on a shared visual event |

> ⚠️ **Biggest reviewer risk in the paper.** Hand-held + ship-borne + ~1 s offset means the *background* also shifts between the two frames of a pair, not just moving vessels. §3.4, §4.2 and §6 were rewritten on 2026-07-26 to say this explicitly rather than imply a static-background model. Highest-value fix for any future capture: hardware trigger / clapper-style shared visual event / audio alignment.
| **"fixed stride of ten frames"** | → **time-based 0.5 s sampling** | ✏️ | pipeline changed; text must be updated |

### §6 Ethics + back-matter (author answers 2026-07-26)
| placeholder | value | status | source |
|---|---|---|---|
| vessel identifiability policy | **retained, not blurred** — justified in-text (commercial vessels in public waterways, not identifiable persons; hull markings are part of the visible-band signal a detector must handle) | ✅ | author |
| Funding | **"This research received no external funding."** | ✅ | author |
| Supervision (Author Contributions) | **D.L. (Dohoon Lee)** | ✅ | author |
| Conflicts of Interest | none declared + manufacturer-non-involvement statement; still 🙋 confirm no author has a financial relationship with Hanwha | ⚠️ | author to confirm |
| §3.1 LWIR spectral band | **"not publicly specified by the manufacturer"** (honest fallback; no inference from imagery) | ✅ | author/vendor |
| §3.2 HDR / stabilization | — | 🙋 | author (iPhone capture setting, not in file header) |
| Zenodo DOI | — | ⏳ | mint after upload |

## 🔴 CRITICAL FINDING (2026-07-25/28, school PC): automated alignment fails on this data
`scripts/01` completed over all N=9660 pairs: **aligned 2 / 9660 (0.02%)**; day 0.04%, night 0.00%.
The 2 successes are accurate (reproj 0.95 px, median inliers 26) — the method isn't buggy,
cross-modal SIFT just can't find ≥15 inliers between wide iPhone RGB and zoomed 640×480 IR.
Diagnostic sweep (6 representative pairs; downscale-SIFT / gradient-map-SIFT / AKAZE):
best variant peaks at ~6 inliers (threshold 15) → NOT a tuning problem.
**Home-session context makes it harder:** cameras were HAND-HELD on a moving vessel with ~1 s
sync offset → homography varies per frame (per-session fixed H is invalid), and even a perfect
per-frame H can't fix temporal displacement. Options (user decision pending, question was dismissed):
(a) learned cross-modal matcher (LoFTR/SuperGlue-class) per frame, (b) reframe paper: release raw
pairs + small manually-aligned landmark GT subset; report auto-align failure rate as a finding.
Note: paper §3.4/§4.2/§6 were already rewritten at home (2026-07-26) to state the hand-held /
parallax / offset reality — consistent with (b), but (a) can still be attempted on top.
Unblocked regardless: rgb_only baseline, autolabel, dataset stats. Blocked: ir_only (needs warped
labels or native IR labels), early/late/middle fusion, alignment-quality metrics.

### → RESOLVED (2026-07-28): XoFTR learned matcher works
User picked option (a). Tested LoFTR-outdoor (kornia): FAILED (2–9 matches, garbage warps —
visible-visible training can't bridge the modality gap). Tested **XoFTR** (CVPR24W, visible↔thermal
-trained, github.com/OnderT/XoFTR + official weights): **SUCCESS** on all 6 diagnostic pairs —
113–464 matches, RANSAC inliers 11–53 vs SIFT's 0–4, reproj 1.9–2.8 px at original resolution,
warp overlays visually verified correct (bridge/skyline/horizon/vessels all line up).
Integration (committed to repo):
- `third_party/XoFTR/` vendored (git dir stripped); weights at `weights/xoftr/*.ckpt` (gitignored)
- `udias/data/align.py`: `align.method: sift|xoftr` dispatch, lazy singleton, same min-inlier gates
- `scripts/01 --realign`: updates ONLY alignment fields on the existing manifest (preserves
  label_path/split written by 02/03 — do NOT full-rebuild after autolabel)
- `config/default.yaml`: `align.method: xoftr`, repo/ckpt paths, thresholds
- deps added to torch_env: kornia 0.8.3 (LoFTR test), einops/yacs/loguru (XoFTR); gdown (weights DL)
Full-corpus realign queued behind the autolabel→smoke chain (GPU contention). Borderline pairs
(open-water 14, night 11 inliers vs threshold 15): keep 15 for the first pass; consider 840-res
weights or lower coarse_thr only with visual re-verification.

### §4.3 / §5 Benchmark result tables (pages 4–6, 8)
| group | status | source |
|---|---|---|
| Dataset stats (counts, per-scene, ship-size dist.) | ⏳ | scripts/01,02,14 — N=9660 pairs measured |
| Alignment metrics (success rate, inliers, reproj) | ✅ | **XoFTR full realign (2026-07-28): 58.4% (5,645/9,660); day 72.9% / night 42.9%; mean reproj 2.64 px; median inliers 44.** Filled into §4.2 (4 placeholders), verified in compiled PDF. Landmark-based metric (i) still needs manual correspondence subset. |
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

## Splits result (2026-07-28, after two-stage near-dup fix): PASS
train 6554 fr / 15 videos (day 4015, night 2539) · val 1486 / 3 (246, 1240) · test 1620 / 4
(753, 867) · total 9660 / 22. Video-level leakage ✓, near-dup: 6 pHash candidates → 0 confirmed
by NCC ✓. ⚠️ scene_type is 'unknown' for ALL frames → stratification currently degenerates to
time_of_day only. 🙋 Author must tag per-video scene types (open_water / nearshore / bridge)
— then re-run 02 (deterministic, same seed) for true scene-stratified splits + per-scene eval.

## Splits leakage check: false-positive fix (2026-07-28, school PC)
First `scripts/02` run FAILED: 6 "near-duplicate" train↔holdout pairs (Night_10 ↔ Night_02,
pHash≤6). Visual inspection (montage sent to author): all 6 are DIFFERENT scenes — dark
night frames have so little entropy that pHash collides. Measured NCC: false positives
0.886–0.900 vs same-video adjacent frames 0.942.
FIX: `udias/data/splits.py check_near_duplicates` is now two-stage — pHash candidates are
confirmed by zero-mean NCC (≥0.92, the measured separation boundary). Candidate and confirmed
counts are both logged. Uniform/unreadable frames are conservatively treated as duplicates.
→ Worth one sentence in §4.4 (leakage-controlled splits): "perceptual-hash candidates are
verified by pixel-level correlation to remove low-entropy night-frame hash collisions."

## Overleaf edits round 2 (2026-07-28, school PC, via Chrome find-replace — verified in compiled PDF)
15 edits, all rendering confirmed on pages 3–4 (citations [13]/[14] resolve):
- §3.3 filled: hand-held aboard vessel (10–20 cm baseline) · distance not instrumented (ship-size
  distribution as proxy) · 9 day / 13 night source-video pairs · single winter day 17 Feb 2023
  (day 14:17–15:32, night 21:03–21:49 KST, container timestamps) · scene types incl. bridge
- §3.4 filled: manual simultaneous start · residual offset ~1 s (> 0.5 s sampling), stated as
  least-controlled aspect · stale FRAME_INTERVAL=10 comments replaced
- Table 1: Ours = 9,660 pairs
- §4.2 REWRITTEN: SIFT fails 2/9,660 (0.02%) + diagnostics (AKAZE/gradient/LoFTR fail) reported
  as a finding → released reference homographies via XoFTR + RANSAC; SIFT kept as documented baseline
- Fig. 1 caption: SIFT+RANSAC → learned cross-modal matching (XoFTR) with RANSAC
- §4.4: two-stage near-dup check (pHash → NCC ≥ 0.92) + split counts 6,554/1,486/1,620 (15/3/4)
- Bib: added sun2021loftr (CVPR'21) + tuzcuoglu2024xoftr (CVPRW'24)
Remaining red: alignment success rates (realign pending), review/IAA/benchmark numbers, and
author-only: LWIR band wording, iPhone HDR setting, GPS coordinates.

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
