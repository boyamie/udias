# udias 데이터 파이프라인 런북

원본 영상 → 데이터셋 → 벤치마크 → 논문 수치까지의 실행 순서. 논문
(`Dual-Spectrum Image Alignment Dataset`)의 `\needsdata` 채우기가 최종 목표다.

> **주의**: `scripts/01`–`03`의 docstring 예시는 `configs/default.yaml`로 돼 있으나
> 실제 설정 파일은 **`config/default.yaml`**(단수)이다. 모든 명령에 경로를 명시할 것.

범례: `[auto]` 자동 · `[semi]` 자동+사람 검수 · `[manual]` 사람 주석/입력

---

## 0. 사전 준비

```bash
conda activate cs310            # torch 있는 환경 (base/cs 에는 torch 없음)
pip install -U ultralytics      # 04/05 용. 4채널 학습엔 최신 필수. torch/cv2/matplotlib 은 이미 있음
```

- 원본 영상을 `config/default.yaml` 의 `raw_video_dir`(기본 `data/raw_videos`, `day`/`night` 하위)에 배치
- 프레임 추출: `python preprocessing/video_to_frames.py` (stride=10) → `data/frames/rgb`, `data/frames/ir`
- 검증만 먼저: `python scripts/11_selftest_extensions.py` 와 `python scripts/08_selftest_middle_fusion.py` 가 PASS 하는지 확인

---

## A. 데이터셋 구축 → 분할·규모 수치

| 단계 | 명령 | 산출 / 채우는 `\needsdata` |
|---|---|---|
| 01 `[auto]` | `python scripts/01_build_manifest_and_align.py config/default.yaml` | 페어 매니페스트 + SIFT/RANSAC 정렬(H 캐싱). §4.2 정렬 성공률·재투영 오차 1차 |
| 02 `[auto]` | `python scripts/02_make_splits.py config/default.yaml` | 영상단위 층화 분할 + 누수검사. §4.4 총 페어·영상 수·train/val/test·Table 1 "Ours N" |
| 03 `[semi]` | `python scripts/03_autolabel.py config/default.yaml` → **Label Studio 검수** | 초벌 라벨 + 가시성 플래그. 검수 후 §4.3 검수 완료율·주석자 수 |

---

## B. 학습

| 단계 | 명령 | 비고 |
|---|---|---|
| 04 `[auto]` | `python scripts/04_train_baselines.py config/default.yaml` | rgb / ir / early_stack4(주) / early_pixel / early_stack4_noalign × seeds. ultralytics 필요 |
| 06 `[auto]` | `python scripts/06_train_middle_fusion.py config/default.yaml` | middle fusion. torch 만 (ultralytics 불필요) |

---

## C. 평가 → 벤치마크 표·강건성

| 단계 | 명령 | 산출 |
|---|---|---|
| 05 `[auto]` | `python scripts/05_eval_benchmark.py config/default.yaml` | `runs/benchmark.json` (단일 / early 2종 / late) |
| 07 `[auto]` | `python scripts/07_eval_middle_fusion.py config/default.yaml` | 위 json 에 middle_fusion 행 합류 |
| 09 `[auto]` | `python scripts/09_eval_robustness.py config/default.yaml` | `runs/robustness.json` (§5.4: clean/drop_rgb/drop_ir/ir_contrast40/ir_noise03) |

---

## D. 데이터셋 통계·정렬·주석 품질 → §4.2/4.3 수치 + M14 그림

병목은 아래 **수동 주석 3종**(native IR · 이중주석 · landmark)이다. 서로 독립이라 병렬 진행 가능하고,
각 주석이 끝나는 대로 해당 스크립트만 재실행하면 된다.

| 단계 | 명령 / 작업 | 채우는 것 |
|---|---|---|
| 12 `[auto]` | `python scripts/12_select_ir_native_subset.py config/default.yaml` | native IR worklist → `splits/ir_native_worklist.txt` |
| ↳ `[manual]` | worklist 페어를 **IR 좌표계로 독립 주석** → `data/labels_ir_native/{pair_id}.txt` (확장 YOLO) | 지표(ii) 재료 |
| 13a `[auto]` | `python scripts/13_eval_iaa.py config/default.yaml` | 이중 주석 worklist → `splits/iaa_worklist.txt` |
| ↳ `[manual]` | 두 주석자가 동일 페어 표기 → `data/labels_iaa/annotator_a/`, `.../annotator_b/` | IAA 재료 |
| 13b `[auto]` | `python scripts/13_eval_iaa.py config/default.yaml` (재실행) | §4.3 IAA: box F1 · 평균 IoU · 가시성/uncertain κ → `runs/iaa.json` |
| ↳ `[manual]` | landmark 대응점 표기 → `data/landmarks/{pair_id}.json` (`{"rgb":[[x,y]...],"ir":[[x,y]...]}`) | 지표(i) 재료 |
| 10 `[auto]` | `python scripts/10_eval_alignment.py config/default.yaml` | §4.2: landmark 오차 · native warp IoU · 코퍼스 정렬 통계 → `runs/alignment_eval.json` |
| 14 `[auto]` | `python scripts/14_dataset_figures.py config/default.yaml` | M14 그림 → `runs/figures/fig-size-dist.pdf`, `fig-align-by-tod.pdf` |
| 15 `[semi]` | 원인 태그 `selection.json` 작성 후 `python scripts/15_qualitative_gallery.py config/default.yaml selection.json runs/benchmark_preds.json` | M14 정성 갤러리 → `runs/figures/fig-qualitative-gallery.pdf` |

`selection.json` 형식: `{"waves":["pair_id",...], "docks":[...], "glare":[...], "duplicate":[...], "crossover":[...]}`

---

## E. 논문 반영

1. `runs/*.json`(alignment_eval / benchmark / iaa / robustness)의 값으로 **자동 산출 `\needsdata`** 채우기
   (정렬 성공률·재투영 오차·분할 수·라벨 완료율·IAA·크기 분포·벤치마크 mAP).
2. `runs/figures/*.pdf` 3개를 Overleaf `figures/` 에 올리고 `main.tex` 의 **주석 처리된 figure 슬롯 3개 해제**.
3. 로컬 `main.tex` 갱신 → 클립보드-붙여넣기로 Overleaf 프로젝트에 덮어쓰기(Overwrite) → 재컴파일.

---

## 파이프라인으로 안 채워지는 수동 `\needsdata` (직접 입력)

- **§3.1–3.4**: IR 해상도·fps·코덱·스펙트럼 밴드; RGB 해상도·fps·HDR; 플랫폼·카메라 baseline·촬영 거리·
  세션 수·위치·날짜·장면유형; 동기화 방법·잔여 시간 오프셋.
- **back-matter**: funding 문구, **Zenodo DOI**(업로드 후 발급), supervision 저자(D.C./D.L. 중 확정), conflicts 확정.
- **§6 윤리**: 선박 식별정보(선체 표식/등록번호) 처리 정책(블러 / 근거와 함께 보존).

---

## 교차검증 체크 (제출 전)

- 정렬 성공률 × 총 페어 수 = aligned 페어 수 (매니페스트와 일치)
- §3.3 카메라 baseline 수치가 §4.2 "small baseline" 주장과 정합
- `grep -rn needsdata main.tex` → 0 건
- 그림 7개(기존) + M14 3개 모두 `figures/` 존재
