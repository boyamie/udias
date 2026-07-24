"""② : 정렬 평가 — 지표 (i) landmark error, (ii) native warp IoU, 코퍼스 통계.

usage: python scripts/10_eval_alignment.py config/default.yaml

입력 규약 (사람 주석):
  {landmarks_dir}/{pair_id}.json          — {"rgb": [[x,y],...], "ir": [[x,y],...]}
                                            (대응 순서 동일; CVAT 등에서 export)
  {ir_native_labels_dir}/{pair_id}.txt    — IR '좌표계'에서 독립 표기한 plain YOLO 5열
                                            (RGB 라벨을 투영한 것 금지 — 순환)

동작:
  1. ir_native_labels_dir 를 스캔해 매니페스트의 ir_label_path 필드를 갱신·저장
  2. 지표 (i): landmark error 요약 (n, mean/median px)
  3. 지표 (ii): native warp IoU 요약 (greedy 매칭)
  4. 코퍼스 정렬 통계 (성공률 전체/주야, 평균 재투영 오차)
→ runs/alignment_eval.json  (논문 §4.2 의 \\needsdata 실측값 소스)
"""
import sys, yaml, json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest, save_manifest
from udias.data.align import imread_unicode
from udias.eval.align_metrics import (landmark_error, native_warp_iou_report,
                                      native_subset_coverage, alignment_report)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"))
P = cfg["paths"]
records = load_manifest(P["manifest"])

# ① native IR 주석 파일 스캔 → 매니페스트 갱신 (멱등)
native_dir = Path(P.get("ir_native_labels_dir", "data/labels_ir_native"))
n_native = 0
for rec in records:
    f = native_dir / f"{rec.pair_id}.txt"
    rec.ir_label_path = str(f) if f.exists() else ""
    n_native += bool(rec.ir_label_path)
save_manifest(records, P["manifest"])
print(f"[native] IR 원생 주석 페어: {n_native}개 → 매니페스트 갱신")

# ①' worklist 대비 커버리지 검증 (scripts/12 가 생성) — 목록과 실제 주석의 불일치 감시
wl_path = Path(P.get("ir_native_worklist", "splits/ir_native_worklist.txt"))
coverage = None
if wl_path.exists():
    worklist = {ln.strip() for ln in wl_path.read_text().splitlines() if ln.strip()}
    coverage = native_subset_coverage(records, worklist)
    print(f"[worklist] {coverage['annotated_in_worklist']}/{coverage['worklist']} 완료"
          + (f", 미완료 {len(coverage['missing'])}개" if coverage['missing'] else "")
          + (f", 목록 밖 주석 {len(coverage['annotated_off_worklist'])}개"
             if coverage['annotated_off_worklist'] else ""))
else:
    print(f"[worklist] {wl_path} 없음 — scripts/12 로 먼저 대상 페어를 선정하세요.")

# ② 지표 (i): landmark error
lm_dir = Path(P.get("landmarks_dir", "data/landmarks"))
errs, pts = [], 0
for rec in records:
    f = lm_dir / f"{rec.pair_id}.json"
    if not f.exists():
        continue
    e = landmark_error(rec, f)
    if e is not None:
        errs.append(e)
        pts += len(json.loads(f.read_text())["rgb"])
landmark_summary = {
    "pairs": len(errs), "points": pts,
    "mean_px": float(np.mean(errs)) if errs else None,
    "median_px": float(np.median(errs)) if errs else None,
}
print(f"[i ] landmark error: {landmark_summary}")

# ③ 지표 (ii): native warp IoU
plain_labels = Path(P["labels_dir"]) / "plain"
native_summary = native_warp_iou_report(records, plain_labels, imread_unicode)
print(f"[ii] native warp IoU: {native_summary}")

# ④ 코퍼스 정렬 통계
corpus = alignment_report(records)
print(f"[통계] {corpus}")

out = Path(P["outputs_dir"]) / "alignment_eval.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"landmark_error": landmark_summary,
                           "native_warp_iou": native_summary,
                           "native_subset_coverage": coverage,
                           "corpus": corpus}, indent=2))
print(f"\n결과: {out}  (논문 §4.2 실측값 소스)")
