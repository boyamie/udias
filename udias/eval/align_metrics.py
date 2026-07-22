"""②/⑥ 정렬 평가 — 정렬을 1급 태스크로 승격.

보고서 7장: "Alignment can be evaluated using landmark error,
intersection-over-union after warping, or downstream detection changes."

세 가지 평가를 제공한다 (논문 §4.2 의 지표 i/ii/iii):
  1. landmark error: 사람이 클릭한 대응점(소규모 검증셋)에 대한 재투영 오차
     — 정렬 ground truth (지표 i)
  2. native warp IoU: RGB GT 박스 vs (IR 좌표계에서 '독립적으로' 사람이 표기한
     native IR 박스를 H로 투영한 박스)의 IoU (지표 ii).
     ※ RGB 라벨을 투영해 만든 IR 라벨은 여기 쓰면 순환(자기 자신과 비교) —
       반드시 rec.ir_label_path 의 native 주석만 사용한다.
  3. downstream: 정렬 on/off로 탐지 mAP 비교 (scripts/05 의 noalign ablation)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..data.manifest import PairRecord


def landmark_error(rec: PairRecord, landmark_json: str | Path) -> float | None:
    """landmark_json 형식: {"rgb": [[x,y],...], "ir": [[x,y],...]} (대응 순서 동일)

    검증 서브셋(예: 각 장면유형별 20페어)에 대해 CVAT 등으로 대응점 5~10개를
    수동 표기해두고, H가 그 점들을 얼마나 잘 옮기는지 측정한다.
    """
    if not rec.aligned:
        return None
    d = json.loads(Path(landmark_json).read_text())
    pts_ir = np.float32(d["ir"]).reshape(-1, 1, 2)
    pts_rgb = np.float32(d["rgb"]).reshape(-1, 2)
    H = np.array(rec.H_ir_to_rgb)
    proj = cv2.perspectiveTransform(pts_ir, H).reshape(-1, 2)
    return float(np.linalg.norm(proj - pts_rgb, axis=1).mean())


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area if area > 0 else 0.0


def warp_iou(boxes_rgb: np.ndarray, boxes_ir: np.ndarray, rec: PairRecord) -> list[float]:
    """[주의] 순서 대응(zip)을 가정하는 저수준 유틸. 논문 지표 (ii)에는 쓰지 말 것 —
    boxes_ir 에 'RGB 라벨을 투영해 만든' IR 라벨을 넣으면 자기 자신과 비교하는
    순환이 된다 (리뷰 M5). 지표 (ii)는 native_warp_iou_report() 를 사용."""
    if not rec.aligned:
        return []
    H = np.array(rec.H_ir_to_rgb)
    ious = []
    for b_rgb, b_ir in zip(boxes_rgb, boxes_ir):
        pts = np.float32([[b_ir[0], b_ir[1]], [b_ir[2], b_ir[1]],
                          [b_ir[2], b_ir[3]], [b_ir[0], b_ir[3]]]).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        warped = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
        ious.append(box_iou(np.array(b_rgb, dtype=float), warped))
    return ious


def load_plain_boxes(path: str | Path, w: int, h: int) -> np.ndarray:
    """plain YOLO 5열(class cx cy w h, 정규화) → 절대 xyxy (N,4)."""
    boxes = []
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            cx, cy, bw, bh = (float(f[1]) * w, float(f[2]) * h,
                              float(f[3]) * w, float(f[4]) * h)
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def _warp_boxes(boxes_ir: np.ndarray, H: np.ndarray) -> np.ndarray:
    """IR 좌표계 xyxy → H 로 RGB 좌표계 투영 후 axis-aligned 근사 (N,4)."""
    out = []
    for b in boxes_ir:
        pts = np.float32([[b[0], b[1]], [b[2], b[1]],
                          [b[2], b[3]], [b[0], b[3]]]).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        out.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
    return np.asarray(out, dtype=float).reshape(-1, 4)


def native_warp_iou_report(records: list[PairRecord], rgb_label_dir: str | Path,
                           imread) -> dict:
    """지표 (ii): native IR 주석이 있는 서브셋에서 warp IoU (greedy 1:1 매칭).

    독립 주석이므로 박스 순서 대응이 없다 → IoU 내림차순 greedy 매칭 후
    매칭된 쌍의 IoU 와 매칭률을 보고한다. rec.ir_label_path == "" 이거나
    정렬 실패 페어는 제외 (제외 수는 리포트에 기록).
    """
    ious, n_pairs, n_ir, n_rgb, n_match = [], 0, 0, 0, 0
    skipped_unaligned = 0
    for rec in records:
        if not rec.ir_label_path or not Path(rec.ir_label_path).exists():
            continue
        if not rec.aligned:
            skipped_unaligned += 1
            continue
        img_ir = imread(rec.ir_path)
        img_rgb = imread(rec.rgb_path)
        if img_ir is None or img_rgb is None:
            continue
        b_ir = load_plain_boxes(rec.ir_label_path,
                                img_ir.shape[1], img_ir.shape[0])
        b_rgb = load_plain_boxes(Path(rgb_label_dir) / f"{rec.pair_id}.txt",
                                 img_rgb.shape[1], img_rgb.shape[0])
        if len(b_ir) == 0 and len(b_rgb) == 0:
            continue
        n_pairs += 1
        n_ir += len(b_ir)
        n_rgb += len(b_rgb)
        warped = _warp_boxes(b_ir, np.array(rec.H_ir_to_rgb))
        if len(warped) == 0 or len(b_rgb) == 0:
            continue
        iou_mat = np.array([[box_iou(w, g) for g in b_rgb] for w in warped])
        while iou_mat.size and iou_mat.max() > 0:
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            ious.append(float(iou_mat[i, j]))
            n_match += 1
            iou_mat[i, :] = -1
            iou_mat[:, j] = -1
    return {
        "pairs_evaluated": n_pairs,
        "skipped_unaligned": skipped_unaligned,
        "ir_boxes": n_ir,
        "rgb_boxes": n_rgb,
        "matched": n_match,
        "match_rate_ir": n_match / n_ir if n_ir else None,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "median_iou": float(np.median(ious)) if ious else None,
    }


def alignment_report(records: list[PairRecord]) -> dict:
    """매니페스트 전체의 정렬 성공률/품질 요약 — 데이터카드 수치."""
    total = len(records)
    ok = [r for r in records if r.aligned]
    by_tod = {}
    for tod in ("day", "night", "unknown"):
        sub = [r for r in records if r.time_of_day == tod]
        if sub:
            by_tod[tod] = sum(r.aligned for r in sub) / len(sub)
    errs = [r.align_reproj_error for r in ok if r.align_reproj_error >= 0]
    return {
        "total_pairs": total,
        "aligned": len(ok),
        "align_rate": len(ok) / total if total else 0,
        "align_rate_by_time_of_day": by_tod,
        "mean_reproj_error_px": float(np.mean(errs)) if errs else None,
        "median_inliers": float(np.median([r.align_num_inliers for r in ok])) if ok else None,
    }
