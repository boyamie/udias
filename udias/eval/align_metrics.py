"""②/⑥ 정렬 평가 — 정렬을 1급 태스크로 승격.

보고서 7장: "Alignment can be evaluated using landmark error,
intersection-over-union after warping, or downstream detection changes."

세 가지 평가를 제공한다:
  1. landmark error: 사람이 클릭한 대응점(소규모 검증셋)에 대한 재투영 오차
  2. warp IoU: RGB 기준 박스 vs (IR 기준 박스를 H로 투영한 박스)의 IoU
  3. downstream: 정렬 on/off로 탐지 mAP 비교 (scripts/07에서 수행)
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
    """같은 타깃에 대해 RGB 라벨 박스와, IR 라벨 박스를 H로 RGB 좌표계에 투영한
    박스의 IoU. 타깃 폭보다 잔여 오프셋이 큰지(보고서 4.1의 fine alignment) 판정 가능."""
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
