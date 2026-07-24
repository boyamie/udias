"""데이터카드 통계 — 그림·표 산출용 순수 함수 (논문 §4.5, M10/M14).

matplotlib 없이 수치만 뽑는다 (그림 렌더는 scripts/14 가 담당). 모든 함수는
매니페스트와 plain 라벨만 읽으므로 학습된 모델이 필요 없다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .align_metrics import load_plain_boxes


def ship_size_areas(records, plain_label_dir, img_size_lookup, split="test"):
    """대상 split 의 모든 선박 박스 면적(px^2) 배열. 크기 분포·소형 편중 진단용."""
    plain = Path(plain_label_dir)
    areas = []
    for rec in records:
        if split is not None and rec.split != split:
            continue
        w, h = img_size_lookup(rec)
        for b in load_plain_boxes(plain / f"{rec.pair_id}.txt", w, h):
            areas.append(float((b[2] - b[0]) * (b[3] - b[1])))
    return np.asarray(areas, dtype=float)


def size_bin_counts(areas, buckets):
    """면적 배열 → {'small':n, 'medium':n, 'large':n} (COCO 관례; buckets=config eval.size_buckets)."""
    out = {}
    for name, (lo, hi) in buckets.items():
        hi = float("inf") if hi in (".inf", "inf", None) else float(hi)
        out[name] = int(np.sum((areas >= float(lo)) & (areas < hi)))
    return out


def alignment_by_tod(records):
    """주야별 정렬 성공률 + 재투영 오차 분포. {tod: {'rate','n','reproj':[...]}}."""
    out = {}
    for tod in sorted({r.time_of_day for r in records}):
        sub = [r for r in records if r.time_of_day == tod]
        ok = [r for r in sub if r.aligned]
        out[tod] = {
            "n": len(sub),
            "rate": len(ok) / len(sub) if sub else 0.0,
            "reproj": [r.align_reproj_error for r in ok if r.align_reproj_error >= 0],
        }
    return out


def split_composition(records):
    """split × (time_of_day, scene_type) 페어 수. 데이터카드 층화 분포용."""
    out = {}
    for r in records:
        out.setdefault(r.split, {}).setdefault(
            (r.time_of_day, r.scene_type), 0)
        out[r.split][(r.time_of_day, r.scene_type)] += 1
    return out
