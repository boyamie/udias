"""주석 품질 — 주석자 간 일치도 (IAA, 논문 §4.3, 리뷰 B3).

이중 주석 서브셋: 같은 페어를 두 주석자가 RGB 프레임에서 독립 표기한다.
확장 YOLO 포맷(class cx cy w h v u)을 로드해 세 가지 일치도를 산출:

  1. detection  : 두 주석자의 박스를 IoU>=thr 로 greedy 1:1 매칭 → F1
                  (일치도는 대칭이라 한쪽을 GT로 두지 않고 F1 = 매칭쌍 기준)
  2. localization: 매칭된 박스쌍의 평균 IoU
  3. flags      : 매칭된 타깃에서 가시성 v∈{rgb,ir,both} 및 uncertain u 의
                  단순 일치율 + Cohen's kappa (우연 일치 보정)

두 주석자 중 한쪽만 표기한 페어는 제외한다(개수 리포트).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .align_metrics import box_iou


def _greedy_match(boxes_a: np.ndarray, boxes_b: np.ndarray, thr: float):
    """IoU>=thr 로 greedy 1:1 매칭 → (매칭쌍 [(i,j,iou)], a미매칭 수, b미매칭 수)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return [], len(boxes_a), len(boxes_b)
    iou = np.array([[box_iou(a, b) for b in boxes_b] for a in boxes_a])
    pairs = []
    ai, bj = set(), set()
    while True:
        i, j = np.unravel_index(iou.argmax(), iou.shape)
        if iou[i, j] < thr:
            break
        pairs.append((int(i), int(j), float(iou[i, j])))
        ai.add(int(i)); bj.add(int(j))
        iou[i, :] = -1; iou[:, j] = -1
    return pairs, len(boxes_a) - len(ai), len(boxes_b) - len(bj)


def _kappa(labels_a, labels_b) -> float | None:
    """Cohen's kappa (범주형). 표본 없거나 한 범주뿐이면 None."""
    a, b = list(labels_a), list(labels_b)
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:                          # 모두 한 범주 → 우연 일치 100%, kappa 정의 불가
        return None
    return (po - pe) / (1 - pe)


def iaa_report(records, dir_a, dir_b, img_size_lookup, load_extended_labels,
               iou_thr: float = 0.5) -> dict:
    """이중 주석된 페어 전체에 대한 IAA 요약.

    dir_a/dir_b: 두 주석자의 확장 라벨 디렉토리 ({pair_id}.txt).
    img_size_lookup(rec)->(w,h): 정규화 좌표 복원용 (RGB 프레임 크기).
    """
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    ious, n_pairs = [], 0
    tp = fa = fb = 0                       # 매칭쌍 / a단독 / b단독
    vis_a, vis_b, unc_a, unc_b = [], [], [], []
    skipped_single = 0
    for rec in records:
        pa, pb = dir_a / f"{rec.pair_id}.txt", dir_b / f"{rec.pair_id}.txt"
        if not (pa.exists() and pb.exists()):
            if pa.exists() or pb.exists():
                skipped_single += 1
            continue
        w, h = img_size_lookup(rec)
        ba, va, ua = load_extended_labels(pa, w, h)
        bb, vb, ub = load_extended_labels(pb, w, h)
        n_pairs += 1
        pairs, ua_n, ub_n = _greedy_match(ba, bb, iou_thr)
        tp += len(pairs); fa += ua_n; fb += ub_n
        for i, j, iou in pairs:
            ious.append(iou)
            vis_a.append(int(va[i])); vis_b.append(int(vb[j]))
            unc_a.append(int(ua[i])); unc_b.append(int(ub[j]))

    f1 = (2 * tp / (2 * tp + fa + fb)) if (2 * tp + fa + fb) else None
    vis_agree = (sum(x == y for x, y in zip(vis_a, vis_b)) / len(vis_a)
                 if vis_a else None)
    unc_agree = (sum(x == y for x, y in zip(unc_a, unc_b)) / len(unc_a)
                 if unc_a else None)
    return {
        "pairs_double_annotated": n_pairs,
        "skipped_single_annotator": skipped_single,
        "box_f1": f1,
        "matched": tp, "only_a": fa, "only_b": fb,
        "mean_iou_matched": float(np.mean(ious)) if ious else None,
        "visibility_agreement": vis_agree,
        "visibility_kappa": _kappa(vis_a, vis_b),
        "uncertain_agreement": unc_agree,
        "uncertain_kappa": _kappa(unc_a, unc_b),
    }
