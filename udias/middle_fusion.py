"""④ Late fusion — RGB-only / IR-only 탐지기의 박스 출력을 병합.

구현이 가장 싼 융합 베이스라인. 특징-수준 정합이 필요 없어(보고서 7장)
정렬 품질이 나쁜 페어에서도 동작한다는 점이 비교 포인트.

절차:
  1. RGB 모델 → RGB 이미지 추론 (RGB 좌표계 박스)
  2. IR 모델 → 정렬된 IR 이미지 추론 (이미 RGB 좌표계) — 정렬 실패 페어는
     resize된 IR 사용 (late fusion의 강건성 주장 근거)
  3. Weighted Box Fusion으로 병합

의존성: pip install ensemble-boxes
"""
from __future__ import annotations

import numpy as np


def wbf_merge(boxes_list, scores_list, img_w, img_h, iou_thr=0.55,
              weights=None, conf_type="avg"):
    """boxes_list: [rgb_boxes(N,4 xyxy px), ir_boxes(M,4 xyxy px)] → 병합 결과(px)"""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as e:
        raise ImportError("pip install ensemble-boxes 필요") from e

    norm_boxes, norm_scores, labels = [], [], []
    for boxes, scores in zip(boxes_list, scores_list):
        b = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
        b[:, [0, 2]] /= img_w
        b[:, [1, 3]] /= img_h
        norm_boxes.append(np.clip(b, 0, 1).tolist())
        norm_scores.append(list(map(float, scores)))
        labels.append([0] * len(scores))

    fb, fs, _ = weighted_boxes_fusion(norm_boxes, norm_scores, labels,
                                      weights=weights, iou_thr=iou_thr,
                                      conf_type=conf_type)
    fb = np.asarray(fb)
    if len(fb):
        fb[:, [0, 2]] *= img_w
        fb[:, [1, 3]] *= img_h
    return fb, np.asarray(fs)


def predict_pair_late(rec, model_rgb, model_ir, cfg_late, imread, warp_ir_to_rgb):
    """한 페어에 대한 late fusion 추론. eval 스크립트에서 호출."""
    import cv2

    img_rgb = imread(rec.rgb_path)
    img_ir = imread(rec.ir_path)
    h, w = img_rgb.shape[:2]

    def run(model, img):
        r = model.predict(img, conf=cfg_late["conf_thr"], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    b_rgb, s_rgb = run(model_rgb, img_rgb)

    if rec.aligned:
        ir_in = warp_ir_to_rgb(rec, img_ir, img_rgb.shape)
    else:
        ir_in = cv2.resize(img_ir, (w, h))
    if ir_in.ndim == 2:
        ir_in = cv2.cvtColor(ir_in, cv2.COLOR_GRAY2BGR)
    b_ir, s_ir = run(model_ir, ir_in)

    return wbf_merge([b_rgb, b_ir], [s_rgb, s_ir], w, h,
                     iou_thr=cfg_late["iou_thr"])
