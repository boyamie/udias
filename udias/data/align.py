"""② 정렬 — SIFT + RANSAC 로 IR->RGB 호모그래피(H)를 추정·캐싱.

설계 원칙(보고서 4장 / config.align):
  - CLAHE(RGB) + min-max 정규화(IR) 로 야간/저대비 대응
  - Lowe ratio test + RANSAC(reproj thresh)
  - inlier 가 min_inliers 미만이면 aligned=False 로 '명시적 실패' 기록.
    silent fallback(resize) 금지 → 정렬 통계가 왜곡되지 않는다(보고서 주장과 일치).
"""
from __future__ import annotations

import cv2
import numpy as np


def imread_unicode(path):
    """유니코드(한글) 경로 이미지 로드 → BGR 3채널(실패 시 None)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _clahe_bgr(img, clip: float = 3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def warp_ir_to_rgb(rec, ir_img, rgb_shape):
    """캐시된 H 로 IR 을 RGB 좌표계로 워프. rec.aligned=False 면 원본을 그대로 반환."""
    h, w = int(rgb_shape[0]), int(rgb_shape[1])
    if not rec.aligned or rec.H_ir_to_rgb is None:
        return ir_img
    H = np.asarray(rec.H_ir_to_rgb, dtype=np.float64)
    return cv2.warpPerspective(ir_img, H, (w, h))


def align_record(rec, cfg_align: dict) -> None:
    """rec 의 RGB/IR 를 정렬해 H 와 품질 지표를 채운다(제자리 수정)."""
    ratio = float(cfg_align.get("ratio_test", 0.75))
    reproj = float(cfg_align.get("ransac_reproj_thresh", 5.0))
    min_inliers = int(cfg_align.get("min_inliers", 15))
    use_clahe = bool(cfg_align.get("clahe", True))
    clip = float(cfg_align.get("clahe_clip", 3.0))

    rec.align_method = "sift+ratio+ransac"
    img_rgb = imread_unicode(rec.rgb_path)
    img_ir = imread_unicode(rec.ir_path)
    if img_rgb is None or img_ir is None:
        rec.aligned = False
        return

    rgb_p = _clahe_bgr(img_rgb, clip) if use_clahe else img_rgb
    gray_rgb = cv2.cvtColor(rgb_p, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.normalize(gray_ir, None, 0, 255, cv2.NORM_MINMAX)

    sift = cv2.SIFT_create()
    kp_rgb, des_rgb = sift.detectAndCompute(gray_rgb, None)
    kp_ir, des_ir = sift.detectAndCompute(gray_ir, None)
    if des_rgb is None or des_ir is None or len(kp_rgb) < 4 or len(kp_ir) < 4:
        rec.aligned = False
        rec.align_num_inliers = 0
        return

    # KNN 매칭 + Lowe ratio test. IR=query, RGB=train  →  H: IR -> RGB
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_ir, des_rgb, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < max(4, min_inliers):
        rec.aligned = False
        rec.align_num_inliers = len(good)
        return

    src = np.float32([kp_ir[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)   # IR
    dst = np.float32([kp_rgb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # RGB
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
    if H is None or mask is None:
        rec.aligned = False
        rec.align_num_inliers = 0
        return

    inl = mask.ravel().astype(bool)
    n_inliers = int(inl.sum())
    if n_inliers < min_inliers:                 # 명시적 실패 — fallback 없음
        rec.aligned = False
        rec.align_num_inliers = n_inliers
        rec.H_ir_to_rgb = None
        return

    proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
    err = float(np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1).mean())

    rec.aligned = True
    rec.H_ir_to_rgb = H.tolist()
    rec.align_num_inliers = n_inliers
    rec.align_reproj_error = err
