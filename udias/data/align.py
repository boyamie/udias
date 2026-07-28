"""② 정렬 — IR->RGB 호모그래피(H)를 추정·캐싱. method: sift | xoftr.

설계 원칙(보고서 4장 / config.align):
  - inlier 가 min_inliers 미만이면 aligned=False 로 '명시적 실패' 기록.
    silent fallback(resize) 금지 → 정렬 통계가 왜곡되지 않는다(보고서 주장과 일치).
  - sift  : CLAHE(RGB) + min-max 정규화(IR), Lowe ratio + RANSAC.
            2026-07-25 실측 — 본 데이터에서 9660쌍 중 2쌍만 정렬(0.02%). 교차모달
            (가시↔열) 기술자 불일치 + FOV 격차로 실패. 비교/재현용으로만 유지.
  - xoftr : 학습 기반 교차모달 준밀집 매처(XoFTR, CVPR24W Image Matching).
            가시↔열화상 전용으로 학습된 모델. 2026-07-28 진단 6쌍 실측 —
            매치 113~464개, RANSAC inlier 11~53 (SIFT 0~4 대비), 워프 시각 검증 통과.
            third_party/XoFTR + weights/xoftr/*.ckpt 필요 (경로는 config.align).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_XOFTR_MATCHER = None                      # 프로세스 전역 1회 로드 (9660쌍 반복 호출용)


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


def _get_xoftr(cfg_align: dict):
    """XoFTR 매처 lazy-singleton. GPU 있으면 자동 사용(DataIOWrapper 내부)."""
    global _XOFTR_MATCHER
    if _XOFTR_MATCHER is None:
        repo = str(Path(cfg_align.get("xoftr_repo", "third_party/XoFTR")).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.xoftr import XoFTR                      # noqa: XoFTR repo 패키지명이 'src'
        from src.config.default import get_cfg_defaults
        from src.utils.data_io import DataIOWrapper, lower_config
        config = lower_config(get_cfg_defaults(inference=True))
        config["xoftr"]["match_coarse"]["thr"] = float(cfg_align.get("xoftr_coarse_thr", 0.3))
        config["xoftr"]["fine"]["thr"] = float(cfg_align.get("xoftr_fine_thr", 0.1))
        matcher = XoFTR(config=config["xoftr"])
        resize = int(cfg_align.get("xoftr_resize", 640))
        config["test"]["img0_resize"] = resize
        config["test"]["img1_resize"] = resize
        ckpt = str(Path(cfg_align.get("xoftr_ckpt", "weights/xoftr/weights_xoftr_640.ckpt")).resolve())
        _XOFTR_MATCHER = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)
    return _XOFTR_MATCHER


def _align_record_xoftr(rec, cfg_align: dict) -> None:
    """XoFTR 매칭 → RANSAC 호모그래피. 좌표는 원본 해상도(래퍼가 복원)."""
    reproj = float(cfg_align.get("ransac_reproj_thresh", 5.0))
    min_inliers = int(cfg_align.get("min_inliers", 15))
    rec.align_method = "xoftr+ransac"

    matcher = _get_xoftr(cfg_align)
    try:
        out = matcher.from_paths(str(rec.rgb_path), str(rec.ir_path), read_color=True)
    except Exception:
        rec.aligned = False
        rec.align_num_inliers = 0
        return
    k_rgb, k_ir = out["mkpts0"], out["mkpts1"]           # img0=RGB, img1=IR
    if len(k_rgb) < max(4, min_inliers):
        rec.aligned = False
        rec.align_num_inliers = 0
        return

    H, mask = cv2.findHomography(k_ir, k_rgb, cv2.RANSAC, reproj)   # H: IR -> RGB
    if H is None or mask is None:
        rec.aligned = False
        rec.align_num_inliers = 0
        return
    inl = mask.ravel().astype(bool)
    n_inliers = int(inl.sum())
    if n_inliers < min_inliers:                          # 명시적 실패 — fallback 없음
        rec.aligned = False
        rec.align_num_inliers = n_inliers
        rec.H_ir_to_rgb = None
        return

    proj = cv2.perspectiveTransform(k_ir[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.linalg.norm(proj - k_rgb[inl], axis=1).mean())
    rec.aligned = True
    rec.H_ir_to_rgb = H.tolist()
    rec.align_num_inliers = n_inliers
    rec.align_reproj_error = err


def align_record(rec, cfg_align: dict) -> None:
    """rec 의 RGB/IR 를 정렬해 H 와 품질 지표를 채운다(제자리 수정).

    cfg_align["method"]: "sift"(기본) | "xoftr"
    """
    if str(cfg_align.get("method", "sift")).lower() == "xoftr":
        _align_record_xoftr(rec, cfg_align)
        return
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
