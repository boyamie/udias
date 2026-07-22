"""⑥ 강건성 테스트 — 보고서 7~8장의 요구사항을 스크립트화.

시나리오:
  - missing_ir / missing_rgb : 한 모달리티 탈락 (보고서 8장 missing-modality)
  - night_only / day_only    : 장면 조건별 성능
  - low_contrast_ir          : IR 대비 축소 시뮬레이션
  - noisy_ir                 : speckle 유사 곱셈 노이즈

각 시나리오는 "입력 변환 함수"로 정의되어 어떤 모델에도 동일 적용된다.
"""
from __future__ import annotations

import numpy as np


def degrade_low_contrast(img, factor=0.4):
    mean = img.mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def degrade_multiplicative_noise(img, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, sigma, img.shape[:2]).astype(np.float32)
    if img.ndim == 3:
        noise = noise[..., None]
    return np.clip(img.astype(np.float32) * noise, 0, 255).astype(np.uint8)


SCENARIOS = {
    # name: (record_filter, rgb_transform, ir_transform, drop_modality)
    "clean":            (None, None, None, None),
    "missing_ir":       (None, None, None, "ir"),
    "missing_rgb":      (None, None, None, "rgb"),
    "night_only":       (lambda r: r.time_of_day == "night", None, None, None),
    "day_only":         (lambda r: r.time_of_day == "day", None, None, None),
    "low_contrast_ir":  (None, None, degrade_low_contrast, None),
    "noisy_ir":         (None, None, degrade_multiplicative_noise, None),
}


def apply_scenario(name, records):
    flt, t_rgb, t_ir, drop = SCENARIOS[name]
    recs = [r for r in records if flt is None or flt(r)]
    return recs, t_rgb, t_ir, drop
