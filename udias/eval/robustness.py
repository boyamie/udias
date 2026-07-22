"""⑥ 강건성 시나리오 — 결정론적 입력 변환 (논문 §5.4).

논문 규정: "Each scenario is defined as a deterministic input transform,
so it applies identically to every baseline."

시나리오:
  clean         — 무변환 기준선
  drop_rgb      — RGB 스트림 완전 제거(0 이미지)
  drop_ir       — IR 스트림 완전 제거(0 이미지) + ir_valid=False (middle 게이트)
  ir_contrast40 — IR 대비를 프레임 평균 중심으로 40%로 축소
  ir_noise03    — IR 에 곱셈형 가우시안 노이즈 (σ=0.3)

결정론성: 노이즈 난수는 pair_id 의 CRC32 로 시드되므로 어떤 모델·어떤 실행에서도
같은 페어에는 같은 노이즈가 적용된다 (day/night/harbor 서브셋 평가는 변환이 아니라
det_metrics.evaluate_by_scene 의 report_by 분리 리포트가 담당).
"""
from __future__ import annotations

import zlib

import numpy as np

SCENARIOS = ("clean", "drop_rgb", "drop_ir", "ir_contrast40", "ir_noise03")


def _rng(pair_id: str, scenario: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{scenario}:{pair_id}".encode()))


def ir_contrast(ir: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """프레임 평균 중심 대비 축소: mean + factor·(ir − mean)."""
    m = float(ir.mean())
    return np.clip(m + factor * (ir.astype(np.float32) - m), 0, 255).astype(np.uint8)


def ir_mult_noise(ir: np.ndarray, sigma: float, pair_id: str,
                  scenario: str = "ir_noise03") -> np.ndarray:
    """곱셈형 가우시안 노이즈: ir · (1 + N(0, σ)). 공간 노이즈는 채널 간 공유."""
    noise = 1.0 + _rng(pair_id, scenario).normal(0.0, sigma, ir.shape[:2]).astype(np.float32)
    x = ir.astype(np.float32)
    if x.ndim == 3:
        noise = noise[..., None]
    return np.clip(x * noise, 0, 255).astype(np.uint8)


def apply_scenario(name: str, rec, rgb: np.ndarray, ir: np.ndarray,
                   ir_valid: bool, cfg: dict | None = None):
    """(rgb, ir, ir_valid) → 변환된 (rgb, ir, ir_valid).

    PairDataset(transform=...) 훅 및 ultralytics 평가 경로 양쪽에서 동일하게
    호출된다. cfg 는 config 의 robustness 섹션 (없으면 논문 기본값).
    """
    c = cfg or {}
    if name == "clean":
        return rgb, ir, ir_valid
    if name == "drop_rgb":
        return np.zeros_like(rgb), ir, ir_valid
    if name == "drop_ir":
        return rgb, np.zeros_like(ir), False
    if name == "ir_contrast40":
        return rgb, ir_contrast(ir, float(c.get("ir_contrast_factor", 0.4))), ir_valid
    if name == "ir_noise03":
        return rgb, ir_mult_noise(ir, float(c.get("ir_noise_sigma", 0.3)),
                                  rec.pair_id), ir_valid
    raise ValueError(f"unknown scenario: {name}")
