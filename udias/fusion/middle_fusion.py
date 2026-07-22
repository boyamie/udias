"""⑤ Middle fusion — 듀얼 스트림 백본 + attention 융합 블록 (Figure 7 계획 대응).

ultralytics 순정으로는 듀얼 입력이 안 되므로 두 가지 경로가 있다:

  경로 A (권장, 여기 구현): 순수 PyTorch로 듀얼 백본 + CBAM 융합 + 경량 헤드.
    - 백본은 timm에서 가져와 재현성 확보. 헤드는 anchor-free 단순 구현 or
      torchvision FCOS/RetinaNet 헤드 재사용.
  경로 B: ultralytics 커스텀 model yaml에 두 백본을 정의하고 중간 concat.
    - 프레임워크 훅에 의존해 유지보수가 어려움. 논문 재현성 관점에서 A 권장.

이 파일은 경로 A의 골격이다. 핵심 설계 결정(보고서 5장 반영):
  - modality-specific encoder 유지 (SAR/IR 통계에 맞는 별도 stem)
  - 채널 attention(SE/CBAM)으로 모달리티 가중 (게이팅: 야간 RGB 억제 등)
  - 정렬 실패 페어 대응: ir_valid 마스크로 IR 스트림을 0으로 드롭
    → 학습 중 modality dropout으로도 사용 (⑥ 강건성 실험과 연결)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // reduction), nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


class CBAMFusion(nn.Module):
    """RGB/IR 피처를 concat → 채널·공간 attention → 축소. 융합 블록의 최소 형태."""

    def __init__(self, ch: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1), nn.BatchNorm2d(ch), nn.SiLU(inplace=True))
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, f_rgb, f_ir, ir_valid=None):
        if ir_valid is not None:  # (B,) bool — 정렬 실패/modality dropout 시 IR 차단
            f_ir = f_ir * ir_valid.view(-1, 1, 1, 1).float()
        x = self.reduce(torch.cat([f_rgb, f_ir], dim=1))
        return self.sa(self.ca(x))


class DualStreamBackbone(nn.Module):
    """timm 백본 2개(모달리티별) + 스케일별 CBAM 융합 → FPN 스타일 멀티스케일 피처.

    사용 예:
        bb = DualStreamBackbone("resnet18")   # pip install timm
        feats = bb(rgb, ir, ir_valid)          # [P3, P4, P5]
    이후 torchvision.models.detection의 헤드(FCOS 등)에 연결하거나
    자체 anchor-free 헤드를 붙인다.
    """

    def __init__(self, backbone_name: str = "resnet18", out_indices=(2, 3, 4),
                 pretrained: bool = True):
        super().__init__()
        import timm
        self.enc_rgb = timm.create_model(backbone_name, features_only=True,
                                         out_indices=out_indices, pretrained=pretrained)
        self.enc_ir = timm.create_model(backbone_name, features_only=True,
                                        out_indices=out_indices, pretrained=pretrained,
                                        in_chans=1)
        chs = self.enc_rgb.feature_info.channels()
        self.fusions = nn.ModuleList([CBAMFusion(c) for c in chs])
        self.out_channels = chs

    def forward(self, rgb, ir, ir_valid=None):
        fr = self.enc_rgb(rgb)
        fi = self.enc_ir(ir)
        return [fuse(a, b, ir_valid) for fuse, a, b in zip(self.fusions, fr, fi)]


# ── 학습 루프 스케치 ─────────────────────────────────────────
# dataset.py의 PairDataset이 (rgb, ir_aligned, ir_valid, targets)를 반환한다고 가정.
# 1) torchvision FCOS 헤드 연결 예:
#    from torchvision.models.detection.fcos import FCOSHead 를 참고해
#    DualStreamBackbone 출력 채널을 공통 채널로 lateral conv 후 헤드에 전달.
# 2) modality dropout: 학습 배치마다 p=0.15로 ir_valid를 False로 설정
#    → ⑥ 강건성 실험(단일 모달 탈락)과 직접 비교 가능한 모델이 된다.
# 3) 비교 공정성: single/early/late와 동일한 split·라벨·epochs·seed 집합 사용.
