"""⑤ Middle fusion — 듀얼 스트림 백본 + CBAM 융합 + FCOS 헤드(완성본).

설계(보고서 5장):
  - modality-specific encoder 2개(RGB 3ch / IR 1ch) — torchvision ResNet 사용
    (timm 의존 제거: torch + torchvision 만 있으면 동작).
  - 스케일별 CBAM 융합 + ir_valid 게이트(정렬 실패/한 모달 탈락 시 IR 스트림 0).
    학습 중 modality dropout(p=0.15)으로도 재사용 → ⑥ 강건성 실험과 직접 연결.
  - FPN(채널 통일) + anchor-free FCOS 헤드(1클래스: Ship).

의존성: torch, torchvision.  (timm 불필요)
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FeaturePyramidNetwork, nms
from torchvision.models._utils import IntermediateLayerGetter


# ────────────────────────── CBAM 융합 블록 ──────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        r = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(inplace=True),
            nn.Linear(r, ch), nn.Sigmoid())

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
    """RGB/IR 피처 concat → 채널·공간 attention → 축소. ir_valid 로 IR 차단."""

    def __init__(self, ch: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1), nn.BatchNorm2d(ch), nn.SiLU(inplace=True))
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, f_rgb, f_ir, ir_valid=None):
        if ir_valid is not None:  # (B,) bool — 정렬 실패/modality dropout 시 IR 차단
            f_ir = f_ir * ir_valid.view(-1, 1, 1, 1).to(f_ir.dtype)
        x = self.reduce(torch.cat([f_rgb, f_ir], dim=1))
        return self.sa(self.ca(x))


# ────────────────────────── 듀얼 백본 ──────────────────────────
def _make_encoder(name: str, in_ch: int, pretrained: bool):
    ctor = getattr(torchvision.models, name)
    try:
        net = ctor(weights="DEFAULT" if pretrained else None)
    except TypeError:                      # 구버전 torchvision
        net = ctor(pretrained=pretrained)
    if in_ch != 3:                         # IR: 1채널 stem 교체
        old = net.conv1
        net.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                              stride=old.stride, padding=old.padding, bias=False)
    return IntermediateLayerGetter(net, {"layer2": "0", "layer3": "1", "layer4": "2"})


class DualStreamBackbone(nn.Module):
    """RGB/IR 별도 인코더 + 스케일별 CBAM 융합 → [P3, P4, P5] (strides 8/16/32)."""

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.enc_rgb = _make_encoder(backbone_name, 3, pretrained)
        self.enc_ir = _make_encoder(backbone_name, 1, pretrained)
        with torch.no_grad():
            chs = [v.shape[1] for v in self.enc_rgb(torch.zeros(1, 3, 64, 64)).values()]
        self.fusions = nn.ModuleList([CBAMFusion(c) for c in chs])
        self.out_channels = chs

    def forward(self, rgb, ir, ir_valid=None):
        fr = list(self.enc_rgb(rgb).values())
        fi = list(self.enc_ir(ir).values())
        return [fuse(a, b, ir_valid) for fuse, a, b in zip(self.fusions, fr, fi)]


# ────────────────────────── FCOS 헤드 ──────────────────────────
class Scale(nn.Module):
    def __init__(self, v: float = 1.0):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(float(v)))

    def forward(self, x):
        return x * self.s


class FCOSHead(nn.Module):
    def __init__(self, in_ch: int = 256, num_classes: int = 1, n_convs: int = 2,
                 n_levels: int = 3):
        super().__init__()

        def tower():
            layers = []
            for _ in range(n_convs):
                layers += [nn.Conv2d(in_ch, in_ch, 3, padding=1),
                           nn.GroupNorm(32, in_ch), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        self.cls = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.reg = nn.Conv2d(in_ch, 4, 3, padding=1)
        self.ctr = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(n_levels)])
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))  # focal prior

    def forward(self, feats):
        cls_out, reg_out, ctr_out = [], [], []
        for i, f in enumerate(feats):
            ct = self.cls_tower(f)
            rt = self.reg_tower(f)
            cls_out.append(self.cls(ct))
            ctr_out.append(self.ctr(rt))
            reg_out.append(self.scales[i](self.reg(rt)))   # 원시 거리(스트라이드 단위), relu는 손실/추론에서
        return cls_out, reg_out, ctr_out


# ────────────────────────── 손실 유틸(버전 안전, 직접 구현) ──────────────────────────
def sigmoid_focal(logits, targets, alpha: float = 0.25, gamma: float = 2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    return w * (1 - pt).pow(gamma) * ce


def giou_loss(pred, tgt, eps: float = 1e-7):
    x1 = torch.max(pred[:, 0], tgt[:, 0]); y1 = torch.max(pred[:, 1], tgt[:, 1])
    x2 = torch.min(pred[:, 2], tgt[:, 2]); y2 = torch.min(pred[:, 3], tgt[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    ap = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    at = (tgt[:, 2] - tgt[:, 0]).clamp(min=0) * (tgt[:, 3] - tgt[:, 1]).clamp(min=0)
    union = ap + at - inter + eps
    iou = inter / union
    cx1 = torch.min(pred[:, 0], tgt[:, 0]); cy1 = torch.min(pred[:, 1], tgt[:, 1])
    cx2 = torch.max(pred[:, 2], tgt[:, 2]); cy2 = torch.max(pred[:, 3], tgt[:, 3])
    carea = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps
    return 1 - (iou - (carea - union) / carea)


# ────────────────────────── 완성 탐지기 ──────────────────────────
class MiddleFusionDetector(nn.Module):
    """듀얼 인코더 + CBAM 융합 + FPN + FCOS 헤드. forward/compute_loss/predict 제공."""

    strides = (8, 16, 32)
    level_ranges = ((0, 64), (64, 128), (128, 1e9))

    def __init__(self, backbone_name: str = "resnet18", num_classes: int = 1,
                 pretrained: bool = True, fpn_out: int = 256):
        super().__init__()
        self.backbone = DualStreamBackbone(backbone_name, pretrained)
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, fpn_out)
        self.head = FCOSHead(fpn_out, num_classes, n_levels=len(self.strides))
        self.num_classes = num_classes

    def forward(self, rgb, ir, ir_valid=None):
        feats = self.backbone(rgb, ir, ir_valid)
        d = OrderedDict((str(i), f) for i, f in enumerate(feats))
        feats = list(self.fpn(d).values())
        return self.head(feats)

    @staticmethod
    def _grid(h, w, stride, device):
        gy, gx = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device), indexing="ij")
        px = (gx.reshape(-1).float() + 0.5) * stride
        py = (gy.reshape(-1).float() + 0.5) * stride
        return torch.stack([px, py], dim=1)                     # (HW,2)

    def compute_loss(self, outs, targets):
        cls_out, reg_out, ctr_out = outs
        device = cls_out[0].device
        B = cls_out[0].shape[0]

        points, strides, ranges = [], [], []
        cls_f, reg_f, ctr_f = [], [], []
        for lvl, (c, r, ct) in enumerate(zip(cls_out, reg_out, ctr_out)):
            _, _, H, W = c.shape
            s = self.strides[lvl]
            m = H * W
            points.append(self._grid(H, W, s, device))
            strides.append(torch.full((m,), float(s), device=device))
            lo, hi = self.level_ranges[lvl]
            ranges.append(torch.tensor([[lo, hi]], device=device).repeat(m, 1))
            cls_f.append(c.permute(0, 2, 3, 1).reshape(B, m, -1))
            reg_f.append(r.permute(0, 2, 3, 1).reshape(B, m, 4))
            ctr_f.append(ct.permute(0, 2, 3, 1).reshape(B, m, 1))
        points = torch.cat(points, 0)                            # (M,2)
        strides = torch.cat(strides, 0)                          # (M,)
        ranges = torch.cat(ranges, 0)                            # (M,2)
        cls_f = torch.cat(cls_f, 1); reg_f = torch.cat(reg_f, 1); ctr_f = torch.cat(ctr_f, 1)
        M = points.shape[0]

        cls_t = torch.zeros(B, M, self.num_classes, device=device)
        reg_t = torch.zeros(B, M, 4, device=device)
        ctr_t = torch.zeros(B, M, device=device)
        pos = torch.zeros(B, M, dtype=torch.bool, device=device)

        px = points[:, 0:1]; py = points[:, 1:2]
        for b in range(B):
            boxes = targets[b]["boxes"].to(device)
            labels = targets[b]["labels"].to(device)
            if boxes.numel() == 0:
                continue
            x1 = boxes[:, 0][None, :]; y1 = boxes[:, 1][None, :]
            x2 = boxes[:, 2][None, :]; y2 = boxes[:, 3][None, :]
            l = px - x1; t = py - y1; r = x2 - px; bt = y2 - py       # (M,N)
            ltrb = torch.stack([l, t, r, bt], -1)                     # (M,N,4)
            inside = ltrb.min(-1).values > 0
            maxd = ltrb.max(-1).values
            in_range = (maxd >= ranges[:, 0:1]) & (maxd <= ranges[:, 1:2])
            cxb = (x1 + x2) / 2; cyb = (y1 + y2) / 2
            radius = 1.5 * strides[:, None]
            in_center = ((px - cxb).abs() < radius) & ((py - cyb).abs() < radius)
            cand = inside & in_range & in_center                      # (M,N)
            areas = ((x2 - x1) * (y2 - y1)).expand(M, -1).clone()
            areas[~cand] = 1e18
            min_area, gt_idx = areas.min(1)
            has = min_area < 1e18
            pos[b] = has
            if has.any():
                sel = gt_idx[has]
                cls_t[b, has, labels[sel]] = 1.0
                chosen = ltrb[has, sel]                                # (P,4)
                reg_t[b, has] = chosen / strides[has, None]
                lr = chosen[:, [0, 2]]; tb = chosen[:, [1, 3]]
                ctr_t[b, has] = torch.sqrt(
                    (lr.min(1).values / lr.max(1).values.clamp(min=1e-6)) *
                    (tb.min(1).values / tb.max(1).values.clamp(min=1e-6)))

        num_pos = pos.sum().clamp(min=1)
        cls_loss = sigmoid_focal(cls_f, cls_t).sum() / num_pos
        if pos.any():
            pts_b = points[None].expand(B, -1, -1)[pos]               # (P,2)
            st_b = strides[None].expand(B, -1)[pos]                   # (P,)
            pred_ltrb = F.relu(reg_f[pos]) * st_b[:, None]
            tgt_ltrb = reg_t[pos] * st_b[:, None]

            def to_box(p, d):
                return torch.stack([p[:, 0] - d[:, 0], p[:, 1] - d[:, 1],
                                    p[:, 0] + d[:, 2], p[:, 1] + d[:, 3]], 1)

            w = ctr_t[pos]
            reg_loss = (giou_loss(to_box(pts_b, pred_ltrb),
                                  to_box(pts_b, tgt_ltrb)) * w).sum() / w.sum().clamp(min=1e-6)
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_f[pos].squeeze(-1), ctr_t[pos], reduction="mean")
        else:
            reg_loss = reg_f.sum() * 0.0
            ctr_loss = ctr_f.sum() * 0.0
        loss = cls_loss + reg_loss + ctr_loss
        return {"loss": loss, "cls": cls_loss.detach(),
                "reg": reg_loss.detach(), "ctr": ctr_loss.detach()}

    @torch.no_grad()
    def predict(self, rgb, ir, ir_valid=None, score_thr: float = 0.05,
                nms_thr: float = 0.6, topk: int = 100):
        cls_out, reg_out, ctr_out = self.forward(rgb, ir, ir_valid)
        device = cls_out[0].device
        B = cls_out[0].shape[0]
        results = []
        for b in range(B):
            boxes_all, scores_all = [], []
            for lvl in range(len(cls_out)):
                s = self.strides[lvl]
                c = cls_out[lvl][b]; r = reg_out[lvl][b]; ct = ctr_out[lvl][b]
                C, H, W = c.shape
                pts = self._grid(H, W, s, device)
                cls_s = torch.sigmoid(c).permute(1, 2, 0).reshape(H * W, C)
                ctr_s = torch.sigmoid(ct).reshape(H * W)
                score, _ = cls_s.max(1)
                score = torch.sqrt(score * ctr_s)
                ltrb = F.relu(r).permute(1, 2, 0).reshape(H * W, 4) * s
                box = torch.stack([pts[:, 0] - ltrb[:, 0], pts[:, 1] - ltrb[:, 1],
                                   pts[:, 0] + ltrb[:, 2], pts[:, 1] + ltrb[:, 3]], 1)
                keep = score > score_thr
                box, sc = box[keep], score[keep]
                if sc.numel() > topk:
                    sc, idx = sc.topk(topk)
                    box = box[idx]
                boxes_all.append(box); scores_all.append(sc)
            boxes = torch.cat(boxes_all, 0); scores = torch.cat(scores_all, 0)
            if boxes.numel():
                k = nms(boxes, scores, nms_thr)[:topk]
                results.append((boxes[k].cpu(), scores[k].cpu()))
            else:
                results.append((boxes.cpu(), scores.cpu()))
        return results
