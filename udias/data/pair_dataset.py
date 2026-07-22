"""듀얼 입력 데이터셋 — middle fusion 학습/평가용.

한 샘플 = (rgb 3ch, ir 1ch, ir_valid, targets). 라벨은 RGB 좌표계의 plain YOLO
(5열)을 절대 xyxy 로 변환한다. IR 은 aligned 면 H 로 워프, 아니면 resize 하고
ir_valid=aligned 로 표시(정렬 실패 시 CBAM 게이트가 IR 을 차단).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .align import imread_unicode, warp_ir_to_rgb

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _load_plain_labels(path, w, h):
    boxes = []
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            cx, cy, bw, bh = (float(f[1]) * w, float(f[2]) * h,
                              float(f[3]) * w, float(f[4]) * h)
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return boxes


class PairDataset(Dataset):
    def __init__(self, records, plain_label_dir, img_size: int = 640, split=None):
        self.records = [r for r in records if (split is None or r.split == split)]
        self.label_dir = Path(plain_label_dir)
        self.img = int(img_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        rgb = imread_unicode(rec.rgb_path)
        if rgb is None:
            rgb = np.zeros((self.img, self.img, 3), np.uint8)
        h0, w0 = rgb.shape[:2]
        ir = imread_unicode(rec.ir_path)
        if ir is None:
            ir = np.zeros((h0, w0, 3), np.uint8)
        ir_valid = bool(rec.aligned)
        ir = warp_ir_to_rgb(rec, ir, rgb.shape) if rec.aligned else cv2.resize(ir, (w0, h0))

        boxes = _load_plain_labels(self.label_dir / f"{rec.pair_id}.txt", w0, h0)

        rgb_r = cv2.resize(rgb, (self.img, self.img))
        ir_g = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY) if ir.ndim == 3 else ir
        ir_r = cv2.resize(ir_g, (self.img, self.img))
        sx, sy = self.img / w0, self.img / h0
        boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]

        rgb_t = torch.from_numpy(rgb_r[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0
        for c in range(3):
            rgb_t[c] = (rgb_t[c] - _MEAN[c]) / _STD[c]
        ir_t = torch.from_numpy(ir_r).float().unsqueeze(0) / 255.0

        return {
            "rgb": rgb_t,
            "ir": ir_t,
            "ir_valid": ir_valid,
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.zeros(len(boxes), dtype=torch.long),
            "pair_id": rec.pair_id,
            "orig_hw": (h0, w0),
        }


def collate(batch):
    rgb = torch.stack([b["rgb"] for b in batch])
    ir = torch.stack([b["ir"] for b in batch])
    ir_valid = torch.tensor([b["ir_valid"] for b in batch], dtype=torch.bool)
    targets = [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch]
    meta = [{"pair_id": b["pair_id"], "orig_hw": b["orig_hw"]} for b in batch]
    return rgb, ir, ir_valid, targets, meta
