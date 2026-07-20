"""④ Early fusion — 융합을 '변환 함수'로 분리. 데이터셋에 굽지 않는다.

원칙: 디스크의 원본은 RGB/IR 페어 + H(매니페스트) 하나뿐이고,
early fusion 학습용 이미지는 필요할 때 매니페스트에서 파생 생성한다.
이렇게 하면 융합 방식을 바꿔도 라벨/split은 그대로 재사용된다.

두 가지 변형 제공:
  - pixel_fusion:  (RGB + ε) · IR / 255   (기존 image_fusion.py 식 유지, 3ch 출력)
  - stack4:        RGB 3ch + IR 1ch 스택   (4ch 입력, ultralytics model yaml에 ch: 4)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from ..data.align import imread_unicode, warp_ir_to_rgb
from ..data.manifest import PairRecord


def pixel_fusion(img_rgb: np.ndarray, ir_aligned: np.ndarray, epsilon: float = 10.0) -> np.ndarray:
    ir = ir_aligned if ir_aligned.ndim == 2 else cv2.cvtColor(ir_aligned, cv2.COLOR_BGR2GRAY)
    ir3 = cv2.merge([ir, ir, ir]).astype(np.float32)
    fused = (img_rgb.astype(np.float32) + epsilon) * (ir3 / 255.0)
    return np.clip(fused, 0, 255).astype(np.uint8)


def stack4(img_rgb: np.ndarray, ir_aligned: np.ndarray) -> np.ndarray:
    ir = ir_aligned if ir_aligned.ndim == 2 else cv2.cvtColor(ir_aligned, cv2.COLOR_BGR2GRAY)
    return np.dstack([img_rgb, ir])  # HxWx4


def export_yolo_dataset(records: list[PairRecord], plain_label_dir: str | Path,
                        out_root: str | Path, mode: str, *,
                        epsilon: float = 10.0, use_alignment: bool = True) -> Path:
    """매니페스트 → ultralytics 학습용 디렉토리 생성.

    mode: 'rgb' | 'ir' | 'early'   (stack4는 커스텀 로더 필요 → dataset.py 참고)
    use_alignment=False 는 '정렬 없는 융합' ablation용 (IR 단순 resize).
    정렬 실패(aligned=False) 페어는 early 모드에서 제외하고 개수를 리포트.
    """
    out_root = Path(out_root)
    skipped = 0
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for rec in tqdm(records, desc=f"export[{mode}]"):
        if rec.split not in ("train", "val", "test"):
            continue
        img_rgb = imread_unicode(rec.rgb_path)
        if img_rgb is None:
            continue

        if mode == "rgb":
            out_img = img_rgb
        elif mode == "ir":
            img_ir = imread_unicode(rec.ir_path)
            if img_ir is None:
                continue
            # IR-only 라벨은 RGB 기준 라벨을 그대로 쓰려면 IR을 RGB 좌표계로 warp
            if rec.aligned and use_alignment:
                out_img = warp_ir_to_rgb(rec, img_ir, img_rgb.shape)
            else:
                out_img = cv2.resize(img_ir, (img_rgb.shape[1], img_rgb.shape[0]))
            if out_img.ndim == 2:
                out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        elif mode == "early":
            img_ir = imread_unicode(rec.ir_path)
            if img_ir is None:
                continue
            if use_alignment:
                if not rec.aligned:
                    skipped += 1
                    continue  # 조용한 fallback 금지 — 제외하고 카운트
                ir_al = warp_ir_to_rgb(rec, img_ir, img_rgb.shape)
            else:
                ir_al = cv2.resize(img_ir, (img_rgb.shape[1], img_rgb.shape[0]))
            out_img = pixel_fusion(img_rgb, ir_al, epsilon)
        else:
            raise ValueError(mode)

        cv2.imwrite(str(out_root / "images" / rec.split / f"{rec.pair_id}.jpg"), out_img)
        lbl = Path(plain_label_dir) / f"{rec.pair_id}.txt"
        if lbl.exists():
            (out_root / "labels" / rec.split / lbl.name).write_text(lbl.read_text())

    if skipped:
        print(f"[info] 정렬 실패로 제외된 페어: {skipped}개 (데이터카드에 기록)")

    data_yaml = {"path": str(out_root.resolve()), "train": "images/train",
                 "val": "images/val", "test": "images/test",
                 "nc": 1, "names": ["Ship"]}
    yp = out_root / "data.yaml"
    yp.write_text(yaml.dump(data_yaml))
    return yp
