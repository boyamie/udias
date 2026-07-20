"""③ 라벨 QC — auto-label은 초벌, 사람 검수가 정답을 만든다.

기존 make_yolo_labels.py의 문제:
  - COCO 사전학습 YOLO의 출력을 그대로 정답으로 사용 → 라벨이 특정 모델의
    편향을 상속. 데이터셋 논문에서는 치명적.
  - 융합 이미지 위에 라벨 생성 → 라벨의 기준 좌표계가 불명확.

여기서의 설계:
  1. 라벨은 항상 RGB 원본 좌표계 기준으로 정의 (매니페스트 규약과 일치)
  2. auto-label은 RGB와 (H로 정렬된) IR 각각에 대해 수행
     → 두 결과를 IoU 매칭해 타깃별 visible_in ∈ {rgb, ir, both} 초벌 플래그 생성
     → 보고서 3장: "each annotation should record whether the target is
        visible in SAR, EO, or both modalities"
  3. Label Studio(YOLO 포맷 import 지원)로 내보내 사람이 검수
     → 검수 완료 시 rec.label_verified = True
  4. 라벨 파일 확장 포맷 (YOLO 5열 + 플래그 2열):
       class cx cy w h visible_in{0=rgb,1=ir,2=both} uncertain{0,1}
     학습 시에는 앞 5열만 잘라 쓰면 ultralytics와 호환.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.align import imread_unicode, warp_ir_to_rgb
from ..data.manifest import PairRecord
from ..eval.align_metrics import box_iou


def autolabel_pair(rec: PairRecord, model, conf: float, boat_cls: int,
                   out_dir: str | Path) -> Path | None:
    """RGB와 정렬된 IR 각각에 초벌 탐지 → visible_in 플래그 부여."""
    img_rgb = imread_unicode(rec.rgb_path)
    if img_rgb is None:
        return None

    def detect(img):
        res = model.predict(img, conf=conf, verbose=False)[0]
        boxes = []
        if res.boxes is not None:
            for b in res.boxes:
                if int(b.cls[0]) == boat_cls:
                    boxes.append(b.xyxy[0].tolist())
        return np.array(boxes).reshape(-1, 4)

    boxes_rgb = detect(img_rgb)

    boxes_ir = np.zeros((0, 4))
    if rec.aligned:
        img_ir = imread_unicode(rec.ir_path)
        if img_ir is not None:
            ir_warped = warp_ir_to_rgb(rec, img_ir, img_rgb.shape)
            boxes_ir = detect(ir_warped)  # 이미 RGB 좌표계

    # IoU 매칭으로 visible_in 결정
    merged = []  # (xyxy, visible_in)
    used_ir = set()
    for br in boxes_rgb:
        flag = 0  # rgb only
        for j, bi in enumerate(boxes_ir):
            if j not in used_ir and box_iou(br, bi) > 0.3:
                flag = 2  # both
                used_ir.add(j)
                break
        merged.append((br, flag))
    for j, bi in enumerate(boxes_ir):
        if j not in used_ir:
            merged.append((bi, 1))  # ir only

    # 확장 YOLO 포맷으로 저장 (uncertain은 검수 단계에서 사람이 표기 → 초기 0)
    h, w = img_rgb.shape[:2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec.pair_id}.txt"
    lines = []
    for (x1, y1, x2, y2), flag in merged:
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {flag} 0")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    rec.label_path = str(out_path)
    return out_path


def load_extended_labels(label_path: str | Path, img_w: int, img_h: int):
    """확장 라벨 로드 → (xyxy Nx4, visible_in N, uncertain N)"""
    boxes, vis, unc = [], [], []
    p = Path(label_path)
    if not p.exists():
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0, int)
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        _, cx, cy, w, h = f[:5]
        cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        vis.append(int(f[5]) if len(f) > 5 else 2)
        unc.append(int(f[6]) if len(f) > 6 else 0)
    return np.array(boxes), np.array(vis), np.array(unc)


def to_plain_yolo(ext_label_dir: str | Path, out_dir: str | Path) -> None:
    """확장 포맷 → 표준 YOLO 5열 (ultralytics 학습용). uncertain=1 타깃 제외."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in Path(ext_label_dir).glob("*.txt"):
        keep = []
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) >= 7 and f[6] == "1":
                continue  # 불확실 타깃은 학습에서 제외 (평가 시 ignore 처리 권장)
            keep.append(" ".join(f[:5]))
        (out_dir / p.name).write_text("\n".join(keep) + ("\n" if keep else ""))
