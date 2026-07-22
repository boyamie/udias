"""⑥ 탐지 평가 프로토콜 — mAP@[.5:.95], 크기별(S/M/L), 장면유형별, 다중 시드.

ultralytics 내장 지표 대신 예측을 COCO 포맷으로 모아 pycocotools로 평가한다.
이유: (1) late/middle fusion 등 프레임워크 밖 모델과 동일 기준 비교,
     (2) 장면유형별(주/야, 근해/항만) 서브셋 평가, (3) uncertain 타깃 ignore 처리.

의존성: pip install pycocotools
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_coco_gt(records, plain_label_dir, img_size_lookup, split="test",
                  subset_key=None, subset_val=None):
    """매니페스트 + 라벨 → COCO GT dict. subset_key로 장면유형 필터."""
    images, annotations = [], []
    ann_id = 1
    for i, rec in enumerate(r for r in records if r.split == split):
        if subset_key and getattr(rec, subset_key) != subset_val:
            continue
        w, h = img_size_lookup(rec)
        images.append({"id": i, "file_name": rec.pair_id, "width": w, "height": h,
                       "pair_id": rec.pair_id})
        lbl = Path(plain_label_dir) / f"{rec.pair_id}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                f = line.split()
                if len(f) < 5:
                    continue
                cx, cy, bw, bh = (float(f[1]) * w, float(f[2]) * h,
                                  float(f[3]) * w, float(f[4]) * h)
                annotations.append({
                    "id": ann_id, "image_id": i, "category_id": 1,
                    "bbox": [cx - bw / 2, cy - bh / 2, bw, bh],
                    "area": bw * bh, "iscrowd": 0})
                ann_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "Ship"}]}


def evaluate_coco(gt_dict: dict, predictions: list[dict]) -> dict:
    """predictions: [{"image_id", "category_id":1, "bbox":[x,y,w,h], "score"}]
    반환: mAP, mAP50, AP_small/medium/large, AR
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f)
        gt_path = f.name
    coco_gt = COCO(gt_path)
    if not predictions:
        return {k: 0.0 for k in ("mAP", "mAP50", "AP_small", "AP_medium", "AP_large", "AR100")}
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP": s[0], "mAP50": s[1], "AP_small": s[3], "AP_medium": s[4],
            "AP_large": s[5], "AR100": s[8]}


def evaluate_by_scene(records, plain_label_dir, img_size_lookup,
                      predictions_by_pair: dict[str, list], report_by=("time_of_day",)):
    """전체 + 장면유형별 분리 리포트. 보고서 7장:
    'results should be reported by scene type, not only as a single overall number.'
    """
    results = {}

    def run(subset_key=None, subset_val=None, tag="overall"):
        gt = build_coco_gt(records, plain_label_dir, img_size_lookup,
                           subset_key=subset_key, subset_val=subset_val)
        id_of = {im["pair_id"]: im["id"] for im in gt["images"]}
        preds = []
        for pid, plist in predictions_by_pair.items():
            if pid in id_of:
                for p in plist:
                    preds.append({**p, "image_id": id_of[pid], "category_id": 1})
        results[tag] = evaluate_coco(gt, preds)

    run()
    for key in report_by:
        for val in sorted({getattr(r, key) for r in records if r.split == "test"}):
            run(key, val, tag=f"{key}={val}")
    return results


def aggregate_seeds(per_seed_results: list[dict]) -> dict:
    """시드별 결과 → mean ± std. 보고서 7장 통계 보고 요구사항."""
    out = {}
    for tag in per_seed_results[0]:
        out[tag] = {}
        for metric in per_seed_results[0][tag]:
            vals = [r[tag][metric] for r in per_seed_results]
            out[tag][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def format_benchmark_table(results_by_model: dict[str, dict]) -> str:
    """모델별 결과 → 논문용 마크다운 표."""
    metrics = ["mAP", "mAP50", "AP_small", "AP_medium", "AP_large"]
    lines = ["| Model | " + " | ".join(metrics) + " |",
             "|" + "---|" * (len(metrics) + 1)]
    for model, res in results_by_model.items():
        ov = res["overall"]
        cells = []
        for m in metrics:
            v = ov[m]
            cells.append(f"{v['mean']:.3f}±{v['std']:.3f}" if isinstance(v, dict)
                         else f"{v:.3f}")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
