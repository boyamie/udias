"""⑥ : 전 모델(단일모달/early/late) 통일 평가 → 장면별·크기별·시드 집계 → 벤치마크 표.

usage: python scripts/05_eval_benchmark.py configs/default.yaml
"""
import sys, yaml, json, cv2
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode, warp_ir_to_rgb
from udias.fusion.late import predict_pair_late
from udias.fusion.early import pixel_fusion
from udias.eval.det_metrics import (evaluate_by_scene, aggregate_seeds,
                                    format_benchmark_table)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"))
P, T, E = cfg["paths"], cfg["train"], cfg["eval"]
plain_labels = Path(P["labels_dir"]) / "plain"
records = [r for r in load_manifest(P["manifest"]) if r.split == "test"]

_size_cache = {}
def img_size_lookup(rec):
    if rec.pair_id not in _size_cache:
        im = imread_unicode(rec.rgb_path)
        _size_cache[rec.pair_id] = (im.shape[1], im.shape[0])
    return _size_cache[rec.pair_id]

from ultralytics import YOLO

def predict_all(model, make_input):
    preds = defaultdict(list)
    for rec in records:
        img = make_input(rec)
        if img is None:
            continue
        r = model.predict(img, conf=0.001, verbose=False)[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds[rec.pair_id].append(
                    {"bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(b.conf[0])})
    return preds

def in_rgb(rec):
    return imread_unicode(rec.rgb_path)

def in_ir(rec):
    ir = imread_unicode(rec.ir_path)
    rgb = imread_unicode(rec.rgb_path)
    ir = (warp_ir_to_rgb(rec, ir, rgb.shape) if rec.aligned
          else cv2.resize(ir, (rgb.shape[1], rgb.shape[0])))
    return cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir

def in_early(rec):
    if not rec.aligned:
        return None
    rgb = imread_unicode(rec.rgb_path)
    ir = warp_ir_to_rgb(rec, imread_unicode(rec.ir_path), rgb.shape)
    return pixel_fusion(rgb, ir, cfg["fusion"]["early"]["epsilon"])

MODELS = {"rgb_only": in_rgb, "ir_only": in_ir, "early_aligned": in_early}
all_results = {}
for name, make_input in MODELS.items():
    per_seed = []
    for seed in T["seeds"]:
        w = Path(P["outputs_dir"]) / name / f"seed{seed}" / "weights" / "best.pt"
        preds = predict_all(YOLO(str(w)), make_input)
        per_seed.append(evaluate_by_scene(records, plain_labels, img_size_lookup,
                                          preds, tuple(E["report_by"])))
    all_results[name] = aggregate_seeds(per_seed)

# late fusion: rgb/ir 시드 쌍으로 앙상블
per_seed = []
for seed in T["seeds"]:
    m_rgb = YOLO(str(Path(P["outputs_dir"]) / "rgb_only" / f"seed{seed}" / "weights" / "best.pt"))
    m_ir  = YOLO(str(Path(P["outputs_dir"]) / "ir_only"  / f"seed{seed}" / "weights" / "best.pt"))
    preds = defaultdict(list)
    for rec in records:
        boxes, scores = predict_pair_late(rec, m_rgb, m_ir, cfg["fusion"]["late"],
                                          imread_unicode, warp_ir_to_rgb)
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            preds[rec.pair_id].append({"bbox": [x1, y1, x2 - x1, y2 - y1],
                                       "score": float(s)})
    per_seed.append(evaluate_by_scene(records, plain_labels, img_size_lookup,
                                      preds, tuple(E["report_by"])))
all_results["late_fusion"] = aggregate_seeds(per_seed)

out = Path(P["outputs_dir"]) / "benchmark.json"
out.write_text(json.dumps(all_results, indent=2))
print(format_benchmark_table(all_results))
print(f"\n상세 결과: {out}")
