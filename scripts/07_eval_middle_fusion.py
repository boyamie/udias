"""⑥ : middle fusion 을 다른 베이스라인과 '동일 COCO 기준'으로 평가 → benchmark.json 합류.

usage: python scripts/07_eval_middle_fusion.py config/default.yaml
scripts/05 가 만든 benchmark.json 에 middle_fusion 행을 추가한다.
"""
import sys
import json
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode
from udias.data.pair_dataset import PairDataset, collate
from udias.fusion.middle_fusion import MiddleFusionDetector
from udias.eval.det_metrics import (evaluate_by_scene, aggregate_seeds,
                                    format_benchmark_table)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"))
P, T, E = cfg["paths"], cfg["train"], cfg["eval"]
M = cfg.get("middle", {})
img = T["img_size"]
plain = Path(P["labels_dir"]) / "plain"
records = [r for r in load_manifest(P["manifest"]) if r.split == "test"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_size = {}
def img_size_lookup(rec):
    if rec.pair_id not in _size:
        im = imread_unicode(rec.rgb_path)
        _size[rec.pair_id] = (im.shape[1], im.shape[0])
    return _size[rec.pair_id]

per_seed = []
for seed in T["seeds"]:
    w = Path(P["outputs_dir"]) / "middle_fusion" / f"seed{seed}" / "weights" / "best.pt"
    model = MiddleFusionDetector(M.get("backbone", "resnet18"), 1, pretrained=False).to(device)
    model.load_state_dict(torch.load(w, map_location=device))
    model.eval()

    dl = DataLoader(PairDataset(records, plain, img, "test"),
                    batch_size=T["batch_size"], shuffle=False, collate_fn=collate)
    preds = {}
    with torch.no_grad():
        for rgb, ir, ir_valid, targets, meta in dl:
            res = model.predict(rgb.to(device), ir.to(device), ir_valid.to(device),
                                score_thr=0.05, nms_thr=cfg["fusion"]["late"]["iou_thr"])
            for (boxes, scores), m in zip(res, meta):
                h0, w0 = m["orig_hw"]
                sx, sy = w0 / img, h0 / img       # 리사이즈 좌표 → 원본 좌표
                preds[m["pair_id"]] = [
                    {"bbox": [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy],
                     "score": float(s)}
                    for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist())]
    per_seed.append(evaluate_by_scene(records, plain, img_size_lookup, preds,
                                      tuple(E["report_by"])))

agg = aggregate_seeds(per_seed)
out = Path(P["outputs_dir"]) / "benchmark.json"
allr = json.loads(out.read_text()) if out.exists() else {}
allr["middle_fusion"] = agg
out.write_text(json.dumps(allr, indent=2))
print(format_benchmark_table(allr))
print(f"\nmiddle_fusion 행 추가 → {out}")
