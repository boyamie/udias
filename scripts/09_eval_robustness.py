"""⑥ : 강건성 벤치마크 — 시나리오(모달 제거·IR 열화) × 전 베이스라인 (논문 §5.4).

usage: python scripts/09_eval_robustness.py config/default.yaml

시나리오는 udias.eval.robustness 의 결정론적 변환. 학습된 가중치가 없는 모델은
건너뛰고 표시한다. 결과는 runs/robustness.json:
  {model: {scenario: aggregate_seeds(...)}}
day/night/harbor 서브셋은 각 셀 안의 report_by 분리 리포트로 이미 포함된다.
"""
import sys, yaml, json, cv2
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode, warp_ir_to_rgb
from udias.fusion.early import pixel_fusion, stack4
from udias.fusion.late import wbf_merge
from udias.eval.robustness import SCENARIOS, apply_scenario
from udias.eval.det_metrics import (evaluate_by_scene, aggregate_seeds,
                                    format_benchmark_table)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"))
P, T, E = cfg["paths"], cfg["train"], cfg["eval"]
R = cfg.get("robustness", {})
scenarios = tuple(R.get("scenarios", SCENARIOS))
plain_labels = Path(P["labels_dir"]) / "plain"
records = [r for r in load_manifest(P["manifest"]) if r.split == "test"]
outputs = Path(P["outputs_dir"])

_size_cache = {}
def img_size_lookup(rec):
    if rec.pair_id not in _size_cache:
        im = imread_unicode(rec.rgb_path)
        _size_cache[rec.pair_id] = (im.shape[1], im.shape[0])
    return _size_cache[rec.pair_id]


def transformed_pair(rec, scen):
    """디스크 원본 → (변환된 rgb, 변환된 warped-IR(RGB 좌표계), ir_valid)."""
    rgb = imread_unicode(rec.rgb_path)
    ir = imread_unicode(rec.ir_path)
    ir = (warp_ir_to_rgb(rec, ir, rgb.shape) if rec.aligned
          else cv2.resize(ir, (rgb.shape[1], rgb.shape[0])))
    return apply_scenario(scen, rec, rgb, ir, bool(rec.aligned), R)


def weights(name, seed):
    w = outputs / name / f"seed{seed}" / "weights" / "best.pt"
    return w if w.exists() else None


# ── ultralytics 계열 (단일모달 + early) ──────────────────────────
def in_rgb(rec, scen):
    rgb, _, _ = transformed_pair(rec, scen)
    return rgb

def in_ir(rec, scen):
    _, ir, _ = transformed_pair(rec, scen)
    return cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir

def in_early_pixel(rec, scen):
    if not rec.aligned:
        return None
    rgb, ir, _ = transformed_pair(rec, scen)
    return pixel_fusion(rgb, ir, cfg["fusion"]["early"]["epsilon"])

def in_stack4(rec, scen):
    # 4ch 는 numpy predict 전처리가 채널을 뒤집으므로 TIFF 경로로 추론 (05와 동일)
    if not rec.aligned:
        return None
    p = outputs / "eval_cache" / f"stack4_rob_{scen}" / f"{rec.pair_id}.tiff"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        rgb, ir, _ = transformed_pair(rec, scen)
        cv2.imwrite(str(p), stack4(rgb, ir))
    return str(p)

ULTRA_MODELS = {"rgb_only": in_rgb, "ir_only": in_ir,
                "early_stack4": in_stack4, "early_pixel": in_early_pixel}

all_results = {}
from ultralytics import YOLO

for name, make_input in ULTRA_MODELS.items():
    if weights(name, T["seeds"][0]) is None:
        print(f"[skip] {name}: 가중치 없음")
        continue
    all_results[name] = {}
    for scen in scenarios:
        per_seed = []
        for seed in T["seeds"]:
            model = YOLO(str(weights(name, seed)))
            preds = defaultdict(list)
            for rec in records:
                img = make_input(rec, scen)
                if img is None:
                    continue
                r = model.predict(img, conf=0.001, verbose=False)[0]
                if r.boxes is not None:
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        preds[rec.pair_id].append(
                            {"bbox": [x1, y1, x2 - x1, y2 - y1],
                             "score": float(b.conf[0])})
            per_seed.append(evaluate_by_scene(records, plain_labels,
                                              img_size_lookup, preds,
                                              tuple(E["report_by"])))
        all_results[name][scen] = aggregate_seeds(per_seed)
        print(f"[{name}][{scen}] 완료")

# ── late fusion (rgb/ir 시드 쌍 WBF) ─────────────────────────────
if weights("rgb_only", T["seeds"][0]) and weights("ir_only", T["seeds"][0]):
    all_results["late_fusion"] = {}
    L = cfg["fusion"]["late"]
    for scen in scenarios:
        per_seed = []
        for seed in T["seeds"]:
            m_rgb = YOLO(str(weights("rgb_only", seed)))
            m_ir = YOLO(str(weights("ir_only", seed)))
            preds = defaultdict(list)
            for rec in records:
                rgb, ir, _ = transformed_pair(rec, scen)
                if ir.ndim == 2:
                    ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
                h, w = rgb.shape[:2]

                def run(model, img):
                    r = model.predict(img, conf=L["conf_thr"], verbose=False)[0]
                    if r.boxes is None or len(r.boxes) == 0:
                        return np.zeros((0, 4)), np.zeros(0)
                    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

                b_rgb, s_rgb = run(m_rgb, rgb)
                b_ir, s_ir = run(m_ir, ir)
                boxes, scores = wbf_merge([b_rgb, b_ir], [s_rgb, s_ir], w, h,
                                          iou_thr=L["iou_thr"])
                for (x1, y1, x2, y2), s in zip(boxes, scores):
                    preds[rec.pair_id].append(
                        {"bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
            per_seed.append(evaluate_by_scene(records, plain_labels,
                                              img_size_lookup, preds,
                                              tuple(E["report_by"])))
        all_results["late_fusion"][scen] = aggregate_seeds(per_seed)
        print(f"[late_fusion][{scen}] 완료")
else:
    print("[skip] late_fusion: rgb/ir 가중치 없음")

# ── middle fusion (PairDataset transform 훅) ─────────────────────
mw = outputs / "middle_fusion" / f"seed{T['seeds'][0]}" / "weights" / "best.pt"
if mw.exists():
    import torch
    from torch.utils.data import DataLoader
    from udias.data.pair_dataset import PairDataset, collate
    from udias.fusion.middle_fusion import MiddleFusionDetector

    M = cfg.get("middle", {})
    img = T["img_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results["middle_fusion"] = {}
    for scen in scenarios:
        per_seed = []
        for seed in T["seeds"]:
            w = outputs / "middle_fusion" / f"seed{seed}" / "weights" / "best.pt"
            model = MiddleFusionDetector(M.get("backbone", "resnet18"), 1,
                                         pretrained=False).to(device)
            model.load_state_dict(torch.load(w, map_location=device))
            model.eval()
            dl = DataLoader(
                PairDataset(records, plain_labels, img, "test",
                            transform=lambda rec, rgb, ir, v, _s=scen:
                                apply_scenario(_s, rec, rgb, ir, v, R)),
                batch_size=T["batch_size"], shuffle=False, collate_fn=collate)
            preds = {}
            with torch.no_grad():
                for rgb, ir, ir_valid, targets, meta in dl:
                    res = model.predict(rgb.to(device), ir.to(device),
                                        ir_valid.to(device), score_thr=0.05,
                                        nms_thr=cfg["fusion"]["late"]["iou_thr"])
                    for (boxes, scores), m in zip(res, meta):
                        h0, w0 = m["orig_hw"]
                        sx, sy = w0 / img, h0 / img
                        preds[m["pair_id"]] = [
                            {"bbox": [x1 * sx, y1 * sy,
                                      (x2 - x1) * sx, (y2 - y1) * sy],
                             "score": float(s)}
                            for (x1, y1, x2, y2), s
                            in zip(boxes.tolist(), scores.tolist())]
            per_seed.append(evaluate_by_scene(records, plain_labels,
                                              img_size_lookup, preds,
                                              tuple(E["report_by"])))
        all_results["middle_fusion"][scen] = aggregate_seeds(per_seed)
        print(f"[middle_fusion][{scen}] 완료")
else:
    print("[skip] middle_fusion: 가중치 없음")

out = outputs / "robustness.json"
out.write_text(json.dumps(all_results, indent=2))
print()
for name, by_scen in all_results.items():
    print(format_benchmark_table({f"{name}@{s}": r for s, r in by_scen.items()}))
    print()
print(f"상세 결과: {out}")
