"""정성 갤러리 생성 — 원인별 오류 예시 격자 (논문 §5.4, M14).

usage: python scripts/15_qualitative_gallery.py config/default.yaml \
           <selection.json> <predictions.json> [out.pdf]

selection.json : {"waves": ["pair_id", ...], "docks": [...], "glare": [...],
                  "duplicate": [...], "crossover": [...]}  (사람이 지정)
predictions.json: {pair_id: [{"bbox":[x,y,w,h], "score":...}, ...]}  (scripts/05·07 산출을
                  저장해두면 재사용; 없으면 GT만 그려짐)

각 셀에 GT(녹색)·예측(빨강 점선) 박스를 그려 원인별로 비교한다.
"""
import sys, yaml, json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode
from udias.eval.align_metrics import load_plain_boxes
from udias.eval.gallery import render_gallery

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
cfg = yaml.safe_load(open(sys.argv[1]))
P = cfg["paths"]
selection = json.loads(Path(sys.argv[2]).read_text())
preds = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(P["outputs_dir"]) / "figures" / "fig-qualitative-gallery.pdf"

plain = Path(P["labels_dir"]) / "plain"
records = {r.pair_id: r for r in load_manifest(P["manifest"])}

def load_image(pid):
    rec = records.get(pid)
    if rec is None:
        return None
    im = imread_unicode(rec.rgb_path)
    return im[:, :, ::-1] if im is not None else None      # BGR→RGB

def gt_lookup(pid):
    rec = records.get(pid)
    if rec is None:
        return np.zeros((0, 4))
    im = imread_unicode(rec.rgb_path)
    if im is None:
        return np.zeros((0, 4))
    return load_plain_boxes(plain / f"{pid}.txt", im.shape[1], im.shape[0])

def pred_lookup(pid):
    out = []
    for p in preds.get(pid, []):
        x, y, w, h = p["bbox"]
        out.append([x, y, x + w, y + h])
    return np.asarray(out, dtype=float).reshape(-1, 4)

rendered = render_gallery(selection, load_image, gt_lookup, pred_lookup, out,
                          per_tag=int(cfg.get("gallery", {}).get("per_tag", 3)))
print(f"[갤러리] {out}  렌더: {rendered}")
if not preds:
    print("[주의] predictions 인자 없음 → GT 박스만 표시됨.")
