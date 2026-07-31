"""⑯ : 정성 비교 figure — 같은 장면에서 RGB-only / IR-only / 4ch fusion 검출 차이.

논문 Figure 2 교체용: "fusion 이 단독 모달이 놓치는 표적을 잡는다"를
한 장면에서 시각적으로 보여준다. 학습 산출물(best.pt, seed0)만 필요 —
추가 학습 없음, 실행 ~수 분.

usage: python scripts/16_qualitative_comparison.py config/default.yaml
출력: <outputs_dir>/figures/fig-qual-comparison.png / .pdf

선택 로직(결정론):
  aligned 테스트 페어 전체에 3개 모델을 돌리고, GT 매칭 히트 수 기준으로
  (fusion 히트 − RGB 히트)가 최대인 야간 프레임 1개 + 주간 프레임 1개를 고른다.
  → 2행(주간/야간) × 3열(RGB-only / IR-only / fusion) 그리드, GT 초록·예측 빨강.
"""
import sys, yaml
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode, warp_ir_to_rgb
from udias.fusion.early import stack4

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml", encoding="utf-8"))
P = cfg["paths"]
plain_labels = Path(P["labels_dir"]) / "plain"
out_dir = Path(P["outputs_dir"]) / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

CONF = 0.25          # 표시용 신뢰도 임계값
IOU_HIT = 0.5        # GT 매칭 판정


def load_gt(rec, w, h):
    lbl = plain_labels / f"{rec.pair_id}.txt"
    boxes = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
    return boxes


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def hits(preds, gts):
    used, n = set(), 0
    for p in preds:
        for gi, g in enumerate(gts):
            if gi not in used and iou(p, g) >= IOU_HIT:
                used.add(gi)
                n += 1
                break
    return n


def predict(model, img_or_path):
    r = model.predict(img_or_path, conf=CONF, verbose=False)[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append([x1, y1, x2 - x1, y2 - y1])
    return out


def main():
    w_root = Path(P["outputs_dir"])
    models = {}
    for name in ("rgb_only", "ir_only", "early_stack4"):
        w = w_root / name / "seed0" / "weights" / "best.pt"
        if not w.exists():
            sys.exit(f"[abort] {w} 없음 — 학습 완료 후 실행하세요")
        models[name] = YOLO(str(w))

    records = [r for r in load_manifest(P["manifest"])
               if r.split == "test" and r.aligned]
    cache_dir = w_root / "eval_cache" / "qual_stack4"
    cache_dir.mkdir(parents=True, exist_ok=True)

    scored = []          # (gain, tod, rec, inputs, preds, gts)
    for rec in records:
        rgb = imread_unicode(rec.rgb_path)
        ir_raw = imread_unicode(rec.ir_path)
        if rgb is None or ir_raw is None:
            continue
        ir = warp_ir_to_rgb(rec, ir_raw, rgb.shape)
        ir3 = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir
        tiff = cache_dir / f"{rec.pair_id}.tiff"
        if not tiff.exists():
            cv2.imwrite(str(tiff), stack4(rgb, ir))
        gts = load_gt(rec, rgb.shape[1], rgb.shape[0])
        if not gts:
            continue
        pr = {"rgb_only": predict(models["rgb_only"], rgb),
              "ir_only": predict(models["ir_only"], ir3),
              "early_stack4": predict(models["early_stack4"], str(tiff))}
        gain = hits(pr["early_stack4"], gts) - hits(pr["rgb_only"], gts)
        tod = getattr(rec, "time_of_day", "day")
        scored.append((gain, tod, rec, {"rgb": rgb, "ir": ir3}, pr, gts))

    if not scored:
        sys.exit("[abort] 후보 페어 없음")

    def pick(tod):
        cand = [s for s in scored if s[1] == tod]
        return max(cand, key=lambda s: s[0]) if cand else None

    rows = [p for p in (pick("day"), pick("night")) if p]
    print("선택된 프레임:")
    for gain, tod, rec, *_ in rows:
        print(f"  {tod}: {rec.pair_id} (fusion−RGB 히트 이득 {gain:+d})")

    # ---- 렌더링: 행=장면, 열=모델. GT 초록, 예측 빨강 ----
    PANEL_W = 640
    col_names = [("rgb_only", "RGB-only"), ("ir_only", "IR-only (warped)"),
                 ("early_stack4", "Early fusion (4-ch)")]
    grid_rows = []
    for gain, tod, rec, inputs, pr, gts in rows:
        panels = []
        for key, title in col_names:
            base = inputs["ir"] if key == "ir_only" else inputs["rgb"]
            img = base.copy()
            for g in gts:
                x, y, bw, bh = (int(v) for v in g)
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 200, 0), max(2, img.shape[1] // 640))
            for p in pr[key]:
                x, y, bw, bh = (int(v) for v in p)
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), max(2, img.shape[1] // 640))
            s = PANEL_W / img.shape[1]
            img = cv2.resize(img, (PANEL_W, round(img.shape[0] * s)))
            cv2.putText(img, f"{title}  ({tod})", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, f"{title}  ({tod})", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            panels.append(img)
        h = min(p.shape[0] for p in panels)
        grid_rows.append(np.hstack([p[:h] for p in panels]))

    w = min(r.shape[1] for r in grid_rows)
    grid = np.vstack([r[:, :w] for r in grid_rows])
    png = out_dir / "fig-qual-comparison.png"
    cv2.imwrite(str(png), grid)
    print(f"저장: {png}")
    try:                       # PDF 는 matplotlib 있으면 함께 (벡터 아님, 삽입 편의용)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4 * len(grid_rows)), dpi=200)
        plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        fig.tight_layout(pad=0.1)
        fig.savefig(out_dir / "fig-qual-comparison.pdf", bbox_inches="tight")
        print(f"저장: {out_dir / 'fig-qual-comparison.pdf'}")
    except ImportError:
        print("(matplotlib 없음 — PNG 만 저장)")


if __name__ == "__main__":
    main()
