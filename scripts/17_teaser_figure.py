"""⑰ : 1페이지 티저 figure — 데이터셋의 novelty 를 한 줄에 보여준다.

구성(가로 4패널, acmart teaserfigure 전폭용):
  [RGB + GT]  [warped IR + 투영 라벨]  [체커보드 오버레이(정렬 증빙)]  [릴리스 카드]
릴리스 카드: 페어 수·주/야·homography+품질 메타데이터·visibility 라벨·CC BY-NC,
"road-scene RGB-T 는 많지만 maritime 은 공개 정렬 페어가 없었다" 포지셔닝 문구.

usage: python scripts/17_teaser_figure.py config/default.yaml [pair_id]
  pair_id 생략 시: 주간·aligned·GT 2개 이상·inlier 최다 페어 자동 선정.
출력: <outputs_dir>/figures/fig-teaser.png / .pdf
"""
import sys, yaml
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode, warp_ir_to_rgb

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml", encoding="utf-8"))
P = cfg["paths"]
plain_labels = Path(P["labels_dir"]) / "plain"
out_dir = Path(P["outputs_dir"]) / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
PICK = sys.argv[2] if len(sys.argv) > 2 else None

PANEL_H = 480          # 패널 공통 높이(px)
GREEN, ORANGE = (60, 190, 60), (0, 150, 255)


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


def draw_boxes(img, boxes, color):
    t = max(2, img.shape[1] // 700)
    for b in boxes:
        x, y, w, h = (int(v) for v in b)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, t)
    return img


def fit(img):
    s = PANEL_H / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * s), PANEL_H))


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (30, 30, 30), -1)
    cv2.putText(img, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def checkerboard(a, b, n=8):
    h, w = a.shape[:2]
    out = a.copy()
    ys, xs = h // n, w // n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                out[i * ys:(i + 1) * ys, j * xs:(j + 1) * xs] = \
                    b[i * ys:(i + 1) * ys, j * xs:(j + 1) * xs]
    return out


def release_card(h, w):
    card = np.full((h, w, 3), 248, np.uint8)
    cv2.rectangle(card, (0, 0), (w - 1, h - 1), (120, 120, 120), 2)
    lines = [
        ("Maritime RGB-T Alignment Dataset", 0.95, (20, 20, 20), 2),
        ("", 0.5, (0, 0, 0), 1),
        ("9,660 paired RGB+thermal frames", 0.72, (30, 30, 30), 2),
        ("22 videos  ·  day + night  ·  open water,", 0.62, (60, 60, 60), 1),
        ("nearshore, bridge", 0.62, (60, 60, 60), 1),
        ("reference homography H per pair", 0.62, (60, 60, 60), 1),
        ("alignment-quality + visibility labels", 0.62, (60, 60, 60), 1),
        ("leakage-controlled fixed splits", 0.62, (60, 60, 60), 1),
        ("", 0.5, (0, 0, 0), 1),
        ("Road RGB-T benchmarks exist;", 0.62, (30, 30, 30), 1),
        ("maritime autonomy had no public", 0.62, (30, 30, 30), 1),
        ("aligned pairs - until now.", 0.62, (30, 30, 30), 1),
        ("", 0.5, (0, 0, 0), 1),
        ("OPEN RELEASE  ·  CC BY-NC 4.0", 0.7, (150, 60, 0), 2),
        ("github.com/boyamie/udias", 0.62, (120, 60, 0), 1),
    ]
    y = 52
    for text, scale, color, th in lines:
        if text:
            cv2.putText(card, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, th, cv2.LINE_AA)
        y += int(34 * scale + 10)
    return card


def main():
    records = [r for r in load_manifest(P["manifest"])
               if r.aligned and r.time_of_day == "day"]
    if PICK:
        records = [r for r in records if r.pair_id == PICK]
        if not records:
            sys.exit(f"[abort] pair_id {PICK} 없음/비정렬")

    best, best_key = None, (-1, -1)
    for rec in records:
        rgb = imread_unicode(rec.rgb_path)
        if rgb is None:
            continue
        gts = load_gt(rec, rgb.shape[1], rgb.shape[0])
        key = (min(len(gts), 4), rec.align_num_inliers or 0)
        if len(gts) >= 2 and key > best_key:
            best, best_key, best_rgb, best_gts = rec, key, rgb, gts
        if PICK:
            best, best_rgb, best_gts = rec, rgb, gts
            break
    if best is None:
        sys.exit("[abort] 조건(주간·aligned·GT>=2) 페어 없음")
    rec, rgb, gts = best, best_rgb, best_gts
    print(f"선택: {rec.pair_id} (GT {len(gts)}, inliers {rec.align_num_inliers})")

    ir = warp_ir_to_rgb(rec, imread_unicode(rec.ir_path), rgb.shape)
    ir3 = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir
    ir_vis = cv2.applyColorMap(cv2.cvtColor(ir3, cv2.COLOR_BGR2GRAY),
                               cv2.COLORMAP_INFERNO)

    p1 = label(fit(draw_boxes(rgb.copy(), gts, GREEN)), "Visible (RGB) + labels")
    p2 = label(fit(draw_boxes(ir_vis.copy(), gts, ORANGE)),
               "Thermal (warped by H) + projected labels")
    p3 = label(fit(checkerboard(rgb, ir_vis)), "Alignment check (checkerboard)")
    used = p1.shape[1] + p2.shape[1] + p3.shape[1]
    p4 = release_card(PANEL_H, max(430, int(used * 0.28)))

    gap = np.full((PANEL_H, 6, 3), 255, np.uint8)
    strip = np.hstack([p1, gap, p2, gap, p3, gap, p4])
    png = out_dir / "fig-teaser.png"
    cv2.imwrite(str(png), strip)
    print(f"저장: {png}  ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
