"""데이터카드 통계 그림 생성 (논문 §4.5, M10 크기편중·M14 통계그림).

usage: python scripts/14_dataset_figures.py config/default.yaml [out_dir]

산출 (out_dir 기본 = runs/figures):
  fig-size-dist.pdf     — 선박 크기(면적) 분포 히스토그램 + small/med/large 경계
  fig-align-by-tod.pdf  — 주야별 정렬 성공률 막대 + 재투영 오차 분포
모델 불필요 (매니페스트 + plain 라벨만). 논문 figures/ 로 복사해 쓰면 됨.
"""
import sys, yaml
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")               # 헤드리스 렌더
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode
from udias.eval.dataset_stats import (ship_size_areas, size_bin_counts,
                                      alignment_by_tod)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"))
P, E = cfg["paths"], cfg["eval"]
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(P["outputs_dir"]) / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plain = Path(P["labels_dir"]) / "plain"
records = load_manifest(P["manifest"])

_size = {}
def img_size_lookup(rec):
    if rec.pair_id not in _size:
        im = imread_unicode(rec.rgb_path)
        _size[rec.pair_id] = (im.shape[1], im.shape[0])
    return _size[rec.pair_id]

# ── 그림 1: 크기 분포 (전 split) ────────────────────────────────
areas = ship_size_areas(records, plain, img_size_lookup, split=None)
if len(areas):
    bins = size_bin_counts(areas, E["size_buckets"])
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    side = np.sqrt(areas)                       # sqrt(면적)=대표 변 길이(px), 로그 스케일
    ax.hist(side, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.3)
    for thr, lab in [(32, "small|med (32px)"), (96, "med|large (96px)")]:
        ax.axvline(thr, color="#C44E52", ls="--", lw=1)
        ax.text(thr, ax.get_ylim()[1] * 0.92, lab, rotation=90,
                va="top", ha="right", fontsize=7, color="#C44E52")
    ax.set_xlabel(r"ship size $\sqrt{\mathrm{area}}$ (px)")
    ax.set_ylabel("count")
    ax.set_title(f"Ship-size distribution  (S/M/L = "
                 f"{bins['small']}/{bins['medium']}/{bins['large']})", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig-size-dist.pdf")
    plt.close(fig)
    print(f"[그림] fig-size-dist.pdf  (n={len(areas)}, S/M/L={bins})")
else:
    print("[skip] 크기 분포 — plain 라벨 없음")

# ── 그림 2: 주야별 정렬 성공률 ──────────────────────────────────
tod = alignment_by_tod(records)
tod = {k: v for k, v in tod.items() if v["n"] > 0}
if tod:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 3.0))
    labels = list(tod.keys())
    rates = [tod[k]["rate"] * 100 for k in labels]
    a1.bar(labels, rates, color="#55A868", edgecolor="white")
    for i, (k, r) in enumerate(zip(labels, rates)):
        a1.text(i, r + 1, f"{r:.0f}%\n(n={tod[k]['n']})", ha="center", fontsize=7)
    a1.set_ylim(0, 105); a1.set_ylabel("alignment success (%)")
    a1.set_title("Success rate by time of day", fontsize=8)
    reproj = [tod[k]["reproj"] for k in labels if tod[k]["reproj"]]
    if reproj:
        a2.boxplot(reproj, labels=[k for k in labels if tod[k]["reproj"]],
                   showfliers=False)
        a2.set_ylabel("mean inlier reproj. error (px)")
        a2.set_title("Alignment error by time of day", fontsize=8)
    else:
        a2.text(0.5, 0.5, "no reproj data", ha="center", va="center")
        a2.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "fig-align-by-tod.pdf")
    plt.close(fig)
    print(f"[그림] fig-align-by-tod.pdf  ({ {k: round(v['rate'], 2) for k, v in tod.items()} })")
else:
    print("[skip] 정렬 성공률 — 매니페스트 비어있음")

print(f"\n출력: {out_dir}  (논문 figures/ 로 복사)")
