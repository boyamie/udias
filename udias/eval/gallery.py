"""정성 갤러리 — 원인별 탐지 오류 예시 격자 렌더 (논문 §5.4, M14).

논문: "Qualitative analysis groups errors by cause --- waves, docks, glare,
duplicate boxes, thermal crossover --- with representative examples."

오류 원인은 자동 판정이 어려우므로, 사람이 pair_id → 태그를 지정한 셀렉션 파일
(JSON: {"waves": ["p001", ...], "glare": [...], ...})을 입력으로 받아 각 태그의
대표 예시를 격자로 렌더한다. 각 셀에 GT(녹색)·예측(빨강) 박스를 그린다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def draw_boxes(ax, img, gt_boxes, pred_boxes):
    """한 축에 이미지 + GT(녹)·예측(빨강, 점선) 박스."""
    from matplotlib.patches import Rectangle
    ax.imshow(img)
    for b in gt_boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               fill=False, edgecolor="#2ca02c", linewidth=1.2))
    for b in pred_boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               fill=False, edgecolor="#d62728", linewidth=1.0,
                               linestyle="--"))
    ax.set_xticks([]); ax.set_yticks([])


def render_gallery(selection: dict, load_image, gt_lookup, pred_lookup,
                   out_path, per_tag: int = 3):
    """원인 태그별 대표 예시 격자 → out_path (PDF).

    selection: {tag: [pair_id, ...]}
    load_image(pair_id) -> HxWx3 RGB (또는 None)
    gt_lookup(pair_id) / pred_lookup(pair_id) -> (N,4) xyxy 절대좌표
    반환: 실제 렌더된 (행=태그, 셀) 개수 dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t, ids in selection.items() if ids]
    if not tags:
        return {}
    ncol = per_tag
    fig, axes = plt.subplots(len(tags), ncol,
                             figsize=(ncol * 2.2, len(tags) * 2.2),
                             squeeze=False)
    rendered = {}
    for r, tag in enumerate(tags):
        ids = list(selection[tag])[:ncol]
        rendered[tag] = 0
        for c in range(ncol):
            ax = axes[r][c]
            if c < len(ids) and (img := load_image(ids[c])) is not None:
                draw_boxes(ax, img, gt_lookup(ids[c]), pred_lookup(ids[c]))
                rendered[tag] += 1
                if c == 0:
                    ax.set_ylabel(tag, fontsize=8, rotation=90, labelpad=4)
            else:
                ax.axis("off")
    fig.suptitle("Qualitative errors by cause  (GT green, prediction red-dashed)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return rendered
