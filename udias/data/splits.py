"""① 누수 방지 분할 — 영상(세션) 단위 층화 분할 + 검사 + 고정 파일 배포.

프레임은 강하게 상관되므로 프레임 단위 랜덤 분할은 누수를 만든다.
→ 같은 video_id 의 모든 프레임은 반드시 같은 split 에 배정한다.
검사: (1) video-level 누수 없음, (2) pHash near-duplicate 가 train/holdout 경계를
넘지 않음. 하나라도 실패하면 스크립트가 학습을 막는다(sys.exit).
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def _video_stratum(recs_of_video, stratify_keys):
    r0 = recs_of_video[0]
    return tuple(getattr(r0, k, "unknown") for k in stratify_keys)


def assign_splits(records, ratios: dict, stratify_keys, seed: int = 42):
    """video_id 단위 그룹핑 후, (time_of_day, scene_type) 층별로 train/val/test 배정.

    표본이 작아 정확한 층화가 어려우면 train 을 우선 채운다(보고서 §4.4의 완화 규칙).
    """
    by_video = defaultdict(list)
    for r in records:
        by_video[r.video_id].append(r)

    by_stratum = defaultdict(list)
    for vid, recs in by_video.items():
        by_stratum[_video_stratum(recs, stratify_keys)].append(vid)

    rng = random.Random(seed)
    tr, va = float(ratios["train"]), float(ratios["val"])
    split_of_video = {}
    for _stratum, vids in by_stratum.items():
        vids = sorted(vids)
        rng.shuffle(vids)
        n = len(vids)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        for i, vid in enumerate(vids):
            split_of_video[vid] = ("train" if i < n_tr
                                   else "val" if i < n_tr + n_va
                                   else "test")
    for r in records:
        r.split = split_of_video[r.video_id]
    return records


def check_video_leakage(records) -> bool:
    """같은 video_id 가 둘 이상의 split 에 걸치면 실패."""
    seen = defaultdict(set)
    for r in records:
        seen[r.video_id].add(r.split)
    bad = {v: s for v, s in seen.items() if len(s) > 1}
    if bad:
        print("[누수] 한 영상이 여러 split 에 배정됨:")
        for v, s in bad.items():
            print(f"   - {v}: {sorted(s)}")
        return False
    print("[검사] video-level 누수 없음 ✓")
    return True


def _phash(img_gray) -> int:
    """32x32 DCT 기반 64-bit perceptual hash."""
    import cv2
    x = cv2.resize(img_gray, (32, 32)).astype(np.float32)
    d = cv2.dct(x)[:8, :8]
    med = float(np.median(d.flatten()[1:]))   # DC 제외 중앙값
    val = 0
    for b in (d > med).flatten():
        val = (val << 1) | int(b)
    return val


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def check_near_duplicates(records, phash_threshold: int = 6) -> bool:
    """train vs holdout(val+test) 프레임의 pHash 해밍거리 <= 임계면 near-duplicate."""
    import cv2
    from .align import imread_unicode

    def phash_of(rec):
        img = imread_unicode(rec.rgb_path)
        if img is None:
            return None
        return _phash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    h_train = [(r.pair_id, phash_of(r)) for r in records if r.split == "train"]
    h_hold = [(r.pair_id, phash_of(r)) for r in records if r.split in ("val", "test")]
    h_train = [(p, h) for p, h in h_train if h is not None]
    h_hold = [(p, h) for p, h in h_hold if h is not None]

    dups = []
    for pid_h, hh in h_hold:
        for pid_t, ht in h_train:
            if _hamming(hh, ht) <= phash_threshold:
                dups.append((pid_t, pid_h))
                break
    if dups:
        print(f"[누수] near-duplicate {len(dups)}쌍 (train↔holdout, pHash≤{phash_threshold}):")
        for t, h in dups[:20]:
            print(f"   - train:{t}  ~  holdout:{h}")
        if len(dups) > 20:
            print(f"   ... 외 {len(dups) - 20}쌍")
        return False
    print(f"[검사] near-duplicate 없음 (pHash≤{phash_threshold}) ✓")
    return True


def export_split_files(records, splits_dir) -> None:
    """고정 split 파일(train/val/test.txt, pair_id 목록) 배포."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        ids = [r.pair_id for r in records if r.split == split]
        (splits_dir / f"{split}.txt").write_text(
            "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    print(f"[배포] 고정 split 파일 → {splits_dir}/(train|val|test).txt")


def summarize(records) -> None:
    by_split = defaultdict(int)
    by_tod = defaultdict(lambda: defaultdict(int))
    by_scene = defaultdict(lambda: defaultdict(int))
    videos = defaultdict(set)
    for r in records:
        by_split[r.split] += 1
        by_tod[r.split][r.time_of_day] += 1
        by_scene[r.split][r.scene_type] += 1
        videos[r.split].add(r.video_id)
    print("\n=== Split 요약 ===")
    for split in ("train", "val", "test"):
        print(f"[{split}] 프레임 {by_split[split]} · 영상 {len(videos[split])}")
        print(f"   time_of_day: {dict(by_tod[split])}")
        print(f"   scene_type : {dict(by_scene[split])}")
    print(f"총 프레임 {sum(by_split.values())} · 총 영상 {sum(len(v) for v in videos.values())}")
