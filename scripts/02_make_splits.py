"""① : 영상 단위 층화 분할 + 누수 검사 + 고정 split 파일 배포.

usage: python scripts/02_make_splits.py configs/default.yaml
"""
import sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest, save_manifest
from udias.data.splits import (assign_splits, check_video_leakage,
                               check_near_duplicates, export_split_files, summarize)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"))
P, S = cfg["paths"], cfg["split"]

records = load_manifest(P["manifest"])
records = assign_splits(records, S["ratios"], S["stratify_keys"], S["seed"])

ok = check_video_leakage(records)
if S.get("near_duplicate_check", False):
    ok = check_near_duplicates(records, S["phash_threshold"]) and ok
if not ok:
    print("\n!! 누수 검사 실패 — 위 로그를 해결하기 전까지 학습 금지 !!")
    sys.exit(1)

save_manifest(records, P["manifest"])
export_split_files(records, P["splits_dir"])
summarize(records)
