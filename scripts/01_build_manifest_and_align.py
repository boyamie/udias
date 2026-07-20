"""① 절반 + ② : 페어 스캔 → 매니페스트 생성 → 정렬(H 캐싱).

usage: python scripts/01_build_manifest_and_align.py configs/default.yaml
"""
import sys, yaml, json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import build_pairs, save_manifest
from udias.data.align import align_record
from udias.eval.align_metrics import alignment_report

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"))
P, S = cfg["paths"], cfg["split"]

records = build_pairs(P["rgb_dir"], P["ir_dir"],
                      S["video_id_regex"], S["video_id_template"])
print(f"페어 후보: {len(records)}개")

for rec in tqdm(records, desc="align (SIFT+RANSAC)"):
    align_record(rec, cfg["align"])

save_manifest(records, P["manifest"])
rep = alignment_report(records)
print(json.dumps(rep, indent=2, ensure_ascii=False))
Path(P["manifest"]).with_suffix(".align_report.json").write_text(
    json.dumps(rep, indent=2, ensure_ascii=False))
