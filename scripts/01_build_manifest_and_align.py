"""① 절반 + ② : 페어 스캔 → 매니페스트 생성 → 정렬(H 캐싱).

usage: python scripts/01_build_manifest_and_align.py configs/default.yaml
       python scripts/01_build_manifest_and_align.py configs/default.yaml --realign
  --realign: 기존 매니페스트를 읽어 '정렬 필드만' 갱신 (label_path/split 등
             03/02 가 채운 필드 보존). 정렬 method 교체 후 재실행용.
"""
import sys, yaml, json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import build_pairs, load_manifest, save_manifest
from udias.data.align import align_record
from udias.eval.align_metrics import alignment_report

args = [a for a in sys.argv[1:] if not a.startswith("--")]
REALIGN = "--realign" in sys.argv
cfg = yaml.safe_load(open(args[0] if args else "configs/default.yaml", encoding="utf-8"))
P, S = cfg["paths"], cfg["split"]

if REALIGN:
    records = load_manifest(P["manifest"])
    print(f"--realign: 기존 매니페스트 {len(records)}개 레코드의 정렬 필드만 갱신 "
          f"(method={cfg['align'].get('method', 'sift')})")
else:
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
