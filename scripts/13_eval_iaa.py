"""주석 품질 — 이중 주석 서브셋 선정 + 주석자 간 일치도(IAA) 산출 (§4.3, B3).

usage: python scripts/13_eval_iaa.py config/default.yaml

동작:
  1. (scene_type, time_of_day) 층화로 이중 주석 대상 페어를 고정 시드 선정
     → {iaa_worklist} 저장. 두 주석자가 이 목록의 페어를 RGB 프레임에서 독립 표기
     (확장 YOLO: class cx cy w h v u) → {iaa_labels_a}, {iaa_labels_b}
  2. 두 디렉토리에 모두 라벨이 있는 페어에 대해 detection F1 / 박스 IoU /
     가시성·uncertain 플래그 일치율 및 Cohen's kappa 산출 → runs/iaa.json

라벨이 아직 없으면 1단계(worklist)만 수행하고 안내한다.
"""
import sys, yaml, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.data.align import imread_unicode
from udias.labeling.qc import load_extended_labels
from udias.eval.align_metrics import stratified_subset
from udias.eval.agreement import iaa_report

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"))
P, L = cfg["paths"], cfg["labeling"]
records = load_manifest(P["manifest"])

# ① 이중 주석 대상 선정 (전체 페어 대상 — RGB 프레임 주석이라 정렬 불필요)
picked, stats = stratified_subset(
    records, per_cell=int(L.get("iaa_per_cell", 10)),
    seed=int(L.get("iaa_seed", 7)), aligned_only=False, tag="iaa")
wl = Path(P.get("iaa_worklist", "splits/iaa_worklist.txt"))
wl.parent.mkdir(parents=True, exist_ok=True)
wl.write_text("\n".join(picked) + ("\n" if picked else ""))
print(f"[worklist] 이중 주석 대상 {len(picked)}개 → {wl}")
for cell, (took, avail) in stats.items():
    print(f"   {cell[0]:<12} {cell[1]:<8}: {took:>3}/{avail}")

# ② 두 주석자 라벨이 준비됐으면 일치도 산출
dir_a, dir_b = Path(P["iaa_labels_a"]), Path(P["iaa_labels_b"])
_size = {}
def img_size_lookup(rec):
    if rec.pair_id not in _size:
        im = imread_unicode(rec.rgb_path)
        _size[rec.pair_id] = (im.shape[1], im.shape[0])
    return _size[rec.pair_id]

if dir_a.exists() and dir_b.exists() and any(dir_a.glob("*.txt")):
    rep = iaa_report(records, dir_a, dir_b, img_size_lookup, load_extended_labels,
                     iou_thr=float(L.get("iaa_iou_thr", 0.5)))
    print(f"[IAA] {rep}")
    out = Path(P["outputs_dir"]) / "iaa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\n결과: {out}  (논문 §4.3 IAA 실측값 소스)")
else:
    print(f"[대기] 두 주석자 라벨({dir_a}, {dir_b})이 준비되면 재실행하면 IAA가 산출됩니다.")
