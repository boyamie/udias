"""② : native IR 주석 대상 페어 선정 → worklist 생성 (논문 §4.3, 지표 ii).

usage: python scripts/12_select_ir_native_subset.py config/default.yaml

규칙(재현 가능): 정렬 성공(aligned=True) 페어를 (scene_type, time_of_day) 셀로
층화한 뒤, 셀마다 최대 native_ir_per_cell 개를 고정 시드로 무작위 추출.
→ {ir_native_worklist} 파일에 pair_id 목록 저장. 주석자는 이 목록의 페어만
IR 좌표계에서 독립 표기하면 되고, scripts/10 이 목록 대비 완료율을 검증한다.

주석 산출물은 {ir_native_labels_dir}/{pair_id}.txt (IR 좌표계 plain YOLO 5열).
"""
import sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.eval.align_metrics import select_native_ir_subset

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"))
P, L = cfg["paths"], cfg["labeling"]
records = load_manifest(P["manifest"])

picked, stats = select_native_ir_subset(
    records, per_cell=int(L.get("native_ir_per_cell", 20)),
    seed=int(L.get("native_ir_seed", 42)))

wl = Path(P.get("ir_native_worklist", "splits/ir_native_worklist.txt"))
wl.parent.mkdir(parents=True, exist_ok=True)
wl.write_text("\n".join(picked) + ("\n" if picked else ""))

print(f"[worklist] {len(picked)}개 페어 선정 → {wl}")
print("[층화] (scene_type, time_of_day): 선정/가용")
short = 0
for cell, (took, avail) in stats.items():
    flag = "  ← 셀 부족" if took < int(L.get("native_ir_per_cell", 20)) else ""
    if flag:
        short += 1
    print(f"   {cell[0]:<12} {cell[1]:<8}: {took:>3}/{avail}{flag}")
if short:
    print(f"[주의] {short}개 셀이 목표 {L.get('native_ir_per_cell', 20)}개 미만 — "
          f"데이터카드에 셀별 실제 개수를 보고하세요.")
print(f"\n다음: 주석자가 {P['ir_native_labels_dir']}/<pair_id>.txt 를 IR 좌표계로 표기 "
      f"→ python scripts/10_eval_alignment.py {sys.argv[1] if len(sys.argv) > 1 else 'config/default.yaml'}")
