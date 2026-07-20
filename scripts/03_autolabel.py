"""③ : 초벌 라벨 + visible_in 플래그 생성. 이후 Label Studio 검수 → verified 갱신.

usage: python scripts/03_autolabel.py configs/default.yaml
"""
import sys, yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest, save_manifest
from udias.labeling.qc import autolabel_pair, to_plain_yolo

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"))
P, L = cfg["paths"], cfg["labeling"]

from ultralytics import YOLO
model = YOLO(L["autolabel_model"])

records = load_manifest(P["manifest"])
ext_dir = Path(P["labels_dir"]) / "extended"
for rec in tqdm(records, desc="autolabel"):
    autolabel_pair(rec, model, L["autolabel_conf"], L["coco_boat_class"], ext_dir)
save_manifest(records, P["manifest"])

to_plain_yolo(ext_dir, Path(P["labels_dir"]) / "plain")
print("초벌 라벨 완료. Label Studio로 extended/를 검수한 뒤 "
      "label_verified 플래그를 갱신하고 to_plain_yolo를 재실행하세요.")
