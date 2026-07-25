"""TEMP smoke test for the training path (rgb_only, 3 epochs, auto-batch).
Validates: autolabel-produced labels -> YOLO dataset export -> VRAM fit ->
training loop -> outputs. Prints the auto-selected batch size so the full
15-run sweep can be launched with a batch that fits 8 GB.
Delete after use. Run from code/ with the torch_env python.
"""
import sys, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from udias.data.manifest import load_manifest
from udias.fusion.early import export_yolo_dataset
from ultralytics import YOLO

cfg = yaml.safe_load(open(ROOT / "config/default.yaml", encoding="utf-8"))
P, T = cfg["paths"], cfg["train"]
plain_labels = Path(P["labels_dir"]) / "plain"
records = load_manifest(P["manifest"])
print(f"[smoke] manifest records: {len(records)}")
print(f"[smoke] labels dir: {plain_labels}  exists={plain_labels.exists()}")

data_yaml = export_yolo_dataset(
    records, plain_labels,
    Path(P["outputs_dir"]) / "datasets" / "_smoke_rgb",
    mode="rgb", use_alignment=True,
    epsilon=cfg["fusion"]["early"]["epsilon"],
)
print(f"[smoke] data.yaml: {data_yaml}")

model = YOLO(T["model"])
model.train(
    data=str(data_yaml), epochs=3, imgsz=T["img_size"],
    batch=-1,                       # auto-batch: report max that fits ~60% VRAM
    device=T["device"], seed=0,
    project=str(Path(P["outputs_dir"]) / "_smoke"), name="rgb_only",
    exist_ok=True, deterministic=True,
)
print("SMOKE_OK")
