"""④ : RGB-only / IR-only / early(정렬 有·無) 4개 베이스라인 × 다중 시드 학습.

usage: python scripts/04_train_baselines.py configs/default.yaml
"""
import sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.fusion.early import export_yolo_dataset

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"))
P, T = cfg["paths"], cfg["train"]
plain_labels = Path(P["labels_dir"]) / "plain"
records = load_manifest(P["manifest"])

EXPERIMENTS = {                      # 보고서 5장의 4-조합 ablation
    "rgb_only":        dict(mode="rgb",   use_alignment=True),
    "ir_only":         dict(mode="ir",    use_alignment=True),
    "early_aligned":   dict(mode="early", use_alignment=True),
    "early_noalign":   dict(mode="early", use_alignment=False),  # 정렬 ablation
}

from ultralytics import YOLO
for name, kw in EXPERIMENTS.items():
    data_yaml = export_yolo_dataset(records, plain_labels,
                                    Path(P["outputs_dir"]) / "datasets" / name,
                                    epsilon=cfg["fusion"]["early"]["epsilon"], **kw)
    for seed in T["seeds"]:
        model = YOLO(T["model"])
        model.train(data=str(data_yaml), epochs=T["epochs"], imgsz=T["img_size"],
                    batch=T["batch_size"], device=T["device"], seed=seed,
                    project=str(Path(P["outputs_dir"]) / name),
                    name=f"seed{seed}", exist_ok=True, deterministic=True)
