"""⑤ : middle fusion(듀얼 인코더 + CBAM + FCOS) 다중 시드 학습.

usage: python scripts/06_train_middle_fusion.py config/default.yaml
"""
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.fusion.middle_train import train

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"))
for seed in cfg["train"]["seeds"]:
    train(cfg, seed)
