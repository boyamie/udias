"""④ : 단일모달/early(4ch 주·곱셈형 변형·정렬無) 5개 베이스라인 × 다중 시드 학습.

논문 Table 2 와 1:1 대응:
  rgb_only / ir_only            — 단일 모달
  early_stack4                  — Early fusion (primary): 4ch RGB+IR concat
  early_pixel                   — Early fusion (variant): (RGB+ε)·IR/255
  early_stack4_noalign          — Early, no alignment: 4ch, IR resize (정렬 ablation)

usage: python scripts/04_train_baselines.py config/default.yaml

주의(stack4): 4채널 학습은 ultralytics 의 multichannel 지원(TIFF 입력 +
data.yaml 의 `channels:` 키)을 사용한다. 구버전 ultralytics 는 이 키를
조용히 무시하고 3채널로 읽어버리므로, 지원 여부를 먼저 검사해 미지원이면
명시적으로 실패시킨다 (silent 3ch 학습 금지).
"""
import inspect
import sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import load_manifest
from udias.fusion.early import export_yolo_dataset

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml", encoding="utf-8"))
P, T = cfg["paths"], cfg["train"]
plain_labels = Path(P["labels_dir"]) / "plain"
records = load_manifest(P["manifest"])

EXPERIMENTS = {                      # 논문 5장 Table 2 의 행 순서
    "rgb_only":             dict(mode="rgb",    use_alignment=True),
    "ir_only":              dict(mode="ir",     use_alignment=True),
    "early_stack4":         dict(mode="stack4", use_alignment=True),   # 주 (M8)
    "early_pixel":          dict(mode="early",  use_alignment=True),   # 변형
    "early_stack4_noalign": dict(mode="stack4", use_alignment=False),  # 정렬 ablation
}

from ultralytics import YOLO


def check_multichannel_support() -> None:
    """ultralytics 가 data.yaml `channels:` 를 실제로 읽는지 검사."""
    from ultralytics.data.utils import check_det_dataset
    if "channels" not in inspect.getsource(check_det_dataset):
        raise RuntimeError(
            "이 ultralytics 버전은 4채널(multichannel) 학습을 지원하지 않습니다. "
            "`pip install -U ultralytics` 후 재실행하세요 "
            "(silent 3ch 학습을 막기 위해 중단합니다).")


def main():
    """Windows spawn-mode DataLoader 워커가 본 모듈을 재임포트하므로
    학습 루프는 반드시 __main__ 가드 안에서 실행한다.

    2단계 구조 (2026-07-28):
      1) 데이터셋 5종을 먼저 전부 export (idempotent — 있으면 스킵)
      2) 학습은 seed-바깥 루프: seed0 를 5개 baseline 전부 → seed1 → seed2.
         부분 완주 시에도 Table 2 의 '행 커버리지'가 먼저 확보된다.
    """
    yamls, train_kws = {}, {}
    for name, kw in EXPERIMENTS.items():
        if kw["mode"] == "stack4":
            check_multichannel_support()
        yamls[name] = export_yolo_dataset(records, plain_labels,
                                          Path(P["outputs_dir"]) / "datasets" / name,
                                          epsilon=cfg["fusion"]["early"]["epsilon"],
                                          max_side=cfg["fusion"].get("export_max_side"), **kw)
        # 색공간 augment 는 4ch 에서 의미가 없고 버전에 따라 실패 → 명시적 off
        train_kws[name] = (dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
                           if kw["mode"] == "stack4" else {})

    out_root = Path(P["outputs_dir"])
    for seed in T["seeds"]:
        for name in EXPERIMENTS:
            # 완료 판정: results.csv 의 epoch 행 수 ≥ 목표 epochs.
            # (best.pt 는 학습 도중에도 생기므로 완료 마커로 쓰면 중단 런을 건너뛴다)
            rcsv = out_root / name / f"seed{seed}" / "results.csv"
            if rcsv.exists():
                n_done = max(0, len(rcsv.read_text(encoding="utf-8").splitlines()) - 1)
                if n_done >= T["epochs"]:
                    print(f"[skip] {name}/seed{seed} 이미 완료 ({n_done} epochs)")
                    continue
            model = YOLO(T["model"])
            model.train(data=str(yamls[name]), epochs=T["epochs"], imgsz=T["img_size"],
                        batch=T["batch_size"], device=T["device"], seed=seed,
                        workers=T.get("workers", 2),   # 4K 원본 디코드 × 다워커 = RAM 고갈 방지
                        project=str(Path(P["outputs_dir"]) / name),
                        name=f"seed{seed}", exist_ok=True, deterministic=True,
                        **train_kws[name])


if __name__ == "__main__":
    main()
