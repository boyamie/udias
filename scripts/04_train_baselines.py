"""④ : 단일모달/early(4ch 주·곱셈형 변형·정렬無) 5개 베이스라인 × 다중 시드 학습.

논문 Table 2 와 1:1 대응:
  rgb_only / ir_only            — 단일 모달
  early_stack4                  — Early fusion (primary): 4ch RGB+IR concat
  early_pixel                   — Early fusion (variant): (RGB+ε)·IR/255
  early_stack4_noalign          — Early, no alignment: 4ch, IR resize (정렬 ablation)

usage: python scripts/04_train_baselines.py config/default.yaml [--force name1,name2|all]

--force: 완료 판정을 무시하고 지정 베이스라인을 재학습한다.
  예) --force early_stack4_noalign
  (2026-07-31 진단: noalign 이 sweep 에서 한 번도 돌지 않은 원인은 W&B 활성화
  이전(≤07-27)의 구버전 학습이 남긴 results.csv(≥100행)를 완료로 오인한 것.
  구 export/구 정렬 기준 결과라 현재 수치와 비교 불가 → --force 로 재학습할 것.)

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

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
FORCE = set()
for _i, _a in enumerate(sys.argv):
    if _a == "--force" and _i + 1 < len(sys.argv):
        FORCE = set(sys.argv[_i + 1].split(","))
    elif _a.startswith("--force="):
        FORCE = set(_a.split("=", 1)[1].split(","))
cfg = yaml.safe_load(open(_args[0] if _args else "config/default.yaml", encoding="utf-8"))
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

    # ---- 프리플라이트: 모든 (name, seed) 의 완료 상태와 그 근거를 먼저 출력 ----
    # (noalign 이 조용히 스킵되던 사고 방지: 스킵 사유가 반드시 화면에 남는다)
    import datetime
    print("=== preflight ===")
    for name in EXPERIMENTS:
        for seed in T["seeds"]:
            run_dir = out_root / name / f"seed{seed}"
            rcsv, done = run_dir / "results.csv", run_dir / "DONE"
            if done.exists():
                st = f"DONE marker ({done.read_text(encoding='utf-8').strip()})"
            elif rcsv.exists():
                rows = max(0, len(rcsv.read_text(encoding="utf-8").splitlines()) - 1)
                mt = datetime.datetime.fromtimestamp(rcsv.stat().st_mtime).strftime("%m-%d %H:%M")
                st = f"results.csv rows={rows} mtime={mt}" + (" -> WILL SKIP (legacy)" if rows >= T["epochs"] else " -> will train (incomplete)")
            else:
                st = "missing -> will train"
            forced = " [FORCED retrain]" if (name in FORCE or "all" in FORCE) else ""
            print(f"  {name}/seed{seed}: {st}{forced}")

    for seed in T["seeds"]:
        for name in EXPERIMENTS:
            run_dir = out_root / name / f"seed{seed}"
            rcsv, done = run_dir / "results.csv", run_dir / "DONE"
            if name not in FORCE and "all" not in FORCE:
                # 완료 판정: DONE 마커(신규, early-stop 도 완료로 인정) 또는
                # 레거시 판정(results.csv 행 수 ≥ epochs — early-stop 런을 놓치는
                # 구버전 규칙이지만 마커 없는 기존 완주 런의 하위호환용으로 유지)
                if done.exists():
                    print(f"[skip] {name}/seed{seed} DONE 마커 존재")
                    continue
                if rcsv.exists():
                    n_done = max(0, len(rcsv.read_text(encoding="utf-8").splitlines()) - 1)
                    if n_done >= T["epochs"]:
                        print(f"[skip] {name}/seed{seed} 레거시 완료 판정 ({n_done} epochs) — 재학습하려면 --force {name}")
                        continue
            model = YOLO(T["model"])
            model.train(data=str(yamls[name]), epochs=T["epochs"], imgsz=T["img_size"],
                        batch=T["batch_size"], device=T["device"], seed=seed,
                        workers=T.get("workers", 2),   # 4K 원본 디코드 × 다워커 = RAM 고갈 방지
                        project=str(Path(P["outputs_dir"]) / name),
                        name=f"seed{seed}", exist_ok=True, deterministic=True,
                        **train_kws[name])
            # 정상 반환 = 완주(early stopping 포함). 마커를 남겨 재시작 시
            # 행 수와 무관하게 스킵되게 한다 (rgb_only seed0 이중 학습 사고 방지).
            done.write_text(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
            # 한 번 강제 재학습했으면 같은 sweep 내 다음 seed 부터는 정상 판정
            # 로직이 새 결과를 존중한다 (마커가 방금 생겼으므로 자연 스킵 없음).


if __name__ == "__main__":
    main()
