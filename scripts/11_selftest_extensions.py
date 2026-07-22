"""자체 검증 ② — M5/M8/§5.4 확장분 (ultralytics 불필요, torch+cv2 만 있으면 됨).

usage: python scripts/11_selftest_extensions.py

검증 항목:
  1. stack4 export: 4ch TIFF 왕복(저장→로드) 무손실, data.yaml 에 channels: 4
  2. 강건성 변환: 결정론성(같은 pair_id → 같은 노이즈), 대비 축소 공식,
     drop_ir 의 ir_valid=False 전파
  3. native warp IoU: 항등 H 에서 IoU=1, 이동 H 에서 감소, greedy 1:1 매칭
  4. PairDataset transform 훅 → middle fusion 순전파 (drop 시나리오 포함)
"""
import sys, tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.data.manifest import PairRecord
from udias.fusion.early import stack4, export_yolo_dataset
from udias.eval.robustness import apply_scenario, ir_contrast, ir_mult_noise
from udias.eval.align_metrics import native_warp_iou_report, load_plain_boxes

tmp = Path(tempfile.mkdtemp())


def make_pair(i, aligned=True, H=None):
    rgb = np.random.default_rng(i).integers(0, 255, (128, 160, 3), np.uint8)
    ir = np.random.default_rng(100 + i).integers(0, 255, (64, 80, 3), np.uint8)
    rp, ip = tmp / f"rgb_{i}.png", tmp / f"ir_{i}.png"
    cv2.imwrite(str(rp), rgb)
    cv2.imwrite(str(ip), ir)
    return PairRecord(pair_id=f"p{i}", video_id="v0", rgb_path=str(rp),
                      ir_path=str(ip), aligned=aligned,
                      H_ir_to_rgb=H or [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                      split="train")


# ── 1. stack4 export ───────────────────────────────────────────
recs = [make_pair(0), make_pair(1)]
lbl_dir = tmp / "plain"
lbl_dir.mkdir()
for r in recs:
    (lbl_dir / f"{r.pair_id}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
data_yaml = export_yolo_dataset(recs, lbl_dir, tmp / "ds_stack4", mode="stack4")
import yaml
dy = yaml.safe_load(data_yaml.read_text())
assert dy["channels"] == 4, dy
t = cv2.imread(str(tmp / "ds_stack4" / "images" / "train" / "p0.tiff"),
               cv2.IMREAD_UNCHANGED)
assert t is not None and t.shape[2] == 4, t.shape if t is not None else None
rgb0 = cv2.imread(str(recs[0].rgb_path))
ir0 = cv2.imread(str(recs[0].ir_path))
from udias.data.align import warp_ir_to_rgb
expect = stack4(rgb0, warp_ir_to_rgb(recs[0], ir0, rgb0.shape))
assert np.array_equal(t, expect), "TIFF 왕복 무손실 실패"
print("1. stack4 export OK  (4ch TIFF 무손실, channels: 4)")

# ── 2. 강건성 변환 ─────────────────────────────────────────────
rec = recs[0]
ir_g = cv2.cvtColor(ir0, cv2.COLOR_BGR2GRAY)
n1 = ir_mult_noise(ir_g, 0.3, "p0")
n2 = ir_mult_noise(ir_g, 0.3, "p0")
n3 = ir_mult_noise(ir_g, 0.3, "p1")
assert np.array_equal(n1, n2), "같은 pair_id 인데 노이즈 다름 (결정론 위반)"
assert not np.array_equal(n1, n3), "다른 pair_id 인데 노이즈 동일"
c = ir_contrast(ir_g, 0.4)
assert abs(float(c.std()) / float(ir_g.std()) - 0.4) < 0.05, "대비 40% 공식 오차"
r_rgb, r_ir, r_v = apply_scenario("drop_ir", rec, rgb0, ir_g, True)
assert r_ir.sum() == 0 and r_v is False and r_rgb is rgb0
r_rgb, r_ir, r_v = apply_scenario("drop_rgb", rec, rgb0, ir_g, True)
assert r_rgb.sum() == 0 and r_v is True
r2 = apply_scenario("ir_noise03", rec, rgb0, ir_g, True)
assert np.array_equal(r2[1], ir_mult_noise(ir_g, 0.3, "p0")), "시나리오 경로 결정론 위반"
print("2. robustness OK     (결정론·대비 공식·drop 게이트)")

# ── 3. native warp IoU ─────────────────────────────────────────
# 항등 H + IR 좌표 = RGB 좌표 절반 스케일 → H=2배 확대가 정확히 복원 → IoU 1.0
nat_dir = tmp / "native"
nat_dir.mkdir()
# RGB 라벨: (40,32)-(80,64) 절대 → 정규화 w=160,h=128
(lbl_dir / "p0.txt").write_text("0 0.375 0.375 0.25 0.25\n")
# native IR 라벨: 같은 박스의 IR 좌표(절반 스케일, w=80,h=64) → H(×2)로 정확 복원
(nat_dir / "p0.txt").write_text("0 0.375 0.375 0.25 0.25\n")
recs[0].ir_label_path = str(nat_dir / "p0.txt")
rep = native_warp_iou_report(recs, lbl_dir, lambda p: cv2.imread(str(p)))
assert rep["pairs_evaluated"] == 1 and rep["matched"] == 1, rep
assert rep["mean_iou"] > 0.99, rep
# 이동 H → IoU 감소 확인
recs[0].H_ir_to_rgb = [[2, 0, 12], [0, 2, 12], [0, 0, 1]]
rep2 = native_warp_iou_report(recs, lbl_dir, lambda p: cv2.imread(str(p)))
assert rep2["mean_iou"] < rep["mean_iou"], (rep, rep2)
print(f"3. native warp IoU OK (항등 IoU={rep['mean_iou']:.3f}, "
      f"이동 IoU={rep2['mean_iou']:.3f})")

# ── 4. PairDataset transform 훅 → middle fusion ────────────────
import torch
from torch.utils.data import DataLoader
from udias.data.pair_dataset import PairDataset, collate
from udias.fusion.middle_fusion import MiddleFusionDetector

recs[0].H_ir_to_rgb = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
ds = PairDataset(recs, lbl_dir, img_size=128, split="train",
                 transform=lambda rec, rgb, ir, v: apply_scenario(
                     "drop_ir", rec, rgb, ir, v))
s = ds[0]
assert float(s["ir"].abs().sum()) == 0.0 and s["ir_valid"] is False
dl = DataLoader(ds, batch_size=2, collate_fn=collate)
rgb_b, ir_b, v_b, targets, meta = next(iter(dl))
assert v_b.tolist() == [False, False]
m = MiddleFusionDetector(pretrained=False)
with torch.no_grad():
    out = m.predict(rgb_b, ir_b, v_b.float())
assert len(out) == 2
print("4. transform hook OK (drop_ir → ir_valid=False → 게이트 순전파)")

print("\nSELFTEST-EXTENSIONS PASS")
