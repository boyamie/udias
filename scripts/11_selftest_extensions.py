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

# ── 5. native IR 서브셋 선정 (B4) — 층화·aligned-only·결정론 ────
from udias.eval.align_metrics import select_native_ir_subset, native_subset_coverage
srecs = []
for i in range(30):
    srecs.append(PairRecord(pair_id=f"hd{i}", video_id="v", rgb_path="", ir_path="",
                            scene_type="harbor", time_of_day="day", aligned=True))
for i in range(5):
    srecs.append(PairRecord(pair_id=f"on{i}", video_id="v", rgb_path="", ir_path="",
                            scene_type="open_water", time_of_day="night", aligned=True))
for i in range(10):
    srecs.append(PairRecord(pair_id=f"un{i}", video_id="v", rgb_path="", ir_path="",
                            scene_type="harbor", time_of_day="day", aligned=False))
picked, stats = select_native_ir_subset(srecs, per_cell=20, seed=42)
assert stats[("harbor", "day")] == (20, 30), stats          # 30중 20 (상한)
assert stats[("open_water", "night")] == (5, 5), stats       # 부족 셀은 전부
assert all(not p.startswith("un") for p in picked), "unaligned 페어가 선정됨"
assert picked == select_native_ir_subset(srecs, per_cell=20, seed=42)[0], "비결정론"
print(f"5. native subset OK  (층화 {stats}, aligned-only, 결정론)")

# ── 6. IAA 주석자 간 일치도 (B3) ───────────────────────────────
from udias.labeling.qc import load_extended_labels
from udias.eval.agreement import iaa_report, _kappa
iaa_dir = tmp / "iaa"
(iaa_dir / "a").mkdir(parents=True)
(iaa_dir / "b").mkdir(parents=True)
irec = [PairRecord(pair_id="q0", video_id="v", rgb_path="", ir_path="",
                   scene_type="harbor", time_of_day="day", aligned=True)]
(iaa_dir / "a" / "q0.txt").write_text(
    "0 0.3 0.3 0.2 0.2 2 0\n0 0.7 0.7 0.2 0.2 0 0\n0 0.5 0.1 0.1 0.1 2 0\n")
(iaa_dir / "b" / "q0.txt").write_text(
    "0 0.3 0.3 0.2 0.2 2 0\n0 0.71 0.7 0.2 0.2 1 0\n")
rep = iaa_report(irec, iaa_dir / "a", iaa_dir / "b", lambda r: (100, 100),
                 load_extended_labels, iou_thr=0.5)
assert rep["matched"] == 2 and rep["only_a"] == 1 and abs(rep["box_f1"] - 0.8) < 1e-9, rep
assert abs(rep["visibility_agreement"] - 0.5) < 1e-9, rep
assert abs(_kappa([0, 1, 0, 1], [0, 1, 0, 1]) - 1.0) < 1e-9
assert _kappa([0, 0, 0], [0, 0, 0]) is None
print(f"6. IAA OK            (F1={rep['box_f1']}, vis일치={rep['visibility_agreement']}, kappa 보정)")

# ── 7. M14 데이터카드 통계 + 갤러리 렌더 ──────────────────────
from udias.eval.dataset_stats import ship_size_areas, size_bin_counts, alignment_by_tod
from udias.eval.gallery import render_gallery
grecs = []
for i in range(12):
    tod = "day" if i % 2 else "night"
    gp = tmp / f"g{i}.png"
    cv2.imwrite(str(gp), np.zeros((200, 300, 3), np.uint8))
    (lbl_dir / f"g{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.05 0.05\n")
    grecs.append(PairRecord(pair_id=f"g{i}", video_id="v", rgb_path=str(gp),
                            ir_path="", scene_type="harbor", time_of_day=tod,
                            aligned=(i % 3 != 0), align_reproj_error=2.0, split="test"))
gareas = ship_size_areas(grecs, lbl_dir, lambda r: (300, 200), split="test")
gbins = size_bin_counts(gareas, {"small": [0, 1024], "medium": [1024, 9216],
                                 "large": [9216, ".inf"]})
assert len(gareas) == 24 and sum(gbins.values()) == 24, (gareas.shape, gbins)
gtod = alignment_by_tod(grecs)
assert set(gtod) == {"day", "night"} and gtod["day"]["n"] == 6, gtod
grend = render_gallery({"waves": ["g0", "g2"], "glare": ["g1"]},
                       lambda p: np.zeros((200, 300, 3), np.uint8),
                       lambda p: np.array([[40.0, 40.0, 120.0, 120.0]]),
                       lambda p: np.array([[45.0, 45.0, 125.0, 118.0]]),
                       tmp / "gallery.pdf", per_tag=3)
assert grend == {"waves": 2, "glare": 1} and (tmp / "gallery.pdf").exists(), grend
print(f"7. M14 figures OK    (크기 S/M/L={gbins}, 갤러리 렌더={grend})")

print("\nSELFTEST-EXTENSIONS PASS")
