"""middle fusion 자체 검증 — 합성 1샘플 과적합으로 손실 감소 + 박스 예측 확인.

torch 환경에서:
    python scripts/08_selftest_middle_fusion.py
CPU로 ~30초. 'SELFTEST PASS' 가 나오면 모델/손실/추론 배선이 정상.
(실데이터·pretrained 불필요 — 구조 검증용)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from udias.fusion.middle_fusion import MiddleFusionDetector

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MiddleFusionDetector("resnet18", num_classes=1, pretrained=False).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 합성 1샘플: 가운데에 상자 하나
img = 256
rgb = torch.randn(1, 3, img, img, device=device)
ir = torch.rand(1, 1, img, img, device=device)
ir_valid = torch.tensor([True], device=device)
target = [{"boxes": torch.tensor([[80., 80., 176., 176.]], device=device),
           "labels": torch.zeros(1, dtype=torch.long, device=device)}]

model.train()
first = last = None
for step in range(80):
    loss = model.compute_loss(model(rgb, ir, ir_valid), target)["loss"]
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 0:
        first = float(loss)
    last = float(loss)
    if step % 20 == 0:
        print(f"step {step:3d}  loss {float(loss):.4f}")

model.eval()
boxes, scores = model.predict(rgb, ir, ir_valid, score_thr=0.2)[0]
print(f"\nloss: {first:.4f} -> {last:.4f}   예측 박스 수: {len(boxes)}")
if boxes.numel():
    print("best box:", [round(v, 1) for v in boxes[scores.argmax()].tolist()],
          "score", round(float(scores.max()), 3))

ok = (last < first * 0.5) and (boxes.numel() >= 1)
# modality dropout 경로도 죽지 않는지(전 배치 IR 차단) 확인
_ = model.compute_loss(model(rgb, ir, torch.tensor([False], device=device)), target)["loss"]
print("\nSELFTEST PASS" if ok else "\nSELFTEST FAIL")
sys.exit(0 if ok else 1)
