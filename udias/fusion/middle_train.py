"""⑤ Middle fusion 학습 루프 — modality dropout 포함, 시드별 best.pt 저장.

출력 레이아웃은 단일/early 베이스라인과 동일:
    {outputs_dir}/middle_fusion/seed{seed}/weights/best.pt
→ scripts/07_eval_middle_fusion.py 가 그대로 찾아 평가한다.
비교 공정성: split·라벨·epochs·seed 집합을 다른 베이스라인과 동일하게 사용.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.manifest import load_manifest
from ..data.pair_dataset import PairDataset, collate
from .middle_fusion import MiddleFusionDetector


def train(cfg: dict, seed: int = 0) -> str:
    torch.manual_seed(seed)
    P, T = cfg["paths"], cfg["train"]
    M = cfg.get("middle", {})
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{T['device']}" if use_cuda else "cpu")

    records = load_manifest(P["manifest"])
    plain = Path(P["labels_dir"]) / "plain"
    dl_tr = DataLoader(PairDataset(records, plain, T["img_size"], "train"),
                       batch_size=T["batch_size"], shuffle=True,
                       collate_fn=collate, num_workers=M.get("workers", 0))
    dl_va = DataLoader(PairDataset(records, plain, T["img_size"], "val"),
                       batch_size=T["batch_size"], shuffle=False,
                       collate_fn=collate, num_workers=M.get("workers", 0))

    model = MiddleFusionDetector(M.get("backbone", "resnet18"), num_classes=1,
                                 pretrained=M.get("pretrained", True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=M.get("lr", 1e-4), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T["epochs"])
    p_drop = float(M.get("modality_dropout", 0.15))

    out = Path(P["outputs_dir"]) / "middle_fusion" / f"seed{seed}" / "weights"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for ep in range(T["epochs"]):
        model.train()
        tot, n = 0.0, 0
        for rgb, ir, ir_valid, targets, _ in dl_tr:
            rgb, ir, ir_valid = rgb.to(device), ir.to(device), ir_valid.to(device)
            if p_drop > 0:                          # modality dropout (⑥ 강건성과 연결)
                ir_valid = ir_valid & (torch.rand(len(ir_valid), device=device) >= p_drop)
            loss = model.compute_loss(model(rgb, ir, ir_valid), targets)["loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        sched.step()

        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for rgb, ir, ir_valid, targets, _ in dl_va:
                rgb, ir, ir_valid = rgb.to(device), ir.to(device), ir_valid.to(device)
                vtot += float(model.compute_loss(model(rgb, ir, ir_valid), targets)["loss"])
                vn += 1
        vl = vtot / max(vn, 1)
        print(f"[middle][seed{seed}] epoch {ep + 1}/{T['epochs']} "
              f"train {tot / max(n, 1):.4f} val {vl:.4f}")
        torch.save(model.state_dict(), out / "last.pt")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), out / "best.pt")
    print(f"[middle][seed{seed}] best val {best:.4f} → {out / 'best.pt'}")
    return str(out / "best.pt")
