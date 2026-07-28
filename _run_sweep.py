"""세션 독립 sweep 러너 (야간 무인 실험용).

- Windows 절전 방지: SetThreadExecutionState(ES_SYSTEM_REQUIRED) — 시스템 설정을
  바꾸지 않고 이 프로세스가 살아있는 동안만 잠들지 않게 한다.
- OOM 자동 강등: batch 8 → 6 → 4 (config 의 batch_size 를 바꿔가며 04 재실행).
  scripts/04 는 완료된 런을 스킵하므로 재시도가 진행분을 잃지 않는다.
- 로그: ../sweep_run.log (기존 규약 유지 — 뷰어/워처 호환)

실행:  Start-Process 로 detach (VS Code/Claude 세션과 무관하게 계속 돈다)
중단:  작업관리자에서 이 python 프로세스 종료
"""
import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # code/
LOG = ROOT.parent / "sweep_run.log"
PY = sys.executable

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def set_batch(b: int) -> None:
    cfg = ROOT / "config/default.yaml"
    txt = cfg.read_text(encoding="utf-8")
    txt = re.sub(r"^(  batch_size:) \d+", rf"\g<1> {b}", txt, count=1, flags=re.M)
    cfg.write_text(txt, encoding="utf-8")


def main() -> int:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    env = dict(os.environ,
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    try:
        with open(LOG, "a", encoding="utf-8", errors="replace") as log:
            for batch in (8, 6, 4):
                log.write(f"\n=== ATTEMPT batch={batch} workers=2 (detached, wandb on) ===\n")
                log.flush()
                set_batch(batch)
                r = subprocess.run([PY, "scripts/04_train_baselines.py", "config/default.yaml"],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if r.returncode == 0:
                    log.write(f"SWEEP_EXIT=0 (batch={batch})\n")
                    return 0
                tail = LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
                if "OutOfMemoryError" in tail or "Insufficient memory" in tail:
                    log.write(f"OOM at batch={batch}, retrying smaller\n")
                    log.flush()
                    continue
                log.write(f"SWEEP_EXIT={r.returncode} (non-OOM failure, stopping)\n")
                return r.returncode
            log.write("SWEEP_EXIT=1 (all batches failed)\n")
            return 1
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    raise SystemExit(main())
