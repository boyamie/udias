"""VS Code 터미널용 sweep 실시간 뷰어 (읽기 전용 — 언제든 Ctrl+C 로 닫아도 학습에 영향 없음).

detached 러너가 쓰는 ../sweep_run.log 를 1초마다 읽어
tqdm 진행바(export/epoch)를 제자리 갱신으로 미러링하고,
epoch 요약·ATTEMPT·완료/실패 라인은 새 줄로 남긴다.

사용:  python _watch_sweep.py   (code/ 에서)
"""
import re
import sys
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "sweep_run.log"

KEEP = re.compile(
    r"ATTEMPT|SWEEP_EXIT|OOM|OutOfMemoryError|Traceback|\[skip\]|epochs completed"
    r"|Results saved|mAP50|Starting training|New cache|정렬 실패"
)
PROGRESS = re.compile(r"(export\[[a-z0-9]+\]|Epoch|it/s|\d+/\d+ \[)")


def main() -> None:
    print(f"[watch] {LOG}  (Ctrl+C 로 종료 — 학습은 계속 돕니다)")
    seen_lines = 0
    last_prog = ""
    while True:
        try:
            raw = LOG.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            time.sleep(1)
            continue
        # \r 진행바 조각과 일반 라인을 모두 개별 토큰으로
        tokens = raw.replace("\r", "\n").split("\n")
        lines = [t for t in tokens if t.strip()]
        # 새로 확정된 '중요' 라인은 자기 줄로 출력
        for ln in lines[seen_lines:]:
            if KEEP.search(ln) and not PROGRESS.search(ln):
                sys.stdout.write("\x1b[2K\r" + ln.strip() + "\n")
        seen_lines = len(lines)
        # 마지막 진행 토큰은 제자리 갱신
        prog = next((t for t in reversed(lines) if PROGRESS.search(t)), "")
        if prog and prog != last_prog:
            sys.stdout.write("\x1b[2K\r" + prog.strip()[:160])
            sys.stdout.flush()
            last_prog = prog
        if "SWEEP_EXIT" in raw[-2000:]:
            tail = [t for t in lines if "SWEEP_EXIT" in t]
            if tail:
                sys.stdout.write("\n[watch] 종료 감지: " + tail[-1].strip() + "\n")
                return
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[watch] 뷰어만 종료했습니다 — 학습은 계속 돕니다.")
