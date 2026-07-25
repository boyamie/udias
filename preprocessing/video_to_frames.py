"""원본 영상 → 프레임 추출 (config 기반, iPhone RGB + 한화 IR 두 카메라 대응).

기존 문제(수정됨):
  - BASE_DIR 하드코딩(/workspace/uda/data) → config/default.yaml 로 대체
  - 출력 폴더가 config(rgb_dir/ir_dir)와 불일치 → config 값으로 통일
  - 파일명 파서가 영어명(ir_day_1.mp4, rgb_day_3.MOV)·번호를 놓침 → 한글/영어 모두 파싱
  - 프레임 '인덱스' 간격 → 카메라 fps 가 다르면 시간이 어긋남
      ⇒ '경과시간(초)' 기준 샘플링. RGB/IR 둘 다 k*SAMPLE_SEC 초 지점을 저장하므로
         같은 저장이름(Day_04_00000.jpg)이 같은 경과시각을 가리킴(두 녹화 시작이 동기라는 가정).

usage:
  python preprocessing/video_to_frames.py [config/default.yaml] [--sec 0.5]

pair 성립: <rgb_dir>/Day_04_00000.jpg  <->  <ir_dir>/Day_04_00000.jpg  (manifest.build_pairs)
그 위에서 align.py(SIFT+RANSAC)가 iPhone↔한화 공간 정합(homography)을 계산.
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import yaml

try:
    from tqdm import tqdm
except Exception:                       # tqdm 없으면 무동작 패스스루
    def tqdm(x=None, *a, **k):
        return x if x is not None else None


def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_src_dirs(base: Path):
    """time_of_day -> 원본 영상 폴더. 실제 폴더명(day/night) 및 레거시(day_origin 등) 모두 수용."""
    out = {}
    for tod, cands in (("Day", ("day", "day_origin", "주간")),
                       ("Night", ("night", "night_origin", "야간"))):
        for c in cands:
            p = base / c
            if p.exists():
                out[tod] = p
                break
    return out


def classify_modality(name: str):
    """파일명 → 'rgb' | 'ir' | None. 한글/영어/카메라 명명 모두 대응."""
    low = name.lower()
    if "rgb" in low or "visible" in low or "가시" in low:
        return "rgb"
    # 'ir' 이 다른 글자에 붙지 않은 토큰일 때만 (infrared/irregular 오탐 방지)
    if re.search(r"(?<![a-z])ir(?![a-z])", low) or "infrared" in low or "적외" in low:
        return "ir"
    return None


def extract_number(stem: str):
    """파일명 stem 의 '마지막' 정수 = 영상 번호. (주간-IR-4→4, ir_day_1→1, night-IR-10→10)"""
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else None


def sample_stride(fps: float, sample_sec: float) -> int:
    if not fps or fps != fps or fps <= 0:   # NaN/0 방어 → 원본과 동일하게 매 10프레임
        return 10
    return max(1, round(fps * sample_sec))


def process_video(video_path: Path, save_dir: Path, prefix: str, sample_sec: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERR] 열기 실패: {video_path.name}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = sample_stride(fps, sample_sec)
    fidx = saved = 0
    pbar = tqdm(total=total, desc=prefix, leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % stride == 0:
            cv2.imwrite(str(save_dir / f"{prefix}_{saved:05d}.jpg"), frame)
            saved += 1
        fidx += 1
        if pbar is not None:
            pbar.update(1)
    cap.release()
    if pbar is not None:
        pbar.close()
    print(f"  {prefix:<9} fps={fps:5.1f} stride={stride:>3} → {saved} 프레임  ({video_path.name})")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config/default.yaml")
    ap.add_argument("--sec", type=float, default=0.5,
                    help="샘플링 간격(초). 두 카메라 공통 경과시각으로 저장.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    root = cfg_path.resolve().parent.parent          # config/ 의 상위 = 리포 루트
    p = cfg["paths"]
    raw_base = (root / p["raw_video_dir"]).resolve()
    rgb_out = (root / p["rgb_dir"]).resolve()
    ir_out = (root / p["ir_dir"]).resolve()
    rgb_out.mkdir(parents=True, exist_ok=True)
    ir_out.mkdir(parents=True, exist_ok=True)

    src = resolve_src_dirs(raw_base)
    if not src:
        print(f"[ERR] 원본 폴더 없음: {raw_base} 아래 day/night(또는 day_origin/night_origin)")
        sys.exit(1)

    print(f"원본: {raw_base}")
    print(f"출력: RGB→{rgb_out}  IR→{ir_out}   (샘플 {args.sec}s)")
    total_saved = {"rgb": 0, "ir": 0}
    skipped = []
    for tod, sdir in src.items():
        files = sorted(sdir.glob("*"))
        print(f"\n[{tod}] {sdir}  (파일 {len(files)}개)")
        for f in files:
            if not f.is_file():
                continue
            mod = classify_modality(f.name)
            num = extract_number(f.stem)
            if mod is None or num is None:
                skipped.append(f.name)
                continue
            prefix = f"{tod}_{num:02d}"
            out = rgb_out if mod == "rgb" else ir_out
            total_saved[mod] += process_video(f, out, prefix, args.sec)

    print(f"\n완료: RGB {total_saved['rgb']}장, IR {total_saved['ir']}장")
    if skipped:
        print(f"[주의] 분류 실패로 건너뜀 {len(skipped)}개: {skipped}")
    print("다음: python scripts/01_build_manifest_and_align.py", args.config)


if __name__ == "__main__":
    main()
