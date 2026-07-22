"""① 페어 매니페스트 — 데이터셋의 primary unit.

레코드 하나 = RGB/IR 프레임 한 쌍 + 정렬/라벨/분할 메타데이터.
JSON Lines(한 줄에 한 레코드)로 저장하며, 모든 스크립트는 이 파일만 읽는다.
필드 규약은 보고서 Table 1과 일치.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class PairRecord:
    pair_id: str
    video_id: str
    rgb_path: str
    ir_path: str
    time_of_day: str = "unknown"          # day | night | unknown
    scene_type: str = "unknown"           # open_water | nearshore | harbor | unknown
    capture_time: str = ""                # optional ISO timestamp
    aligned: bool = False
    H_ir_to_rgb: Optional[list] = None    # 3x3, IR -> RGB
    align_num_inliers: int = 0
    align_reproj_error: float = -1.0      # px; -1 = 미측정/실패
    align_method: str = ""
    label_path: str = ""
    label_verified: bool = False
    ir_label_path: str = ""               # IR 좌표계 '원생(native)' 주석 (독립 표기, 정렬 지표 ii용); "" = 없음
    split: str = ""                       # train | val | test
    preprocessing_version: str = "v2"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PairRecord":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


def _derive_time_of_day(name: str) -> str:
    low = name.lower()
    if "night" in low or "야간" in name:
        return "night"
    if "day" in low or "주간" in name:
        return "day"
    return "unknown"


def _make_video_id(stem: str, regex: str, template: str) -> Optional[str]:
    """파일명 stem에 regex를 적용하고 template({1},{2},...)에 그룹을 채워 video_id 조립."""
    m = re.search(regex, stem)
    if not m:
        return None
    out = template
    for i, g in enumerate(m.groups(), start=1):
        out = out.replace("{%d}" % i, str(g))
    return out


def build_pairs(rgb_dir, ir_dir, video_id_regex: str, video_id_template: str):
    """RGB/IR 프레임 폴더를 스캔해 '같은 파일명' 기준으로 페어를 만든다.

    규약: video_to_frames.py가 두 폴더에 동일한 이름으로 저장한다.
        <rgb_dir>/Day_01_00000.jpg  <->  <ir_dir>/Day_01_00000.jpg
    video_id 는 파일명에 regex 를 적용해 조립한다(같은 영상의 프레임은 같은 video_id).
    """
    rgb_dir, ir_dir = Path(rgb_dir), Path(ir_dir)
    records = []
    for rgb_path in sorted(rgb_dir.rglob("*")):
        if rgb_path.suffix.lower() not in IMG_EXTS:
            continue
        ir_path = ir_dir / rgb_path.name
        if not ir_path.exists():
            continue
        stem = rgb_path.stem
        vid = _make_video_id(stem, video_id_regex, video_id_template)
        if vid is None:
            vid = stem  # regex 실패 안전장치: 프레임 하나를 하나의 영상으로 취급(누수 방지 보수적)
        records.append(PairRecord(
            pair_id=stem,
            video_id=vid,
            rgb_path=str(rgb_path),
            ir_path=str(ir_path),
            time_of_day=_derive_time_of_day(stem),
        ))
    return records


def save_manifest(records, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def load_manifest(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(PairRecord.from_dict(json.loads(line)))
    return records
