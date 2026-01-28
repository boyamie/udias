import cv2
import shutil
import re
import os
from pathlib import Path
from tqdm import tqdm

# --- [윈도우 경로 설정] ---
# 사용자님 PC 경로에 맞췄습니다.
BASE_DIR = Path(r"C:\Users\BohyunKim\Documents\udias\data")

# 원본 동영상이 있는 폴더 (탐색기 이름을 기준으로 설정)
# 만약 폴더명이 영어(day_origin)라면 아래 한글 부분을 영어로 바꿔주세요.
SRC_DIRS = {
    "Day": BASE_DIR / "주간_원본",   # 또는 "day_origin"
    "Night": BASE_DIR / "야간_원본"  # 또는 "night_origin"
}

# 기존 라벨이 있는 폴더 (우리가 다운받았던 압축파일 푼 곳)
# 보통 'data/labels' 또는 'data/dataset/labels' 에 있습니다. 확인 필요!
LABEL_DIR = BASE_DIR / "labels" 

# 결과 저장 경로 (바탕화면에 'aligned_dataset' 폴더 생성)
OUTPUT_BASE = Path.home() / "Desktop" / "aligned_dataset"
NEW_RGB_DIR = OUTPUT_BASE / "visible_images_aligned"
NEW_IR_DIR = OUTPUT_BASE / "infrared_images_aligned"
NEW_LABEL_DIR = OUTPUT_BASE / "labels_aligned"
# ------------------------

def parse_label_info(filename):
    """ 라벨 파일명에서 정보 추출 (Fused_야간-RGB-13_frame_00114.txt) """
    is_night = "야간" in filename or "Night" in filename
    
    # 비디오 ID 추출
    match_id = re.search(r'RGB-(\d+)', filename)
    if not match_id: return None
    video_id = int(match_id.group(1))
    
    # 프레임 번호 추출
    match_frame = re.search(r'frame_(\d+)', filename)
    if not match_frame: return None
    frame_idx = int(match_frame.group(1))
    
    return is_night, video_id, frame_idx

def find_video_path(is_night, video_id, mode="RGB"):
    """ 동영상 파일 찾기 """
    dir_key = "Night" if is_night else "Day"
    src_dir = SRC_DIRS[dir_key]
    
    if not src_dir.exists():
        # 혹시 영어 폴더명일 경우 대비
        alt_dir = BASE_DIR / ("night_origin" if is_night else "day_origin")
        if alt_dir.exists():
            src_dir = alt_dir
        else:
            return None

    # 파일명 패턴 (깨진 문자 무시하고 숫자와 모드로 찾기)
    # 예: *-RGB-1.MOV 또는 *-IR-1.mp4
    pattern = f"*-{mode}-{video_id}.*"
    candidates = list(src_dir.glob(pattern))
    
    if candidates:
        return candidates[0]
    return None

def main():
    # 폴더 생성
    for p in [NEW_RGB_DIR, NEW_IR_DIR, NEW_LABEL_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"🚀 [윈도우] 데이터 정제 시작...")
    print(f" - 라벨 폴더: {LABEL_DIR}")
    
    label_files = list(LABEL_DIR.glob("*.txt"))
    if not label_files:
        # 혹시 dataset/labels 경로에 있을 경우
        LABEL_DIR_ALT = BASE_DIR / "dataset" / "labels"
        label_files = list(LABEL_DIR_ALT.glob("*.txt"))
        if not label_files:
            print("❌ 라벨 파일(.txt)을 찾을 수 없습니다! 경로를 확인해주세요.")
            return
        else:
            print(f" -> 경로 자동 수정됨: {LABEL_DIR_ALT}")

    print(f" - 총 {len(label_files)}개의 라벨 처리 시작")
    
    success = 0
    fail = 0

    for label_path in tqdm(label_files):
        info = parse_label_info(label_path.name)
        if not info: continue
            
        is_night, vid_id, frame_idx = info
        
        # 동영상 찾기
        rgb_vid = find_video_path(is_night, vid_id, "RGB")
        ir_vid = find_video_path(is_night, vid_id, "IR")
        
        if not rgb_vid or not ir_vid:
            fail += 1
            continue

        # 프레임 추출
        try:
            # RGB
            cap_rgb = cv2.VideoCapture(str(rgb_vid))
            cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_r, frame_rgb = cap_rgb.read()
            cap_rgb.release()

            # IR
            cap_ir = cv2.VideoCapture(str(ir_vid))
            cap_ir.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_i, frame_ir = cap_ir.read()
            cap_ir.release()

            if not ret_r or not ret_i:
                fail += 1
                continue
                
            # 저장 (이름 통일: Night_01_00114.jpg)
            prefix = "Night" if is_night else "Day"
            new_name = f"{prefix}_{vid_id:02d}_{frame_idx:05d}"
            
            cv2.imwrite(str(NEW_RGB_DIR / f"{new_name}.jpg"), frame_rgb)
            cv2.imwrite(str(NEW_IR_DIR / f"{new_name}.jpg"), frame_ir)
            shutil.copy(str(label_path), str(NEW_LABEL_DIR / f"{new_name}.txt"))
            
            success += 1
            
        except Exception as e:
            print(f"Error: {e}")
            fail += 1

    print(f"\n✅ 완료! 바탕화면의 [aligned_dataset] 폴더를 확인하세요.")
    print(f" - 성공: {success}쌍 / 실패: {fail}쌍")

if __name__ == "__main__":
    main()