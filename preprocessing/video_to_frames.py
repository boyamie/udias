import cv2
import os
import re
from pathlib import Path
from tqdm import tqdm

# --- [설정: 여기를 꼭 확인하세요!] ---
BASE_DIR = Path("/workspace/uda/data")  # 서버 경로여야 함

# 아까 영어로 바꾼 폴더 이름 (day_origin, night_origin)
SRC_CONFIG = {
    "Day": BASE_DIR / "day_origin",
    "Night": BASE_DIR / "night_origin"
}

# 저장할 폴더
RGB_SAVE_DIR = BASE_DIR / "dataset/visible_images"
IR_SAVE_DIR = BASE_DIR / "dataset/infrared_images"

# 프레임 저장 간격 (10프레임마다 1장)
FRAME_INTERVAL = 10
# ----------------

def extract_number(filename):
    """ 파일명 끝의 숫자 추출 (예: ...-IR-1.mp4 -> 1) """
    match = re.search(r'-(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return None

def process_video(video_path, save_dir, save_prefix):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path.name}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved_count = 0
    
    # 진행바 표시
    pbar = tqdm(total=total_frames, desc=f"{save_prefix}", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % FRAME_INTERVAL == 0:
            # 파일명: {접두사}_{프레임번호}.jpg  (예: Day_01_00000.jpg)
            save_name = f"{save_prefix}_{saved_count:05d}.jpg"
            cv2.imwrite(str(save_dir / save_name), frame)
            saved_count += 1
            
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    return saved_count

def main():
    # 1. 저장 폴더 생성
    RGB_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IR_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 동영상 -> 이미지 변환 및 분류 시작...")
    print(f" - 소스 경로: {BASE_DIR}")

    for time_type, src_dir in SRC_CONFIG.items():
        if not src_dir.exists():
            print(f"⚠️ 폴더 없음, 건너뜀: {src_dir}")
            continue

        # 폴더 내 모든 파일 검색
        files = list(src_dir.glob("*"))
        
        # 숫자 기준으로 정렬
        files.sort(key=lambda x: extract_number(x.name) if extract_number(x.name) is not None else 9999)

        print(f"\n📂 [{time_type}] 폴더 처리 중... (파일 {len(files)}개)")

        for vid_path in files:
            fname = vid_path.name
            
            # 숫자 추출
            vid_id = extract_number(fname)
            if vid_id is None:
                continue 

            # 공통 이름 생성 (예: Day_01)
            new_name_prefix = f"{time_type}_{vid_id:02d}"

            # 분류 및 처리
            if "-IR-" in fname:
                print(f"   🔥 [IR] {fname} -> {new_name_prefix}_xxxxx.jpg")
                process_video(vid_path, IR_SAVE_DIR, new_name_prefix)
                
            elif "-RGB-" in fname:
                print(f"   🌈 [RGB] {fname} -> {new_name_prefix}_xxxxx.jpg")
                process_video(vid_path, RGB_SAVE_DIR, new_name_prefix)

    print("\n✅ 모든 작업 완료!")
    print(f" - 가시광선 저장소: {RGB_SAVE_DIR}")
    print(f" - 적외선 저장소: {IR_SAVE_DIR}")

if __name__ == "__main__":
    main()