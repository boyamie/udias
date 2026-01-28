import cv2
from ultralytics import YOLO
from pathlib import Path
import shutil

# --- [설정] ---
# 1. 모델 경로 (바탕화면)
MODEL_PATH = Path.home() / "Desktop" / "yolo11s_best.pt"

# 2. 데이터 기본 경로 (사용자 경로)
BASE_DIR = Path(r"C:\Users\BohyunKim\Documents\udias\data")

# 3. 동영상이 들어있는 폴더 이름들 (정확한 폴더명 입력)
TARGET_DIRS = ["주간_원본", "야간_원본"]

# 4. 결과 저장 경로 (자동 생성됨)
SAVE_DIR = BASE_DIR / "new_8000_dataset"

# 5. 프레임 간격 (몇 프레임마다 1장씩 뽑을지)
FRAME_INTERVAL = 15  # 15프레임마다 저장 (너무 자주 뽑으면 데이터 중복됨)
# -------------

def main():
    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        return

    print(f"🔥 모델 로드 중... ({MODEL_PATH.name})")
    model = YOLO(str(MODEL_PATH))

    # 저장 폴더 생성
    (SAVE_DIR / "images").mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "labels").mkdir(parents=True, exist_ok=True)

    print(f"🚀 동영상 자동 라벨링 시작! (간격: {FRAME_INTERVAL}프레임)")
    print(f"   대상 폴더: {TARGET_DIRS}")
    print(f"   저장 위치: {SAVE_DIR}")

    total_saved = 0
    
    for folder_name in TARGET_DIRS:
        src_dir = BASE_DIR / folder_name
        if not src_dir.exists():
            print(f"⚠️ 폴더 없음, 건너뜀: {src_dir}")
            continue
            
        # 동영상 파일 검색
        video_files = list(src_dir.glob("*"))
        # 확장자 필터링 (mp4, mov, avi 등)
        video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']]

        for vid_path in video_files:
            print(f"   🎥 처리 중: {vid_path.name}")
            
            cap = cv2.VideoCapture(str(vid_path))
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 지정한 간격마다 처리
                if frame_count % FRAME_INTERVAL == 0:
                    # YOLO 추론 (이미지 한 장)
                    results = model.predict(
                        frame, 
                        conf=0.6,       # 60% 이상 확신할 때만 저장 (엄격하게)
                        verbose=False,
                        device=0        # RTX 4060
                    )
                    
                    # 드론이 감지된 경우에만 저장
                    if len(results[0].boxes) > 0:
                        # 파일명 생성: 폴더명_파일명_프레임번호
                        safe_name = f"{folder_name}_{vid_path.stem}_{frame_count:06d}"
                        
                        # 1. 이미지 저장
                        cv2.imwrite(str(SAVE_DIR / "images" / f"{safe_name}.jpg"), frame)
                        
                        # 2. 라벨 저장 (YOLO 포맷 txt)
                        with open(SAVE_DIR / "labels" / f"{safe_name}.txt", "w") as f:
                            for box in results[0].boxes:
                                # class x_center y_center width height
                                cls = int(box.cls[0])
                                x, y, w, h = box.xywhn[0]
                                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        
                        total_saved += 1
                        print(f"      [+] 저장됨: {safe_name}.jpg (Total: {total_saved})", end='\r')
                
                frame_count += 1
            
            cap.release()
            print("") # 줄바꿈

    print(f"\n✅ 모든 작업 완료!")
    print(f"총 {total_saved}장의 라벨링된 데이터가 생성되었습니다.")
    print(f"폴더 위치: {SAVE_DIR}")

if __name__ == "__main__":
    main()