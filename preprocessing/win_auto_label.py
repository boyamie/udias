from ultralytics import YOLO
from pathlib import Path
import sys

# --- [설정: 사용자 환경에 맞게 자동 설정] ---
# 1. 모델 경로 (바탕화면에 다운로드 받은 파일)
MODEL_PATH = Path.home() / "Desktop" / "yolo11s_best.pt"

# 2. 라벨링할 8000장 이미지가 있는 폴더 (경로가 맞는지 꼭 확인하세요!)
# 문자열 앞에 r을 붙여야 윈도우 경로(\) 에러가 안 납니다.
IMAGE_DIR = Path(r"C:\Users\BohyunKim\Documents\udias\data\raw_images_8000")

# 3. 라벨이 저장될 위치 (이미지 폴더 안에 'auto_labels'라는 폴더가 생깁니다)
OUTPUT_DIR = IMAGE_DIR / "auto_labels"
# ----------------------------------------

def main():
    # 0. 경로 확인
    if not MODEL_PATH.exists():
        print(f"❌ 모델을 찾을 수 없습니다: {MODEL_PATH}")
        print("바탕화면에 'yolo11s_best.pt' 파일이 있는지 확인해주세요.")
        return

    if not IMAGE_DIR.exists():
        print(f"❌ 이미지 폴더를 찾을 수 없습니다: {IMAGE_DIR}")
        print("폴더 경로가 정확한지 확인해주세요.")
        return

    # 1. 모델 로드
    print(f"🔥 모델 로드 중... ({MODEL_PATH.name})")
    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"❌ 모델 로드 실패. ultralytics가 설치되어 있나요? 에러: {e}")
        return

    # 2. 이미지 파일 찾기 (jpg, png 등)
    print(f"📂 이미지 검색 중: {IMAGE_DIR}")
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.MOV", "*.mp4"] # 동영상 파일이 섞여있을 경우 대비
    image_files = []
    for ext in extensions:
        image_files.extend(list(IMAGE_DIR.glob(ext)))
    
    # 이미지 파일만 골라내기 (동영상 제외, 확장자 필터링)
    valid_images = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    if not valid_images:
        print("⚠️ 처리할 이미지가 없습니다. 폴더가 비어있나요?")
        return

    print(f"🚀 총 {len(valid_images)}장의 이미지에 대해 라벨링을 시작합니다.")
    print(f"   - 사용 장비: GPU (device=0, RTX 4060)")
    print(f"   - 저장 위치: {OUTPUT_DIR}")

    # 3. 추론 및 라벨 저장 (Auto-Labeling)
    # batch=16 정도로 설정하여 속도 향상
    model.predict(
        source=str(IMAGE_DIR), 
        conf=0.5,        # 50% 이상 확신하는 드론만 라벨링 (오탐지 방지)
        iou=0.45,        # 중복 박스 제거 기준
        save_txt=True,   # 라벨 파일(.txt) 저장 필수
        save_conf=False, # 라벨 파일에 정확도 점수는 뺌 (YOLO 학습용 포맷 준수)
        save=False,      # 이미지 위에 박스 그린 그림은 저장 안 함 (용량/시간 절약)
        device=0,        # 0번 GPU 사용 (RTX 4060)
        project=str(IMAGE_DIR), # 저장 경로 (상위 폴더)
        name="auto_labels",     # 저장 경로 (하위 폴더 이름)
        exist_ok=True,    # 폴더가 있어도 덮어쓰기 가능
        stream=True       # 대량 데이터 처리 시 메모리 절약 (필수)
    )
    
    # predict()를 stream=True로 실행하면 제너레이터가 반환되므로, 
    # 루프를 돌려야 실제로 실행됩니다.
    print("⏳ 진행 중...")
    count = 0
    # 진행 상황을 보기 위해 tqdm 대신 간단한 로직 사용 (윈도우 호환성 위함)
    results = model.predict(
        source=str(IMAGE_DIR),
        conf=0.5,
        save_txt=True,
        save_conf=False,
        save=False,
        device=0,
        project=str(IMAGE_DIR),
        name="auto_labels",
        exist_ok=True,
        stream=True
    )
    
    for _ in results:
        count += 1
        if count % 100 == 0:
            print(f"   -> {count} / {len(valid_images)} 장 처리 완료", end='\r')

    print(f"\n\n✅ 모든 작업 완료!")
    print(f"결과 확인: {IMAGE_DIR / 'auto_labels' / 'labels'}")

if __name__ == "__main__":
    main()