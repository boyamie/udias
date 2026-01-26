import os
import glob
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import cv2

def auto_label(source_dir, output_dir, conf_threshold=0.3):
    """
    사전 학습된 YOLO 모델을 사용하여 초벌 라벨링을 수행하는 함수
    """
    # 1. 모델 로드 (가벼운 nano 모델 사용)
    # 처음 실행 시 자동으로 가중치 파일을 다운로드합니다.
    model = YOLO('yolo11n.pt') 
    
    # COCO 데이터셋 기준 'boat'는 클래스 ID 8번입니다.
    TARGET_CLASS_ID = 8 
    
    # 이미지가 있는 폴더 경로
    image_files = glob.glob(os.path.join(source_dir, "*.jpg"))
    
    print(f"--- 자동 라벨링 시작 ---")
    print(f"모델: YOLOv11n (COCO Pre-trained)")
    print(f"대상: {len(image_files)}장")
    print(f"저장: {output_dir}")
    
    # 결과 저장 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    
    for img_path in tqdm(image_files):
        # 2. 추론 (Inference)
        # verbose=False로 로그 줄임
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        # 저장할 텍스트 파일 경로 (이미지와 이름 동일, 확장자만 txt)
        file_name = os.path.basename(img_path)
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        save_path = os.path.join(output_dir, txt_name)
        
        # 3. 결과 파싱 및 저장
        detected_objects = []
        result = results[0] # 한 장이므로 첫 번째 결과
        
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                
                # 'Boat'(8번) 클래스만 저장 (필요시 2:car 등 추가 가능하지만 선박탐지 목적)
                if cls_id == TARGET_CLASS_ID:
                    # YOLO 포맷 좌표 (Normalized xywh: 0~1 사이 값)
                    x, y, w, h = box.xywhn[0].tolist()
                    
                    # 우리가 학습할 때는 'Ship'이 0번 클래스가 되어야 함
                    # 따라서 맨 앞을 0으로 고정
                    line = f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                    detected_objects.append(line)
        
        # 탐지된 객체가 있을 때만 txt 파일 생성 (Empty file 방지 옵션)
        if detected_objects:
            with open(save_path, "w") as f:
                f.writelines(detected_objects)
            count += 1

    print(f"\n[완료] 총 {len(image_files)}장 중 {count}장에 라벨이 생성되었습니다.")
    print(f"라벨이 없는 {len(image_files) - count}장은 배가 탐지되지 않은 이미지입니다.")

if __name__ == "__main__":
    # 경로 설정
    PROJECT_ROOT = Path(r"C:\Users\BohyunKim\Documents\udias")
    
    # 융합된 이미지가 있는 곳
    IMAGE_DIR = PROJECT_ROOT / "data" / "dataset" / "fused_images"
    
    # 라벨을 저장할 곳 (같은 폴더 혹은 labels 폴더 분리)
    # YOLO 학습을 편하게 하려면 보통 images 폴더와 형제 폴더인 labels에 저장합니다.
    LABEL_DIR = PROJECT_ROOT / "data" / "dataset" / "labels"
    
    auto_label(IMAGE_DIR, LABEL_DIR, conf_threshold=0.25)