import os
import shutil
import random
import glob
from pathlib import Path
from tqdm import tqdm

def split_dataset(root_dir, train_ratio=0.8):
    # 1. 경로 설정
    # 현재: data/dataset/fused_images, data/dataset/labels
    src_images = root_dir / "fused_images"
    src_labels = root_dir / "labels"
    
    # 목표: data/dataset/yolo_data/images/train, .../val
    dest_root = root_dir / "yolo_data"
    
    for split in ["train", "val"]:
        for dtype in ["images", "labels"]:
            os.makedirs(dest_root / dtype / split, exist_ok=True)
            
    # 2. 파일 리스트 확보
    print("파일 목록을 읽는 중...")
    image_files = glob.glob(str(src_images / "*.jpg"))
    random.shuffle(image_files) # 랜덤으로 섞기
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"총 {len(image_files)}장 -> 학습용(Train): {len(train_files)}장, 검증용(Val): {len(val_files)}장")
    
    # 3. 파일 이동 (복사) 함수
    def copy_files(files, split_name):
        print(f"Moving {split_name} data...")
        for img_path in tqdm(files):
            # 이미지 복사
            file_name = os.path.basename(img_path)
            shutil.copy(img_path, dest_root / "images" / split_name / file_name)
            
            # 라벨 복사 (있으면)
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            label_src = src_labels / txt_name
            
            if label_src.exists():
                shutil.copy(label_src, dest_root / "labels" / split_name / txt_name)

    # 4. 실행
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    
    print(f"\n✅ 데이터 준비 완료! 저장 경로: {dest_root}")
    
    # 5. data.yaml 자동 생성 (학습 설정 파일)
    yaml_content = f"""
path: {dest_root.absolute().as_posix()} # 데이터셋 루트 경로
train: images/train  # 학습 이미지 (path 기준 상대 경로)
val: images/val      # 검증 이미지

# 클래스 정보
nc: 1          # 클래스 개수 (Ship 하나뿐)
names: ['Ship'] # 클래스 이름
"""
    yaml_path = dest_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"📄 설정 파일 생성됨: {yaml_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path("/workspace/uda")
    DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
    
    split_dataset(DATASET_DIR)