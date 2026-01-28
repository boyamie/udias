import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

# --- [설정] ---
BASE_DIR = Path("/workspace/uda/data")
# 윈도우에서 올린 정렬된 데이터 경로
RGB_SRC = BASE_DIR / "dataset/aligned_dataset/visible_images_aligned"
IR_SRC = BASE_DIR / "dataset/aligned_dataset/infrared_images_aligned"
LABEL_SRC = BASE_DIR / "dataset/aligned_dataset/labels_aligned"

OUTPUT_DIR = BASE_DIR / "dataset/yolo_data_simple_aligned"
# ----------------

def simple_fuse(rgb_path, ir_path):
    img_rgb = cv2.imread(str(rgb_path))
    img_ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    
    if img_rgb is None or img_ir is None: return None
    
    # 크기만 맞춤 (정렬 X)
    img_ir_resized = cv2.resize(img_ir, (img_rgb.shape[1], img_rgb.shape[0]))
    img_ir_3ch = cv2.merge([img_ir_resized]*3)
    
    # 단순 평균 (50:50) -> 물 위의 선박은 열화상대비가 뚜렷해서 단순 융합도 효과적일 수 있음
    fused = cv2.addWeighted(img_rgb, 0.5, img_ir_3ch, 0.5, 0)
    return fused

def main():
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    files = list(RGB_SRC.glob("*.jpg"))
    import random
    random.shuffle(files)
    
    split_idx = int(len(files) * 0.8)
    
    print("🚀 [Ship Detection] 단순 융합 데이터셋 생성 중...")
    
    for i, rgb_path in enumerate(tqdm(files)):
        fname = rgb_path.name
        ir_path = IR_SRC / fname
        label_path = LABEL_SRC / fname.replace(".jpg", ".txt")
        
        if not ir_path.exists() or not label_path.exists(): continue
        
        split = "train" if i < split_idx else "val"
        
        fused = simple_fuse(rgb_path, ir_path)
        if fused is not None:
            cv2.imwrite(str(OUTPUT_DIR / "images" / split / fname), fused)
            shutil.copy(str(label_path), str(OUTPUT_DIR / "labels" / split / label_path.name))

    # [중요] 클래스 이름을 'Ship'으로 설정
    yaml_content = {
        'path': str(OUTPUT_DIR), 
        'train': 'images/train', 
        'val': 'images/val', 
        'nc': 1, 
        'names': ['Ship'] 
    }
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f)
        
    print("✅ 완료! data.yaml의 클래스 이름이 ['Ship']으로 설정되었습니다.")

if __name__ == "__main__":
    main()