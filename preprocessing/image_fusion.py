import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import yaml

# --- [설정] ---
BASE_DIR = Path("/workspace/uda/data/dataset")
RGB_DIR = BASE_DIR / "visible_images"
IR_DIR = BASE_DIR / "infrared_images"
SAVE_DIR = BASE_DIR / "yolo_data_paper_fusion"  # 새로운 데이터셋 저장 경로

# 논문 파라미터 (Section 3.1.1 & 3.2.2)
K_SHARP = 1.0  # 선명화 계수 (k)
EPSILON = 10   # 조절 파라미터 (epsilon)

def sharpen_image(image, k=K_SHARP):
    """논문 식 (1): Laplacian을 이용한 이미지 선명화"""
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian = cv2.filter2D(image, -1, kernel)
    sharpened = cv2.addWeighted(image, 1.0, laplacian, k, 0)
    return sharpened

def align_and_fuse(rgb_path, ir_path):
    # 1. 이미지 로드
    img_rgb = cv2.imread(str(rgb_path))
    img_ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    
    if img_rgb is None or img_ir is None:
        return None

    # 2. 선명화 (Sharpening) - 논문 Section 3.1.1
    img_rgb_sharp = sharpen_image(img_rgb)
    img_ir_sharp = sharpen_image(img_ir)

    # 3. SURF 특징점 매칭 및 정렬 (Alignment) - 논문 Section 3.1.2
    # 논문의 복잡한 Pixel Remapping 대신 Homography로 효율적 구현
    surf = cv2.xfeatures2d.SURF_create(400)
    
    # RGB를 그레이스케일로 변환해 특징점 찾기
    kp1, des1 = surf.detectAndCompute(cv2.cvtColor(img_rgb_sharp, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = surf.detectAndCompute(img_ir_sharp, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # 매칭 실패 시 그냥 리사이즈해서 융합 (Fallback)
        img_ir_aligned = cv2.resize(img_ir, (img_rgb.shape[1], img_rgb.shape[0]))
    else:
        # 매칭
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 상위 매칭점 추출
        good_matches = matches[:50]
        if len(good_matches) < 4:
             img_ir_aligned = cv2.resize(img_ir, (img_rgb.shape[1], img_rgb.shape[0]))
        else:
            src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Homography 행렬 계산 및 IR 이미지를 RGB 시점으로 변환
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                 img_ir_aligned = cv2.resize(img_ir, (img_rgb.shape[1], img_rgb.shape[0]))
            else:
                img_ir_aligned = cv2.warpPerspective(img_ir, M, (img_rgb.shape[1], img_rgb.shape[0]))

    # 4. 픽셀 재구성 (Pixel Reconstruction) - 논문 식 (5)
    # RGB_Enhanced = (RGB + epsilon) * Infrared / 255
    
    img_rgb_float = img_rgb.astype(np.float32)
    img_ir_float = img_ir_aligned.astype(np.float32)
    
    # IR 이미지를 3채널로 확장 (계산을 위해)
    img_ir_3ch = cv2.merge([img_ir_float, img_ir_float, img_ir_float])

    # 수식 적용
    fused = (img_rgb_float + EPSILON) * (img_ir_3ch / 255.0)
    
    # 클리핑 (0~255 범위 맞추기)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    
    return fused

def process_dataset():
    # 저장 폴더 생성
    for split in ["train", "val"]:
        (SAVE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 기존 yolo_data(참조용)에서 파일 리스트 가져오기
    REF_DIR = BASE_DIR / "yolo_data"
    
    print(f"🚀 논문 방식(Huang et al. 2026) 데이터 생성 시작...")
    
    for split in ["train", "val"]:
        image_files = list((REF_DIR / "images" / split).glob("*.jpg"))
        
        for img_file in tqdm(image_files, desc=f"Processing {split}"):
            fname = img_file.name
            
            # 원본 RGB, IR 파일 경로
            rgb_path = RGB_DIR / fname
            ir_path = IR_DIR / fname
            
            if not rgb_path.exists() or not ir_path.exists():
                continue
                
            # 1. 융합 (Alignment + Fusion)
            fused_img = align_and_fuse(rgb_path, ir_path)
            
            if fused_img is not None:
                # 2. 이미지 저장
                cv2.imwrite(str(SAVE_DIR / "images" / split / fname), fused_img)
                
                # 3. 라벨 복사 (기존 라벨 그대로 사용)
                label_name = fname.replace(".jpg", ".txt")
                src_label = REF_DIR / "labels" / split / label_name
                dst_label = SAVE_DIR / "labels" / split / label_name
                
                if src_label.exists():
                    import shutil
                    shutil.copy(src_label, dst_label)

    # data.yaml 생성
    yaml_content = {
        'path': str(SAVE_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['Ship']
    }
    with open(SAVE_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f)

    print(f"✅ 데이터셋 생성 완료: {SAVE_DIR}")

if __name__ == "__main__":
    process_dataset()