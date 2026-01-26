import cv2
import numpy as np
import os
import glob
from pathlib import Path

def imread_korean(path):
    """ 한글 경로 이미지 읽기 """
    try:
        stream = open(path.encode("utf-8"), "rb")
    except:
        stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpy_array = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

def apply_clahe(img):
    """ 
    이미지 명암비 향상 (야간 데이터 필수 전처리)
    RGB 이미지를 LAB 색공간으로 변환 후 L(밝기) 채널에만 CLAHE 적용
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE 객체 생성 (Limit: 3.0, Grid: 8x8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def find_ir_counterpart(rgb_path, search_dir):
    """ RGB -> IR 파일 찾기 """
    rgb_name = os.path.basename(rgb_path)
    if "RGB" in rgb_name:
        ir_name = rgb_name.replace("RGB", "IR")
    elif "rgb" in rgb_name:
        ir_name = rgb_name.replace("rgb", "ir")
    else:
        return None
    
    for root, dirs, files in os.walk(search_dir):
        if ir_name in files:
            return os.path.join(root, ir_name)
    return None

def align_images(img_rgb, img_ir, save_debug_path=None):
    """ RGB-IR 정렬 (CLAHE 적용 + 디버깅 강화) """
    
    # [전처리] 야간 데이터 대응을 위해 밝기/대비 향상
    img_rgb_enhanced = apply_clahe(img_rgb)
    img_ir_enhanced = cv2.normalize(img_ir, None, 0, 255, cv2.NORM_MINMAX) # IR도 정규화

    # 1. 흑백 변환
    gray_rgb = cv2.cvtColor(img_rgb_enhanced, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.cvtColor(img_ir_enhanced, cv2.COLOR_BGR2GRAY)

    # 2. 특징점 검출기 (SIFT 사용 - 특허 문제 없음)
    detector = cv2.SIFT_create()
    
    # 3. 키포인트 검출
    kp1, des1 = detector.detectAndCompute(gray_rgb, None)
    kp2, des2 = detector.detectAndCompute(gray_ir, None)

    # [디버깅] 특징점 개수 확인
    print(f"   -> 특징점 개수: RGB({len(kp1)}), IR({len(kp2)})")
    
    # 특징점이 너무 적으면 실패 처리 (FLANN 에러 방지)
    if len(kp1) < 5 or len(kp2) < 5:
        print("   [Fail] 특징점이 너무 적어 매칭 불가.")
        return None, None

    # 4. 매칭 (FLANN)
    index_params = dict(algorithm=1, trees=5) # KD-Tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception as e:
        print(f"   [Error] 매칭 중 FLANN 에러: {e}")
        return None, None

    # 5. 좋은 매칭점 선별 (Ratio Test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"   -> 유효 매칭 점(Good Matches): {len(good_matches)}개")

    # [디버깅] 매칭 결과 이미지를 저장 (어디가 매칭됐는지 확인용)
    if save_debug_path and len(good_matches) > 0:
        debug_img = cv2.drawMatches(gray_rgb, kp1, gray_ir, kp2, good_matches[:20], None, flags=2)
        cv2.imwrite(save_debug_path + "_match_debug.jpg", debug_img)

    if len(good_matches) < 4:
        print("   [Fail] 호모그래피 계산을 위한 매칭 점 부족.")
        return None, None

    # 6. 호모그래피 계산
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("   [Fail] 호모그래피 행렬을 찾을 수 없습니다.")
        return None, None

    # 7. 변환
    height, width = img_rgb.shape[:2]
    aligned_ir = cv2.warpPerspective(img_ir, H, (width, height))

    return aligned_ir, H

def visualize_overlay(img1, img2, save_path):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    
    extension = os.path.splitext(save_path)[1]
    result, encoded_img = cv2.imencode(extension, overlay)
    if result:
        with open(save_path, "wb") as f:
            encoded_img.tofile(f)

if __name__ == "__main__":
    PROJECT_ROOT = Path(r"C:\Users\BohyunKim\Documents\udias")
    IMAGE_DIR = PROJECT_ROOT / "data" / "images"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "aligned_test"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # [중요] 주간 데이터를 먼저 찾도록 정렬 (Reverse=True하면 주간이 먼저 올 확률 높음 or 명시적 필터)
    all_folders = glob.glob(str(IMAGE_DIR / "*"))
    
    # '주간'이 포함된 폴더를 우선적으로 리스트 앞쪽으로 배치
    sorted_folders = sorted(all_folders, key=lambda x: "주간" not in os.path.basename(x))

    print(f"검색된 폴더: {len(sorted_folders)}개")
    
    for folder in sorted_folders:
        folder_name = os.path.basename(folder)
        rgb_images = glob.glob(os.path.join(folder, "**", "*RGB*.jpg"), recursive=True)
        
        if not rgb_images:
            continue
            
        print(f"\n📂 폴더 테스트: '{folder_name}' ({len(rgb_images)}장)")
        
        # 각 폴더에서 5장씩 테스트
        success_count = 0
        for i, rgb_path in enumerate(rgb_images[:5]): 
            print(f"\n[{i+1}] {os.path.basename(rgb_path)}")
            
            ir_path = find_ir_counterpart(rgb_path, IMAGE_DIR)
            
            if ir_path and os.path.exists(ir_path):
                img_rgb = imread_korean(rgb_path)
                img_ir = imread_korean(ir_path)
                
                # 디버그 이미지 저장 경로
                debug_path = os.path.join(OUTPUT_DIR, f"test_{folder_name}_{i}")
                
                aligned_ir, H = align_images(img_rgb, img_ir, save_debug_path=debug_path)
                
                if aligned_ir is not None:
                    save_name = debug_path + "_result.jpg"
                    visualize_overlay(img_rgb, aligned_ir, save_name)
                    print(f"   -> [성공] 저장됨: {save_name}")
                    success_count += 1
                else:
                    print("   -> 정렬 실패")
            else:
                print("   -> 짝꿍 IR 없음")
        
        if success_count > 0:
            print("\n✅ 성공했습니다! data/aligned_test 폴더를 확인하세요.")
            break