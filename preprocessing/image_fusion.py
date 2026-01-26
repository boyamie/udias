import cv2
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm

# --- [1. 헬퍼 함수 정의] ---

def imread_korean(path):
    """ 한글 경로 지원 이미지 읽기 """
    try:
        stream = open(path.encode("utf-8"), "rb")
    except:
        stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpy_array = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

def imwrite_korean(path, img):
    """ 한글 경로 지원 이미지 저장 """
    extension = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(extension, img)
    if result:
        with open(path, "wb") as f:
            encoded_img.tofile(f)
            return True
    return False

def apply_clahe(img):
    """ 야간 데이터 특징점 검출을 위한 대비 향상 """
    if img is None: return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def find_ir_counterpart(rgb_path, search_dir):
    """ RGB 파일명에 대응하는 IR 파일 찾기 """
    rgb_name = os.path.basename(rgb_path)
    if "RGB" in rgb_name:
        ir_name = rgb_name.replace("RGB", "IR")
    elif "rgb" in rgb_name:
        ir_name = rgb_name.replace("rgb", "ir")
    else:
        return None
    
    # 동일 폴더 내 검색 (속도 최적화)
    parent_dir = os.path.dirname(rgb_path)
    # IR 폴더가 RGB 폴더와 형제 관계일 경우를 대비해 상위 폴더 이름 치환
    # 예: .../Day_RGB/... -> .../Day_IR/...
    if "RGB" in parent_dir:
        ir_parent = parent_dir.replace("RGB", "IR")
    else:
        # 폴더 구조가 불명확할 경우 전체 검색 (느림)
        for root, dirs, files in os.walk(search_dir):
            if ir_name in files:
                return os.path.join(root, ir_name)
        return None

    candidate = os.path.join(ir_parent, ir_name)
    if os.path.exists(candidate):
        return candidate
    
    # 못 찾았으면 재귀 검색으로 fallback
    for root, dirs, files in os.walk(search_dir):
        if ir_name in files:
            return os.path.join(root, ir_name)
    return None

def process_alignment_and_fusion(img_rgb, img_ir):
    """ 정렬 및 융합 수행 (성공 시 융합 이미지 반환, 실패 시 None) """
    
    # 1. 특징점 추출을 위한 전처리 (CLAHE)
    img_rgb_clahe = apply_clahe(img_rgb)
    img_ir_norm = cv2.normalize(img_ir, None, 0, 255, cv2.NORM_MINMAX)

    gray_rgb = cv2.cvtColor(img_rgb_clahe, cv2.COLOR_BGR2GRAY)
    gray_ir = cv2.cvtColor(img_ir_norm, cv2.COLOR_BGR2GRAY)

    # 2. SIFT 특징점 검출
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(gray_rgb, None)
    kp2, des2 = detector.detectAndCompute(gray_ir, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return None # 특징점 부족

    # 3. 매칭
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return None

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 최소 매칭 점 개수 기준 (너무 적으면 정렬이 부정확함)
    if len(good_matches) < 10: 
        return None

    # 4. 호모그래피 계산
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    # 5. 정렬 (Warp)
    h, w = img_rgb.shape[:2]
    aligned_ir = cv2.warpPerspective(img_ir, H, (w, h))

    # 6. 융합 (Fusion) - 50:50 가중치 합
    # 논문 베이스 연구를 위해 가장 기초적인 Pixel-level Fusion 적용
    fused_img = cv2.addWeighted(img_rgb, 0.5, aligned_ir, 0.5, 0)

    return fused_img


# --- [2. 메인 실행부] ---

if __name__ == "__main__":
    # 경로 설정
    PROJECT_ROOT = Path(r"C:\Users\BohyunKim\Documents\udias")
    IMAGE_DIR = PROJECT_ROOT / "data" / "images"
    
    # 결과 저장 경로 (YOLO 학습용 데이터셋 폴더)
    SAVE_DIR = PROJECT_ROOT / "data" / "dataset" / "fused_images"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"=== 듀얼 스펙트럼 융합 데이터셋 생성 시작 ===")
    print(f"소스 경로: {IMAGE_DIR}")
    print(f"저장 경로: {SAVE_DIR}")

    # 모든 RGB 이미지 검색
    rgb_files = glob.glob(os.path.join(IMAGE_DIR, "**", "*RGB*.jpg"), recursive=True)
    print(f"총 처리 대상 파일: {len(rgb_files)}개")

    success_count = 0
    fail_count = 0

    # TQDM으로 진행률 표시
    pbar = tqdm(rgb_files, unit="img")
    
    for rgb_path in pbar:
        # 파일명 파싱
        file_name = os.path.basename(rgb_path)
        
        # 1. IR 짝꿍 찾기
        ir_path = find_ir_counterpart(rgb_path, IMAGE_DIR)
        
        if not ir_path or not os.path.exists(ir_path):
            fail_count += 1
            continue

        # 2. 이미지 읽기
        img_rgb = imread_korean(rgb_path)
        img_ir = imread_korean(ir_path)

        # 3. 정렬 및 융합 시도
        fused_result = process_alignment_and_fusion(img_rgb, img_ir)
        
        if fused_result is not None:
            # 4. 저장
            # 파일명 규칙: Fused_원본이름.jpg
            save_name = f"Fused_{file_name}"
            save_path = os.path.join(SAVE_DIR, save_name)
            
            imwrite_korean(save_path, fused_result)
            success_count += 1
            pbar.set_description(f"성공: {success_count} / 실패: {fail_count}")
        else:
            fail_count += 1
            # 실패한 경우 로그를 너무 많이 남기지 않고 진행 바에만 표시

    print("\n" + "="*30)
    print(f"작업 완료!")
    print(f" - 성공적으로 생성된 이미지: {success_count}장")
    print(f" - 실패(매칭부족/짝없음): {fail_count}장")
    print(f" - 저장 위치: {SAVE_DIR}")
    print("="*30)