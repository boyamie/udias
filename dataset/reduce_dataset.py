import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

def reduce_dataset(source_img_dir, source_lbl_dir, dest_dir, interval=10):
    # 1. 저장할 폴더 생성 (images, labels 분리)
    dest_img_dir = os.path.join(dest_dir, "images")
    dest_lbl_dir = os.path.join(dest_dir, "labels")
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)
    
    # 2. 이미지 리스트 가져오기 (정렬 필수!)
    images = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")))
    
    print(f"--- 데이터셋 축소 시작 ---")
    print(f"원본 개수: {len(images)}장")
    print(f"추출 간격: {interval}장마다 1장")
    
    count = 0
    
    # 3. 간격에 맞춰 복사
    for i in tqdm(range(0, len(images), interval)):
        img_path = images[i]
        file_name = os.path.basename(img_path)
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        
        # 라벨 경로
        lbl_path = os.path.join(source_lbl_dir, txt_name)
        
        # (옵션) 라벨이 있는 파일만 가져갈 것인가?
        # 연구 초기에는 라벨이 있는(배가 있는) 데이터가 중요하므로
        # 라벨 파일이 없으면 스킵하는 전략도 가능합니다.
        # 여기서는 배가 있든 없든 10장마다 가져옵니다 (배경 이미지 포함).
        
        # 복사 수행
        shutil.copy(img_path, os.path.join(dest_img_dir, file_name))
        
        # 라벨이 있으면 같이 복사
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dest_lbl_dir, txt_name))
            
        count += 1
        
    print(f"\n[완료] 총 {count}장의 데이터가 '{dest_dir}'로 이동되었습니다.")
    print(f"이제 LabelImg에서 '{dest_dir}/images' 폴더를 여세요.")

if __name__ == "__main__":
    PROJECT_ROOT = Path(r"C:\Users\BohyunKim\Documents\udias")
    
    # 원본 위치
    SRC_IMG = PROJECT_ROOT / "data" / "dataset" / "fused_images"
    SRC_LBL = PROJECT_ROOT / "data" / "dataset" / "labels"
    
    # 이사갈 위치 (최종 학습용)
    DEST = PROJECT_ROOT / "data" / "dataset" / "final_train_data"
    
    reduce_dataset(SRC_IMG, SRC_LBL, DEST, interval=10)