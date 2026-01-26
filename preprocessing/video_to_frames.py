import cv2
import os
import glob
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_root, interval=30):
    """
    동영상을 읽어 일정 간격으로 프레임을 저장하는 함수 (tqdm 적용)
    """
    v_path = Path(video_path)
    
    # 1. 저장 경로 설정
    parent_folder = v_path.parent.name
    file_stem = v_path.stem
    save_dir = Path(output_root) / parent_folder / file_stem
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 동영상 로드
    cap = cv2.VideoCapture(str(v_path))
    
    if not cap.isOpened():
        print(f"[Error] 파일을 열 수 없습니다: {v_path}")
        return

    # [중요] 전체 프레임 수 가져오기 (이 부분이 누락되어 에러가 발생했습니다)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        total_frames = 0 # 프레임 수를 못 읽을 경우를 대비한 예외처리

    frame_idx = 0
    saved_count = 0
    
    # tqdm 설정: total=total_frames가 반드시 필요합니다.
    with tqdm(total=total_frames, desc=f"Processing {v_path.name}", unit="frame", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # interval 마다 저장
            if frame_idx % interval == 0:
                save_name = f"{file_stem}_frame_{saved_count:05d}.jpg"
                save_path = save_dir / save_name
                
                # 한글 경로 호환 저장
                extension = os.path.splitext(save_name)[1]
                result, encoded_img = cv2.imencode(extension, frame)
                if result:
                    with open(save_path, "wb") as f:
                        encoded_img.tofile(f)
                
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)

    cap.release()
    tqdm.write(f" -> [완료] {v_path.name}: {saved_count}장 저장됨")


if __name__ == "__main__":
    # --- [설정 구간] ---
    # 프로젝트 루트 (현재 위치 기준 상위 폴더로 가정)
    # 필요하다면 절대 경로(예: r"C:\Users\BohyunKim\Documents\udias")를 직접 입력하세요.
    PROJECT_ROOT = Path(r"C:\Users\BohyunKim\Documents\udias")
    
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = DATA_DIR / "images"
    
    TARGET_FOLDERS = ["야간_원본", "주간_원본"]
    FRAME_INTERVAL = 15
    # ------------------

    print(f"[시작] 프레임 추출을 시작합니다. (간격: {FRAME_INTERVAL} 프레임)\n")
    
    for folder_name in TARGET_FOLDERS:
        search_pattern = DATA_DIR / folder_name / "*"
        video_files = glob.glob(str(search_pattern))
        
        valid_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        target_videos = [f for f in video_files if os.path.splitext(f)[-1].lower() in valid_extensions]
        
        if not target_videos:
            print(f"⚠️  '{folder_name}' 폴더에 동영상이 없습니다. 경로를 확인해주세요: {search_pattern}")
            continue

        print(f"📂 폴더 '{folder_name}' 처리 중 (총 {len(target_videos)}개 파일)")
        
        for i, video_path in enumerate(target_videos, 1):
            print(f"[{i}/{len(target_videos)}] ", end="")
            extract_frames(video_path, OUTPUT_DIR, interval=FRAME_INTERVAL)
            
    print("\n[종료] 모든 작업이 끝났습니다.")