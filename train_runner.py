import yaml
import os
from ultralytics import YOLO
from pathlib import Path

def train_yolo(config_path):
    # 1. 설정 파일 로드
    print(f"⚙️ 설정 파일을 읽는 중: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 모델 초기화
    print(f"🚢 모델 로드 중: {cfg['model_type']}")
    model = YOLO(cfg['model_type'])

    # 3. 학습 시작
    print("🚀 학습을 시작합니다! (설정된 파라미터 적용)")
    print(f"   - Epochs: {cfg['epochs']}")
    print(f"   - Batch: {cfg['batch_size']}")
    print(f"   - Device: {cfg['device']}")
    print(f"   - Workers: {cfg['workers']} (Windows 최적화)")

    results = model.train(
        data=cfg['data_path'],
        epochs=cfg['epochs'],
        imgsz=cfg['img_size'],
        batch=cfg['batch_size'],
        device=cfg['device'],
        workers=cfg['workers'], # 윈도우 필수
        project=cfg['project_dir'],
        name=cfg['experiment_name'],
        pretrained=cfg['pretrained'],
        optimizer=cfg['optimizer'],
        verbose=cfg['verbose']
    )
    
    print(f"\n✅ 학습 완료! 결과 저장 위치: {cfg['project_dir']}/{cfg['experiment_name']}")

if __name__ == "__main__":
    # 프로젝트 루트 기준 경로 설정
    # 현재 위치(code/) 기준으로 config 파일 찾기
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_FILE = BASE_DIR / "config" / "train_config.yaml"
    
    # PyYAML 라이브러리가 필요합니다 (없으면 pip install pyyaml)
    try:
        train_yolo(CONFIG_FILE)
    except ImportError:
        print("❌ 'pyyaml' 모듈이 없습니다. 'pip install pyyaml'을 실행해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")