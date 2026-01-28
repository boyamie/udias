from ultralytics import YOLO

# --- [설정] ---
MODEL_NAME = 'yolo11m.pt' 
DATA_YAML = '/workspace/uda/data/dataset/yolo_data_paper_fusion/data.yaml'
PROJECT_DIR = '/workspace/uda/data/runs/paper_experiment'

print("🔥 논문 방식(Paper Fusion) 실험 시작...")

model = YOLO(MODEL_NAME)
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    device=3,  # H100 사용
    project=PROJECT_DIR,
    name='train_paper_method',
    exist_ok=True
)

print("✅ 실험 종료!")