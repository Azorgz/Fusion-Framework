from ultralytics import YOLO

# -------------------------------------------------
# Configuration
# -------------------------------------------------
VIDEO_PATH = "/home/godeta/PycharmProjects/FusionMethods/results/videos/intro2.mp4"
OUTPUT_PATH = "/home/godeta/PycharmProjects/FusionMethods/results/videos/intro_detected.mp4"
MODEL_NAME = "yolo26x.pt"
DEVICE = "cuda"

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = YOLO(MODEL_NAME, task='detect')

model.predict(
    source=VIDEO_PATH,
    save=True,
    classes=list(range(1)),
    conf=0.6)