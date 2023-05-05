from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

# Load a model

model = YOLO('E:/ultralytics-main/runs/segment/train61/weights/best.pt')  # load a custom model



# Predict with the model
results = model('E:/ultralytics-main/Testing Kit/Scene4-Test1-Random.jpg', imgsz=640, show=True, conf=0.5)  # predict on an image
#results = model.predict(source='E:/ultralytics-main/Testing Kit/Test Vid bottles.mp4', imgsz=640, show=True, conf=0.5)

