from ultralytics import YOLO
from ultralytics.yolo.engine.model import YOLO

model = YOLO('yolov8s-seg.pt')  

if __name__ == '__main__':
    results = model.train(data='data.yaml', epochs=200, imgsz=640, batch=8)  # train the model
 
 
 



