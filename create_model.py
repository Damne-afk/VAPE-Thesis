from ultralytics import YOLO
import yaml
from ultralytics.yolo.engine.model import YOLO



model = YOLO("E:/ultralytics-main/yolov8-seg.yaml")  # build a YOLOv8n model from scratch
# YOLO("model.pt")  use pre-trained model if available
model.info()  # display model information


if __name__ == '__main__':
   model.train(data="data.yaml", epochs=3,name=('yolov8n_custom'))  # train the model










#load model

#model = YOLO('yolov8-seg.yaml')
 
# Training.
#results = model.train(
   #data='data.yaml',
   #imgsz=640,
   #epochs=10,
   #batch=8,
   #name='yolov8n_custom')