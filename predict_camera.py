from ultralytics import YOLO
from PIL import Image
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2


model = YOLO('E:/V7/ultralytics-main/runs/segment/1 Solid Waste Training  100 epochs/weights/best.pt')
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0", show=True, save=True, hide_labels=False,conf=0.5) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("can.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("can.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
results = model.predict(source=[im1, im2])
 