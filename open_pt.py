import torch
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('E:/ultralytics-main/yolov8n.pt')

# Access the YOLOv8n model's attributes
model = model.model

# Modify the nc attribute to 3
#model.nc = 3

# Save the modified model
#torch.save(model.state_dict(), 'modified_yolov8n-seg.pt')