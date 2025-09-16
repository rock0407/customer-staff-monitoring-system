from ultralytics import YOLO
import numpy as np
from config_loader import YOLO_MODEL_PATH

class PersonDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Use larger input size for better detection accuracy
        # This will help with detecting people at different distances
        results = self.model(frame, imgsz=640)  # Use 640x640 instead of default 384x640
        boxes = []
        confidences = []
        for r in results:
            for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(cls) == 0:  # class 0 is 'person' in COCO
                    boxes.append(box)
                    confidences.append(conf)
        return np.array(boxes), np.array(confidences) 