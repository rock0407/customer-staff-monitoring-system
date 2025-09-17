from ultralytics import YOLO
import numpy as np
import logging
from config_loader import YOLO_MODEL_PATH

class PersonDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Run YOLO detection and return Nx4 xyxy and confidences arrays for person class.
        Handles Ultralytics API variations where results/boxes can differ across versions.
        """
        try:
            results = self.model.predict(frame, imgsz=640, verbose=False)
        except Exception:
            results = self.model(frame, imgsz=640)
        boxes_out = []
        confs_out = []

        # Ensure we can iterate results consistently
        try:
            results_iter = list(results) if not isinstance(results, (list, tuple)) else results
        except Exception:
            results_iter = [results]

        for res in results_iter:
            if res is None:
                continue
            boxes_obj = getattr(res, 'boxes', None)
            if boxes_obj is None:
                continue

            # Most stable path across versions: boxes.data -> Nx6 [x1,y1,x2,y2,conf,cls]
            try:
                data = getattr(boxes_obj, 'data', None)
                if data is not None:
                    data_np = data.cpu().numpy() if hasattr(data, 'cpu') else np.array(data)
                    if data_np.size:
                        # Filter for class 0 (person)
                        person_mask = (data_np[:, 5].astype(int) == 0)
                        if person_mask.any():
                            sel = data_np[person_mask]
                            boxes_out.extend(sel[:, 0:4])
                            confs_out.extend(sel[:, 4].tolist())
                        continue
            except Exception as e:
                logging.debug(f"YOLO boxes.data parse failed: {e}")
            # Path B (vector attributes): boxes.xyxy / boxes.conf / boxes.cls
            try:
                xyxy = boxes_obj.xyxy
                conf = boxes_obj.conf
                cls = boxes_obj.cls
                # Move to CPU numpy regardless of tensor backend
                xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else np.array(xyxy)
                conf_np = conf.cpu().numpy() if hasattr(conf, 'cpu') else np.array(conf)
                cls_np = cls.cpu().numpy() if hasattr(cls, 'cpu') else np.array(cls)
                m = min(len(xyxy_np), len(conf_np), len(cls_np))
                if m > 0:
                    xyxy_np = xyxy_np[:m]
                    conf_np = conf_np[:m]
                    cls_np = cls_np[:m]
                    mask = (cls_np.astype(int) == 0)
                    if mask.any():
                        boxes_out.extend(xyxy_np[mask])
                        confs_out.extend(conf_np[mask].astype(float).tolist())
                continue
            except Exception:
                pass

            # Fallback: iterate over per-box objects (may be list-like)
            try:
                for b in boxes_obj:
                    try:
                        # b could be an object or a list/ndarray; handle both
                        if hasattr(b, 'xyxy') and hasattr(b, 'conf') and hasattr(b, 'cls'):
                            b_xyxy = b.xyxy
                            b_conf = b.conf
                            b_cls = b.cls
                            b_xyxy_np = b_xyxy.cpu().numpy().squeeze() if hasattr(b_xyxy, 'cpu') else np.array(b_xyxy).squeeze()
                            b_conf_f = float(b_conf.cpu().numpy().squeeze() if hasattr(b_conf, 'cpu') else b_conf)
                            b_cls_i = int(b_cls.cpu().numpy().squeeze() if hasattr(b_cls, 'cpu') else b_cls)
                        else:
                            # Try treat as array-like [x1,y1,x2,y2,conf,cls]
                            b_np = np.array(b).squeeze()
                            if b_np.shape[-1] < 6:
                                continue
                            b_xyxy_np = b_np[0:4]
                            b_conf_f = float(b_np[4])
                            b_cls_i = int(b_np[5])
                        if b_xyxy is None or b_conf is None or b_cls is None:
                            continue
                        if b_cls_i == 0:
                            boxes_out.append(b_xyxy_np)
                            confs_out.append(b_conf_f)
                    except Exception:
                        continue
            except Exception:
                continue

        try:
            return np.array(boxes_out), np.array(confs_out)
        except Exception as e:
            logging.error(f"Detector output conversion error: {e}")
            return np.zeros((0,4), dtype=float), np.zeros((0,), dtype=float)