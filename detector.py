from ultralytics import YOLO
import numpy as np
import logging
from config_loader import YOLO_MODEL_PATH

class PersonDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        # Fix PyTorch weights_only issue by monkey-patching torch.load
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Run YOLO detection and return Nx4 xyxy and confidences arrays for person class.
        Handles Ultralytics API variations where results/boxes can differ across versions.
        """
        results = self._run_yolo_prediction(frame)
        boxes_out = []
        confs_out = []

        results_iter = self._prepare_results_iter(results)
        for res in results_iter:
            if res is None:
                continue
            boxes_obj = getattr(res, 'boxes', None)
            if boxes_obj is None:
                continue

            # Try different parsing methods
            if self._try_parse_boxes_data(boxes_obj, boxes_out, confs_out):
                continue
            if self._try_parse_boxes_attributes(boxes_obj, boxes_out, confs_out):
                continue
            self._try_parse_boxes_iteration(boxes_obj, boxes_out, confs_out)

        return self._format_detection_output(boxes_out, confs_out)

    def _run_yolo_prediction(self, frame):
        """Run YOLO prediction on the frame."""
        try:
            # Use original frame size to avoid coordinate scaling issues
            # Note: YOLO will automatically adjust to multiples of 32
            height, width = frame.shape[:2]
            # Force CPU device to avoid CUDA CUBLAS issues
            return self.model.predict(frame, imgsz=max(height, width), verbose=False, device='cpu')
        except Exception:
            return self.model(frame, imgsz=max(frame.shape[:2]))

    def _prepare_results_iter(self, results):
        """Prepare results for iteration."""
        try:
            return list(results) if not isinstance(results, (list, tuple)) else results
        except Exception:
            return [results]

    def _try_parse_boxes_data(self, boxes_obj, boxes_out, confs_out):
        """Try parsing boxes using data attribute."""
        try:
            data = getattr(boxes_obj, 'data', None)
            if data is not None:
                data_np = data.cpu().numpy() if hasattr(data, 'cpu') else np.array(data)
                if data_np.size:
                    person_mask = (data_np[:, 5].astype(int) == 0)
                    if person_mask.any():
                        sel = data_np[person_mask]
                        boxes_out.extend(sel[:, 0:4])
                        confs_out.extend(sel[:, 4].tolist())
                    return True
        except Exception as e:
            logging.debug(f"YOLO boxes.data parse failed: {e}")
        return False

    def _try_parse_boxes_attributes(self, boxes_obj, boxes_out, confs_out):
        """Try parsing boxes using individual attributes."""
        try:
            xyxy = boxes_obj.xyxy
            conf = boxes_obj.conf
            cls = boxes_obj.cls
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
            return True
        except Exception:
            return False

    def _try_parse_boxes_iteration(self, boxes_obj, boxes_out, confs_out):
        """Try parsing boxes by iteration."""
        try:
            for b in boxes_obj:
                self._parse_single_box(b, boxes_out, confs_out)
        except Exception:
            pass

    def _parse_single_box(self, b, boxes_out, confs_out):
        """Parse a single box object."""
        try:
            if hasattr(b, 'xyxy') and hasattr(b, 'conf') and hasattr(b, 'cls'):
                self._parse_box_object(b, boxes_out, confs_out)
            else:
                self._parse_box_array(b, boxes_out, confs_out)
        except Exception:
            pass

    def _parse_box_object(self, b, boxes_out, confs_out):
        """Parse box as object with attributes."""
        b_xyxy = b.xyxy
        b_conf = b.conf
        b_cls = b.cls
        b_xyxy_np = b_xyxy.cpu().numpy().squeeze() if hasattr(b_xyxy, 'cpu') else np.array(b_xyxy).squeeze()
        b_conf_f = float(b_conf.cpu().numpy().squeeze() if hasattr(b_conf, 'cpu') else b_conf)
        b_cls_i = int(b_cls.cpu().numpy().squeeze() if hasattr(b_cls, 'cpu') else b_cls)
        
        if b_xyxy is not None and b_conf is not None and b_cls is not None and b_cls_i == 0:
            boxes_out.append(b_xyxy_np)
            confs_out.append(b_conf_f)

    def _parse_box_array(self, b, boxes_out, confs_out):
        """Parse box as array-like object."""
        b_np = np.array(b).squeeze()
        if b_np.shape[-1] < 6:
            return
        b_xyxy_np = b_np[0:4]
        b_conf_f = float(b_np[4])
        b_cls_i = int(b_np[5])
        
        if b_cls_i == 0:
            boxes_out.append(b_xyxy_np)
            confs_out.append(b_conf_f)

    def _format_detection_output(self, boxes_out, confs_out):
        """Format the final detection output."""
        try:
            # Scale coordinates back to original frame size if needed
            # YOLO might be using a different input size internally
            if len(boxes_out) > 0:
                # Check if coordinates are outside expected range
                max_coord = max([max(box) for box in boxes_out])
                if max_coord > 1920:  # If coordinates are too large, scale them down
                    scale_factor = 1920 / max_coord
                    boxes_out = [[coord * scale_factor for coord in box] for box in boxes_out]
                    logging.debug(f"Scaled coordinates by factor: {scale_factor}")
            
            return np.array(boxes_out), np.array(confs_out)
        except Exception as e:
            logging.error(f"Detector output conversion error: {e}")
            return np.zeros((0,4), dtype=float), np.zeros((0,), dtype=float)