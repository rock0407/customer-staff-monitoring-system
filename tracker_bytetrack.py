"""
ByteTrack tracker wrapper compatible with the project's tracker interface.

Exposes class PersonTrackerBYTE with update(boxes, confidences, frame) -> [{id, xyxy}].
Requires ultralytics with ByteTrack available.
"""

from typing import List, Dict
import numpy as np

try:
    # Ultralytics ByteTrack implementation
    from ultralytics.trackers.byte_tracker import BYTETracker, STrack
    UL_BYTE_AVAILABLE = True
except Exception:
    UL_BYTE_AVAILABLE = False


def _xyxy_to_tlwh(box_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    return np.array([x1, y1, w, h], dtype=float)


class PersonTrackerBYTE:
    def __init__(self,
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 mot20: bool = False):
        if not UL_BYTE_AVAILABLE:
            raise ImportError("Ultralytics BYTETracker not available. Ensure ultralytics is installed and up-to-date.")

        # Create tracker in a version-compatible way (constructor signatures vary across releases)
        try:
            # Newer builds may support no-arg constructor
            self.tracker = BYTETracker()
        except TypeError:
            # Older builds expect an args namespace and optional frame_rate
            try:
                from types import SimpleNamespace
                args = SimpleNamespace(
                    track_thresh=track_thresh,
                    track_buffer=track_buffer,
                    match_thresh=match_thresh,
                    mot20=mot20
                )
                self.tracker = BYTETracker(args, frame_rate=30)
            except Exception as e:
                raise ImportError(f"Failed to initialize BYTETracker with compatible signature: {e}")

    def update(self, boxes: np.ndarray, confidences: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """
        Update ByteTrack with current-frame detections.
        boxes: Nx4 xyxy (float)
        confidences: N scores (float)
        frame: HxWx3 BGR image
        Returns: List[{ 'id': int, 'xyxy': np.ndarray(shape=(4,), dtype=int) }]
        """
        import logging

        if boxes is None or len(boxes) == 0:
            # Still step tracker with empty detections
            self.tracker.update([], (frame.shape[0], frame.shape[1]), (frame.shape[0], frame.shape[1]))
            return []

        # Build data once
        xyxy_list, conf_list, cls_list = [], [], []
        for box, score in zip(boxes, confidences):
            if box is None or len(box) != 4:
                continue
            if score is None:
                continue
            if np.any(np.isnan(box)) or np.any(np.isinf(box)):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(float(score))
            cls_list.append(0.0)  # person
        H, W = frame.shape[0], frame.shape[1]
        xyxy_arr = np.array(xyxy_list, dtype=float) if len(xyxy_list) else np.zeros((0, 4), dtype=float)
        conf_arr = np.array(conf_list, dtype=float) if len(conf_list) else np.zeros((0,), dtype=float)
        cls_arr = np.array(cls_list, dtype=float) if len(cls_list) else np.zeros((0,), dtype=float)
        from types import SimpleNamespace
        # Prepare three variants accepted by different Ultralytics versions
        # 1) root-level xyxy/xywh + conf/cls
        if xyxy_arr.size:
            xywh_arr = np.column_stack([
                (xyxy_arr[:, 0] + xyxy_arr[:, 2]) / 2.0,
                (xyxy_arr[:, 1] + xyxy_arr[:, 3]) / 2.0,
                (xyxy_arr[:, 2] - xyxy_arr[:, 0]),
                (xyxy_arr[:, 3] - xyxy_arr[:, 1])
            ])
        else:
            xywh_arr = np.zeros((0, 4), dtype=float)

        detections_root = SimpleNamespace(xyxy=xyxy_arr, xywh=xywh_arr, conf=conf_arr, cls=cls_arr)
        boxes_shim = SimpleNamespace(xyxy=xyxy_arr, conf=conf_arr, cls=cls_arr)
        detections_boxes = SimpleNamespace(boxes=boxes_shim)
        detections_nd = np.hstack([xyxy_arr, conf_arr.reshape(-1, 1), cls_arr.reshape(-1, 1)]) if xyxy_arr.size else np.zeros((0, 6), dtype=float)

        update_result = None
        # Try root-level fields
        try:
            update_result = self.tracker.update(detections_root, (H, W), (H, W))
        except Exception:
            update_result = None
        # Try boxes-level fields
        if update_result is None:
            try:
                update_result = self.tracker.update(detections_boxes, (H, W), (H, W))
            except Exception:
                update_result = None
        # Try ndarray Nx6 [x1,y1,x2,y2,conf,cls]
        if update_result is None:
            try:
                update_result = self.tracker.update(detections_nd, (H, W), (H, W))
            except Exception as e:
                logging.debug(f"BYTETracker update error: {e}. Returning empty tracks.")
                return []
        # Normalize return to a list of tracks
        if isinstance(update_result, (list, tuple)) and len(update_result) == 4:
            activated_tracks, refind_tracks, _, _ = update_result
            current_tracks = list(activated_tracks) + list(refind_tracks)
        else:
            current_tracks = update_result if isinstance(update_result, list) else []

        results: List[Dict] = []
        for t in current_tracks:
            try:
                tlwh = t.tlwh
                x1 = int(tlwh[0])
                y1 = int(tlwh[1])
                x2 = int(tlwh[0] + tlwh[2])
                y2 = int(tlwh[1] + tlwh[3])
                results.append({
                    'id': int(t.track_id),
                    'xyxy': np.array([x1, y1, x2, y2], dtype=int)
                })
            except Exception:
                continue

        return results


