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
                 track_thresh: float = 0.6,      # Higher threshold for better quality
                 track_buffer: int = 50,         # Longer buffer for CSI use case
                 match_thresh: float = 0.7,      # Balanced matching threshold
                 mot20: bool = False,
                 frame_rate: int = 30):
        if not UL_BYTE_AVAILABLE:
            raise ImportError("Ultralytics BYTETracker not available. Ensure ultralytics is installed and up-to-date.")

        # Store parameters for monitoring
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        # Initialize tracking statistics
        self.total_detections = 0
        self.total_tracks = 0
        self.track_loss_count = 0
        self.frame_count = 0

        # Create tracker with optimized parameters for CSI
        try:
            from types import SimpleNamespace
            args = SimpleNamespace(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                mot20=mot20,
                frame_rate=frame_rate,
                # Additional required attributes for ultralytics 8.0.196
                track_high_thresh=min(track_thresh + 0.2, 0.9),
                track_low_thresh=max(track_thresh - 0.2, 0.1),
                new_track_thresh=track_thresh,
                match_thresh_high=min(match_thresh + 0.1, 0.9),
                match_thresh_low=max(match_thresh - 0.1, 0.1)
            )
            self.tracker = BYTETracker(args, frame_rate=frame_rate)
            print("✅ ByteTracker initialized with optimized parameters:")
            print(f"   - Track threshold: {track_thresh}")
            print(f"   - Track buffer: {track_buffer} frames")
            print(f"   - Match threshold: {match_thresh}")
            print(f"   - Frame rate: {frame_rate} FPS")
        except Exception as e:
            try:
                # Fallback with minimal args
                from types import SimpleNamespace
                minimal_args = SimpleNamespace(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    mot20=False,
                    track_high_thresh=0.7,
                    track_low_thresh=0.3,
                    new_track_thresh=0.5,
                    match_thresh_high=0.9,
                    match_thresh_low=0.3
                )
                self.tracker = BYTETracker(minimal_args, frame_rate=30)
                print(f"⚠️ Using minimal ByteTracker parameters due to: {e}")
            except Exception as e2:
                raise ImportError(f"Failed to initialize BYTETracker: {e2}")

    def update(self, boxes: np.ndarray, confidences: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """
        Update ByteTrack with current-frame detections.
        boxes: Nx4 xyxy (float)
        confidences: N scores (float)
        frame: HxWx3 BGR image
        Returns: List[{ 'id': int, 'xyxy': np.ndarray(shape=(4,), dtype=int) }]
        """
        import logging
        import time

        self.frame_count += 1
        start_time = time.time()

        # Validate inputs
        if boxes is None or boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=float)
        if confidences is None or confidences.size == 0:
            confidences = np.zeros((0,), dtype=float)
        if frame is None or frame.size == 0:
            logging.warning("⚠️ Invalid frame received by tracker")
            return []

        # Validate frame dimensions
        H, W = frame.shape[0], frame.shape[1]
        if H <= 0 or W <= 0:
            logging.warning("⚠️ Invalid frame dimensions")
            return []

        # Filter and validate detections
        valid_detections = []
        for i, (box, score) in enumerate(zip(boxes, confidences)):
            if self._is_valid_detection(box, score, W, H):
                valid_detections.append((box, score))

        # Update statistics
        self.total_detections += len(valid_detections)
        
        if not valid_detections:
            # No valid detections, return empty tracks
            return []

        # Convert to tracker format efficiently
        xyxy_arr = np.array([det[0] for det in valid_detections], dtype=float)
        conf_arr = np.array([det[1] for det in valid_detections], dtype=float)
        cls_arr = np.zeros(len(valid_detections), dtype=float)  # All person class

        # Use the correct ByteTracker update method for ultralytics 8.0.196
        try:
            # Convert to tlwh format (top-left-width-height) as expected by ByteTracker
            tlwh_arr = np.column_stack([
                xyxy_arr[:, 0],                           # x1 (top-left x)
                xyxy_arr[:, 1],                           # y1 (top-left y)
                xyxy_arr[:, 2] - xyxy_arr[:, 0],         # w (width)
                xyxy_arr[:, 3] - xyxy_arr[:, 1]          # h (height)
            ])
            
            logging.debug(f"Input xyxy_arr: {xyxy_arr}")
            logging.debug(f"Converted tlwh_arr: {tlwh_arr}")
            
            # Create a results-like object that ByteTracker expects
            from types import SimpleNamespace
            results = SimpleNamespace()
            results.data = np.column_stack([tlwh_arr, conf_arr, cls_arr])  # [x, y, w, h, conf, cls]
            results.conf = conf_arr
            results.cls = cls_arr
            results.xyxy = xyxy_arr
            results.xywh = tlwh_arr
            
            logging.debug(f"Results.data shape: {results.data.shape}")
            logging.debug(f"Results.data: {results.data}")
            
            # New API: update(results, img=None)
            update_result = self.tracker.update(results, frame)
            logging.debug(f"Update result type: {type(update_result)}")
            logging.debug(f"Update result: {update_result}")
                
        except Exception as e:
            logging.error(f"❌ ByteTracker update failed: {e}")
            return []

        # Process results
        current_tracks = self._extract_tracks(update_result)
        self.total_tracks += len(current_tracks)

        # Log performance periodically
        if self.frame_count % 100 == 0:
            processing_time = time.time() - start_time
            track_ratio = len(current_tracks) / max(1, len(valid_detections))
            logging.info(f"📊 Tracker Stats - Frame {self.frame_count}: "
                        f"{len(valid_detections)} detections → {len(current_tracks)} tracks "
                        f"(ratio: {track_ratio:.2f}, time: {processing_time:.3f}s)")

        return current_tracks

    def _is_valid_detection(self, box, score, img_w, img_h):
        """Validate a single detection."""
        if box is None or len(box) != 4:
            return False
        if score is None or score <= 0:
            return False
        if np.any(np.isnan(box)) or np.any(np.isinf(box)):
            return False
        
        x1, y1, x2, y2 = box
        # Check if box is within image bounds
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            return False
        # Check if box has reasonable size
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False
        # Check if box is not too large (likely false detection)
        if (x2 - x1) > img_w * 0.8 or (y2 - y1) > img_h * 0.8:
            return False
            
        return True

    def _extract_tracks(self, update_result):
        """Extract tracks from ByteTracker result."""
        current_tracks = []
        
        # Handle different ByteTracker output formats
        if isinstance(update_result, np.ndarray) and update_result.size > 0:
            current_tracks = self._extract_tracks_from_array(update_result)
        elif isinstance(update_result, (list, tuple)) and len(update_result) == 4:
            current_tracks = self._extract_tracks_from_tuple(update_result)
        else:
            current_tracks = self._extract_tracks_from_list(update_result)

        return current_tracks

    def _extract_tracks_from_array(self, update_result):
        """Extract tracks from numpy array format."""
        current_tracks = []
        for track_data in update_result:
            try:
                if len(track_data) >= 6:
                    # ByteTracker output format is actually: [x1, y1, x2, y2, track_id, conf, ...]
                    # NOT [x, y, w, h, track_id, conf, ...] as previously assumed
                    x1, y1, x2, y2, track_id, _ = track_data[:6]
                    
                    # Coordinates are already in xyxy format, just ensure they're integers
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = int(x2)
                    y2 = int(y2)
                    
                    current_tracks.append({
                        'id': int(track_id),
                        'xyxy': np.array([x1, y1, x2, y2], dtype=int)
                    })
            except Exception as e:
                import logging
                logging.debug(f"Error extracting track from array: {e}")
                continue
        return current_tracks

    def _extract_tracks_from_tuple(self, update_result):
        """Extract tracks from tuple format (activated, refind, lost, removed)."""
        current_tracks = []
        activated_tracks, refind_tracks, _, _ = update_result
        all_tracks = list(activated_tracks) + list(refind_tracks)
        
        for track in all_tracks:
            track_data = self._extract_single_track(track)
            if track_data:
                current_tracks.append(track_data)
        return current_tracks

    def _extract_tracks_from_list(self, update_result):
        """Extract tracks from direct list format."""
        current_tracks = []
        all_tracks = update_result if isinstance(update_result, list) else []
        
        for track in all_tracks:
            track_data = self._extract_single_track(track)
            if track_data:
                current_tracks.append(track_data)
        return current_tracks

    def _extract_single_track(self, track):
        """Extract data from a single track object."""
        try:
            if hasattr(track, 'tlwh') and hasattr(track, 'track_id'):
                tlwh = track.tlwh
                x1 = max(0, int(tlwh[0]))
                y1 = max(0, int(tlwh[1]))
                x2 = int(tlwh[0] + tlwh[2])
                y2 = int(tlwh[1] + tlwh[3])
                
                return {
                    'id': int(track.track_id),
                    'xyxy': np.array([x1, y1, x2, y2], dtype=int)
                }
        except Exception as e:
            logging.debug(f"Error extracting track: {e}")
        return None

    def get_tracking_stats(self):
        """Get tracking performance statistics."""
        if self.frame_count == 0:
            return {}
        
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'total_tracks': self.total_tracks,
            'avg_detections_per_frame': self.total_detections / self.frame_count,
            'avg_tracks_per_frame': self.total_tracks / self.frame_count,
            'track_efficiency': self.total_tracks / max(1, self.total_detections)
        }


