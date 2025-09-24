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
                match_thresh_low=max(match_thresh - 0.1, 0.1),
                # Critical missing attributes that were causing the error
                fuse_score=0.5,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                with_reid=False,
                fast_reid_config="",
                fast_reid_weights="",
                device="",
                half=False,
                per_class=False
            )
            
            # Try different initialization methods for different ultralytics versions
            try:
                self.tracker = BYTETracker(args, frame_rate=frame_rate)
            except TypeError:
                # Fallback for older versions that don't accept frame_rate parameter
                self.tracker = BYTETracker(args)
                
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
                    match_thresh_low=0.3,
                    # Critical missing attributes for fallback
                    fuse_score=0.5,
                    proximity_thresh=0.5,
                    appearance_thresh=0.25,
                    with_reid=False,
                    fast_reid_config="",
                    fast_reid_weights="",
                    device="",
                    half=False,
                    per_class=False
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

        # Validate inputs and get valid detections
        valid_detections = self._validate_and_filter_detections(boxes, confidences, frame)
        if not valid_detections:
            return []

        # Update tracker and get results
        try:
            update_result = self._update_tracker(valid_detections, frame)
        except Exception as e:
            self._log_tracker_error(e, frame, boxes, confidences)
            return []

        # Process and validate results
        current_tracks = self._process_tracker_results(update_result)
        self.total_tracks += len(current_tracks)

        # Log performance periodically
        self._log_performance_stats(start_time, len(valid_detections), len(current_tracks))

        return current_tracks

    def _validate_and_filter_detections(self, boxes, confidences, frame):
        """Validate inputs and filter valid detections."""
        import logging
        
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
        return valid_detections

    def _update_tracker(self, valid_detections, frame):
        """Update the ByteTracker with valid detections."""
        import logging
        
        # Convert to tracker format efficiently
        xyxy_arr = np.array([det[0] for det in valid_detections], dtype=float)
        conf_arr = np.array([det[1] for det in valid_detections], dtype=float)
        cls_arr = np.zeros(len(valid_detections), dtype=float)  # All person class

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
        
        # Debug input data format for first few frames
        self._debug_input_data(xyxy_arr, tlwh_arr, conf_arr, results)
        
        logging.debug(f"Results.data shape: {results.data.shape}")
        logging.debug(f"Results.data: {results.data}")
        
        # New API: update(results, img=None)
        update_result = self.tracker.update(results, frame)
        logging.debug(f"Update result type: {type(update_result)}")
        logging.debug(f"Update result: {update_result}")
        
        # Debug coordinate format for first few frames
        self._debug_output_data(update_result)
        
        return update_result

    def _debug_input_data(self, xyxy_arr, tlwh_arr, conf_arr, results):
        """Debug input data format for first few frames."""
        import logging
        if self.frame_count <= 3:
            logging.info(f"🔍 ByteTracker input debug (frame {self.frame_count}):")
            logging.info(f"   - Input xyxy_arr: {xyxy_arr}")
            logging.info(f"   - Input tlwh_arr: {tlwh_arr}")
            logging.info(f"   - Input conf_arr: {conf_arr}")
            logging.info(f"   - Results.data: {results.data}")

    def _debug_output_data(self, update_result):
        """Debug output data format for first few frames."""
        import logging
        if self.frame_count <= 5 and hasattr(update_result, '__len__') and len(update_result) > 0:
            logging.info(f"🔍 ByteTracker output format debug (frame {self.frame_count}):")
            logging.info(f"   - Result type: {type(update_result)}")
            if isinstance(update_result, np.ndarray) and update_result.size > 0:
                logging.info(f"   - Array shape: {update_result.shape}")
                logging.info(f"   - First track data: {update_result[0] if len(update_result) > 0 else 'Empty'}")
            elif isinstance(update_result, (list, tuple)):
                logging.info(f"   - List/tuple length: {len(update_result)}")
                if len(update_result) > 0:
                    logging.info(f"   - First element: {update_result[0]}")

    def _log_tracker_error(self, e, frame, boxes, confidences):
        """Log tracker error with detailed information."""
        import logging
        logging.error(f"❌ ByteTracker update failed: {e}")
        logging.error(f"   - Error type: {type(e).__name__}")
        logging.error(f"   - Frame shape: {frame.shape if frame is not None else 'None'}")
        logging.error(f"   - Boxes shape: {boxes.shape if boxes is not None else 'None'}")
        logging.error(f"   - Confidences shape: {confidences.shape if confidences is not None else 'None'}")

    def _process_tracker_results(self, update_result):
        """Process tracker results and validate coordinates."""
        import logging
        
        # Process results
        current_tracks = self._extract_tracks(update_result)
        
        # Validate all tracks have proper coordinates
        validated_tracks = []
        for track in current_tracks:
            if self._validate_track_coordinates(track):
                validated_tracks.append(track)
            else:
                logging.warning(f"⚠️ Invalid track coordinates detected: {track}")
        
        return validated_tracks

    def _log_performance_stats(self, start_time, num_detections, num_tracks):
        """Log performance statistics periodically."""
        import logging
        import time
        
        if self.frame_count % 100 == 0:
            processing_time = time.time() - start_time
            track_ratio = num_tracks / max(1, num_detections)
            logging.info(f"📊 Tracker Stats - Frame {self.frame_count}: "
                        f"{num_detections} detections → {num_tracks} tracks "
                        f"(ratio: {track_ratio:.2f}, time: {processing_time:.3f}s)")

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

    def _validate_track_coordinates(self, track):
        """Validate that track coordinates are reasonable."""
        try:
            if 'id' not in track or 'xyxy' not in track:
                return False
            
            xyxy = track['xyxy']
            if not isinstance(xyxy, np.ndarray) or xyxy.shape != (4,):
                return False
            
            x1, y1, x2, y2 = xyxy
            # Check if coordinates are reasonable
            if x1 >= x2 or y1 >= y2:
                return False
            if x1 < 0 or y1 < 0:
                return False
            if (x2 - x1) < 5 or (y2 - y1) < 5:  # Too small
                return False
            if (x2 - x1) > 2000 or (y2 - y1) > 2000:  # Too large
                return False
                
            return True
        except Exception:
            return False

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
                    track = self._extract_single_track_from_array(track_data)
                    if track:
                        current_tracks.append(track)
            except Exception as e:
                import logging
                logging.debug(f"Error extracting track from array: {e}")
                continue
        return current_tracks

    def _extract_single_track_from_array(self, track_data):
        """Extract a single track from array data, trying both coordinate formats."""
        # ByteTracker output format varies by version - try both formats
        # Format 1: [x1, y1, x2, y2, track_id, conf, ...]
        # Format 2: [x, y, w, h, track_id, conf, ...]
        
        # Try format 1 first (xyxy)
        track = self._try_xyxy_format(track_data)
        if track:
            return track
        
        # Try format 2 (xywh - convert to xyxy)
        track = self._try_xywh_format(track_data)
        if track:
            return track
        
        return None

    def _try_xyxy_format(self, track_data):
        """Try extracting track using xyxy format."""
        try:
            x1, y1, x2, y2, track_id, _ = track_data[:6]
            # Validate coordinates make sense (x2 > x1, y2 > y1)
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = int(x2)
                y2 = int(y2)
                
                return {
                    'id': int(track_id),
                    'xyxy': np.array([x1, y1, x2, y2], dtype=int)
                }
        except (ValueError, IndexError, TypeError):
            pass
        return None

    def _try_xywh_format(self, track_data):
        """Try extracting track using xywh format and convert to xyxy."""
        try:
            x, y, w, h, track_id, _ = track_data[:6]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = int(x + w)
            y2 = int(y + h)
            
            # Validate coordinates
            if x2 > x1 and y2 > y1 and w > 0 and h > 0:
                return {
                    'id': int(track_id),
                    'xyxy': np.array([x1, y1, x2, y2], dtype=int)
                }
        except (ValueError, IndexError, TypeError):
            pass
        return None

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


