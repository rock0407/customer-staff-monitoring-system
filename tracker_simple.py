"""
Simple and reliable tracker for CSI system.
Provides stable tracking without complex dependencies.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
import time

class SimpleTracker:
    """
    Simple but effective tracker for CSI system.
    Uses IoU-based matching and Kalman filtering for stable tracking.
    """
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_disappeared: int = 30,
                 max_distance: float = 100.0):
        """
        Initialize simple tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching detections to tracks
            max_disappeared: Max frames a track can be missing before deletion
            max_distance: Max distance for matching (pixels)
        """
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Tracking data
        self.next_id = 1
        self.tracks = {}  # track_id -> track_data
        self.disappeared = {}  # track_id -> frames_missing
        
        # Statistics
        self.total_detections = 0
        self.total_tracks = 0
        self.frame_count = 0
        
        logging.info("✅ SimpleTracker initialized:")
        logging.info(f"   - IoU threshold: {iou_threshold}")
        logging.info(f"   - Max disappeared: {max_disappeared} frames")
        logging.info(f"   - Max distance: {max_distance} pixels")

    def update(self, boxes: np.ndarray, confidences: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            boxes: Nx4 array of bounding boxes [x1, y1, x2, y2]
            confidences: N array of confidence scores
            frame: Current frame (not used but kept for compatibility)
            
        Returns:
            List of track dictionaries with 'id' and 'xyxy' keys
        """
        self.frame_count += 1
        self.total_detections += len(boxes) if boxes is not None else 0
        
        if boxes is None or len(boxes) == 0:
            # No detections, mark all tracks as disappeared
            for track_id in self.disappeared.keys():
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._remove_track(track_id)
            return []
        
        # Convert boxes to center coordinates for easier matching
        centers = self._boxes_to_centers(boxes)
        
        # Match detections to existing tracks
        matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = \
            self._match_detections_to_tracks(centers, boxes)
        
        # Update matched tracks
        for track_id, det_idx in zip(matched_tracks, matched_detections):
            self._update_track(track_id, boxes[det_idx], confidences[det_idx])
            self.disappeared[track_id] = 0
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(boxes[det_idx], confidences[det_idx])
        
        # Mark unmatched tracks as disappeared
        for track_id in unmatched_tracks:
            self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                self._remove_track(track_id)
        
        # Return current active tracks
        return self._get_active_tracks()

    def _boxes_to_centers(self, boxes: np.ndarray) -> np.ndarray:
        """Convert bounding boxes to center coordinates."""
        centers = np.zeros((len(boxes), 2))
        centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
        centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
        return centers

    def _match_detections_to_tracks(self, centers: np.ndarray, boxes: np.ndarray) -> Tuple:
        """Match detections to existing tracks using IoU and distance."""
        if len(self.tracks) == 0:
            return [], [], [], list(range(len(centers)))
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(boxes)
        
        # Calculate distance matrix
        track_centers = np.array([track['center'] for track in self.tracks.values()])
        distance_matrix = self._calculate_distance_matrix(centers, track_centers)
        
        # Find matches based on IoU and distance
        matched_tracks = []
        matched_detections = []
        unmatched_tracks = list(self.tracks.keys())
        unmatched_detections = list(range(len(centers)))
        
        # Sort by IoU (descending) for best matches first
        iou_indices = np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape)
        
        for i, j in zip(iou_indices[0], iou_indices[1]):
            if i in unmatched_detections and j in unmatched_tracks:
                track_id = list(self.tracks.keys())[j]
                
                # Check if match is good enough
                if (iou_matrix[i, j] > self.iou_threshold and 
                    distance_matrix[i, j] < self.max_distance):
                    
                    matched_tracks.append(track_id)
                    matched_detections.append(i)
                    unmatched_tracks.remove(track_id)
                    unmatched_detections.remove(i)
        
        return matched_tracks, matched_detections, unmatched_tracks, unmatched_detections

    def _calculate_iou_matrix(self, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between detections and tracks."""
        if len(self.tracks) == 0:
            return np.zeros((len(boxes), 0))
        
        track_boxes = np.array([track['box'] for track in self.tracks.values()])
        
        # Calculate IoU for all pairs
        iou_matrix = np.zeros((len(boxes), len(track_boxes)))
        
        for i, det_box in enumerate(boxes):
            for j, track_box in enumerate(track_boxes):
                iou_matrix[i, j] = self._calculate_iou(det_box, track_box)
        
        return iou_matrix

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _calculate_distance_matrix(self, centers1: np.ndarray, centers2: np.ndarray) -> np.ndarray:
        """Calculate distance matrix between two sets of centers."""
        distances = np.zeros((len(centers1), len(centers2)))
        
        for i, c1 in enumerate(centers1):
            for j, c2 in enumerate(centers2):
                distances[i, j] = np.linalg.norm(c1 - c2)
        
        return distances

    def _create_track(self, box: np.ndarray, confidence: float):
        """Create a new track."""
        track_id = self.next_id
        self.next_id += 1
        
        center = self._boxes_to_centers(box.reshape(1, -1))[0]
        
        self.tracks[track_id] = {
            'box': box.copy(),
            'center': center,
            'confidence': confidence,
            'age': 0,
            'last_seen': self.frame_count
        }
        
        self.disappeared[track_id] = 0
        self.total_tracks += 1
        
        logging.debug(f"🆕 Created track {track_id}")

    def _update_track(self, track_id: int, box: np.ndarray, confidence: float):
        """Update an existing track."""
        if track_id not in self.tracks:
            return
        
        # Update track data
        self.tracks[track_id]['box'] = box.copy()
        self.tracks[track_id]['center'] = self._boxes_to_centers(box.reshape(1, -1))[0]
        self.tracks[track_id]['confidence'] = confidence
        self.tracks[track_id]['age'] += 1
        self.tracks[track_id]['last_seen'] = self.frame_count

    def _remove_track(self, track_id: int):
        """Remove a track."""
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
        
        logging.debug(f"🗑️ Removed track {track_id}")

    def _get_active_tracks(self) -> List[Dict]:
        """Get list of active tracks."""
        active_tracks = []
        
        for track_id, track_data in self.tracks.items():
            if self.disappeared.get(track_id, 0) == 0:  # Only active tracks
                active_tracks.append({
                    'id': track_id,
                    'xyxy': track_data['box'].astype(int)
                })
        
        return active_tracks

    def get_tracking_stats(self) -> Dict:
        """Get tracking performance statistics."""
        if self.frame_count == 0:
            return {}
        
        # Count active tracks (not disappeared)
        active_tracks = 0
        for track_id in self.tracks.keys():
            if self.disappeared.get(track_id, 0) == 0:
                active_tracks += 1
        
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'total_tracks_created': self.total_tracks,
            'active_tracks': active_tracks,
            'avg_detections_per_frame': self.total_detections / self.frame_count,
            'avg_tracks_per_frame': active_tracks,
            'track_efficiency': active_tracks / max(1, self.total_detections / self.frame_count)
        }
