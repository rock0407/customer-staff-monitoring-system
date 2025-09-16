import numpy as np
import time
import logging
import cv2
from config_loader import LOG_FILE, INTERACTION_THRESHOLD, UNATTENDED_THRESHOLD, MIN_TRACKING_DURATION_FOR_ALERT

class InteractionLogger:
    def __init__(self, log_file=LOG_FILE, threshold=INTERACTION_THRESHOLD, min_duration=2.0):
        self.log_file = log_file
        self.threshold = threshold  # pixel distance for interaction
        self.min_duration = min_duration  # minimum interaction time in seconds
        self.active_interactions = {}  # (staff_id, cust_id): start_time
        self.interaction_history = {}  # (staff_id, cust_id): last_position
        self.person_history = {}  # person_id: list of recent positions for movement analysis
        self.customer_last_attended = {}  # customer_id -> last time seen in interaction
        self.customer_first_detected = {}  # customer_id -> first time detected (for 10-minute rule)
        self.history_length = 10  # Number of frames to track for movement analysis
        self.proximity_zones = {
            'close': 100,      # Very close interaction (0-100px)
            'medium': 200,     # Medium distance (100-200px)
            'far': 300         # Far but still interaction (200-300px)
        }
        self.movement_threshold = 20  # Maximum movement in pixels to consider "stationary"
        self.facing_threshold = 0.7   # Cosine similarity threshold for facing direction
        # Hysteresis + debounce controls (seconds)
        self.start_score = 0.30      # score to begin considering start
        self.end_score = 0.20        # score to remain active; below this risks ending
        self.start_min_time = 0.50   # must stay above start_score at least this long to start
        self.end_grace = 0.50        # allow brief dips below end_score before ending
        # State for debounce/end-grace
        self._prestart_since = {}    # (sid,cid) -> time when score first exceeded start_score
        self._end_grace_since = {}   # (sid,cid) -> time when score first dropped below end_score
        logging.info("🔧 Advanced Interaction Logger initialized:")
        logging.info(f"   - Distance threshold: {self.threshold}px")
        logging.info(f"   - Min duration: {self.min_duration}s")
        logging.info(f"   - Proximity zones: {self.proximity_zones}")
        logging.info(f"   - Movement threshold: {self.movement_threshold}px")
        logging.info(f"   - Facing threshold: {self.facing_threshold}")

    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate intersection over union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def estimate_facing_direction(self, bbox1, bbox2):
        """Estimate if two people are facing each other based on bounding box positions."""
        # Simple heuristic: if bounding boxes are roughly aligned horizontally
        # and close to each other, they're likely facing each other
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate centers
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # Calculate direction vector
        direction = np.array(center2) - np.array(center1)
        distance = np.linalg.norm(direction)
        
        if distance == 0:
            return 0.0
        
        # Normalize direction
        direction = direction / distance
        
        # Calculate if boxes are roughly aligned (similar y-coordinates)
        y_diff = abs(center1[1] - center2[1])
        max_y_diff = max(y2_1 - y1_1, y2_2 - y1_2) * 0.5  # Half the height of larger box
        
        if y_diff > max_y_diff:
            return 0.0  # Not aligned, likely not facing
        
        # Calculate facing score based on alignment and proximity
        alignment_score = max(0, 1 - (y_diff / max_y_diff))
        proximity_score = max(0, 1 - (distance / self.threshold))
        
        return (alignment_score + proximity_score) / 2

    def analyze_movement(self, person_id, current_pos):
        """Analyze if a person is stationary (more likely to be in interaction)."""
        if person_id not in self.person_history:
            self.person_history[person_id] = []
        
        self.person_history[person_id].append(current_pos)
        
        # Keep only recent positions
        if len(self.person_history[person_id]) > self.history_length:
            self.person_history[person_id] = self.person_history[person_id][-self.history_length:]
        
        # Calculate movement if we have enough history
        if len(self.person_history[person_id]) >= 3:
            recent_positions = self.person_history[person_id][-3:]
            total_movement = 0
            for i in range(1, len(recent_positions)):
                movement = np.linalg.norm(np.array(recent_positions[i]) - np.array(recent_positions[i-1]))
                total_movement += movement
            
            avg_movement = total_movement / (len(recent_positions) - 1)
            return avg_movement < self.movement_threshold
        
        return True  # Assume stationary if not enough history

    def get_proximity_zone(self, distance):
        """Determine proximity zone based on distance."""
        if distance <= self.proximity_zones['close']:
            return 'close'
        elif distance <= self.proximity_zones['medium']:
            return 'medium'
        elif distance <= self.proximity_zones['far']:
            return 'far'
        else:
            return 'none'

    def calculate_interaction_score(self, staff_bbox, customer_bbox, distance, staff_pos, customer_pos):
        """Calculate comprehensive interaction score based on multiple factors."""
        scores = {}
        
        # 1. Distance score (closer = higher score)
        distance_score = max(0, 1 - (distance / self.threshold))
        scores['distance'] = distance_score
        
        # 2. Bounding box overlap score
        overlap_score = self.calculate_bbox_overlap(staff_bbox, customer_bbox)
        scores['overlap'] = overlap_score
        
        # 3. Facing direction score
        facing_score = self.estimate_facing_direction(staff_bbox, customer_bbox)
        scores['facing'] = facing_score
        
        # 4. Movement analysis (stationary = higher score)
        staff_stationary = self.analyze_movement(f"staff_{staff_pos}", staff_pos)
        customer_stationary = self.analyze_movement(f"customer_{customer_pos}", customer_pos)
        movement_score = 1.0 if (staff_stationary and customer_stationary) else 0.3
        scores['movement'] = movement_score
        
        # 5. Proximity zone score
        zone = self.get_proximity_zone(distance)
        zone_scores = {'close': 1.0, 'medium': 0.7, 'far': 0.4, 'none': 0.0}
        zone_score = zone_scores[zone]
        scores['zone'] = zone_score
        
        # Calculate weighted average
        weights = {
            'distance': 0.20,
            'overlap': 0.20,
            'facing': 0.35,
            'movement': 0.15,
            'zone': 0.10
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        return total_score, scores, zone

    # --- Complexity reduction helpers ---
    def _pair_iter(self, staff, customers):
        for sid, spt, sbbox in staff:
            for cid, cpt, cbbox in customers:
                dist = np.linalg.norm(np.array(spt) - np.array(cpt))
                score, scores, zone = self.calculate_interaction_score(sbbox, cbbox, dist, spt, cpt)
                yield sid, cid, spt, cpt, dist, score, scores, zone

    def _maybe_start(self, key, sid, cid, score, frame_time, spt, cpt, dist, zone, scores, interactions):
        if score < self.start_score:
            self._prestart_since.pop(key, None)
            return
        self._prestart_since.setdefault(key, frame_time)
        if frame_time - self._prestart_since[key] < self.start_min_time:
            return
        self.active_interactions[key] = frame_time
        self.interaction_history[key] = (spt, cpt)
        interactions.append(key)
        logging.info(f"🟢 NEW INTERACTION: Staff {sid} & Customer {cid}")
        logging.info(f"   Distance: {dist:.1f}px | Zone: {zone} | Score: {score:.2f}")
        logging.info(f"   Scores: Distance={scores['distance']:.2f}, Overlap={scores['overlap']:.2f}, Facing={scores['facing']:.2f}, Movement={scores['movement']:.2f}")
        self._prestart_since.pop(key, None)

    def _should_remain_active(self, key, score, frame_time):
        if score >= self.end_score:
            self._end_grace_since.pop(key, None)
            return True
        self._end_grace_since.setdefault(key, frame_time)
        return (frame_time - self._end_grace_since[key]) < self.end_grace

    def _finalize_ended(self, interactions, frame_time):
        ended = [k for k in self.active_interactions if k not in interactions]
        for k in ended:
            start_time = self.active_interactions.pop(k)
            duration = frame_time - start_time
            if duration >= self.min_duration:
                self.log_interaction(k[0], k[1], start_time, frame_time, duration, "VALID")
                logging.info(f"✅ INTERACTION COMPLETED: Staff {k[0]} & Customer {k[1]} | Duration: {duration:.1f}s | Status: VALID")
                self.customer_last_attended[k[1]] = frame_time
            else:
                logging.info(f"❌ INTERACTION TOO SHORT: Staff {k[0]} & Customer {k[1]} | Duration: {duration:.1f}s | Status: IGNORED (min: {self.min_duration}s)")

    def _compute_unattended(self, customers, frame_time):
        unattended_ids = []
        for cid, _pos, _bbox in customers:
            # Track when customer was first detected
            if cid not in self.customer_first_detected:
                self.customer_first_detected[cid] = frame_time
                logging.info(f"👤 Customer {cid} first detected at {frame_time:.1f}s")
            
            # Check if customer has been tracked for minimum duration
            tracking_duration = frame_time - self.customer_first_detected[cid]
            if tracking_duration < MIN_TRACKING_DURATION_FOR_ALERT:
                # Customer hasn't been tracked long enough for alerts
                continue
            
            # Check if customer is unattended (no interaction for threshold time)
            last = self.customer_last_attended.get(cid)
            if last is None:
                self.customer_last_attended[cid] = frame_time
                continue
            
            if frame_time - last >= UNATTENDED_THRESHOLD:
                unattended_ids.append(cid)
                logging.info(f"🚨 Unattended customer {cid}: tracked for {tracking_duration:.1f}s, unattended for {frame_time - last:.1f}s")
        
        if unattended_ids:
            logging.info(f"🚨 Unattended customers: {unattended_ids} (tracked >{MIN_TRACKING_DURATION_FOR_ALERT:.0f}s, unattended >{UNATTENDED_THRESHOLD:.0f}s)")
        return unattended_ids

    def check_and_log(self, staff, customers, frame_time):
        """Advanced interaction detection with multiple criteria."""
        interactions = []
        for sid, cid, spt, cpt, dist, score, scores, zone in self._pair_iter(staff, customers):
            key = (sid, cid)
            if key in self.active_interactions:
                if self._should_remain_active(key, score, frame_time):
                    interactions.append(key)
                    duration = frame_time - self.active_interactions[key]
                    if duration >= 1.0:
                        logging.info(f"🟡 INTERACTION CONTINUES: Staff {sid} & Customer {cid} | Duration: {duration:.1f}s | Score: {score:.2f}")
                    self.customer_last_attended[cid] = frame_time
            else:
                self._maybe_start(key, sid, cid, score, frame_time, spt, cpt, dist, zone, scores, interactions)

        self._finalize_ended(interactions, frame_time)
        unattended_ids = self._compute_unattended(customers, frame_time)
        return interactions, unattended_ids

    def log_interaction(self, staff_id, cust_id, start, end, duration, status="VALID"):
        """Log interaction to file with detailed information."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {status} | Staff {staff_id} & Customer {cust_id} | Start: {start:.2f}s | End: {end:.2f}s | Duration: {duration:.2f}s\n")

    def get_active_interactions(self):
        """Get currently active interactions for visualization."""
        return list(self.active_interactions.keys())

    def get_interaction_duration(self, staff_id, cust_id, current_time=None):
        """Get current duration of an active interaction using the same timebase as check_and_log.
        Pass current_time=frame_time from the main loop for consistent results.
        """
        key = (staff_id, cust_id)
        if key in self.active_interactions:
            start_t = self.active_interactions[key]
            if current_time is not None:
                return max(0.0, current_time - start_t)
        return 0.0 