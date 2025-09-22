import numpy as np
import time
import logging
import cv2
from config_loader import LOG_FILE, INTERACTION_THRESHOLD, UNATTENDED_THRESHOLD, MIN_TRACKING_DURATION_FOR_ALERT, UNATTENDED_CONFIRMATION_TIMER, TIMER_RESET_GRACE_PERIOD

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
        
        # NEW: Timer-based unattended customer system
        self.unattended_timers = {}  # customer_id -> timer start time
        self.confirmed_unattended = set()  # customers confirmed as unattended
        self.unattended_confirmation_timer = UNATTENDED_CONFIRMATION_TIMER  # 60 seconds confirmation
        
        # Tracking stability improvements
        self.customer_position_history = {}  # customer_id -> recent positions for stability
        self.customer_id_mapping = {}  # old_id -> new_id for ID persistence
        self.position_history_length = 5  # Keep last 5 positions
        
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
        logging.info("🔧 Enhanced Interaction Logger initialized:")
        logging.info(f"   - Distance threshold: {self.threshold}px")
        logging.info(f"   - Min duration: {self.min_duration}s")
        logging.info(f"   - Unattended confirmation timer: {self.unattended_confirmation_timer}s")
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

    def _stabilize_customer_tracking(self, customers, frame_time):
        """
        Stabilize customer tracking by maintaining position history and handling ID changes.
        This prevents timer resets due to tracking instability.
        """
        current_customer_positions = self._build_current_positions(customers)
        self._handle_id_changes(current_customer_positions)
        self._update_position_history(current_customer_positions)
        self._cleanup_old_customers(current_customer_positions, frame_time)

    def _build_current_positions(self, customers):
        """Build dictionary of current customer positions."""
        return {cid: pos for cid, pos, bbox in customers}

    def _handle_id_changes(self, current_customer_positions):
        """Handle potential customer ID changes by comparing positions."""
        for old_cid, old_positions in self.customer_position_history.items():
            if old_cid not in current_customer_positions and old_positions:
                self._find_id_mapping(old_cid, old_positions[-1], current_customer_positions)

    def _find_id_mapping(self, old_cid, last_known_pos, current_positions):
        """Find if old customer moved to a new ID."""
        for new_cid, new_pos in current_positions.items():
            if new_cid not in self.customer_position_history:
                distance = np.linalg.norm(np.array(last_known_pos) - np.array(new_pos))
                if distance < 100:  # Within 100 pixels
                    self.customer_id_mapping[new_cid] = old_cid
                    logging.info(f"🔄 Customer ID changed: {old_cid} -> {new_cid} (distance: {distance:.1f}px)")
                    break

    def _update_position_history(self, current_customer_positions):
        """Update position history for current customers."""
        for cid, pos in current_customer_positions.items():
            if cid not in self.customer_position_history:
                self.customer_position_history[cid] = []
            
            self.customer_position_history[cid].append(pos)
            
            # Keep only recent positions
            if len(self.customer_position_history[cid]) > self.position_history_length:
                self.customer_position_history[cid] = self.customer_position_history[cid][-self.position_history_length:]

    def _cleanup_old_customers(self, current_customer_positions, frame_time):
        """Clean up customers not seen for too long."""
        customers_to_remove = self._identify_old_customers(current_customer_positions, frame_time)
        
        for cid in customers_to_remove:
            self._remove_customer_data(cid)
            self._cleanup_customer_interactions(cid)
            logging.info(f"🧹 Cleaned up old customer {cid} and related data")

    def _identify_old_customers(self, current_customer_positions, frame_time):
        """Identify customers that should be removed."""
        customers_to_remove = []
        for cid in self.customer_position_history:
            if cid not in current_customer_positions:
                if cid in self.customer_first_detected:
                    time_since_last_seen = frame_time - self.customer_first_detected[cid]
                    if time_since_last_seen > 30:  # 30 seconds grace period
                        customers_to_remove.append(cid)
        return customers_to_remove

    def _remove_customer_data(self, cid):
        """Remove all data structures related to a customer."""
        self.customer_position_history.pop(cid, None)
        self.customer_id_mapping.pop(cid, None)
        self.customer_first_detected.pop(cid, None)
        self.customer_last_attended.pop(cid, None)
        self.unattended_timers.pop(cid, None)
        self.confirmed_unattended.discard(cid)

    def _cleanup_customer_interactions(self, cid):
        """Clean up interactions involving a specific customer."""
        interactions_to_remove = []
        for (sid, cid_key) in self.active_interactions:
            if cid_key == cid:
                interactions_to_remove.append((sid, cid_key))
        
        for interaction in interactions_to_remove:
            self.active_interactions.pop(interaction, None)
            self.interaction_history.pop(interaction, None)
            self.person_history.pop(cid, None)

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

    def _compute_unattended_with_timer(self, customers, frame_time):
        """
        Enhanced unattended customer detection with confirmation timer.
        Only reports incidents after customer has been consistently unattended for the full timer duration.
        """
        unattended_ids = []
        confirmed_unattended_ids = []
        
        for cid, _pos, _bbox in customers:
            if self._should_process_customer(cid, frame_time):
                result = self._process_customer_unattended_status(cid, frame_time)
                if result:
                    unattended_ids.append(cid)
                    if result == 'confirmed':
                        confirmed_unattended_ids.append(cid)
        
        self._log_unattended_summary(unattended_ids, confirmed_unattended_ids)
        return unattended_ids, confirmed_unattended_ids

    def _should_process_customer(self, cid, frame_time):
        """Check if customer should be processed for unattended detection."""
        if cid not in self.customer_first_detected:
            self.customer_first_detected[cid] = frame_time
            logging.info(f"👤 Customer {cid} first detected at {frame_time:.1f}s")
            return False
        
        # Validate frame_time is reasonable
        if frame_time < 0:
            logging.warning(f"⚠️ Invalid frame_time {frame_time} for customer {cid}")
            return False
            
        tracking_duration = frame_time - self.customer_first_detected[cid]
        
        # Validate tracking duration is reasonable
        if tracking_duration < 0:
            logging.warning(f"⚠️ Negative tracking duration {tracking_duration} for customer {cid}")
            return False
            
        return tracking_duration >= MIN_TRACKING_DURATION_FOR_ALERT

    def _process_customer_unattended_status(self, cid, frame_time):
        """Process unattended status for a single customer."""
        last_attended = self.customer_last_attended.get(cid)
        if last_attended is None:
            self.customer_last_attended[cid] = frame_time
            return None
        
        unattended_duration = frame_time - last_attended
        
        if unattended_duration >= UNATTENDED_THRESHOLD:
            return self._handle_unattended_customer(cid, frame_time, unattended_duration)
        else:
            self._handle_attended_customer(cid, unattended_duration)
            return None

    def _handle_unattended_customer(self, cid, frame_time, unattended_duration):
        """Handle customer who has been unattended for threshold time."""
        if cid not in self.unattended_timers:
            self.unattended_timers[cid] = frame_time
            logging.info(f"⏱️ Started unattended timer for customer {cid} (unattended for {unattended_duration:.1f}s)")
        
        timer_duration = frame_time - self.unattended_timers[cid]
        if timer_duration >= self.unattended_confirmation_timer:
            if cid not in self.confirmed_unattended:
                self._confirm_unattended_customer(cid, timer_duration, unattended_duration)
            return 'confirmed'
        else:
            self._log_confirmation_period(cid, timer_duration)
            return 'pending'

    def _confirm_unattended_customer(self, cid, timer_duration, unattended_duration):
        """Confirm customer as unattended."""
        self.confirmed_unattended.add(cid)
        tracking_duration = self.customer_first_detected.get(cid, 0)
        logging.warning(f"🚨 CONFIRMED UNATTENDED: Customer {cid} - Timer completed ({timer_duration:.1f}s)")
        logging.warning(f"   - Total unattended time: {unattended_duration:.1f}s")
        logging.warning(f"   - Tracking duration: {tracking_duration:.1f}s")

    def _log_confirmation_period(self, cid, timer_duration):
        """Log confirmation period status."""
        remaining_time = self.unattended_confirmation_timer - timer_duration
        logging.info(f"⏳ Customer {cid} in confirmation period: {remaining_time:.1f}s remaining")

    def _handle_attended_customer(self, cid, unattended_duration):
        """Handle customer who was recently attended."""
        if unattended_duration < TIMER_RESET_GRACE_PERIOD:
            if cid in self.unattended_timers:
                logging.info(f"✅ Customer {cid} attended recently - resetting unattended timer")
                del self.unattended_timers[cid]
            if cid in self.confirmed_unattended:
                logging.info(f"✅ Customer {cid} attended recently - removing from confirmed unattended")
                self.confirmed_unattended.discard(cid)
        else:
            # Customer has been unattended for too long, don't reset timers
            # This prevents false resets when customer briefly appears attended
            logging.debug(f"⚠️ Customer {cid} unattended for {unattended_duration:.1f}s - not resetting timers")

    def _log_unattended_summary(self, unattended_ids, confirmed_unattended_ids):
        """Log summary of unattended customers."""
        if unattended_ids:
            confirmed_count = len(confirmed_unattended_ids)
            pending_count = len(unattended_ids) - confirmed_count
            logging.info(f"📊 Unattended Status: {confirmed_count} confirmed, {pending_count} pending confirmation")

    def check_and_log(self, staff, customers, frame_time):
        """Enhanced interaction detection with timer-based unattended customer system."""
        # Stabilize customer tracking first
        self._stabilize_customer_tracking(customers, frame_time)
        
        interactions = []
        for sid, cid, spt, cpt, dist, score, scores, zone in self._pair_iter(staff, customers):
            key = (sid, cid)
            if key in self.active_interactions:
                if self._should_remain_active(key, score, frame_time):
                    interactions.append(key)
                    duration = frame_time - self.active_interactions[key]
                    if duration >= 1.0:
                        logging.info(f"🟡 INTERACTION CONTINUES: Staff {sid} & Customer {cid} | Duration: {duration:.1f}s | Score: {score:.2f}")
                    # Update last attended time when customer is in interaction
                    self.customer_last_attended[cid] = frame_time
            else:
                self._maybe_start(key, sid, cid, score, frame_time, spt, cpt, dist, zone, scores, interactions)

        self._finalize_ended(interactions, frame_time)
        
        # Use enhanced unattended detection with timer
        unattended_ids, confirmed_unattended_ids = self._compute_unattended_with_timer(customers, frame_time)
        
        return interactions, unattended_ids, confirmed_unattended_ids

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

    def get_confirmed_unattended_customers(self):
        """Get customers that have been confirmed as unattended (timer completed)."""
        return list(self.confirmed_unattended)

    def get_unattended_timer_status(self, customer_id):
        """Get timer status for a specific customer."""
        if customer_id in self.unattended_timers:
            return {
                'timer_started': True,
                'timer_start_time': self.unattended_timers[customer_id],
                'is_confirmed': customer_id in self.confirmed_unattended
            }
        return {
            'timer_started': False,
            'timer_start_time': None,
            'is_confirmed': False
        }

    def get_detailed_timing_info(self, frame_time):
        """Get detailed timing information for all customers for incident reporting."""
        timing_info = {
            "customers": {},
            "summary": {
                "total_customers": 0,
                "unattended_customers": 0,
                "confirmed_unattended": 0,
                "pending_confirmation": 0
            }
        }
        
        # Process all customers
        for cid in self.customer_first_detected:
            if cid in self.customer_last_attended:
                last_attended = self.customer_last_attended[cid]
                unattended_duration = frame_time - last_attended
                tracking_duration = frame_time - self.customer_first_detected[cid]
                
                customer_info = {
                    "customer_id": cid,
                    "first_detected": self.customer_first_detected[cid],
                    "last_attended": last_attended,
                    "unattended_duration": unattended_duration,
                    "tracking_duration": tracking_duration,
                    "is_unattended": unattended_duration >= UNATTENDED_THRESHOLD,
                    "is_confirmed": cid in self.confirmed_unattended,
                    "timer_started": cid in self.unattended_timers,
                    "timer_duration": frame_time - self.unattended_timers[cid] if cid in self.unattended_timers else 0
                }
                
                timing_info["customers"][cid] = customer_info
                timing_info["summary"]["total_customers"] += 1
                
                if customer_info["is_unattended"]:
                    timing_info["summary"]["unattended_customers"] += 1
                    
                if customer_info["is_confirmed"]:
                    timing_info["summary"]["confirmed_unattended"] += 1
                elif customer_info["timer_started"]:
                    timing_info["summary"]["pending_confirmation"] += 1
        
        return timing_info 