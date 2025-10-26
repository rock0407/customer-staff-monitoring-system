import numpy as np
import time
import logging
import cv2
from datetime import datetime, timedelta
from config_loader import LOG_FILE, INTERACTION_THRESHOLD, UNATTENDED_THRESHOLD, MIN_TRACKING_DURATION_FOR_ALERT, UNATTENDED_CONFIRMATION_TIMER, TIMER_RESET_GRACE_PERIOD, MIN_QUEUE_DURATION, QUEUE_VALIDATION_PERIOD, QUEUE_STABILITY_THRESHOLD
from metrics_collector import MetricsCollector
from extended_queue_tracker import ExtendedQueueTracker

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
        self.original_to_current = {}  # original_id -> latest current id for display/overlay
        self.position_history_length = 5  # Keep last 5 positions
        
        # Bounding box storage for unattended customers
        self.customer_bounding_boxes = {}  # customer_id -> latest bounding box coordinates
        self.unattended_customer_boxes = {}  # customer_id -> bounding box for confirmed unattended
        
        # Track completed interactions for immediate upload
        self.completed_interactions = []  # List of completed interactions waiting for upload
        
        # ID change rate limiting
        self.last_id_change_time = 0
        self.id_change_cooldown = 1.0  # Minimum 1 second between ID changes
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector("analytics_metrics.json")
        
        # Initialize extended queue tracker for hours-long scenarios
        self.extended_queue_tracker = ExtendedQueueTracker(max_queue_capacity=50, cleanup_interval=300)
        
        # Track queue times with validation
        self.customer_queue_start = {}  # customer_id -> queue start time
        self.customer_queue_end = {}      # customer_id -> queue end time
        self.customer_queue_validation = {}  # customer_id -> validation start time
        self.customer_queue_confirmed = set()  # Set of confirmed queue customers
        
        # Track unattended times
        self.customer_unattended_start = {}  # customer_id -> unattended start time
        self.customer_unattended_end = {}    # customer_id -> unattended end time
        
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
        self.end_score = 0.15        # score to remain active; below this risks ending
        self.start_min_time = 0.50   # must stay above start_score at least this long to start
        self.end_grace = 0.50        # allow brief dips below end_score before ending
        # State for debounce/end-grace
        self._prestart_since = {}    # (sid,cid) -> time when score first exceeded start_score
        self._end_grace_since = {}   # (sid,cid) -> time when score first dropped below end_score
        logging.info("🔧 Interaction Logger initialized - background processing mode")

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
        # Track which old customers we've already processed to avoid duplicate mappings
        processed_old_customers = set()
        
        for old_cid, old_positions in self.customer_position_history.items():
            if old_cid not in current_customer_positions and old_positions and old_cid not in processed_old_customers:
                # Check if we already have a mapping for this old customer
                already_mapped = any(mapped_old == old_cid for mapped_old in self.customer_id_mapping.values())
                if not already_mapped:
                    self._find_id_mapping(old_cid, old_positions[-1], current_customer_positions)
                    processed_old_customers.add(old_cid)

    def _find_id_mapping(self, old_cid, last_known_pos, current_positions):
        """Find if old customer moved to a new ID."""
        # Rate limiting: don't process ID changes too frequently
        current_time = time.time()
        if current_time - self.last_id_change_time < self.id_change_cooldown:
            logging.debug(f"⏰ ID change rate limited for {old_cid} (cooldown: {self.id_change_cooldown}s)")
            return
            
        # Additional safety check: don't process if we already have too many mappings
        if len(self.customer_id_mapping) > 50:
            logging.debug(f"⚠️ Too many ID mappings ({len(self.customer_id_mapping)}), skipping ID change for {old_cid}")
            return
            
        # Find the closest new customer ID that hasn't been mapped yet
        best_match = None
        best_distance = float('inf')
        
        for new_cid, new_pos in current_positions.items():
            # Skip if this new ID is already mapped to someone else
            if new_cid in self.customer_id_mapping:
                continue
                
            # Skip if this new ID already has position history (it's not really new)
            if new_cid in self.customer_position_history:
                continue
                
            distance = np.linalg.norm(np.array(last_known_pos) - np.array(new_pos))
            
            # Use much stricter distance threshold and find the closest match
            if distance < 30 and distance < best_distance:  # Further reduced from 50px to 30px
                best_match = new_cid
                best_distance = distance
        
        # Only create one mapping for the best match, and only if distance is very small
        if best_match is not None and best_distance < 20:  # Only for very close matches
            self.customer_id_mapping[best_match] = old_cid
            self.last_id_change_time = current_time  # Update rate limiting timestamp
            logging.info(f"🔄 Customer ID changed: {old_cid} -> {best_match} (distance: {best_distance:.1f}px)")
            # Maintain reverse mapping for display purposes
            try:
                original_id = self.get_original_customer_id(old_cid)
            except Exception:
                original_id = old_cid
            self.original_to_current[original_id] = best_match
            
            # Clean up old mappings to prevent memory growth
            self._cleanup_old_mappings()
        elif best_match is not None:
            logging.debug(f"🔍 ID change candidate too far: {old_cid} -> {best_match} (distance: {best_distance:.1f}px > 20px)")

    def _cleanup_old_mappings(self):
        """Clean up old mappings to prevent memory growth."""
        # Keep only recent mappings (last 100 entries)
        if len(self.customer_id_mapping) > 100:
            # Remove oldest entries
            items_to_remove = list(self.customer_id_mapping.items())[:-50]  # Keep last 50
            for key, _ in items_to_remove:
                del self.customer_id_mapping[key]
            logging.debug(f"🧹 Cleaned up {len(items_to_remove)} old customer ID mappings")

    def get_original_customer_id(self, cid):
        """Resolve the original (stable) customer id by following mapping chains."""
        visited = set()
        current = cid
        # customer_id_mapping stores new_id -> previous_id
        while current in self.customer_id_mapping and current not in visited:
            visited.add(current)
            current = self.customer_id_mapping[current]
        return current

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
        original_cid = self.get_original_customer_id(cid)
        interactions_to_remove = []
        for (sid, cid_key) in self.active_interactions:
            if cid_key == original_cid:
                interactions_to_remove.append((sid, cid_key))
        
        for interaction in interactions_to_remove:
            self.active_interactions.pop(interaction, None)
            self.interaction_history.pop(interaction, None)
            self.person_history.pop(original_cid, None)

    # --- Complexity reduction helpers ---
    def _pair_iter(self, staff, customers):
        for sid, spt, sbbox in staff:
            for cid, cpt, cbbox in customers:
                dist = np.linalg.norm(np.array(spt) - np.array(cpt))
                score, scores, zone = self.calculate_interaction_score(sbbox, cbbox, dist, spt, cpt)
                # Normalize customer id to original for stable interaction keys
                original_cid = self.get_original_customer_id(cid)
                yield sid, original_cid, spt, cpt, dist, score, scores, zone

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
        logging.info(f"🟢 NEW INTERACTION: Staff {sid} & Customer {cid} (Score: {score:.2f})")
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
                logging.info(f"✅ INTERACTION COMPLETED: Staff {k[0]} & Customer {k[1]} | Duration: {duration:.1f}s")
                self.customer_last_attended[k[1]] = frame_time
                
                # Add to completed interactions list for immediate upload
                self.completed_interactions.append({
                    'staff_id': k[0],
                    'customer_id': k[1],
                    'duration': duration,
                    'start_time': start_time,
                    'end_time': frame_time
                })
                
                # NEW: Report interaction immediately to footfall API
                try:
                    from footfall import report_interaction_analytics
                    success = report_interaction_analytics(
                        staff_id=k[0],
                        customer_id=k[1],
                        duration_seconds=duration
                    )
                    if success:
                        logging.info(f"✅ Footfall API call successful for Staff {k[0]} & Customer {k[1]}")
                    else:
                        logging.error(f"❌ Footfall API call failed for Staff {k[0]} & Customer {k[1]}")
                except Exception as e:
                    logging.error(f"❌ Failed to call footfall API: {e}")
                
            else:
                logging.debug(f"❌ INTERACTION TOO SHORT: Staff {k[0]} & Customer {k[1]} | Duration: {duration:.1f}s")

    def _compute_unattended_with_timer(self, customers, frame_time):
        """
        Enhanced unattended customer detection with confirmation timer.
        Only reports incidents after customer has been consistently unattended for the full timer duration.
        """
        unattended_ids = []
        confirmed_unattended_ids = []
        
        for cid, _pos, bbox in customers:
            # Store bounding box data for all customers
            self.customer_bounding_boxes[cid] = bbox.copy()
            
            if self._should_process_customer(cid, frame_time):
                result = self._process_customer_unattended_status(cid, frame_time)
                if result:
                    unattended_ids.append(cid)
                    if result == 'confirmed':
                        confirmed_unattended_ids.append(cid)
                        # Store bounding box for confirmed unattended customers
                        self.unattended_customer_boxes[cid] = bbox.copy()
        
        self._log_unattended_summary(unattended_ids, confirmed_unattended_ids)
        return unattended_ids, confirmed_unattended_ids

    def _should_process_customer(self, cid, frame_time):
        """Check if customer should be processed for unattended detection."""
        original_cid = self.get_original_customer_id(cid)
        if original_cid not in self.customer_first_detected:
            self.customer_first_detected[original_cid] = frame_time
            logging.debug(f"👤 Customer {original_cid} first detected at {frame_time:.1f}s")
            return False
        
        # Validate frame_time is reasonable
        if frame_time < 0:
            logging.warning(f"⚠️ Invalid frame_time {frame_time} for customer {cid}")
            return False
            
        tracking_duration = frame_time - self.customer_first_detected[original_cid]
        
        # Validate tracking duration is reasonable
        if tracking_duration < 0:
            logging.warning(f"⚠️ Negative tracking duration {tracking_duration} for customer {cid}")
            return False
            
        return tracking_duration >= MIN_TRACKING_DURATION_FOR_ALERT

    def _process_customer_unattended_status(self, cid, frame_time):
        """Process unattended status for a single customer."""
        original_cid = self.get_original_customer_id(cid)
        last_attended = self.customer_last_attended.get(original_cid)
        if last_attended is None:
            self.customer_last_attended[original_cid] = frame_time
            return None
        
        unattended_duration = frame_time - last_attended
        
        if unattended_duration >= UNATTENDED_THRESHOLD:
            return self._handle_unattended_customer(original_cid, frame_time, unattended_duration)
        else:
            self._handle_attended_customer(original_cid, unattended_duration)
            return None

    def _handle_unattended_customer(self, cid, frame_time, unattended_duration):
        """Handle customer who has been unattended for threshold time."""
        if cid not in self.unattended_timers:
            self.unattended_timers[cid] = frame_time
            logging.debug(f"⏱️ Started unattended timer for customer {cid} (unattended for {unattended_duration:.1f}s)")
        
        timer_duration = frame_time - self.unattended_timers[cid]
        if timer_duration >= self.unattended_confirmation_timer:
            if cid not in self.confirmed_unattended:
                self._confirm_unattended_customer(cid, timer_duration, unattended_duration)
            return 'confirmed'
        else:
            # Timer still running - confirmation pending
            return 'pending'

    def _confirm_unattended_customer(self, cid, timer_duration, unattended_duration):
        """Confirm customer as unattended."""
        self.confirmed_unattended.add(cid)
        tracking_duration = self.customer_first_detected.get(cid, 0)
        logging.warning(f"🚨 CONFIRMED UNATTENDED: Customer {cid} - Timer completed ({timer_duration:.1f}s)")
        logging.warning(f"   - Total unattended time: {unattended_duration:.1f}s")
        logging.warning(f"   - Tracking duration: {tracking_duration:.1f}s")
        
        # Start tracking the confirmed unattended customer
        self.start_unattended_tracking(cid)

# _log_confirmation_period function removed - confirmation period logging no longer needed

    def _handle_attended_customer(self, cid, unattended_duration):
        """Handle customer who was recently attended."""
        if unattended_duration < TIMER_RESET_GRACE_PERIOD:
            if cid in self.unattended_timers:
                logging.debug(f"✅ Customer {cid} attended recently - resetting unattended timer")
                del self.unattended_timers[cid]
            if cid in self.confirmed_unattended:
                logging.debug(f"✅ Customer {cid} attended recently - removing from confirmed unattended")
                self.confirmed_unattended.discard(cid)
            # If customer was being tracked as unattended, end the tracking
            if cid in self.customer_unattended_start:
                self.end_unattended_tracking(cid)
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
        for sid, original_cid, spt, cpt, dist, score, scores, zone in self._pair_iter(staff, customers):
            key = (sid, original_cid)
            if key in self.active_interactions:
                if self._should_remain_active(key, score, frame_time):
                    interactions.append(key)
                    duration = frame_time - self.active_interactions[key]
                    if duration >= 1.0:
                        logging.info(f"🟡 INTERACTION CONTINUES: Staff {sid} & Customer {original_cid} | Duration: {duration:.1f}s | Score: {score:.2f}")
                    # Update last attended time when customer is in interaction
                    self.customer_last_attended[original_cid] = frame_time
            else:
                self._maybe_start(key, sid, original_cid, score, frame_time, spt, cpt, dist, zone, scores, interactions)

        self._finalize_ended(interactions, frame_time)
        
        # Use enhanced unattended detection with timer
        unattended_ids, confirmed_unattended_ids = self._compute_unattended_with_timer(customers, frame_time)
        
        return interactions, unattended_ids, confirmed_unattended_ids

    def log_interaction(self, staff_id, cust_id, start, end, duration, status="VALID", video_evidence=None):
        """Log interaction to file with detailed information and optional video evidence."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {status} | Staff {staff_id} & Customer {cust_id} | Start: {start:.2f}s | End: {end:.2f}s | Duration: {duration:.2f}s\n")
        
        # Add to metrics collector (local JSON storage only)
        # Convert relative timestamps to actual datetime objects
        current_time = datetime.now()
        interaction_start = current_time - timedelta(seconds=(end - start))
        interaction_end = current_time
        self.metrics_collector.add_interaction_time(staff_id, cust_id, interaction_start, interaction_end, video_evidence)
        
        # Update daily summary
        self.metrics_collector.update_daily_summary()

    def link_video_evidence_to_interaction(self, staff_id, customer_id, video_evidence):
        """Link video evidence to an existing interaction"""
        return self.metrics_collector.link_video_evidence_to_interaction(staff_id, customer_id, video_evidence)
    
    def link_video_evidence_to_unattended(self, customer_id, video_evidence):
        """Link video evidence to an existing unattended event"""
        return self.metrics_collector.link_video_evidence_to_unattended(customer_id, video_evidence)

    def get_active_interactions(self):
        """Get currently active interactions for visualization.
        Returns pairs using the latest current customer id when available so overlays match.
        """
        result = []
        for sid, original_cid in self.active_interactions.keys():
            display_cid = self.original_to_current.get(original_cid, original_cid)
            result.append((sid, display_cid))
        return result

    def get_interaction_duration(self, staff_id, cust_id, current_time=None):
        """Get current duration of an active interaction using the same timebase as check_and_log.
        Pass current_time=frame_time from the main loop for consistent results.
        """
        # Normalize provided customer id to original for lookup
        original_cid = self.get_original_customer_id(cust_id)
        key = (staff_id, original_cid)
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

    def get_unattended_customer_bounding_boxes(self, unattended_ids, confirmed_unattended_ids):
        """Get bounding box data for unattended customers."""
        bounding_boxes = {
            'unattended': {},
            'confirmed_unattended': {}
        }
        
        for cid in unattended_ids:
            if cid in self.customer_bounding_boxes:
                bounding_boxes['unattended'][cid] = self.customer_bounding_boxes[cid].tolist()
        
        for cid in confirmed_unattended_ids:
            if cid in self.unattended_customer_boxes:
                bounding_boxes['confirmed_unattended'][cid] = self.unattended_customer_boxes[cid].tolist()
        
        return bounding_boxes

    def get_customer_timing_details(self, customer_ids, frame_time):
        """Get detailed timing information for specific customers."""
        timing_details = {}
        for cid in customer_ids:
            if cid in self.customer_first_detected:
                original_cid = self.get_original_customer_id(cid)
                last_attended = self.customer_last_attended.get(cid, self.customer_first_detected[original_cid])
                timing_details[cid] = {
                    'original_id': original_cid,
                    'first_detected': self.customer_first_detected[original_cid],
                    'last_attended': last_attended,
                    'unattended_duration': frame_time - last_attended,
                    'is_confirmed': cid in self.confirmed_unattended,
                    'timer_duration': frame_time - self.unattended_timers[cid] if cid in self.unattended_timers else 0
                }
        return timing_details

    def get_completed_interactions(self):
        """Get and clear completed interactions for immediate upload."""
        completed = self.completed_interactions.copy()
        self.completed_interactions.clear()
        return completed

    # QUEUE TIME TRACKING METHODS
    def start_customer_queue(self, customer_id):
        """Start queue validation for customer with extended tracking"""
        current_time = datetime.now()
        
        if customer_id not in self.customer_queue_validation and customer_id not in self.customer_queue_start:
            # Start validation period
            self.customer_queue_validation[customer_id] = current_time
            logging.debug(f"⏳ Queue validation started for Customer {customer_id}")
        elif customer_id in self.customer_queue_validation:
            # Check if validation period completed
            validation_start = self.customer_queue_validation[customer_id]
            if (current_time - validation_start).total_seconds() >= QUEUE_VALIDATION_PERIOD:
                # Queue confirmed - start actual queue tracking
                self.customer_queue_start[customer_id] = validation_start  # Use validation start time
                self.customer_queue_confirmed.add(customer_id)
                del self.customer_queue_validation[customer_id]
                
                # Add to extended queue tracker
                queue_result = self.extended_queue_tracker.add_customer_to_queue(customer_id)
                logging.info(f"✅ Queue confirmed for Customer {customer_id} - {queue_result['message']}")
        # If already in queue_start, do nothing (already confirmed)

    def end_customer_queue(self, customer_id):
        """End queue time tracking for customer with extended analytics"""
        # Clean up validation if customer was in validation phase
        if customer_id in self.customer_queue_validation:
            del self.customer_queue_validation[customer_id]
            logging.debug(f"⏭️ Queue validation cancelled for Customer {customer_id}")
            return
            
        if customer_id in self.customer_queue_start:
            queue_start = self.customer_queue_start[customer_id]
            queue_end = datetime.now()
            queue_duration = (queue_end - queue_start).total_seconds()
            
            # Remove from extended queue tracker
            queue_result = self.extended_queue_tracker.remove_customer_from_queue(customer_id)
            
            # Only record if queue duration meets minimum threshold
            if queue_duration >= MIN_QUEUE_DURATION:
                self.metrics_collector.add_queue_time(customer_id, queue_start, queue_end)
                logging.info(f"✅ Valid queue recorded: Customer {customer_id}, Duration: {queue_duration:.2f}s")
                
                # Record service time for analytics
                if queue_duration > 0:
                    self.extended_queue_tracker.record_service_time(queue_duration)
            else:
                logging.debug(f"⏭️ Queue too short: Customer {customer_id}, Duration: {queue_duration:.2f}s (min: {MIN_QUEUE_DURATION}s)")
            
            # Clean up
            del self.customer_queue_start[customer_id]
            self.customer_queue_confirmed.discard(customer_id)
            
            # Clean up extended data periodically
            self.extended_queue_tracker.cleanup_extended_data()

    # UNATTENDED TIME TRACKING METHODS
    def start_unattended_tracking(self, customer_id):
        """Start tracking unattended time for customer"""
        self.customer_unattended_start[customer_id] = datetime.now()
        logging.info(f"🚨 Unattended tracking started for Customer {customer_id}")

    def end_unattended_tracking(self, customer_id):
        """End unattended time tracking for customer"""
        if customer_id in self.customer_unattended_start:
            unattended_start = self.customer_unattended_start[customer_id]
            unattended_end = datetime.now()
            unattended_duration = (unattended_end - unattended_start).total_seconds()
            
            # Only log meaningful unattended durations (≥5 seconds to filter out noise)
            if unattended_duration >= 5.0:
                # Record unattended time metric
                self.metrics_collector.add_unattended_time(customer_id, unattended_start, unattended_end)
                logging.info(f"✅ Unattended tracking ended for Customer {customer_id} - Duration: {unattended_duration:.1f}s")
            else:
                logging.debug(f"🔇 Skipped logging brief unattended duration for Customer {customer_id}: {unattended_duration:.1f}s")
            
            # Clean up
            del self.customer_unattended_start[customer_id]
            
            # Remove from confirmed unattended set
            self.confirmed_unattended.discard(customer_id)

    # METRICS AND ANALYTICS METHODS
    def get_metrics_summary(self):
        """Get current metrics summary"""
        return self.metrics_collector.get_metrics_summary()

    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        return self.metrics_collector.generate_analytics_report()
    
    def get_extended_queue_analytics(self):
        """Get extended queue analytics for hours-long scenarios"""
        return self.extended_queue_tracker.get_extended_analytics()
    
    def get_queue_status(self):
        """Get current queue status with positions and wait times"""
        return self.extended_queue_tracker.get_queue_status()
    
    def get_customer_queue_info(self, customer_id):
        """Get detailed queue information for a specific customer"""
        return self.extended_queue_tracker.get_customer_queue_info(customer_id)
    
    def export_extended_analytics(self, filename=None):
        """Export extended queue analytics to JSON file"""
        return self.extended_queue_tracker.export_analytics(filename)

    def update_daily_summary(self):
        """Update daily summary metrics"""
        self.metrics_collector.update_daily_summary()

