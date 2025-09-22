import cv2
import time
import signal
import sys
import logging
import csv
from config_loader import (VIDEO_SOURCE, LINE_COORDS, HEADLESS, MIN_INTERACTION_DURATION, 
                   INTERACTION_THRESHOLD, LOG_LEVEL, DEBUG_SUMMARY_EVERY_N_FRAMES, UNATTENDED_THRESHOLD,
                   MIN_TRACKING_DURATION_FOR_ALERT, ORGANIZATION_NAME, BRANCH_ID, LOCATION, UNATTENDED_CONFIRMATION_TIMER,
                   TRACK_THRESH, TRACK_BUFFER, MATCH_THRESH, TRACKING_FRAME_RATE, ENABLE_TRACKING_STATS)
from detector import PersonDetector
from tracker_bytetrack import PersonTrackerBYTE
from tracker_simple import SimpleTracker
from line_drawer import LineDrawer
from interaction import InteractionLogger
from video_segmenter import VideoSegmenter
 
import numpy as np
from line_calculating import get_line_from_user

# Setup logging
level_map = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR
}
logging.basicConfig(
    level=level_map.get(LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('acsi.log', encoding='utf-8'), 
        logging.StreamHandler(sys.stdout)
    ]
)

# Optionally enable debug logging for tracker issues
import os
if os.environ.get('ACSI_DEBUG_TRACKER', '0') == '1':
    logging.getLogger().setLevel(logging.DEBUG)

# Add this function to reduce logging verbosity
def setup_quiet_logging():
    """Configure logging for production with less verbosity."""
    # Set YOLO logging to WARNING to reduce output
    import logging
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    
    # Flush stdout to prevent garbled output
    sys.stdout.flush()

setup_quiet_logging()

# Initialize CSV file for bounding box data
def init_bbox_csv():
    """Initialize CSV file for bounding box data."""
    with open('bounding_boxes.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'frame', 'time', 'person_id', 'category', 'centroid_x', 'centroid_y', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

# Immediate face/image alerts removed per requirements; incidents are reported via segments only

def log_bbox_data(frame_count, frame_time, staff, customers):
    """Log bounding box data to CSV file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open('bounding_boxes.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Log staff data
        for tid, (cx, cy), xyxy in staff:
            writer.writerow([timestamp, frame_count, f"{frame_time:.2f}", tid, "STAFF", cx, cy, xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
        
        # Log customer data
        for tid, (cx, cy), xyxy in customers:
            writer.writerow([timestamp, frame_count, f"{frame_time:.2f}", tid, "CUSTOMER", cx, cy, xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

# Graceful shutdown flag
running = True
def handle_signal(sig, frame):
    global running
    logging.info('Received shutdown signal. Exiting...')
    running = False
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def open_video():
    # Prefer FFMPEG backend and set sane RTSP options if supported
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO_SOURCE)
    try:
        # Reduce internal buffering; prefer TCP transport for RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    except Exception:
        pass
    if not cap.isOpened():
        logging.error('Cannot open video source')
        return None
    return cap

def draw_text_with_background(img, text, position, font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, 
                  (position[0], position[1] - text_height - baseline),
                  (position[0] + text_width, position[1] + baseline),
                  bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, position, font, font_scale, color, thickness)

def draw_interaction_line(img, pt1, pt2, color=(0, 255, 255), thickness=2):
    """Draw interaction line between two points."""
    cv2.line(img, pt1, pt2, color, thickness)
    # Draw circles at endpoints
    cv2.circle(img, pt1, 3, color, -1)
    cv2.circle(img, pt2, 3, color, -1)

def _add_customer_timer_overlays(frame, customers, interaction_logger, frame_time):
    """Add visual timer overlays for each customer showing unattended duration."""
    for cid, cpt, cbbox in customers:
        # Get customer timing information
        if cid in interaction_logger.customer_last_attended:
            last_attended = interaction_logger.customer_last_attended[cid]
            unattended_duration = frame_time - last_attended
            
            # Only show timer if customer has been unattended for threshold time
            if unattended_duration >= UNATTENDED_THRESHOLD:
                # Get timer status
                timer_status = interaction_logger.get_unattended_timer_status(cid)
                
                # Calculate position for timer display (above customer)
                timer_x = cpt[0]
                timer_y = cpt[1] - 20
                
                # Determine timer color and text based on status
                if timer_status['is_confirmed']:
                    # Confirmed unattended - RED
                    timer_color = (0, 0, 255)  # Red
                    bg_color = (0, 0, 0)  # Black background
                    timer_text = f"C{cid}: CONFIRMED UNATTENDED"
                elif timer_status['timer_started']:
                    # In confirmation period - ORANGE
                    timer_duration = frame_time - timer_status['timer_start_time']
                    remaining_time = UNATTENDED_CONFIRMATION_TIMER - timer_duration
                    timer_color = (0, 165, 255)  # Orange
                    bg_color = (0, 0, 0)  # Black background
                    timer_text = f"C{cid}: Pending ({remaining_time:.1f}s)"
                else:
                    # Just started being unattended - YELLOW
                    timer_color = (0, 255, 255)  # Yellow
                    bg_color = (0, 0, 0)  # Black background
                    timer_text = f"C{cid}: Unattended ({unattended_duration:.1f}s)"
                
                # Draw timer text with background
                draw_text_with_background(frame, timer_text, (timer_x, timer_y), 
                                       font_scale=0.5, color=timer_color, bg_color=bg_color)
                
                # Draw a colored circle around the customer
                circle_radius = 15
                cv2.circle(frame, cpt, circle_radius, timer_color, 3)
                
                # Add total unattended time below
                total_time_text = f"Total: {unattended_duration:.1f}s"
                draw_text_with_background(frame, total_time_text, (timer_x, timer_y + 20), 
                                       font_scale=0.4, color=timer_color, bg_color=bg_color)

cap = open_video()
if cap is None:
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25
# Clamp FPS to a sane range to avoid FFMPEG timebase issues (e.g., 90000 from RTSP)
try:
    if fps is None or fps <= 0 or fps > 120:
        logging.warning(f"⚠️ Invalid/unsupported FPS reported ({fps}). Forcing to 25fps for writer compatibility.")
        fps = 25
except Exception:
    fps = 25
ret, first_frame = cap.read()
if not ret:
    logging.error('Cannot read first frame')
    sys.exit(1)
frame_h, frame_w = first_frame.shape[:2]

logging.warning(f"🎬 Video loaded: {frame_w}x{frame_h} @ {fps}fps")

# 1. Get line coordinates
if HEADLESS or LINE_COORDS:
    if LINE_COORDS is None:
        logging.error('LINE_COORDS must be set in config_loader.py for headless mode.')
        sys.exit(1)
    line_pts = [tuple(LINE_COORDS[0]), tuple(LINE_COORDS[1])]
    logging.warning(f"📍 Using configured line: {line_pts[0]} to {line_pts[1]}")
else:
    # Use the interactive line drawing utility
    result = get_line_from_user(first_frame)
    if result is None:
        logging.error('No line drawn. Exiting.')
        sys.exit(1)
    line_pts = [result[0], result[1]]
    print(f"Copy these coordinates to config_loader.py as LINE_COORDS = {result}")
    logging.warning(f"📍 Line drawn: {line_pts[0]} to {line_pts[1]}")

# 2. Validate configuration and initialize modules
def validate_config():
    """Validate critical configuration parameters."""
    issues = []
    
    if UNATTENDED_CONFIRMATION_TIMER <= 0:
        issues.append("UNATTENDED_CONFIRMATION_TIMER must be positive")
    if UNATTENDED_THRESHOLD <= 0:
        issues.append("UNATTENDED_THRESHOLD must be positive")
    if MIN_TRACKING_DURATION_FOR_ALERT <= 0:
        issues.append("MIN_TRACKING_DURATION_FOR_ALERT must be positive")
    if INTERACTION_THRESHOLD <= 0:
        issues.append("INTERACTION_THRESHOLD must be positive")
    if MIN_INTERACTION_DURATION <= 0:
        issues.append("MIN_INTERACTION_DURATION must be positive")
    
    if issues:
        logging.error("❌ Configuration validation failed:")
        for issue in issues:
            logging.error(f"   - {issue}")
        return False
    
    logging.info("✅ Configuration validation passed")
    return True

if not validate_config():
    logging.error("❌ Invalid configuration, exiting")
    sys.exit(1)

person_detector = PersonDetector()
logging.warning("🔍 Person detector initialized")

# Initialize tracker with fallback to simple tracker
try:
    person_tracker = PersonTrackerBYTE(
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        frame_rate=TRACKING_FRAME_RATE
    )
    logging.warning('🎯 Using optimized ByteTrack tracker for persistent tracking')
except Exception as e:
    logging.warning(f'⚠️ ByteTracker failed to initialize: {e}')
    logging.warning('🔄 Falling back to SimpleTracker')
    person_tracker = SimpleTracker(
        iou_threshold=0.3,
        max_disappeared=30,
        max_distance=100.0
    )

interaction_logger = InteractionLogger(min_duration=MIN_INTERACTION_DURATION, threshold=INTERACTION_THRESHOLD)
logging.warning(f"📊 Interaction logger initialized (min duration: {MIN_INTERACTION_DURATION}s, threshold: {INTERACTION_THRESHOLD}px)")
video_segmenter = VideoSegmenter(fps=int(fps), frame_size=(frame_w, frame_h))
logging.warning("🎥 Video segmenter initialized")

start_time = time.monotonic()
frame_count = 0
total_people_detected = 0

logging.warning("🚀 Starting video processing...")

# Initialize CSV file for bounding box data
init_bbox_csv()
logging.warning("📊 Bounding box CSV file initialized: bounding_boxes.csv")

while running:
    ret, frame = cap.read()
    if not ret:
        logging.warning('Frame grab failed. Attempting to reconnect in 5 seconds...')
        cap.release()
        time.sleep(5)
        cap = open_video()
        if cap is None:
            break
        continue
    
    frame_count += 1
    frame_time = time.monotonic() - start_time

    # 3. Detect people (robust to detector API changes)
    try:
        boxes, confs = person_detector.detect(frame)
        if len(boxes) > 0:
            logging.debug(f"Frame {frame_count}: Detected {len(boxes)} people")
    except Exception as e:
        logging.error(f"Detector error: {e}. Skipping detections this frame.")
        boxes, confs = np.zeros((0,4), dtype=float), np.zeros((0,), dtype=float)
    
    # 4. Track people (guard against tracker exceptions)
    try:
        tracks = person_tracker.update(boxes, confs, frame)
    except Exception as e:
        logging.error(f"Tracker update error: {e}. Using empty tracks this frame.")
        tracks = []
    
    # Validate tracks data structure
    if not isinstance(tracks, list):
        logging.warning("⚠️ Tracker returned invalid data type, converting to list")
        tracks = []
    # Safety net: if tracker yields no tracks but we have detections, display detections
    display_tracks = tracks
    if (not tracks) and len(boxes) > 0:
        display_tracks = []
        for i, b in enumerate(boxes):
            try:
                display_tracks.append({'id': 100000 + i, 'xyxy': b.astype(int)})
            except Exception:
                continue

    # Logical lists used for interaction logic (must be based on real tracker IDs)
    staff = []
    customers = []
    active_interactions = interaction_logger.get_active_interactions()
    
    # 5. Process tracks once: classify, draw, and add to staff/customer lists
    for track in tracks:
        tid = track['id']
        xyxy = track['xyxy'].astype(int)
        
        # Calculate center coordinates once
        cx = int((xyxy[0] + xyxy[2]) / 2)
        cy = int((xyxy[1] + xyxy[3]) / 2)
        
        # Classify based on line position
        (x1, y1), (x2, y2) = line_pts
        side = (x2 - x1)*(cy - y1) - (y2 - y1)*(cx - x1)
        
        if side > 0:
            staff.append((tid, (cx, cy), xyxy))
            color = (255, 0, 0)  # Red for staff
            label = f"S{tid}"
        else:
            customers.append((tid, (cx, cy), xyxy))
            color = (0, 255, 255)  # Cyan for customers
            label = f"C{tid}"
        
        # Draw person indicator (small dot and label)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        draw_text_with_background(frame, f"{label}", (cx+8, cy-8), color=color, bg_color=(0, 0, 0), font_scale=0.5)
        
        # Show interaction indicator if in active interaction
        for (sid, cid) in active_interactions:
            if tid == sid or tid == cid:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                break
    
    # Fallback: if no tracks, use detections for classification only (no drawing)
    if len(staff) == 0 and len(customers) == 0 and display_tracks:
        logging.debug("Tracker returned 0 tracks; falling back to detections for logic this frame.")
        for track in display_tracks:
            tid = track['id']
            xyxy = np.array(track['xyxy']).astype(int)
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)
            (x1, y1), (x2, y2) = line_pts
            side = (x2 - x1)*(cy - y1) - (y2 - y1)*(cx - x1)
            if side > 0:
                staff.append((tid, (cx, cy), xyxy))
            else:
                customers.append((tid, (cx, cy), xyxy))

    # Log bounding box information for staff and customers
    if len(staff) > 0 or len(customers) > 0:
        logging.info(f"📊 Frame {frame_count} ({frame_time:.1f}s):")
        if len(staff) > 0:
            staff_info = []
            for tid, (cx, cy), xyxy in staff:
                staff_info.append(f"Staff {tid}: pos=({cx},{cy}), bbox=({xyxy[0]},{xyxy[1]},{xyxy[2]},{xyxy[3]})")
            logging.info(f"👨‍💼 Staff ({len(staff)}): {' | '.join(staff_info)}")
        if len(customers) > 0:
            cust_info = []
            for tid, (cx, cy), xyxy in customers:
                cust_info.append(f"Customer {tid}: pos=({cx},{cy}), bbox=({xyxy[0]},{xyxy[1]},{xyxy[2]},{xyxy[3]})")
            logging.info(f"👥 Customers ({len(customers)}): {' | '.join(cust_info)}")

    # Log bounding box data to CSV for analysis
    log_bbox_data(frame_count, frame_time, staff, customers)

    # Draw the separation line
    if len(line_pts) == 2:
        cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 3)
        # Add line label
        mid_x = (line_pts[0][0] + line_pts[1][0]) // 2
        mid_y = (line_pts[0][1] + line_pts[1][1]) // 2
        draw_text_with_background(frame, "STAFF AREA", (mid_x-50, mid_y-20), 
                                 color=(255, 0, 0), bg_color=(0, 0, 0))
        draw_text_with_background(frame, "CUSTOMER AREA", (mid_x-50, mid_y+20), 
                                 color=(0, 255, 255), bg_color=(0, 0, 0))

    # 6. Interaction detection and logging (now returns unattended IDs and confirmed unattended)
    interactions_frame, unattended_ids, confirmed_unattended_ids = interaction_logger.check_and_log(staff, customers, frame_time)
    active_interactions = interaction_logger.get_active_interactions()
    
    # Enhanced logging for timer-based system
    if unattended_ids:
        confirmed_count = len(confirmed_unattended_ids)
        pending_count = len(unattended_ids) - confirmed_count
        logging.info(f"📹 Unattended customers: {confirmed_count} confirmed, {pending_count} pending confirmation")
    
    # Draw unattended customers with different indicators (no bounding boxes, just status)
    if unattended_ids:
        for cid, pos, bbox in customers:
            if cid in unattended_ids:
                if cid in confirmed_unattended_ids:
                    # Confirmed unattended - show red status indicator
                    cv2.circle(frame, pos, 15, (0, 0, 255), -1)  # Red filled circle
                    cv2.circle(frame, pos, 20, (0, 0, 255), 2)   # Red border
                    draw_text_with_background(frame, f"CONFIRMED {cid}", (pos[0]-30, pos[1]-30), 
                                           color=(255,255,255), bg_color=(0,0,255))
                else:
                    # Pending confirmation - show orange status indicator
                    cv2.circle(frame, pos, 12, (0, 165, 255), -1)  # Orange filled circle
                    cv2.circle(frame, pos, 17, (0, 165, 255), 2)   # Orange border
                    # Show countdown timer
                    timer_status = interaction_logger.get_unattended_timer_status(cid)
                    if timer_status['timer_started']:
                        remaining_time = UNATTENDED_CONFIRMATION_TIMER - (frame_time - timer_status['timer_start_time'])
                        draw_text_with_background(frame, f"Pending {cid} ({remaining_time:.0f}s)", (pos[0]-40, pos[1]-30), 
                                               color=(255,255,255), bg_color=(0,165,255))
    
    # Draw active interaction lines with enhanced information
    for (sid, cid) in active_interactions:
        # Find staff and customer positions
        staff_pos = None
        cust_pos = None
        staff_bbox = None
        cust_bbox = None
        for tid, pos, bbox in staff:  # Unpack tuple with bounding box
            if tid == sid:
                staff_pos = pos
                staff_bbox = bbox
                break
        for tid, pos, bbox in customers:  # Unpack tuple with bounding box
            if tid == cid:
                cust_pos = pos
                cust_bbox = bbox
                break
        
        if staff_pos and cust_pos and staff_bbox is not None and cust_bbox is not None:
            # Calculate interaction score and zone
            dist = np.linalg.norm(np.array(staff_pos) - np.array(cust_pos))
            score, scores, zone = interaction_logger.calculate_interaction_score(staff_bbox, cust_bbox, dist, staff_pos, cust_pos)
            
            # Draw interaction line with color based on score
            if score > 0.7:
                line_color = (0, 255, 0)
            elif score > 0.5:
                line_color = (0, 255, 255)
            else:
                line_color = (0, 165, 255)
            draw_interaction_line(frame, staff_pos, cust_pos, line_color, 2)
            
            # Show interaction duration and score
            duration = interaction_logger.get_interaction_duration(sid, cid, frame_time)
            mid_x = (staff_pos[0] + cust_pos[0]) // 2
            mid_y = (staff_pos[1] + cust_pos[1]) // 2
            
            # Draw interaction info box
            info_text = f"{duration:.1f}s | {zone} | {score:.2f}"
            draw_text_with_background(frame, info_text, (mid_x-40, mid_y), 
                                     color=line_color, bg_color=(0, 0, 0))
            
            # Draw zone indicators around people
            zone_colors = {'close': (0, 255, 0), 'medium': (0, 255, 255), 'far': (0, 165, 255)}
            zone_color = zone_colors.get(zone, (128, 128, 128))
            
            # Draw zone circles around staff and customer
            cv2.circle(frame, staff_pos, 15, zone_color, 1)  # Inner zone
            cv2.circle(frame, staff_pos, 30, zone_color, 1)  # Outer zone
            cv2.circle(frame, cust_pos, 15, zone_color, 1)   # Inner zone
            cv2.circle(frame, cust_pos, 30, zone_color, 1)   # Outer zone

    # Draw potential interactions (high scores but not yet active)
    for sid, spt, sbbox in staff:
        for cid, cpt, cbbox in customers:
            # Skip if this is already an active interaction
            if (sid, cid) in active_interactions:
                continue
            
            dist = np.linalg.norm(np.array(spt) - np.array(cpt))
            score, scores, zone = interaction_logger.calculate_interaction_score(sbbox, cbbox, dist, spt, cpt)
            
            # Show potential interactions with high scores
            if score > 0.3:  # Show potential interactions
                # Draw dashed line for potential interaction
                line_color = (128, 128, 128)  # Gray for potential
                cv2.line(frame, spt, cpt, line_color, 1, cv2.LINE_AA)
                
                # Show score at midpoint
                mid_x = (spt[0] + cpt[0]) // 2
                mid_y = (spt[1] + cpt[1]) // 2
                score_text = f"{score:.2f}"
                draw_text_with_background(frame, score_text, (mid_x, mid_y), 
                                         color=line_color, bg_color=(0, 0, 0), font_scale=0.4)

    # 7. Add visual timer overlays for each customer
    _add_customer_timer_overlays(frame, customers, interaction_logger, frame_time)
    
    # 8. Add status overlay with timer information
    # Get current segment info for display
    segment_info = video_segmenter.get_current_segment_info()
    
    # Calculate timer status
    confirmed_count = len(confirmed_unattended_ids) if 'confirmed_unattended_ids' in locals() else 0
    pending_count = len(unattended_ids) - confirmed_count if 'unattended_ids' in locals() else 0
    
    status_text = [
        f"Frame: {frame_count} | Time: {frame_time:.1f}s",
        f"Staff: {len(staff)} | Customers: {len(customers)} | Interactions: {len(active_interactions)}",
        f"Unattended: {confirmed_count} confirmed, {pending_count} pending",
        f"Segment: {segment_info['segment_idx']} ({segment_info['duration']:.1f}s)",
        f"Timer: {UNATTENDED_CONFIRMATION_TIMER}s confirmation",
        f"Threshold: {UNATTENDED_THRESHOLD}s unattended"
    ]
    
    y_offset = 30
    for text in status_text:
        draw_text_with_background(frame, text, (10, y_offset), 
                                 color=(255, 255, 255), bg_color=(0, 0, 0))
        y_offset += 25

    # Periodic debug snapshot for diagnosis
    if frame_count % max(1, int(DEBUG_SUMMARY_EVERY_N_FRAMES)) == 0:
        # Calculate processing performance
        elapsed_time = time.monotonic() - start_time
        fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        logging.info(
            f"📌 Snapshot f={frame_count} t={frame_time:.1f}s staff={len(staff)} cust={len(customers)} "
            f"active={len(active_interactions)} unattended={len(unattended_ids)} fps={fps_actual:.1f}"
        )
        
        # Memory usage warning
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            logging.warning(f"⚠️ High memory usage: {memory_percent:.1f}%")
        
        # Tracking performance monitoring
        if ENABLE_TRACKING_STATS:
            tracking_stats = person_tracker.get_tracking_stats()
            if tracking_stats:
                logging.info(f"🎯 Tracking Performance: "
                           f"Efficiency: {tracking_stats.get('track_efficiency', 0):.2f}, "
                           f"Avg Tracks/Frame: {tracking_stats.get('avg_tracks_per_frame', 0):.1f}")

    # 8. Save video segment with interaction info
    # Only mark segment as having interactions when any pair exceeds MIN_INTERACTION_DURATION
    any_met_duration = False
    if active_interactions:
        for (sid, cid) in active_interactions:
            try:
                if interaction_logger.get_interaction_duration(sid, cid) >= MIN_INTERACTION_DURATION:
                    any_met_duration = True
                    break
            except Exception:
                continue
    
    # Get timing details for incident reporting
    timing_details = interaction_logger.get_detailed_timing_info(frame_time)
    video_segmenter.set_unattended_timing_details(timing_details)
    
    # Pass confirmed unattended customers to video segmenter
    video_segmenter.add_frame(frame, active_interactions, any_met_duration, 
                            unattended_present=len(unattended_ids) > 0, 
                            confirmed_unattended_present=len(confirmed_unattended_ids) > 0)

    # 9. Optionally show live video if not headless
    if not HEADLESS:
        cv2.imshow('ACSI - Staff-Customer Interaction Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Graceful shutdown
logging.info("🛑 Shutting down CSI system...")

try:
    # Finalize last segment
    video_segmenter.finalize_segment()
    logging.info("✅ Video segmenter finalized")
except Exception as e:
    logging.error(f"❌ Error finalizing video segmenter: {e}")

try:
    cap.release()
    logging.info("✅ Video capture released")
except Exception as e:
    logging.error(f"❌ Error releasing video capture: {e}")

if not HEADLESS:
    try:
        cv2.destroyAllWindows()
        logging.info("✅ OpenCV windows closed")
    except Exception as e:
        logging.error(f"❌ Error closing OpenCV windows: {e}")

logging.info(f"📊 Processing complete: {frame_count} frames processed")
logging.info('ACSI stopped gracefully.') 