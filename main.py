import cv2
import time
import signal
import sys
import logging
import csv
from config_loader import (VIDEO_SOURCE, LINE_COORDS, HEADLESS, MIN_INTERACTION_DURATION, 
                   INTERACTION_THRESHOLD, LOG_LEVEL, DEBUG_SUMMARY_EVERY_N_FRAMES, UNATTENDED_THRESHOLD,
                   MIN_TRACKING_DURATION_FOR_ALERT, ORGANIZATION_NAME, BRANCH_ID, LOCATION)
from detector import PersonDetector
from tracker_bytetrack import PersonTrackerBYTE
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

# 2. Initialize modules
person_detector = PersonDetector()
logging.warning("🔍 Person detector initialized")

# Initialize ByteTrack tracker
person_tracker = PersonTrackerBYTE()
logging.warning('🎯 Using ByteTrack tracker for persistent tracking')

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
    except Exception as e:
        logging.error(f"Detector error: {e}. Skipping detections this frame.")
        boxes, confs = np.zeros((0,4), dtype=float), np.zeros((0,), dtype=float)
    if len(boxes) > 0:
        logging.debug(f"Frame {frame_count}: Detected {len(boxes)} people")
    
    # 4. Track people (guard against tracker exceptions)
    try:
        tracks = person_tracker.update(boxes, confs, frame)
    except Exception as e:
        logging.error(f"Tracker update error: {e}. Using empty tracks this frame.")
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
    
    # 5. Classify by line for LOGIC using true tracker outputs
    for track in tracks:
        tid = track['id']
        xyxy = track['xyxy'].astype(int)
        cx = int((xyxy[0] + xyxy[2]) / 2)
        cy = int((xyxy[1] + xyxy[3]) / 2)
        
        # Use the same logic as before
        (x1, y1), (x2, y2) = line_pts
        side = (x2 - x1)*(cy - y1) - (y2 - y1)*(cx - x1)
        
        if side > 0:
            staff.append((tid, (cx, cy), xyxy))  # Include bounding box
            color = (255, 0, 0)  # Red for staff
            label = f'Staff {tid}'
            category = "STAFF"
        else:
            customers.append((tid, (cx, cy), xyxy))  # Include bounding box
            color = (0, 255, 255)  # Yellow for customers
            label = f'Cust {tid}'
            category = "CUSTOMER"
    # If tracker produced no tracks but detections exist, fall back to detections for logic
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
        
    # Draw pass can use fallback display_tracks for visualization only (no logic)
    for track in display_tracks:
        tid = track['id']
        xyxy = np.array(track['xyxy']).astype(int)
        cx = int((xyxy[0] + xyxy[2]) / 2)
        cy = int((xyxy[1] + xyxy[3]) / 2)
        (x1, y1), (x2, y2) = line_pts
        side = (x2 - x1)*(cy - y1) - (y2 - y1)*(cx - x1)
        color = (255, 0, 0) if side > 0 else (0, 255, 255)
        label = f"Staff {tid}" if side > 0 else f"Cust {tid}"
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        draw_text_with_background(frame, f"{label}", (xyxy[0], xyxy[1]-10), color=color, bg_color=(0, 0, 0))
        cv2.circle(frame, (cx, cy), 4, color, -1)
        for (sid, cid) in active_interactions:
            if tid == sid or tid == cid:
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)
                break

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

    # 6. Interaction detection and logging (now returns unattended IDs)
    interactions_frame, unattended_ids = interaction_logger.check_and_log(staff, customers, frame_time)
    active_interactions = interaction_logger.get_active_interactions()
    
    # No immediate face/image alerts – segments with unattended will be reported upon finalization
    if unattended_ids:
        logging.info(f"📹 Segment flagged for unattended customers: {len(unattended_ids)}")
    
    # Draw unattended customers indicator
    if unattended_ids:
        for cid, pos, bbox in customers:
            if cid in unattended_ids:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                draw_text_with_background(frame, f"Unattended {cid}", (bbox[0], bbox[1]-25), color=(255,255,255), bg_color=(0,0,255))
    
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

    # 7. Add status overlay
    # Get current segment info for display
    segment_info = video_segmenter.get_current_segment_info()
    
    status_text = [
        f"Frame: {frame_count}",
        f"Time: {frame_time:.1f}s",
        f"Staff: {len(staff)}",
        f"Customers: {len(customers)}",
        f"Active Interactions: {len(active_interactions)}",
        f"Segment: {segment_info['segment_idx']} ({segment_info['duration']:.1f}s)",
        f"Interactions in Segment: {segment_info['interaction_count']}",
        f"Interaction Threshold: {INTERACTION_THRESHOLD}px",
        f"Min Duration: {MIN_INTERACTION_DURATION}s",
        f"Min Tracking for Alert: {MIN_TRACKING_DURATION_FOR_ALERT/60:.1f}min",
        f"Unattended: {len(unattended_ids)}"
    ]
    
    y_offset = 30
    for text in status_text:
        draw_text_with_background(frame, text, (10, y_offset), 
                                 color=(255, 255, 255), bg_color=(0, 0, 0))
        y_offset += 25

    # Periodic debug snapshot for diagnosis
    if frame_count % max(1, int(DEBUG_SUMMARY_EVERY_N_FRAMES)) == 0:
        logging.info(
            f"📌 Snapshot f={frame_count} t={frame_time:.1f}s staff={len(staff)} cust={len(customers)} "
            f"active={len(active_interactions)} unattended={len(unattended_ids)}"
        )

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
    video_segmenter.add_frame(frame, active_interactions, any_met_duration, unattended_present=len(unattended_ids) > 0)

    # 9. Optionally show live video if not headless
    if not HEADLESS:
        cv2.imshow('ACSI - Staff-Customer Interaction Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Finalize last segment
video_segmenter.finalize_segment()
cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()

logging.info(f"📊 Processing complete: {frame_count} frames processed")
logging.info('ACSI stopped.') 