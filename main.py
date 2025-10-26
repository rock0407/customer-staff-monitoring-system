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
# Now using per-interaction footfall API calls
from detector import PersonDetector
from tracker_bytetrack import PersonTrackerBYTE
from tracker_simple import SimpleTracker
from line_drawer import LineDrawer
from interaction import InteractionLogger
from video_segmenter_optimized import VideoSegmenter
from overlay_utils import OverlayRenderer
 
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

# Removed: Bounding box CSV logging - not needed for production

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

# Visual display helper functions removed - no longer needed for background-only operation
# All interaction and timing calculations continue to work without visual elements

# _add_customer_timer_overlays function removed - visual overlays no longer needed
# Timer calculations continue in background through interaction_logger

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
overlay_renderer = OverlayRenderer()
logging.warning("🎨 Overlay renderer initialized for unattended customer visualization")

# Initialize analytics scheduler for 10-minute updates
from analytics_scheduler import AnalyticsScheduler
analytics_scheduler = AnalyticsScheduler(update_interval_minutes=10)
analytics_scheduler.set_interaction_logger(interaction_logger)
analytics_scheduler.start()
logging.warning("📊 Analytics scheduler started - will update JSON every 10 minutes")

# Initialize queue and unattended tracking state
queue_tracking = set()  # Set of customer IDs currently in queue
unattended_tracking = set()  # Set of customer IDs currently being tracked as unattended

# Hourly aggregation is now handled by InteractionLogger
# No need for separate analytics tracking

start_time = time.monotonic()
frame_count = 0
total_people_detected = 0

logging.warning("🚀 Starting video processing...")

# Initialize CSV file for bounding box data
# Removed: Bounding box CSV initialization - not needed for production

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
        
        # Visual indicators removed - calculations continue in background
        # Person detection and tracking data is still processed for interaction analysis
    
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

    # Detailed position logging removed - essential detection data still processed
    # Staff and customer detection continues in background for interaction analysis

    # Removed: Bounding box CSV logging - not needed for production

    # Separation line and area labels removed - line calculation still used for staff/customer classification
    # The line coordinates are still used in the background for determining which side of the line people are on

    # 6. Interaction detection and logging (now returns unattended IDs and confirmed unattended)
    interactions_frame, unattended_ids, confirmed_unattended_ids = interaction_logger.check_and_log(staff, customers, frame_time)
    active_interactions = interaction_logger.get_active_interactions()
    
    # 6.1. IMPROVED QUEUE TRACKING - Track customers with validation and filtering
    current_customer_ids = {cid for cid, _, _ in customers}
    active_customer_ids = {cid for _, cid in active_interactions}
    
    # Start queue validation for customers who are detected but not in interaction
    for cid in current_customer_ids:
        if cid not in active_customer_ids and cid not in queue_tracking:
            interaction_logger.start_customer_queue(cid)
            queue_tracking.add(cid)
            logging.debug(f"⏳ Customer {cid} entered queue validation")
    
    # End queue tracking for customers who start interactions or leave
    for cid in list(queue_tracking):
        if cid in active_customer_ids:
            interaction_logger.end_customer_queue(cid)
            queue_tracking.remove(cid)
            logging.debug(f"✅ Customer {cid} left queue (started interaction)")
        elif cid not in current_customer_ids:
            # Customer left the scene while in queue
            interaction_logger.end_customer_queue(cid)
            queue_tracking.remove(cid)
            logging.debug(f"✅ Customer {cid} left queue (left scene)")
    
    # 6.2. AUTOMATIC UNATTENDED TRACKING - Only track CONFIRMED unattended customers
    # Start tracking only for newly confirmed unattended customers
    for cid in confirmed_unattended_ids:
        if cid not in unattended_tracking:
            interaction_logger.start_unattended_tracking(cid)
            unattended_tracking.add(cid)
            logging.warning(f"🚨 CONFIRMED UNATTENDED: Customer {cid} started unattended tracking")
    
    # End unattended tracking for customers who get attended again or leave scene
    for cid in list(unattended_tracking):
        if cid in active_customer_ids:
            # Customer got attended again - end tracking and log the duration
            interaction_logger.end_unattended_tracking(cid)
            unattended_tracking.remove(cid)
            logging.info(f"✅ Customer {cid} ended unattended tracking (got attended)")
        elif cid not in current_customer_ids:
            # Customer left the scene while unattended - end tracking and log the duration
            interaction_logger.end_unattended_tracking(cid)
            unattended_tracking.remove(cid)
            logging.info(f"✅ Customer {cid} ended unattended tracking (left scene)")
    
    # Hourly aggregation is now handled automatically by InteractionLogger
    # No need to manually track completed interactions
    
    # Enhanced logging for timer-based system
    if unattended_ids:
        confirmed_count = len(confirmed_unattended_ids)
        pending_count = len(unattended_ids) - confirmed_count
        logging.info(f"📹 Unattended customers: {confirmed_count} confirmed, {pending_count} pending confirmation")
    
    # Unattended customer visual indicators removed - calculations continue in background
    # Unattended detection, timing, and confirmation logic still processes all data
    
    # Interaction visual indicators removed - calculations continue in background
    # Interaction detection, scoring, and duration tracking still processes all data

    # Visual timer overlays and status displays removed - calculations continue in background
    # All timing, interaction, and unattended customer calculations still process data

    # Removed: Debug snapshot logging - not needed for production

    # 8. Save video segment with interaction info
    # Only mark segment as having interactions when any pair exceeds MIN_INTERACTION_DURATION
    any_met_duration = False
    if active_interactions:
        for (sid, cid) in active_interactions:
            try:
                # Pass frame_time so duration uses the same timebase as check_and_log
                duration = interaction_logger.get_interaction_duration(sid, cid, current_time=frame_time)
                if duration >= MIN_INTERACTION_DURATION:
                    any_met_duration = True
                    logging.info(f"🎯 QUALIFIED INTERACTION: Staff {sid} & Customer {cid} | Duration: {duration:.1f}s (≥{MIN_INTERACTION_DURATION}s)")
                    break
            except Exception as e:
                logging.error(f"❌ Error checking interaction duration for Staff {sid} & Customer {cid}: {e}")
                continue
    
    # Get timing details for incident reporting
    timing_details = interaction_logger.get_detailed_timing_info(frame_time)
    video_segmenter.set_unattended_timing_details(timing_details)
    
    # Get bounding box details for incident reporting
    if unattended_ids:
        bounding_box_details = interaction_logger.get_unattended_customer_bounding_boxes(unattended_ids, confirmed_unattended_ids)
        video_segmenter.set_unattended_bounding_box_details(bounding_box_details)
    
    # 7. Process unattended customers and add visual overlays
    display_frame = frame.copy()
    if unattended_ids:
        logging.info(f"🎨 DRAWING OVERLAYS: {len(unattended_ids)} unattended customers detected")
        
        # Get bounding box data for unattended customers
        bounding_boxes = interaction_logger.get_unattended_customer_bounding_boxes(unattended_ids, confirmed_unattended_ids)
        logging.info(f"🎨 Bounding boxes: {bounding_boxes}")
        
        # Get detailed timing information for unattended customers
        customer_timing_details = interaction_logger.get_customer_timing_details(unattended_ids + confirmed_unattended_ids, frame_time)
        logging.info(f"🎨 Customer details: {customer_timing_details}")
        
        # Draw visual overlays for unattended customers (no text)
        display_frame = overlay_renderer.draw_unattended_overlays(
            display_frame, customer_timing_details, bounding_boxes
        )
        logging.info("🎨 Overlays drawn successfully")
        
        # Log detailed information for unattended customers (console only)
        for cid in unattended_ids:
            if cid in customer_timing_details:
                details = customer_timing_details[cid]
                status = "CONFIRMED" if details['is_confirmed'] else "PENDING"
                logging.warning(f"🚨 UNATTENDED CUSTOMER {cid}: {status} - "
                              f"Unattended: {details['unattended_duration']:.1f}s, "
                              f"Timer: {details['timer_duration']:.1f}s")
                
                # Log bounding box coordinates
                if cid in bounding_boxes.get('unattended', {}):
                    bbox = bounding_boxes['unattended'][cid]
                    logging.info(f"   📦 Bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                elif cid in bounding_boxes.get('confirmed_unattended', {}):
                    bbox = bounding_boxes['confirmed_unattended'][cid]
                    logging.info(f"   📦 Bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    else:
        logging.debug("🎨 No unattended customers - no overlays to draw")
    
    # Pass frame with overlays to video segmenter
    video_segmenter.add_frame(display_frame, active_interactions, any_met_duration, 
                            unattended_present=len(unattended_ids) > 0, 
                            confirmed_unattended_present=len(confirmed_unattended_ids) > 0)
    
    # Immediate interaction uploads removed: interactions will upload after the segment is saved/closed

    # 9. Optionally show live video if not headless
    if not HEADLESS:
        cv2.imshow('ACSI - Staff-Customer Interaction Tracking', display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


# Graceful shutdown
logging.info("🛑 Shutting down CSI system...")

# Stop analytics scheduler
try:
    analytics_scheduler.stop()
    logging.info("✅ Analytics scheduler stopped")
except Exception as e:
    logging.error(f"❌ Error stopping analytics scheduler: {e}")

# Note: Hourly analytics removed - now using per-interaction footfall API calls
logging.info("✅ Per-interaction footfall API calls enabled")

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