# Configuration file for ACSI project

# Single camera support
VIDEO_SOURCE = 'rtsp://admin:proeffico23@103.117.15.75:8131/cam/realmonitor?channel=1&subtype=0' # Set your video file or RTSP URL here
LINE_COORDS =  ((1422, 13), (1034, 1065))# Set your line coordinates here

# API endpoint to send video segments
API_ENDPOINT = 'https://your.api/endpoint'

# Logging
LOG_FILE = 'interaction_log.txt'
LOG_LEVEL = 'INFO'   # DEBUG, INFO, WARNING, ERROR
DEBUG_SUMMARY_EVERY_N_FRAMES = 50  # periodic high-level snapshot frequency

# Path to save temporary video segments
VIDEO_SEGMENT_PATH = 'segments/'

# Segment duration in seconds (10 minutes)
SEGMENT_DURATION = 120

# YOLOv8 model path (can use 'yolov8n.pt' for nano model)
YOLO_MODEL_PATH = 'yolov8n.pt'

# Headless mode (no GUI, no imshow) - Set to False for initial setup
HEADLESS = False

# Tracker selection: 'deepsort' (recommended), 'sort', or 'bytetrack'
TRACKER_TYPE = 'bytetrack'

# Interaction settings
MIN_INTERACTION_DURATION = 20.0  # Minimum interaction time in seconds (reduces false positives)
INTERACTION_THRESHOLD = 500     # Distance threshold in pixels for interaction detection (scaled zones will derive from this)
UNATTENDED_THRESHOLD = 30.0     # Customer idle time (no interaction) to mark as unattended (seconds)

# Unattended segments destination
UNATTENDED_SEGMENT_PATH = 'segments_unattended/'

# Video upload settings
UPLOAD_ONLY_INTERACTION_VIDEOS = True  # Only upload segments with interactions to API
KEEP_ALL_VIDEOS_LOCALLY = True         # Always save all segments locally (recommended)