# Configuration loader for CSI (Customer-Staff Interaction) project
import os
import json

def load_config():
    """Load configuration from config.json file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"config.json not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in config.json")

# Load the configuration
config = load_config()

# System settings
CURRENT_ORG = config['system_settings']['current_organization']
CURRENT_BRANCH = config['system_settings']['current_branch']
CURRENT_CAMERA = config['system_settings']['current_camera']

# Get current organization and camera configuration
current_org_config = config['organizations'][CURRENT_ORG]
current_camera_config = current_org_config['ip_cameras'][CURRENT_CAMERA]

# Build RTSP URL from camera configuration
def build_rtsp_url(camera_config):
    """Build RTSP URL from camera configuration."""
    params = camera_config['camera_config']['parameters']
    param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    return f"{camera_config['camera_config']['protocol']}://{camera_config['camera_config']['username']}:{camera_config['camera_config']['password']}@{camera_config['camera_config']['host']}:{camera_config['camera_config']['port']}{camera_config['camera_config']['path']}?{param_string}"

# Camera configuration
VIDEO_SOURCE = build_rtsp_url(current_camera_config)
LINE_COORDS = tuple(tuple(coord) for coord in current_camera_config['line_coords'])
HEADLESS = current_camera_config['headless']

# API configuration
API_URLS = {
    'upload_image': config['organizations']['api_url']['upload_image'],
    'report_incident': config['organizations']['api_url']['report_incident'],
    'report_footfall': config['organizations']['api_url']['report_footfall'],
    'test_api': config['organizations']['api_url']['test_api']
}
UPLOAD_API_KEY = config['organizations']['upload_api_key']

# Logging configuration
LOG_FILE = current_org_config['logging_settings']['log_file']
LOG_LEVEL = current_org_config['logging_settings']['log_level']
DEBUG_SUMMARY_EVERY_N_FRAMES = current_org_config['logging_settings']['debug_summary_every_n_frames']

# Video processing configuration
VIDEO_SEGMENT_PATH = current_org_config['directories']['segments']
UNATTENDED_SEGMENT_PATH = current_org_config['directories']['unattended_segments']
SEGMENT_DURATION = current_org_config['processing_settings']['segment_duration']
UPLOAD_ONLY_INTERACTION_VIDEOS = current_org_config['processing_settings']['upload_only_interaction_videos']
KEEP_ALL_VIDEOS_LOCALLY = current_org_config['processing_settings']['keep_all_videos_locally']

# Detection configuration
YOLO_MODEL_PATH = current_org_config['processing_settings']['yolo_model_path']
USE_GPU = current_org_config['processing_settings'].get('use_gpu', True)
GPU_DEVICE = current_org_config['processing_settings'].get('gpu_device', 'cuda:0')
HALF_PRECISION = current_org_config['processing_settings'].get('half_precision', True)

# Tracking settings
TRACK_THRESH = current_org_config['tracking_settings']['track_thresh']
TRACK_BUFFER = current_org_config['tracking_settings']['track_buffer']
MATCH_THRESH = current_org_config['tracking_settings']['match_thresh']
TRACKING_FRAME_RATE = current_org_config['tracking_settings']['frame_rate']
ENABLE_TRACKING_STATS = current_org_config['tracking_settings']['enable_tracking_stats']

# Interaction settings
MIN_INTERACTION_DURATION = current_org_config['interaction_settings']['min_interaction_duration']
INTERACTION_THRESHOLD = current_org_config['interaction_settings']['interaction_threshold']
UNATTENDED_THRESHOLD = current_org_config['interaction_settings']['unattended_threshold']
MIN_TRACKING_DURATION_FOR_ALERT = current_org_config['interaction_settings']['min_tracking_duration_for_alert']
UNATTENDED_CONFIRMATION_TIMER = current_org_config['interaction_settings']['unattended_confirmation_timer']
TIMER_RESET_GRACE_PERIOD = current_org_config['interaction_settings']['timer_reset_grace_period']

# Queue settings
MIN_QUEUE_DURATION = current_org_config['queue_settings']['min_queue_duration']
QUEUE_VALIDATION_PERIOD = current_org_config['queue_settings']['queue_validation_period']
QUEUE_STABILITY_THRESHOLD = current_org_config['queue_settings']['queue_stability_threshold']

# Organization information
ORGANIZATION_NAME = current_org_config['name']
BRANCH_ID = current_camera_config['branch_id']
LOCATION = current_camera_config['location']