# Configuration loader for ACSI (Customer-Staff Interaction) project
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

# Camera configuration
VIDEO_SOURCE = config['camera']['video_source']
LINE_COORDS = tuple(tuple(coord) for coord in config['camera']['line_coords'])
HEADLESS = config['camera']['headless']

# API configuration
API_ENDPOINT = config['api']['endpoint']
API_URLS = {
    'upload_image': config['api']['upload_image'],
    'report_incident': config['api']['report_incident']
}
UPLOAD_API_KEY = config['api']['upload_api_key']

# Logging configuration
LOG_FILE = config['logging']['log_file']
LOG_LEVEL = config['logging']['log_level']
DEBUG_SUMMARY_EVERY_N_FRAMES = config['logging']['debug_summary_every_n_frames']

# Video processing configuration
VIDEO_SEGMENT_PATH = config['video_processing']['segment_path']
UNATTENDED_SEGMENT_PATH = config['video_processing']['unattended_segment_path']
SEGMENT_DURATION = config['video_processing']['segment_duration']
UPLOAD_ONLY_INTERACTION_VIDEOS = config['video_processing']['upload_only_interaction_videos']
KEEP_ALL_VIDEOS_LOCALLY = config['video_processing']['keep_all_videos_locally']

# Detection configuration
YOLO_MODEL_PATH = config['detection']['yolo_model_path']

# Interaction settings
MIN_INTERACTION_DURATION = config['interaction_settings']['min_interaction_duration']
INTERACTION_THRESHOLD = config['interaction_settings']['interaction_threshold']
UNATTENDED_THRESHOLD = config['interaction_settings']['unattended_threshold']
MIN_TRACKING_DURATION_FOR_ALERT = config['interaction_settings']['min_tracking_duration_for_alert']

# Organization information
ORGANIZATION_NAME = config['organization']['name']
BRANCH_ID = config['organization']['branch_id']
LOCATION = config['organization']['location']