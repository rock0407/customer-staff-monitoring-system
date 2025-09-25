import os
import logging
import requests
from datetime import datetime
from config_loader import API_URLS, UPLOAD_API_KEY, BRANCH_ID, LOCATION, ORGANIZATION_NAME
import mimetypes

logging.basicConfig(
    level=logging.INFO,  # This enables INFO and above (INFO, WARNING, ERROR, etc.)
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Constants
DEFAULT_FOOTFALL_API_URL = "https://api.prolificapp.in/api/Addfootfall"
DEFAULT_TEST_API_URL = "https://testapi.prolificapp.in/api/Addfootfall"
FOOTFALL_TYPE = "customer_staff_interaction"  # Consistent footfall type for ALL data - never changes

# Interaction Analytics API Functions
def report_interaction_analytics(staff_id, customer_id, duration_seconds):
    """
    Report interaction duration analytics using the footfall API structure.
    Only sends essential data for customer_staff_interaction type.
    
    Args:
        staff_id: ID of the staff member
        customer_id: ID of the customer
        duration_seconds: Duration of interaction in seconds
    """
    api_url = API_URLS.get('report_footfall', DEFAULT_FOOTFALL_API_URL)
    incident_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Minimal payload - only core footfall API fields
    payload = {
        "branch_id": BRANCH_ID,
        "timestamp": incident_time,
        "camera_name": "cam1",
        "footfall_count_period": int(duration_seconds),
        "footfall_type": FOOTFALL_TYPE,  # Always "customer_staff_interaction"
        "current_value": int(duration_seconds),
        "average_value": int(duration_seconds),
        "created_by": incident_time
    }
    
    try:
        headers = {'X-API-Key': UPLOAD_API_KEY}
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        
        if response.status_code in [200, 201]:
            logging.info(f"✅ Interaction analytics sent: Staff {staff_id} & Customer {customer_id} | Duration: {duration_seconds:.1f}s")
            return True
        else:
            logging.error(f"❌ Failed to send interaction analytics. Status: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Exception in report_interaction_analytics: {str(e)}")
        return False


# for uploading video
def upload_video(video_path):
    api_url = API_URLS['upload_image']
    headers = {'X-API-Key': UPLOAD_API_KEY}
    mime_type, _ = mimetypes.guess_type(video_path)
    mime_type = mime_type or "application/octet-stream"
    try:
        with open(video_path, "rb") as file:
            files = {"file": (os.path.basename(video_path), file, mime_type)}
            response = requests.post(api_url, files=files, headers = headers)
        if response.status_code in [200, 201]:
            logging.info(f"Successfully uploaded {video_path}")
            return os.path.basename(video_path)
        else:
            logging.error(f"Failed to upload {video_path}. Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception in report_incident: {str(e)}")

# for uploading image
def upload_image(image_path):
    api_url = API_URLS['upload_image']
    headers = {'X-API-Key': UPLOAD_API_KEY}
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    try:
        with open(image_path, "rb") as file:
            files = {"file": (os.path.basename(image_path), file, mime_type)}
            response = requests.post(api_url, files=files, headers = headers)
        if response.status_code in [200, 201]:
            logging.info(f"Successfully uploaded {image_path}")
            return os.path.basename(image_path)
        else:
            logging.error(f"Failed to upload {image_path}. Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception in report_incident: {str(e)}")

# send smart alert
def report_incident(image_name, activity_type, branch_id, reported_location, percentage=90):
    api_url2 = API_URLS['report_incident']
    incident_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "branch_id": branch_id, 
        "incident_time": incident_time,
        "reported_location": reported_location,
        "incident_type": activity_type,  
        "status": "Open",
        "image": image_name,
        "percentage": percentage # as no yolo or other model is used confidence can't be predicted
        #"vehicle_template" : "up14bw1302"
    }
    try:
        headers = {'X-API-Key': UPLOAD_API_KEY}
        response = requests.post(api_url2, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            logging.info("Incident reported successfully!")
            return True
        else:
            logging.error(f"Failed to report incident. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logging.error(f"Exception in report_incident: {str(e)}")
