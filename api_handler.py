import os
import logging
import requests
import json
from datetime import datetime
from config_loader import API_URLS, UPLOAD_API_KEY, BRANCH_ID, LOCATION

# Get specialized logger for API operations
logger = logging.getLogger('aif.api')

# --- Main Functions ---

def _upload_image_file(image_path):
    """Internal function to upload an image file."""
    api_url = API_URLS['upload_image']
    headers = {'X-API-Key': UPLOAD_API_KEY}
    
    try:
        response = _perform_image_upload(api_url, headers, image_path)
        response.raise_for_status()
        
        if response.status_code in [200, 201]:
            uploaded_filename = _extract_uploaded_filename(response, image_path)
            logging.info(f"Successfully uploaded image. Using stored reference: {uploaded_filename}")
            return uploaded_filename
        else:
            logging.error(f"Image upload failed with status {response.status_code}. Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Exception during image upload: {e}")
        return None

def _perform_image_upload(api_url, headers, image_path):
    """Perform the actual image upload request."""
    with open(image_path, "rb") as file:
        files = {"file": (os.path.basename(image_path), file, "image/jpeg")}
        return requests.post(api_url, files=files, headers=headers, timeout=20)

def _extract_uploaded_filename(response, image_path):
    """Extract the uploaded filename from the server response."""
    uploaded_filename = None
    
    try:
        data = response.json()
        uploaded_filename = _extract_from_json_response(data)
    except ValueError:
        uploaded_filename = _extract_from_text_response(response)
    
    # Fallback to local basename if server does not return anything usable
    if not uploaded_filename:
        uploaded_filename = os.path.basename(image_path)
    
    return uploaded_filename

def _extract_from_json_response(data):
    """Extract filename from JSON response."""
    for key in ['filename','file','name','stored_name','saved_as','path','url','data','result']:
        if key in data and data[key]:
            result = _extract_value_from_key(data[key])
            if result:
                return result
    return None

def _extract_value_from_key(value):
    """Extract filename from a value that could be dict or string."""
    if isinstance(value, dict):
        for inner in ['filename','file','name','path','url']:
            if inner in value and value[inner]:
                return str(value[inner]).strip()
    else:
        return str(value).strip()
    return None

def _extract_from_text_response(response):
    """Extract filename from text response."""
    text = (response.text or '').strip()
    if text and len(text) < 512:
        return text
    return None

def _report_incident_payload(image_name, incident_type, branch_id, location, confidence_percentage=None, timing_details=None):
    """Internal function to report the incident with the uploaded image's name."""
    api_url = API_URLS['report_incident']
    incident_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Use the simplified payload structure from the user
    payload = {
        "branch_id": branch_id,
        "incident_time": incident_time,
        "reported_location": location,
        "incident_type": incident_type,
        "status": "Open",
        "image": image_name,
        "percentage": int(confidence_percentage * 100) if confidence_percentage is not None else 85, 
        "vehicle_template": "",
        "unattended_customer": 1
    }
    
    # Timing details are collected for internal logging but not sent in payload
    if timing_details:
        logger.info(f"Incident timing details collected: {timing_details}")
    print(payload)
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        logging.info(f"Successfully reported incident for image: {image_name}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(
            "Failed to report incident. "
            f"Exception: {str(e)} | "
            f"Status: {e.response.status_code if isinstance(e, requests.exceptions.HTTPError) and e.response else 'N/A'}, "
            f"Response: {e.response.text if isinstance(e, requests.exceptions.HTTPError) and e.response else 'N/A'}"
        )
        return False

# Face/image-based reporting removed. Incidents are reported only via video segments.

def report_video_segment_incident(video_path, incident_type="unattended_customer", branch_id=None, location=None, timing_details=None):
    """
    Report an incident with a video segment instead of an image.
    
    Args:
        video_path: Path to the video file
        incident_type: Type of incident (default: "unattended_customer")
        branch_id: Branch ID (uses config default if None)
        location: Location (uses config default if None)
        timing_details: Detailed timing information for unattended customers
    """
    # Use config defaults if not provided
    branch_id = branch_id or BRANCH_ID
    location = location or LOCATION
        
    try:
        uploaded_filename = _upload_video_file(video_path)
        if not uploaded_filename:
            return False
            
        # Report the incident with timing details
        report_success = _report_incident_payload(
            uploaded_filename, 
            incident_type, 
            branch_id, 
            location, 
            confidence_percentage=0.85,  # Default confidence for video segments
            timing_details=timing_details
        )
        
        return report_success
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Exception during video upload: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during video segment incident reporting: {e}")
        return False

def _upload_video_file(video_path):
    """Upload video file and return the uploaded filename."""
    api_url = API_URLS['upload_image']  # Assuming the same endpoint handles videos
    headers = {'X-API-Key': UPLOAD_API_KEY}
    
    with open(video_path, "rb") as file:
        files = {"file": (os.path.basename(video_path), file, "video/mp4")}
        response = requests.post(api_url, files=files, headers=headers, timeout=20)
    
    response.raise_for_status()
    
    if response.status_code in [200, 201]:
        uploaded_filename = _extract_uploaded_filename(response, video_path)
        logging.info(f"Successfully uploaded video segment. Stored reference: {uploaded_filename}")
        return uploaded_filename
    else:
        logging.error(f"Video upload failed with status {response.status_code}. Response: {response.text}")
        return None 