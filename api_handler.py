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
        with open(image_path, "rb") as file:
            files = {"file": (os.path.basename(image_path), file, "image/jpeg")}
            response = requests.post(api_url, files=files, headers=headers, timeout=20)
        
        response.raise_for_status() # Will raise an exception for 4xx/5xx errors

        # Try to extract the stored filename or URL from the server response
        if response.status_code in [200, 201]:
            uploaded_filename = None
            try:
                data = response.json()
                # Common keys that might contain the stored name or URL
                for key in [
                    'filename','file','name','stored_name','saved_as','path','url','data','result'
                ]:
                    if key in data and data[key]:
                        # If nested object, try common inner keys
                        if isinstance(data[key], dict):
                            for inner in ['filename','file','name','path','url']:
                                if inner in data[key] and data[key][inner]:
                                    uploaded_filename = str(data[key][inner]).strip()
                                    break
                        else:
                            uploaded_filename = str(data[key]).strip()
                        if uploaded_filename:
                            break
            except ValueError:
                # Not JSON; try raw text
                text = (response.text or '').strip()
                if text and len(text) < 512:
                    uploaded_filename = text

            # Fallback to local basename if server does not return anything usable
            if not uploaded_filename:
                uploaded_filename = os.path.basename(image_path)

            logging.info(f"Successfully uploaded image. Using stored reference: {uploaded_filename}")
            return uploaded_filename
        else:
            # This case might be redundant due to raise_for_status, but it's safe to have.
            logging.error(f"Image upload failed with status {response.status_code}. Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Exception during image upload: {e}")
        return None

def _report_incident_payload(image_name, incident_type, branch_id, location, confidence_percentage=None):
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

def report_video_segment_incident(video_path, incident_type="unattended_customer", branch_id=None, location=None):
    """
    Report an incident with a video segment instead of an image.
    
    Args:
        video_path: Path to the video file
        incident_type: Type of incident (default: "unattended_customer")
        branch_id: Branch ID (uses config default if None)
        location: Location (uses config default if None)
    """
    # Use config defaults if not provided
    if branch_id is None:
        branch_id = BRANCH_ID
    if location is None:
        location = LOCATION
        
    try:
        # Upload the video file
        api_url = API_URLS['upload_image']  # Assuming the same endpoint handles videos
        headers = {'X-API-Key': UPLOAD_API_KEY}
        
        with open(video_path, "rb") as file:
            files = {"file": (os.path.basename(video_path), file, "video/mp4")}
            response = requests.post(api_url, files=files, headers=headers, timeout=20)
        
        response.raise_for_status()
        
        if response.status_code in [200, 201]:
            uploaded_filename = None
            try:
                data = response.json()
                for key in [
                    'filename','file','name','stored_name','saved_as','path','url','data','result'
                ]:
                    if key in data and data[key]:
                        if isinstance(data[key], dict):
                            for inner in ['filename','file','name','path','url']:
                                if inner in data[key] and data[key][inner]:
                                    uploaded_filename = str(data[key][inner]).strip()
                                    break
                        else:
                            uploaded_filename = str(data[key]).strip()
                        if uploaded_filename:
                            break
            except ValueError:
                text = (response.text or '').strip()
                if text and len(text) < 512:
                    uploaded_filename = text

            if not uploaded_filename:
                uploaded_filename = os.path.basename(video_path)

            logging.info(f"Successfully uploaded video segment. Stored reference: {uploaded_filename}")
            
            # Report the incident
            report_success = _report_incident_payload(
                uploaded_filename, 
                incident_type, 
                branch_id, 
                location, 
                confidence_percentage=0.85  # Default confidence for video segments
            )
            
            return report_success
        else:
            logging.error(f"Video upload failed with status {response.status_code}. Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Exception during video upload: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during video segment incident reporting: {e}")
        return False 