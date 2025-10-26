import os
import logging
import requests
import json
import time
from datetime import datetime
from config_loader import API_URLS, UPLOAD_API_KEY, BRANCH_ID, LOCATION
# Removed API integration - metrics are stored locally in JSON only

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
    # First try to get the filename directly
    if 'filename' in data and data['filename']:
        return str(data['filename']).strip()
    
    # Then try other common keys
    for key in ['file','name','stored_name','saved_as','path','url','data','result']:
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

def _report_incident_payload(image_name, incident_type, branch_id, location, confidence_percentage=None, timing_details=None, bounding_box_details=None):
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
        "unattended_customer": 1 if incident_type != "customer_staff_interaction" else 0
    }
    
    # For interaction incidents, do NOT include any extra info beyond base payload
    if incident_type == "customer_staff_interaction":
        pass
    # Include unattended timing details only for unattended incidents
    elif timing_details and isinstance(timing_details, dict):
        try:
            customers = timing_details.get("customers", {})
            confirmed = [info for info in customers.values() if info.get("is_confirmed")]
            unattended_seconds = None
            timer_seconds = None
            if confirmed:
                unattended_seconds = int(max(info.get("unattended_duration", 0) or 0 for info in confirmed))
                timer_seconds = int(max(info.get("timer_duration", 0) or 0 for info in confirmed))
            else:
                # Fallback to any unattended customer if none confirmed yet
                unattended_candidates = [info for info in customers.values() if info.get("is_unattended")]
                if unattended_candidates:
                    unattended_seconds = int(max(info.get("unattended_duration", 0) or 0 for info in unattended_candidates))
                    timer_seconds = int(max(info.get("timer_duration", 0) or 0 for info in unattended_candidates))

            summary = timing_details.get("summary", {})
            payload.update({
                "unattended_duration_seconds": unattended_seconds if unattended_seconds is not None else 0,
                "unattended_timer_seconds": timer_seconds if timer_seconds is not None else 0,
                "unattended_summary": {
                    "total_customers": int(summary.get("total_customers", 0) or 0),
                    "unattended_customers": int(summary.get("unattended_customers", 0) or 0),
                    "confirmed_unattended": int(summary.get("confirmed_unattended", 0) or 0),
                    "pending_confirmation": int(summary.get("pending_confirmation", 0) or 0)
                }
            })
            
            # Add detailed customer information with bounding boxes
            if bounding_box_details and isinstance(bounding_box_details, dict):
                payload["unattended_customer_details"] = []
                
                # Process confirmed unattended customers first
                confirmed_boxes = bounding_box_details.get("confirmed_unattended", {})
                for customer_id, bbox in confirmed_boxes.items():
                    if customer_id in customers:
                        customer_info = customers[customer_id]
                        payload["unattended_customer_details"].append({
                            "customer_id": customer_id,
                            "status": "confirmed_unattended",
                            "bounding_box": bbox,
                            "unattended_duration_seconds": int(customer_info.get("unattended_duration", 0) or 0),
                            "timer_duration_seconds": int(customer_info.get("timer_duration", 0) or 0),
                            "first_detected_time": customer_info.get("first_detected", 0),
                            "last_attended_time": customer_info.get("last_attended", 0)
                        })
                
                # Process pending unattended customers
                pending_boxes = bounding_box_details.get("unattended", {})
                for customer_id, bbox in pending_boxes.items():
                    if customer_id in customers and customer_id not in confirmed_boxes:
                        customer_info = customers[customer_id]
                        payload["unattended_customer_details"].append({
                            "customer_id": customer_id,
                            "status": "pending_confirmation",
                            "bounding_box": bbox,
                            "unattended_duration_seconds": int(customer_info.get("unattended_duration", 0) or 0),
                            "timer_duration_seconds": int(customer_info.get("timer_duration", 0) or 0),
                            "first_detected_time": customer_info.get("first_detected", 0),
                            "last_attended_time": customer_info.get("last_attended", 0)
                        })
        except Exception as parse_error:
            logger.warning(f"Failed to include timing details in payload: {parse_error}")
        finally:
            logger.info(f"Incident timing details collected: {timing_details}")
    
    
    try:
        headers = {'X-API-Key': UPLOAD_API_KEY}
        response = requests.post(api_url, json=payload, headers=headers)
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

def report_video_segment_incident(video_path, incident_type="unattended_customer", branch_id=None, location=None, timing_details=None, bounding_box_details=None):
    """
    Report an incident with a video segment instead of an image.
    
    Args:
        video_path: Path to the video file
        incident_type: Type of incident (default: "unattended_customer")
        branch_id: Branch ID (uses config default if None)
        location: Location (uses config default if None)
        timing_details: Detailed timing information for unattended customers
        bounding_box_details: Bounding box coordinates for unattended customers
    
    Returns:
        dict: Success status and video evidence information
    """
    # Use config defaults if not provided
    branch_id = branch_id or BRANCH_ID
    location = location or LOCATION
    
    logging.info(f"📤 Reporting incident: {incident_type} for video: {video_path}")
        
    try:
        uploaded_filename = _upload_video_file(video_path)
        if not uploaded_filename:
            return {"success": False, "error": "Upload failed"}
            
        # Report the incident with timing details and bounding boxes
        report_success = _report_incident_payload(
            uploaded_filename, 
            incident_type, 
            branch_id, 
            location, 
            confidence_percentage=0.85,  # Default confidence for video segments
            timing_details=timing_details,
            bounding_box_details=bounding_box_details
        )
        
        if report_success:
            # Create video evidence object
            video_evidence = {
                "segment_filename": os.path.basename(video_path),
                "uploaded_filename": uploaded_filename,
                "video_url": f"https://s3-noi.aces3.ai/vizo361/{uploaded_filename}",
                "incident_id": f"incident_{int(time.time())}",
                "upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return {
                "success": True,
                "video_evidence": video_evidence
            }
        else:
            return {"success": False, "error": "Incident reporting failed"}
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Exception during video upload: {e}")
        return {"success": False, "error": f"Upload exception: {e}"}
    except Exception as e:
        logging.error(f"Unexpected error during video segment incident reporting: {e}")
        return {"success": False, "error": f"Unexpected error: {e}"}

def _upload_video_file(video_path):
    """Upload video file and return the uploaded filename with retry logic."""
    api_url = API_URLS['upload_image']  # Assuming the same endpoint handles videos
    headers = {'X-API-Key': UPLOAD_API_KEY}
    
    # Log the original path
    logging.info(f"📁 Original video path: {video_path}")
    logging.info(f"📁 Current working directory: {os.getcwd()}")
    
    # Ensure we have an absolute path
    video_path = os.path.abspath(video_path)
    logging.info(f"📁 Video path resolved to: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        logging.error(f"❌ Video file does not exist: {video_path}")
        # Try to find the file in segments directory
        segments_path = os.path.join(os.getcwd(), "segments", os.path.basename(video_path))
        logging.info(f"🔍 Trying segments path: {segments_path}")
        if os.path.exists(segments_path):
            logging.info(f"✅ Found file in segments directory: {segments_path}")
            video_path = segments_path
        else:
            return None
    
    # Get file size to determine appropriate timeout
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    
    # Calculate timeout based on file size (minimum 60s, +10s per MB)
    timeout_seconds = max(60, int(60 + (file_size_mb * 10)))
    
    logging.info(f"📤 Uploading video segment: {os.path.basename(video_path)} ({file_size_mb:.1f}MB, timeout: {timeout_seconds}s)")
    
    # Retry logic for large files
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(video_path, "rb") as file:
                files = {"file": (os.path.basename(video_path), file, "video/mp4")}
                response = requests.post(api_url, files=files, headers=headers, timeout=timeout_seconds)
            
            response.raise_for_status()
            
            if response.status_code in [200, 201]:
                uploaded_filename = _extract_uploaded_filename(response, video_path)
                logging.info(f"✅ Successfully uploaded video segment. Stored reference: {uploaded_filename}")
                return uploaded_filename
            else:
                logging.error(f"❌ Video upload failed with status {response.status_code}. Response: {response.text}")
                if attempt < max_retries - 1:
                    logging.info(f"🔄 Retrying upload (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(5)  # Wait 5 seconds before retry
                    continue
                return None
                
        except requests.exceptions.Timeout as e:
            logging.error(f"⏰ Video upload timeout (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"🔄 Retrying upload with longer timeout...")
                timeout_seconds += 30  # Increase timeout for retry
                time.sleep(5)
                continue
            return None
        except Exception as e:
            logging.error(f"❌ Video upload error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"🔄 Retrying upload...")
                time.sleep(5)
                continue
            return None
    
    return None

# ANALYTICS FUNCTIONS - LOCAL JSON STORAGE ONLY
# No API integration - metrics are stored locally in JSON file only 