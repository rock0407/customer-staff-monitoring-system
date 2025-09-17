import cv2
import os
import time
import requests
import logging
from config_loader import VIDEO_SEGMENT_PATH, SEGMENT_DURATION, UNATTENDED_SEGMENT_PATH, API_URLS, UPLOAD_API_KEY, ORGANIZATION_NAME, BRANCH_ID, LOCATION
from api_handler import report_video_segment_incident
import numpy as np

class VideoSegmenter:
    def __init__(self, fps, frame_size):
        os.makedirs(VIDEO_SEGMENT_PATH, exist_ok=True)
        os.makedirs(UNATTENDED_SEGMENT_PATH, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.segment_start_time = None
        self.segment_idx = 0
        self.writer = None
        self.frames_written = 0
        self.has_interactions = False  # Track if current segment has interactions
        self.interaction_count = 0     # Count interactions in current segment
        self.has_unattended = False    # Track unattended presence in current segment
        self._unattended_since = None  # When unattended first seen within this segment
        self.segment_file = None
        # Don't start a segment immediately - wait for first frame

    def _start_new_segment(self):
        if self.writer:
            self.writer.release()
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(VIDEO_SEGMENT_PATH, f'segment_{self.segment_idx}_{timestamp}.mp4')
        
        # Try codecs in order of preference (avoid H264 encoder if not available)
        codecs = ['mp4v', 'avc1']
        self.writer = None
        for codec in codecs:
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec), self.fps, self.frame_size)
            try:
                if writer.isOpened():
                    # Test if the writer can be opened successfully
                    self.writer = writer
                    logging.info(f"🎬 Using codec '{codec}' for segment: {os.path.basename(filename)}")
                    break
                else:
                    writer.release()
            except Exception:
                writer.release()
                continue
        if self.writer is None:
            # Final fallback: try 'mp4v' again and log a warning
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)
            if writer.isOpened():
                self.writer = writer
                logging.warning(f"⚠️ All preferred codecs failed, using fallback 'mp4v' for segment: {os.path.basename(filename)}")
            else:
                raise RuntimeError("Could not open video writer with any compatible codec.")
        self.segment_file = filename
        self.segment_start_time = time.time()
        self.frames_written = 0
        self.has_interactions = False
        self.interaction_count = 0
        self.has_unattended = False
        self._unattended_since = None
        self.segment_idx += 1
        
        logging.info(f"🎬 Started new segment: {os.path.basename(filename)}")

    def add_frame(self, frame, active_interactions=None, any_met_duration=False, unattended_present=False):
        """Add frame to current segment and track interactions.
        any_met_duration: mark segment as having interactions only if any pair met the min duration.
        """
        # Start a new segment if this is the first frame
        if self.writer is None:
            self._start_new_segment()
        
        # Write the frame
        self.writer.write(frame)
        self.frames_written += 1
        
        # Track if this frame has qualifying interactions (met min duration)
        if any_met_duration:
            if not self.has_interactions:
                self.has_interactions = True
                logging.info(f"🟢 Qualified interactions detected in segment {self.segment_idx}")
            if active_interactions:
                self.interaction_count = max(self.interaction_count, len(active_interactions))
        # Track unattended presence
        if unattended_present:
            if not self.has_unattended:
                self.has_unattended = True
                self._unattended_since = time.time()
            # Optionally finalize early to ensure unattended evidence saved promptly
            if self._unattended_since and (time.time() - self._unattended_since) >= 5:
                logging.info("⏲️ Unattended detected – finalizing current segment early to save evidence")
                self.finalize_segment()
                self._start_new_segment()
        
        # Check if segment duration is reached
        if time.time() - self.segment_start_time >= SEGMENT_DURATION:
            self.finalize_segment()
            self._start_new_segment()

    def finalize_segment(self):
        """Finalize current segment and decide whether to send to API."""
        if self.writer:
            self.writer.release()
            self.writer = None
            
            if self.segment_file and self.segment_start_time:
                segment_name = os.path.basename(self.segment_file)
                duration = time.time() - self.segment_start_time
                
                # Only log if we actually wrote frames
                if self.frames_written > 0:
                    logging.info(f"💾 Segment saved locally: {segment_name} ({duration:.1f}s, {self.frames_written} frames)")
                else:
                    logging.warning(f"⚠️ Empty segment created: {segment_name} (0 frames written)")
                    # Remove empty segment file
                    try:
                        os.remove(self.segment_file)
                        logging.info(f"🗑️ Removed empty segment: {segment_name}")
                    except Exception as e:
                        logging.error(f"❌ Failed to remove empty segment {segment_name}: {e}")
            
            # Only process segments that have actual content
            if self.frames_written > 0:
                # Handle unattended customer segments - UPLOAD VIDEO TO INCIDENT ALERT
                if self.has_unattended:
                    logging.info(f"🚨 Segment contains unattended customers - UPLOADING VIDEO TO INCIDENT ALERT")
                    try:
                        # Copy to unattended folder
                        base = os.path.basename(self.segment_file)
                        dest = os.path.join(UNATTENDED_SEGMENT_PATH, base)
                        if os.path.abspath(os.path.dirname(self.segment_file)) != os.path.abspath(UNATTENDED_SEGMENT_PATH):
                            # Copy using OpenCV re-write to ensure codec compatibility
                            cap = cv2.VideoCapture(self.segment_file)
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
                                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                ok, frm = cap.read()
                                while ok:
                                    writer.write(frm)
                                    ok, frm = cap.read()
                                writer.release()
                                cap.release()
                                logging.info(f"📁 Copied segment to unattended folder: {dest}")
                        
                        # Report unattended customer incident
                        success = report_video_segment_incident(
                            video_path=self.segment_file,
                            incident_type="unattended_customer",
                            branch_id=BRANCH_ID,
                            location=LOCATION
                        )
                        
                        if success:
                            logging.info("✅ Successfully reported unattended customer incident for segment: %s", segment_name)
                        else:
                            logging.error(f"❌ Failed to report unattended customer incident for segment: {segment_name}")
                            
                    except Exception as e:
                        logging.error(f"❌ Error handling unattended customer segment: {e}")
                
                # Do NOT upload interaction-only segments; keep locally
                else:
                    logging.info("⏭️  Segment has no interactions or unattended customers - keeping locally only")

    def send_to_api(self, filepath):
        """Send video segment to upload endpoint (file + X-API-Key only)."""
        try:
            api_url = API_URLS['upload_image']
            headers = {'X-API-Key': UPLOAD_API_KEY}
            with open(filepath, 'rb') as f:
                files = {'file': (os.path.basename(filepath), f, 'video/mp4')}
                response = requests.post(api_url, files=files, headers=headers, timeout=20)
                if response.status_code in [200, 201]:
                    logging.info(f'✅ Successfully sent {os.path.basename(filepath)} to API')
                else:
                    logging.warning(f'⚠️ Upload responded with status {response.status_code} for {os.path.basename(filepath)}')
        except Exception as e:
            logging.error(f'❌ Failed to send {os.path.basename(filepath)} to API: {e}')

    def get_current_segment_info(self):
        """Get information about current segment."""
        duration = 0.0
        if self.segment_start_time is not None:
            duration = time.time() - self.segment_start_time
        
        return {
            'segment_idx': self.segment_idx,
            'duration': duration,
            'frames': self.frames_written,
            'has_interactions': self.has_interactions,
            'interaction_count': self.interaction_count
        } 