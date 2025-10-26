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
        self.has_confirmed_unattended = False  # Track confirmed unattended presence in current segment
        self._unattended_since = None  # When unattended first seen within this segment
        # Count consecutive frames without unattended after it was seen
        self._unattended_absent_frames = 0
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
        self.has_confirmed_unattended = False
        self._unattended_since = None
        self._unattended_absent_frames = 0
        self.segment_idx += 1
        
        logging.info(f"🎬 Started new segment: {os.path.basename(filename)}")

    def add_frame(self, frame, active_interactions=None, any_met_duration=False, unattended_present=False, confirmed_unattended_present=False):
        """Add frame to current segment and track interactions.
        any_met_duration: mark segment as having interactions only if any pair met the min duration.
        confirmed_unattended_present: track if confirmed unattended customers are present.
        """
        try:
            # Start a new segment if this is the first frame
            if self.writer is None:
                self._start_new_segment()
            
            # Validate frame before writing
            if frame is None or frame.size == 0:
                logging.warning("⚠️ Invalid frame received, skipping")
                return
            
            # Write the frame with error handling
            if self.writer and self.writer.isOpened():
                self.writer.write(frame)
                self.frames_written += 1
            else:
                logging.error("❌ Video writer not available, attempting to restart")
                self._start_new_segment()
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                    self.frames_written += 1
            
            # Track interactions and unattended status
            self._track_interactions(active_interactions, any_met_duration)
            self._track_unattended_status(unattended_present, confirmed_unattended_present)
            
            # Check if segment duration is reached
            if time.time() - self.segment_start_time >= SEGMENT_DURATION:
                self.finalize_segment()
                self._start_new_segment()
                
        except Exception as e:
            logging.error(f"❌ Error in add_frame: {e}")
            # Attempt to recover by restarting segment
            try:
                self.finalize_segment()
                self._start_new_segment()
            except Exception as recovery_error:
                logging.error(f"❌ Recovery failed: {recovery_error}")

    def _track_interactions(self, active_interactions, any_met_duration):
        """Track interaction status for the segment."""
        if any_met_duration:
            if not self.has_interactions:
                self.has_interactions = True
                logging.info(f"🟢 Qualified interactions detected in segment {self.segment_idx}")
            if active_interactions:
                self.interaction_count = max(self.interaction_count, len(active_interactions))

    def _track_unattended_status(self, unattended_present, confirmed_unattended_present):
        """Track unattended customer status for the segment."""
        if unattended_present:
            # Reset absence counter whenever unattended appears
            self._unattended_absent_frames = 0
            self._handle_unattended_detection()
        else:
            # If we previously had unattended in this segment, count absence frames
            if self.has_unattended:
                self._unattended_absent_frames += 1
                # If unattended has been absent for 10 consecutive frames, cut the segment here
                if self._unattended_absent_frames >= 10:
                    logging.info("✂️  Unattended absent for 10 frames – finalizing segment to bound event window")
                    self.finalize_segment()
                    self._start_new_segment()
                    # After cutting, nothing to do further this frame for unattended
                    return

        if confirmed_unattended_present:
            self._handle_confirmed_unattended()

    def _handle_unattended_detection(self):
        """Handle unattended customer detection."""
        if not self.has_unattended:
            self.has_unattended = True
            self._unattended_since = time.time()
            logging.info("⏲️ Unattended customer detected - will be included in full 120s segment")

    def _handle_confirmed_unattended(self):
        """Handle confirmed unattended customer detection."""
        if not self.has_confirmed_unattended:
            self.has_confirmed_unattended = True
            logging.info("🚨 CONFIRMED unattended customers detected in segment")

    def finalize_segment(self):
        """Finalize current segment and decide whether to send to API."""
        if not self.writer:
            return
            
        self.writer.release()
        self.writer = None
        
        if self.segment_file and self.segment_start_time:
            self._process_segment_finalization()

    def _process_segment_finalization(self):
        """Process segment finalization logic."""
        segment_name = os.path.basename(self.segment_file)
        duration = time.time() - self.segment_start_time
        
        if self.frames_written > 0:
            self._log_segment_saved(segment_name, duration)
            self._handle_segment_upload(segment_name)
        else:
            self._handle_empty_segment(segment_name)

    def _log_segment_saved(self, segment_name, duration):
        """Log segment save information."""
        logging.info(f"💾 Segment saved locally: {segment_name} ({duration:.1f}s, {self.frames_written} frames)")

    def _handle_empty_segment(self, segment_name):
        """Handle empty segment removal."""
        logging.warning(f"⚠️ Empty segment created: {segment_name} (0 frames written)")
        try:
            os.remove(self.segment_file)
            logging.info(f"🗑️ Removed empty segment: {segment_name}")
        except Exception as e:
            logging.error(f"❌ Failed to remove empty segment {segment_name}: {e}")

    def _handle_segment_upload(self, segment_name):
        """Handle segment upload logic."""
        if self.has_confirmed_unattended:
            self._upload_confirmed_unattended_segment(segment_name)
        elif self.has_interactions:
            self._upload_interaction_segment(segment_name)
        else:
            logging.info("⏭️  Segment has no confirmed unattended customers or interactions - keeping locally only")

    def _upload_confirmed_unattended_segment(self, segment_name):
        """Upload segment with confirmed unattended customers."""
        logging.info("🚨 Segment contains CONFIRMED unattended customers - UPLOADING VIDEO TO INCIDENT ALERT")
        try:
            self._copy_to_unattended_folder()
            self._report_incident(segment_name)
        except Exception as e:
            logging.error(f"❌ Error handling unattended customer segment: {e}")

    def _upload_interaction_segment(self, segment_name):
        """Upload segment with customer-staff interactions."""
        logging.info("🤝 Segment contains CUSTOMER-STAFF INTERACTIONS - UPLOADING VIDEO TO INCIDENT ALERT")
        try:
            self._copy_to_unattended_folder()
            self._report_incident(segment_name, incident_type="customer_staff_interaction")
        except Exception as e:
            logging.error(f"❌ Error handling interaction segment: {e}")


    def _copy_to_unattended_folder(self):
        """Copy segment to unattended folder."""
        base = os.path.basename(self.segment_file)
        dest = os.path.join(UNATTENDED_SEGMENT_PATH, base)
        
        if os.path.abspath(os.path.dirname(self.segment_file)) != os.path.abspath(UNATTENDED_SEGMENT_PATH):
            self._copy_video_file(self.segment_file, dest)

    def _copy_video_file(self, source, destination):
        """Copy video file using OpenCV for codec compatibility."""
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(destination, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            ok, frm = cap.read()
            while ok:
                writer.write(frm)
                ok, frm = cap.read()
            
            writer.release()
            cap.release()
            logging.info(f"📁 Copied segment to unattended folder: {destination}")

    def _report_incident(self, segment_name, incident_type="unattended_customer"):
        """Report incident (unattended customer or interaction)."""
        # Get timing information for internal logging only
        timing_details = self._get_unattended_timing_details()
        
        # Get bounding box details for unattended customers
        bounding_box_details = self._get_unattended_bounding_box_details()
        
        success = report_video_segment_incident(
            video_path=self.segment_file,
            incident_type=incident_type,
            branch_id=BRANCH_ID,
            location=LOCATION,
            timing_details=timing_details,
            bounding_box_details=bounding_box_details
        )
        
        if success:
            logging.info(f"✅ Successfully reported unattended customer incident for segment: {segment_name}")
            if timing_details:
                logging.info(f"   - Internal timing details logged: {timing_details}")
        else:
            logging.error(f"❌ Failed to report unattended customer incident for segment: {segment_name}")

    def _get_unattended_timing_details(self):
        """Get detailed timing information for unattended customers."""
        # This will be populated by the main loop when calling add_frame
        return getattr(self, '_unattended_timing_details', {})

    def set_unattended_timing_details(self, timing_details):
        """Set timing details for unattended customers in this segment."""
        self._unattended_timing_details = timing_details

    def set_unattended_bounding_box_details(self, bounding_box_details):
        """Set bounding box details for unattended customers in this segment."""
        self._unattended_bounding_box_details = bounding_box_details

    def _get_unattended_bounding_box_details(self):
        """Get bounding box details for unattended customers."""
        return getattr(self, '_unattended_bounding_box_details', {})

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
            'interaction_count': self.interaction_count,
            'has_confirmed_unattended': self.has_confirmed_unattended
        } 