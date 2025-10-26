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
        
        # Event-based tracking
        self.has_interactions = False
        self.interaction_count = 0
        self.has_unattended = False
        self.has_confirmed_unattended = False
        
        # Interaction event tracking
        self.interaction_start_time = None
        self.interaction_end_time = None
        self.current_interaction_pairs = set()  # Track active interaction pairs
        
        # Unattended event tracking
        self.unattended_start_time = None
        self.unattended_end_time = None
        self._unattended_since = None
        self._unattended_absent_frames = 0
        
        # Buffer for pre-interaction frames
        self.frame_buffer = []
        self.buffer_size = 30  # Keep 30 frames (1 second at 30fps) before interaction
        self.buffer_start_time = None
        
        # Minimum segment duration filter
        self.MIN_SEGMENT_DURATION = 3.0  # Minimum 3 seconds for meaningful clips
        
        self.segment_file = None

    def _start_new_segment(self, reason="time_based"):
        """Start a new video segment with proper initialization."""
        if self.writer:
            self.writer.release()
        
        # Create filename with timestamp and reason
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        reason_suffix = reason.replace("_", "-")
        filename = os.path.join(VIDEO_SEGMENT_PATH, f'segment_{self.segment_idx}_{timestamp}_{reason_suffix}.mp4')
        
        # Try codecs in order of preference
        codecs = ['mp4v', 'avc1']
        self.writer = None
        for codec in codecs:
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec), self.fps, self.frame_size)
            try:
                if writer.isOpened():
                    self.writer = writer
                    logging.info(f"🎬 Using codec '{codec}' for segment: {os.path.basename(filename)}")
                    break
                else:
                    writer.release()
            except Exception:
                writer.release()
                continue
                
        if self.writer is None:
            # Final fallback
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)
            if writer.isOpened():
                self.writer = writer
                logging.warning(f"⚠️ Using fallback 'mp4v' for segment: {os.path.basename(filename)}")
            else:
                raise RuntimeError("Could not open video writer with any compatible codec.")
        
        self.segment_file = filename
        self.segment_start_time = time.time()
        self.frames_written = 0
        
        # Reset event tracking
        self.has_interactions = False
        self.interaction_count = 0
        self.has_unattended = False
        self.has_confirmed_unattended = False
        self.interaction_start_time = None
        self.interaction_end_time = None
        self.unattended_start_time = None
        self.unattended_end_time = None
        self._unattended_since = None
        self._unattended_absent_frames = 0
        
        # Clear frame buffer
        self.frame_buffer = []
        self.buffer_start_time = None
        
        self.segment_idx += 1
        logging.info(f"🎬 Started new segment: {os.path.basename(filename)} (reason: {reason})")

    def _write_buffered_frames(self):
        """Write buffered frames to current segment."""
        if not self.writer or not self.writer.isOpened():
            return
            
        for buffered_frame in self.frame_buffer:
            self.writer.write(buffered_frame)
            self.frames_written += 1
        
        logging.info(f"📹 Wrote {len(self.frame_buffer)} buffered frames to segment")

    def add_frame(self, frame, active_interactions=None, any_met_duration=False, unattended_present=False, confirmed_unattended_present=False):
        """Add frame to current segment with event-based segmentation."""
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logging.warning("⚠️ Invalid frame received, skipping")
                return
            
            current_time = time.time()
            
            # Start new segment if this is the first frame
            if self.writer is None:
                self._start_new_segment("initial")
            
            # Check for interaction events
            interaction_event = self._check_interaction_events(active_interactions, any_met_duration, current_time)
            unattended_event = self._check_unattended_events(unattended_present, confirmed_unattended_present, current_time)
            
            # Handle interaction start
            if interaction_event == "start":
                self._handle_interaction_start(frame, current_time)
            # Handle interaction end
            elif interaction_event == "end":
                self._handle_interaction_end(current_time)
            # Handle unattended start
            elif unattended_event == "start":
                self._handle_unattended_start(frame, current_time)
            # Handle unattended end
            elif unattended_event == "end":
                self._handle_unattended_end(current_time)
            
            # Write frame to current segment
            if self.writer and self.writer.isOpened():
                self.writer.write(frame)
                self.frames_written += 1
            else:
                logging.error("❌ Video writer not available, attempting to restart")
                self._start_new_segment("writer_error")
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                    self.frames_written += 1
            
            # Check if we should finalize segment
            self._check_segment_finalization(current_time)
                
        except Exception as e:
            logging.error(f"❌ Error in add_frame: {e}")
            try:
                self.finalize_segment()
                self._start_new_segment("error_recovery")
            except Exception as recovery_error:
                logging.error(f"❌ Recovery failed: {recovery_error}")

    def _check_interaction_events(self, active_interactions, any_met_duration, current_time):
        """Check for interaction start/end events."""
        if any_met_duration and active_interactions:
            # New interaction started
            current_pairs = set(active_interactions)
            if not self.current_interaction_pairs:
                # First interaction in this segment
                return "start"
            elif current_pairs != self.current_interaction_pairs:
                # Different interaction pairs
                return "start"
        elif self.current_interaction_pairs and not active_interactions:
            # All interactions ended
            return "end"
        
        return None

    def _check_unattended_events(self, unattended_present, confirmed_unattended_present, current_time):
        """Check for unattended customer start/end events."""
        if confirmed_unattended_present and not self.has_confirmed_unattended:
            return "start"
        elif self.has_confirmed_unattended and not confirmed_unattended_present:
            return "end"
        
        return None

    def _handle_interaction_start(self, frame, current_time):
        """Handle start of interaction event."""
        if not self.has_interactions:
            # Start new segment for this interaction
            self.finalize_segment()  # Finalize current segment if any
            self._start_new_segment("interaction_start")
            
            # Write buffered frames first
            self._write_buffered_frames()
            
            # Write current frame
            if self.writer and self.writer.isOpened():
                self.writer.write(frame)
                self.frames_written += 1
            
            self.has_interactions = True
            self.interaction_start_time = current_time
            logging.info(f"🎯 INTERACTION SEGMENT STARTED: Staff-Customer interaction detected")
        
        # Update current interaction pairs
        if hasattr(self, 'current_interaction_pairs'):
            self.current_interaction_pairs = set()

    def _handle_interaction_end(self, current_time):
        """Handle end of interaction event."""
        if self.has_interactions:
            self.interaction_end_time = current_time
            interaction_duration = current_time - self.interaction_start_time
            logging.info(f"🎯 INTERACTION SEGMENT ENDED: Duration {interaction_duration:.1f}s")
            
            # Finalize segment after interaction ends
            self.finalize_segment()
            self._start_new_segment("interaction_end")

    def _handle_unattended_start(self, frame, current_time):
        """Handle start of unattended customer event."""
        if not self.has_confirmed_unattended:
            # Start new segment for unattended customer
            self.finalize_segment()  # Finalize current segment if any
            self._start_new_segment("unattended_start")
            
            # Write buffered frames first
            self._write_buffered_frames()
            
            # Write current frame
            if self.writer and self.writer.isOpened():
                self.writer.write(frame)
                self.frames_written += 1
            
            self.has_confirmed_unattended = True
            self.unattended_start_time = current_time
            logging.info(f"🚨 UNATTENDED SEGMENT STARTED: Confirmed unattended customer detected")

    def _handle_unattended_end(self, current_time):
        """Handle end of unattended customer event."""
        if self.has_confirmed_unattended:
            self.unattended_end_time = current_time
            unattended_duration = current_time - self.unattended_start_time
            logging.info(f"🚨 UNATTENDED SEGMENT ENDED: Duration {unattended_duration:.1f}s")
            
            # Finalize segment after unattended event ends
            self.finalize_segment()
            self._start_new_segment("unattended_end")

    def _check_segment_finalization(self, current_time):
        """Check if segment should be finalized based on time or events."""
        # Time-based finalization (fallback)
        if self.segment_start_time and (current_time - self.segment_start_time) >= SEGMENT_DURATION:
            logging.info("⏰ Time-based segment finalization (120s reached)")
            self.finalize_segment()
            self._start_new_segment("time_based")

    def finalize_segment(self):
        """Finalize current segment and decide whether to upload."""
        if not self.writer:
            return
            
        self.writer.release()
        self.writer = None
        
        if self.segment_file and self.segment_start_time:
            self._process_segment_finalization()

    def _process_segment_finalization(self):
        """Process segment finalization with duration filtering."""
        segment_name = os.path.basename(self.segment_file)
        duration = time.time() - self.segment_start_time
        
        # Filter out segments that are too short
        if duration < self.MIN_SEGMENT_DURATION:
            logging.warning(f"⚠️ Segment too short ({duration:.1f}s < {self.MIN_SEGMENT_DURATION}s): {segment_name}")
            self._handle_short_segment(segment_name)
            return
        
        if self.frames_written > 0:
            self._log_segment_saved(segment_name, duration)
            self._handle_segment_upload(segment_name)
        else:
            self._handle_empty_segment(segment_name)

    def _handle_short_segment(self, segment_name):
        """Handle segments that are too short to be meaningful."""
        try:
            os.remove(self.segment_file)
            logging.info(f"🗑️ Removed short segment: {segment_name}")
        except Exception as e:
            logging.error(f"❌ Failed to remove short segment {segment_name}: {e}")

    def _log_segment_saved(self, segment_name, duration):
        """Log segment save information."""
        event_type = "interaction" if self.has_interactions else "unattended" if self.has_confirmed_unattended else "general"
        logging.info(f"💾 {event_type.upper()} segment saved: {segment_name} ({duration:.1f}s, {self.frames_written} frames)")

    def _handle_empty_segment(self, segment_name):
        """Handle empty segment removal."""
        logging.warning(f"⚠️ Empty segment created: {segment_name} (0 frames written)")
        try:
            os.remove(self.segment_file)
            logging.info(f"🗑️ Removed empty segment: {segment_name}")
        except Exception as e:
            logging.error(f"❌ Failed to remove empty segment {segment_name}: {e}")

    def _handle_segment_upload(self, segment_name):
        """Handle segment upload logic with event-based prioritization."""
        if self.has_confirmed_unattended:
            self._upload_confirmed_unattended_segment(segment_name)
        elif self.has_interactions:
            self._upload_interaction_segment(segment_name)
        else:
            logging.info("⏭️ Segment has no confirmed unattended customers or interactions - keeping locally only")

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
        """Report incident with accurate timing information."""
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
            logging.info(f"✅ Successfully reported {incident_type} incident for segment: {segment_name}")
            if timing_details:
                logging.info(f"   - Internal timing details logged: {timing_details}")
        else:
            logging.error(f"❌ Failed to report {incident_type} incident for segment: {segment_name}")

    def _get_unattended_timing_details(self):
        """Get detailed timing information for unattended customers."""
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
            'has_confirmed_unattended': self.has_confirmed_unattended,
            'interaction_start_time': self.interaction_start_time,
            'interaction_end_time': self.interaction_end_time,
            'unattended_start_time': self.unattended_start_time,
            'unattended_end_time': self.unattended_end_time
        }
