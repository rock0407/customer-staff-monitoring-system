#!/usr/bin/env python3
"""
Optimized Video Segmenter - Fixes the issue of long video segments
Properly detects interaction end events and creates precise video segments
"""

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
        
        # Interaction event tracking with precise end detection
        self.interaction_start_time = None
        self.interaction_end_time = None
        self.current_interaction_pairs = set()  # Track active interaction pairs
        self.interaction_validation_start = None  # When interaction first detected
        self.interaction_validation_duration = 15.0  # 15 seconds validation period
        
        # Unattended event tracking with precise end detection
        self.unattended_start_time = None
        self.unattended_end_time = None
        self._unattended_since = None
        self._unattended_absent_frames = 0
        self.unattended_validation_start = None  # When unattended first detected
        self.unattended_validation_duration = 30.0  # 30 seconds validation period
        
        # Buffer for pre-interaction frames
        self.frame_buffer = []
        self.buffer_size = 30  # Keep 30 frames (1 second at 30fps) before interaction
        self.buffer_start_time = None
        
        # Minimum segment duration filter
        self.MIN_SEGMENT_DURATION = 3.0  # Minimum 3 seconds for meaningful clips
        
        # Interaction end detection
        self.interaction_end_grace_period = 1.0  # 1 second grace period after interaction ends for faster response
        self.last_interaction_time = None
        self.interaction_ended = False
        
        # Segment file tracking
        self.segment_file = None

    def _start_new_segment(self, reason="time_based"):
        """Start a new video segment with proper initialization."""
        if self.writer:
            self.writer.release()
        
        # Reset all tracking variables
        self.has_interactions = False
        self.interaction_count = 0
        self.has_unattended = False
        self.has_confirmed_unattended = False
        self.interaction_start_time = None
        self.interaction_end_time = None
        self.current_interaction_pairs = set()
        self.interaction_validation_start = None
        self.unattended_start_time = None
        self.unattended_end_time = None
        self._unattended_since = None
        self._unattended_absent_frames = 0
        self.unattended_validation_start = None
        self.last_interaction_time = None
        self.interaction_ended = False
        
        # Create new segment file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.segment_file = os.path.join(VIDEO_SEGMENT_PATH, f"segment_{self.segment_idx}_{timestamp}_{reason}.mp4")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.segment_file, fourcc, self.fps, self.frame_size)
        
        if not self.writer.isOpened():
            logging.error(f"❌ Failed to initialize video writer for segment {self.segment_idx}")
            return
        
        self.segment_start_time = time.time()
        self.frames_written = 0
        self.segment_idx += 1
        
        logging.info(f"🎬 Started new segment: {os.path.basename(self.segment_file)} (reason: {reason})")

    def _write_buffered_frames(self):
        """Write buffered frames to current segment."""
        if not self.writer or not self.writer.isOpened():
            return
        
        for buffered_frame in self.frame_buffer:
            self.writer.write(buffered_frame)
            self.frames_written += 1
        
        self.frame_buffer.clear()
        logging.debug(f"📦 Wrote {len(self.frame_buffer)} buffered frames to segment")

    def add_frame(self, frame, active_interactions=None, any_met_duration=False, unattended_present=False, confirmed_unattended_present=False):
        """Add frame to video segment with optimized event detection."""
        current_time = time.time()
        
        try:
            # Initialize segment if needed
            if not self.writer:
                self._start_new_segment("initial")
            
            # Buffer frames before events
            if not self.has_interactions and not self.has_confirmed_unattended:
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                self.buffer_start_time = current_time
            
            # Check for interaction events with precise end detection
            interaction_event = self._check_interaction_events(active_interactions, any_met_duration, current_time)
            if interaction_event == "start":
                self._handle_interaction_start(frame, current_time)
            elif interaction_event == "end":
                self._handle_interaction_end(current_time)
            
            # Check for unattended events
            unattended_event = self._check_unattended_events(unattended_present, confirmed_unattended_present, current_time)
            if unattended_event == "start":
                self._handle_unattended_start(frame, current_time)
            elif unattended_event == "end":
                self._handle_unattended_end(current_time)
            
            # Write frame to segment if we have an active segment
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
        """Check for interaction start/end events with precise end detection."""
        if any_met_duration and active_interactions:
            # Interaction has met minimum duration - check if we should start segment
            current_pairs = set(active_interactions)
            if not self.current_interaction_pairs:
                # First valid interaction - start segment immediately
                self.current_interaction_pairs = current_pairs
                self.last_interaction_time = current_time
                self.interaction_ended = False
                logging.info("🎯 INTERACTION SEGMENT STARTED: Valid Staff-Customer interaction detected")
                return "start"
            elif current_pairs != self.current_interaction_pairs:
                # Different interaction pairs - start new segment
                self.current_interaction_pairs = current_pairs
                self.last_interaction_time = current_time
                self.interaction_ended = False
                logging.info("🎯 INTERACTION SEGMENT STARTED: New interaction pair detected")
                return "start"
            else:
                # Same interaction continuing - update last interaction time
                self.last_interaction_time = current_time
                self.interaction_ended = False
        elif self.current_interaction_pairs and not active_interactions:
            # All interactions ended - check grace period
            if not self.interaction_ended:
                self.interaction_ended = True
                self.last_interaction_time = current_time
                logging.info("🎯 INTERACTION SEGMENT ENDING: All interactions ended, starting grace period")
                return "end"
            elif self.last_interaction_time and (current_time - self.last_interaction_time) >= self.interaction_end_grace_period:
                # Grace period expired - definitely end segment
                logging.info("🎯 INTERACTION SEGMENT ENDED: Grace period expired")
                return "end"
        
        return None

    def _check_unattended_events(self, unattended_present, confirmed_unattended_present, current_time):
        """Check for unattended customer start/end events with validation period handling."""
        if confirmed_unattended_present and not self.has_confirmed_unattended:
            # Confirmed unattended customer detected - start segment immediately
            return "start"
        elif self.has_confirmed_unattended and not confirmed_unattended_present:
            # Confirmed unattended customer is no longer present - end segment
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
            logging.info(f"🎯 INTERACTION SEGMENT STARTED: Valid Staff-Customer interaction detected (≥{self.interaction_validation_duration}s)")
        
        # Update current interaction pairs
        self.current_interaction_pairs = set()

    def _handle_interaction_end(self, current_time):
        """Handle end of interaction event with immediate finalization."""
        if self.has_interactions:
            self.interaction_end_time = current_time
            interaction_duration = current_time - self.interaction_start_time
            logging.info(f"🎯 INTERACTION SEGMENT ENDED: Duration {interaction_duration:.1f}s")
            
            # Finalize segment immediately after interaction ends
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
            logging.info(f"🚨 UNATTENDED SEGMENT STARTED: Confirmed unattended customer detected (≥{self.unattended_validation_duration}s)")

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
        # Time-based finalization (fallback) - reduced from 60s to 30s for better responsiveness
        if self.segment_start_time and (current_time - self.segment_start_time) >= 30.0:  # 30 seconds max
            logging.info("⏰ Time-based segment finalization (30s reached)")
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
        logging.info(f"💾 GENERAL segment saved: {segment_name} ({duration:.1f}s, {self.frames_written} frames)")

    def _handle_segment_upload(self, segment_name):
        """Handle segment upload decision."""
        # Check if segment has meaningful events
        has_meaningful_events = (
            self.has_interactions or 
            self.has_confirmed_unattended or 
            self.interaction_count > 0
        )
        
        if has_meaningful_events:
            logging.info(f"📤 Segment has confirmed unattended customers or interactions - will upload")
            self._upload_segment(segment_name)
        else:
            logging.info(f"⏭️ Segment has no confirmed unattended customers or interactions - keeping locally only")

    def _handle_empty_segment(self, segment_name):
        """Handle empty segments."""
        logging.warning(f"⚠️ Empty segment: {segment_name}")
        try:
            os.remove(self.segment_file)
            logging.info(f"🗑️ Removed empty segment: {segment_name}")
        except Exception as e:
            logging.error(f"❌ Failed to remove empty segment {segment_name}: {e}")

    def _upload_segment(self, segment_name):
        """Upload segment to API."""
        try:
            success = report_video_segment_incident(
                segment_name, 
                self.segment_file, 
                "interaction_detected"
            )
            if success:
                logging.info(f"✅ Segment uploaded successfully: {segment_name}")
            else:
                logging.error(f"❌ Failed to upload segment: {segment_name}")
        except Exception as e:
            logging.error(f"❌ Error uploading segment {segment_name}: {e}")

    def get_segment_info(self):
        """Get current segment information."""
        if not self.segment_start_time:
            return None
        
        return {
            "segment_file": self.segment_file,
            "start_time": self.segment_start_time,
            "duration": time.time() - self.segment_start_time,
            "frames_written": self.frames_written,
            "has_interactions": self.has_interactions,
            "has_unattended": self.has_unattended,
            "interaction_count": self.interaction_count
        }

    def set_unattended_timing_details(self, timing_details):
        """Set timing details for unattended customers in this segment."""
        self._unattended_timing_details = timing_details

    def set_unattended_bounding_box_details(self, bounding_box_details):
        """Set bounding box details for unattended customers in this segment."""
        self._unattended_bounding_box_details = bounding_box_details

    def _get_unattended_timing_details(self):
        """Get timing details for unattended customers."""
        return getattr(self, '_unattended_timing_details', {})

    def _get_unattended_bounding_box_details(self):
        """Get bounding box details for unattended customers."""
        return getattr(self, '_unattended_bounding_box_details', {})

    def get_current_segment_info(self):
        """Get information about current segment."""
        duration = 0.0
        if self.segment_start_time is not None:
            duration = time.time() - self.segment_start_time
        
        return {
            "segment_file": self.segment_file,
            "start_time": self.segment_start_time,
            "duration": duration,
            "frames_written": self.frames_written,
            "has_interactions": self.has_interactions,
            "has_unattended": self.has_unattended,
            "interaction_count": self.interaction_count,
            "unattended_timing_details": self._get_unattended_timing_details(),
            "unattended_bounding_box_details": self._get_unattended_bounding_box_details()
        }
