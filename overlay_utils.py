import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

class OverlayRenderer:
    """Utility class for rendering overlays on video frames."""
    
    def __init__(self):
        # Color definitions
        self.colors = {
            'unattended_pending': (0, 255, 255),    # Yellow for pending confirmation
            'unattended_confirmed': (0, 0, 255),    # Red for confirmed unattended
            'staff': (255, 0, 0),                   # Blue for staff
            'customer': (0, 255, 0),                # Green for regular customers
            'text_bg': (0, 0, 0),                   # Black background for text
            'text_fg': (255, 255, 255),             # White text
            'alert_red': (0, 0, 255),               # Bright red for alerts
            'alert_yellow': (0, 255, 255)           # Bright yellow for warnings
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_padding = 5
        
    def draw_unattended_overlays(self, frame: np.ndarray, 
                                unattended_data: Dict, 
                                bounding_boxes: Dict) -> np.ndarray:
        """
        Draw overlays for unattended customers on the frame.
        
        Args:
            frame: Input video frame
            unattended_data: Dictionary containing timing details for unattended customers
            bounding_boxes: Dictionary containing bounding box coordinates
            
        Returns:
            Frame with overlays drawn
        """
        import logging
        # Draw based on available bounding boxes; timing data may be empty in some frames
        total_boxes = len(bounding_boxes.get('unattended', {})) + len(bounding_boxes.get('confirmed_unattended', {}))
        logging.info(f"🎨 OverlayRenderer: Drawing overlays for {total_boxes} customers (by boxes)")
        logging.info(f"🎨 OverlayRenderer: Bounding boxes: {bounding_boxes}")
        
        overlay_frame = frame.copy()
        
        # Draw overlays for pending unattended customers
        if 'unattended' in bounding_boxes:
            for customer_id, bbox in bounding_boxes['unattended'].items():
                # Draw even if timing data is missing for this customer
                timing = unattended_data.get(customer_id) if isinstance(unattended_data, dict) else None
                self._draw_customer_overlay(
                    overlay_frame, bbox, customer_id,
                    timing, 'pending'
                )
        
        # Draw overlays for confirmed unattended customers
        if 'confirmed_unattended' in bounding_boxes:
            for customer_id, bbox in bounding_boxes['confirmed_unattended'].items():
                # Draw even if timing data is missing for this customer
                timing = unattended_data.get(customer_id) if isinstance(unattended_data, dict) else None
                self._draw_customer_overlay(
                    overlay_frame, bbox, customer_id,
                    timing, 'confirmed'
                )
        
        return overlay_frame
    
    def _draw_customer_overlay(self, frame: np.ndarray, bbox: List[int], 
                              customer_id: int, timing_data: Dict, status: str):
        """Draw overlay for a single unattended customer."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on status
            if status == 'confirmed':
                color = self.colors['alert_red']
                status_text = "CONFIRMED UNATTENDED"
                # Draw prominent red confirmation dot
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 15, color, -1)  # Filled red circle
                cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), 2)  # White border
            else:
                color = self.colors['alert_yellow']
                status_text = "UNATTENDED (PENDING)"
                # Draw yellow warning dot
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 12, color, -1)  # Filled yellow circle
                cv2.circle(frame, (center_x, center_y), 15, (0, 0, 0), 2)  # Black border
            
            # Draw thick bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            
            # Draw corner markers for better visibility
            self._draw_corner_markers(frame, (x1, y1), (x2, y2), color)
            
            # Add pulsing effect for confirmed unattended customers
            if status == 'confirmed':
                self._draw_pulsing_ring(frame, (center_x, center_y), color)
            
        except Exception as e:
            logging.error(f"Error drawing overlay for customer {customer_id}: {e}")
    
    def _draw_text_with_background(self, frame: np.ndarray, text_lines: List[str], 
                                  position: Tuple[int, int]):
        """Draw text with background rectangle."""
        x, y = position
        
        # Calculate text dimensions
        max_width = 0
        total_height = 0
        
        for line in text_lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line, self.font, self.font_scale, self.font_thickness
            )
            max_width = max(max_width, text_width)
            total_height += text_height + 5  # 5px spacing between lines
        
        # Draw background rectangle
        bg_x1 = x - self.text_padding
        bg_y1 = y - total_height - self.text_padding
        bg_x2 = x + max_width + self.text_padding
        bg_y2 = y + self.text_padding
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                     self.colors['text_bg'], -1)
        
        # Draw text lines
        current_y = y - 5
        for line in text_lines:
            cv2.putText(frame, line, (x, current_y), self.font, 
                       self.font_scale, self.colors['text_fg'], self.font_thickness)
            current_y -= 20  # Move up for next line
    
    def _draw_corner_markers(self, frame: np.ndarray, top_left: Tuple[int, int], 
                           bottom_right: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw corner markers for better visibility."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        marker_length = 20
        marker_thickness = 3
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + marker_length, y1), color, marker_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + marker_length), color, marker_thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - marker_length, y1), color, marker_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + marker_length), color, marker_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + marker_length, y2), color, marker_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - marker_length), color, marker_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - marker_length, y2), color, marker_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - marker_length), color, marker_thickness)
    
    def _draw_pulsing_ring(self, frame: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a pulsing ring effect around the center point."""
        import time
        center_x, center_y = center
        
        # Create pulsing effect based on current time
        pulse_time = time.time() * 3  # Speed of pulse
        pulse_radius = int(25 + 10 * abs(np.sin(pulse_time)))  # Pulsing radius
        
        # Draw multiple rings for pulsing effect
        for i in range(3):
            radius = pulse_radius + i * 5
            alpha = 0.8 - i * 0.2  # Fade out effect
            ring_color = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, (center_x, center_y), radius, ring_color, 2)

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def draw_status_summary(self, frame: np.ndarray, 
                           unattended_count: int, 
                           confirmed_count: int) -> np.ndarray:
        """Draw status summary in the top-left corner of the frame."""
        # No text display - just return the frame as-is for better readability
        return frame
