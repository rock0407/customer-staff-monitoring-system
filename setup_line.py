#!/usr/bin/env python3
"""
Standalone script to get line coordinates from a video file.
Run this script to draw a line on the first frame and get the coordinates.
"""

import cv2
import sys
from config_loader import VIDEO_SOURCE

def get_line_coordinates(video_path):
    """Get line coordinates from the first frame of a video."""
    print(f"Opening video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Cannot read frame from video")
        return None
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    print("Click two points to draw a line. Press ESC to cancel.")
    
    points = []
    frame_copy = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
                print(f"Point {len(points)}: ({x}, {y})")
    
    cv2.namedWindow("Draw Line on Video Frame")
    cv2.setMouseCallback("Draw Line on Video Frame", mouse_callback)
    
    while True:
        temp = frame_copy.copy()
        
        # Draw points and line
        if len(points) >= 1:
            cv2.circle(temp, points[0], 5, (0, 255, 0), -1)
        if len(points) >= 2:
            cv2.line(temp, points[0], points[1], (0, 255, 0), 2)
            cv2.circle(temp, points[1], 5, (0, 255, 0), -1)
        
        cv2.imshow("Draw Line on Video Frame", temp)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif len(points) == 2:
            break
    
    cv2.destroyAllWindows()
    
    if len(points) == 2:
        return points[0], points[1]
    else:
        return None

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = VIDEO_SOURCE
    
    result = get_line_coordinates(video_path)
    
    if result:
        point1, point2 = result
        print("\n" + "="*50)
        print("LINE COORDINATES OBTAINED:")
        print(f"Point 1: {point1}")
        print(f"Point 2: {point2}")
        print("\nTo use these coordinates in your project:")
        print(f"1. Open config_loader.py")
        print(f"2. Set LINE_COORDS = {result}")
        print(f"3. Set HEADLESS = True (for server mode)")
        print("="*50)
    else:
        print("No line coordinates obtained.")

if __name__ == "__main__":
    main() 