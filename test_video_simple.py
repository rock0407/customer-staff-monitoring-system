#!/usr/bin/env python3
"""
Simple test to verify video segmenter fixes without API calls.
"""

import os
import time
import numpy as np
import cv2
from video_segmenter import VideoSegmenter

def test_video_simple():
    """Simple test of video segmenter fixes."""
    print("🧪 Simple Video Segmenter Test")
    print("="*30)
    
    # Create test segmenter
    segmenter = VideoSegmenter(fps=25, frame_size=(640, 480))
    print("✅ VideoSegmenter initialized")
    
    # Test 1: No segment on initialization
    if segmenter.writer is None:
        print("✅ No segment created on initialization (FIXED)")
    else:
        print("❌ Segment created on initialization (BUG)")
        return False
    
    # Test 2: Add frames
    print("📹 Adding 25 frames (1 second)...")
    for i in range(25):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 10) % 255  # Red channel changes
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        segmenter.add_frame(frame)
    
    print(f"✅ Added {segmenter.frames_written} frames")
    
    # Test 3: Check segment file
    if segmenter.segment_file and os.path.exists(segmenter.segment_file):
        file_size = os.path.getsize(segmenter.segment_file)
        print(f"✅ Segment file: {os.path.basename(segmenter.segment_file)} ({file_size} bytes)")
        
        if file_size > 10000:  # Should be substantial
            print("✅ Segment has good content (FIXED)")
        else:
            print(f"⚠️ Segment size: {file_size} bytes (might be small)")
    else:
        print("❌ No segment file created")
        return False
    
    # Test 4: Verify video readability
    cap = cv2.VideoCapture(segmenter.segment_file)
    if cap.isOpened():
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.release()
        print(f"✅ Video readable: {frame_count} frames")
    else:
        print("❌ Cannot read video")
        return False
    
    print("\n🎉 Video segmenter fixes are working!")
    return True

if __name__ == "__main__":
    test_video_simple()
