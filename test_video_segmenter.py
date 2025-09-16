#!/usr/bin/env python3
"""
Test script to verify video segmenter fixes work correctly.
"""

import os
import time
import numpy as np
import cv2
from video_segmenter import VideoSegmenter

def test_video_segmenter_fixes():
    """Test the video segmenter with our fixes."""
    print("🧪 Testing Video Segmenter Fixes")
    print("="*40)
    
    # Clean up any existing test segments
    test_segments_dir = "test_segments"
    if os.path.exists(test_segments_dir):
        import shutil
        shutil.rmtree(test_segments_dir)
    
    # Create test segmenter with short duration for testing
    segmenter = VideoSegmenter(fps=25, frame_size=(640, 480))
    
    # Override the segment duration for testing
    segmenter.SEGMENT_DURATION = 2  # 2 seconds for quick testing
    
    print("✅ VideoSegmenter initialized")
    
    # Test 1: Verify no segment is created on initialization
    if segmenter.writer is None:
        print("✅ No segment created on initialization (FIXED)")
    else:
        print("❌ Segment created on initialization (BUG)")
        return False
    
    # Test 2: Add some frames and verify segment is created
    print("\n📹 Adding test frames...")
    for i in range(50):  # 2 seconds at 25fps
        # Create a test frame with some content
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 5) % 255  # Red channel changes
        frame[:, :, 1] = 100  # Green channel
        frame[:, :, 2] = 200  # Blue channel
        
        # Add some text to make it more interesting
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        segmenter.add_frame(frame, active_interactions=[(1, 2)], any_met_duration=True)
        
        if i == 0:
            if segmenter.writer is not None:
                print("✅ Segment created on first frame (FIXED)")
            else:
                print("❌ Segment not created on first frame (BUG)")
                return False
    
    print(f"✅ Added {segmenter.frames_written} frames")
    
    # Test 3: Finalize segment and check it was saved properly
    print("\n💾 Finalizing segment...")
    segmenter.finalize_segment()
    
    # Check if segment file exists and has content
    if segmenter.segment_file and os.path.exists(segmenter.segment_file):
        file_size = os.path.getsize(segmenter.segment_file)
        print(f"✅ Segment file created: {os.path.basename(segmenter.segment_file)} ({file_size} bytes)")
        
        if file_size > 1000:  # Should be much larger than the old 4361 byte empty files
            print("✅ Segment has substantial content (FIXED)")
        else:
            print("❌ Segment file too small (BUG)")
            return False
    else:
        print("❌ Segment file not created (BUG)")
        return False
    
    # Test 4: Verify video can be read back
    print("\n🔍 Verifying video can be read...")
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
        if frame_count > 0:
            print("✅ Video contains frames (FIXED)")
        else:
            print("❌ Video contains no frames (BUG)")
            return False
    else:
        print("❌ Cannot open video file (BUG)")
        return False
    
    # Test 5: Test empty segment handling
    print("\n🧪 Testing empty segment handling...")
    segmenter2 = VideoSegmenter(fps=25, frame_size=(640, 480))
    segmenter2.finalize_segment()  # Finalize without adding any frames
    
    if segmenter2.writer is None:
        print("✅ No empty segment file created (FIXED)")
    else:
        print("❌ Empty segment file created (BUG)")
        return False
    
    print("\n" + "="*40)
    print("🎉 All video segmenter tests passed!")
    print("✅ 0-second segment issue has been FIXED")
    return True

if __name__ == "__main__":
    success = test_video_segmenter_fixes()
    if success:
        print("\n🚀 Video segmenter is working correctly!")
    else:
        print("\n❌ Video segmenter still has issues!")
