#!/usr/bin/env python3
"""
Comprehensive test script for ACSI system.
Tests all components and their integration.
"""

import os
import sys
import logging
import cv2
import numpy as np

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import config_loader
        print("✅ config_loader.py imported successfully")
        
        from detector import PersonDetector
        print("✅ PersonDetector imported successfully")
        
        from tracker_bytetrack import PersonTrackerBYTE
        print("✅ PersonTrackerBYTE (ByteTrack) imported successfully")
        
        from interaction import InteractionLogger
        print("✅ InteractionLogger imported successfully")
        
        from video_segmenter import VideoSegmenter
        print("✅ VideoSegmenter imported successfully")
        
        from line_calculating import get_line_from_user
        print("✅ line_calculating imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration file."""
    print("\n🧪 Testing configuration...")
    
    try:
        from config_loader import (VIDEO_SOURCE, LINE_COORDS, HEADLESS,
                           MIN_INTERACTION_DURATION, INTERACTION_THRESHOLD,
                           UPLOAD_ONLY_INTERACTION_VIDEOS, KEEP_ALL_VIDEOS_LOCALLY)
        
        print(f"✅ VIDEO_SOURCE: {VIDEO_SOURCE}")
        print(f"✅ LINE_COORDS: {LINE_COORDS}")
        print(f"✅ HEADLESS: {HEADLESS}")
        print(f"✅ MIN_INTERACTION_DURATION: {MIN_INTERACTION_DURATION}")
        print(f"✅ INTERACTION_THRESHOLD: {INTERACTION_THRESHOLD}")
        print(f"✅ UPLOAD_ONLY_INTERACTION_VIDEOS: {UPLOAD_ONLY_INTERACTION_VIDEOS}")
        print(f"✅ KEEP_ALL_VIDEOS_LOCALLY: {KEEP_ALL_VIDEOS_LOCALLY}")
        
        # Test video source existence
        if os.path.exists(VIDEO_SOURCE):
            print(f"✅ Video source exists: {VIDEO_SOURCE}")
        else:
            print(f"⚠️  Video source not found: {VIDEO_SOURCE}")
            
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_detector():
    """Test person detector."""
    print("\n🧪 Testing person detector...")
    
    try:
        from detector import PersonDetector
        
        detector = PersonDetector()
        print("✅ PersonDetector initialized")
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes, confs = detector.detect(dummy_frame)
        
        print(f"✅ Detection test passed - detected {len(boxes)} objects")
        return True
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        return False

def test_trackers():
    """Test both tracking systems."""
    print("\n🧪 Testing trackers...")
    
    try:
        from tracker_bytetrack import PersonTrackerBYTE
        
        # Test ByteTrack tracker
        bytetrack_tracker = PersonTrackerBYTE()
        print("✅ ByteTrack tracker initialized")
        
        # Test with dummy data
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        dummy_confs = np.array([0.9, 0.8])
        
        tracks = bytetrack_tracker.update(dummy_boxes, dummy_confs, dummy_frame)
        
        print(f"✅ ByteTrack tracking test passed - {len(tracks)} tracks")
        
        return True
        
    except Exception as e:
        print(f"❌ Tracker test failed: {e}")
        return False

def test_interaction_logger():
    """Test interaction logging system."""
    print("\n🧪 Testing interaction logger...")
    
    try:
        from interaction import InteractionLogger
        from config_loader import MIN_INTERACTION_DURATION, INTERACTION_THRESHOLD
        
        logger = InteractionLogger(min_duration=MIN_INTERACTION_DURATION, 
                                 threshold=INTERACTION_THRESHOLD)
        print("✅ InteractionLogger initialized")
        
        # Test with dummy data
        staff = [(1, (100, 100), [90, 90, 110, 110])]
        customers = [(2, (150, 150), [140, 140, 160, 160])]  # Close enough for interaction
        
        logger.check_and_log(staff, customers, 1.0)
        logger.check_and_log(staff, customers, 3.0)  # Should trigger completion
        
        active = logger.get_active_interactions()
        print(f"✅ Interaction logging test passed - {len(active)} active interactions")
        
        return True
        
    except Exception as e:
        print(f"❌ Interaction logger test failed: {e}")
        return False

def test_video_segmenter():
    """Test video segmenter."""
    print("\n🧪 Testing video segmenter...")
    
    try:
        from video_segmenter import VideoSegmenter
        
        segmenter = VideoSegmenter(fps=25, frame_size=(640, 480))
        print("✅ VideoSegmenter initialized")
        
        # Test with dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_interactions = [(1, 2)]
        
        segmenter.add_frame(dummy_frame, dummy_interactions)
        
        info = segmenter.get_current_segment_info()
        print(f"✅ Video segmenter test passed - segment {info['segment_idx']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Video segmenter test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        'main.py',
        'config_loader.py',
        'detector.py',
        'tracker_bytetrack.py',
        'interaction.py',
        'video_segmenter.py',
        'line_calculating.py',
        'setup_line.py',
        'run.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    if not missing_files:
        print("✅ All required files present")
        return True
    else:
        print(f"❌ Missing files: {missing_files}")
        return False

def run_all_tests():
    """Run all system tests."""
    print("🚀 ACSI System Comprehensive Test Suite")
    print("="*50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_config,
        test_detector,
        test_trackers,
        test_interaction_logger,
        test_video_segmenter
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        return True
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 