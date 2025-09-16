#!/usr/bin/env python3
"""
Test script to verify API alert logic and responses.
"""

import os
import time
import numpy as np
import cv2
import requests
import json
from datetime import datetime
from api_handler import (
    _upload_image_file, 
    _report_incident_payload, 
    report_unattended_customer,
    report_video_segment_incident
)
from config_loader import API_URLS, UPLOAD_API_KEY, BRANCH_ID, LOCATION

def test_api_connectivity():
    """Test basic API connectivity."""
    print("🌐 Testing API Connectivity")
    print("="*30)
    
    # Test upload endpoint
    upload_url = API_URLS['upload_image']
    print(f"📤 Upload URL: {upload_url}")
    
    try:
        # Test with a simple GET request to check if endpoint is reachable
        response = requests.get(upload_url, timeout=10)
        print(f"✅ Upload endpoint reachable: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload endpoint not reachable: {e}")
        return False
    
    # Test incident reporting endpoint
    incident_url = API_URLS['report_incident']
    print(f"📋 Incident URL: {incident_url}")
    
    try:
        response = requests.get(incident_url, timeout=10)
        print(f"✅ Incident endpoint reachable: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Incident endpoint not reachable: {e}")
        return False
    
    return True

def test_image_upload():
    """Test image upload functionality."""
    print("\n📸 Testing Image Upload")
    print("="*25)
    
    # Create a test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :, 0] = 255  # Red channel
    cv2.putText(test_image, "TEST", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save test image
    test_filename = f"test_image_{int(time.time())}.jpg"
    success = cv2.imwrite(test_filename, test_image)
    
    if not success:
        print("❌ Failed to create test image")
        return False
    
    print(f"✅ Created test image: {test_filename}")
    
    try:
        # Test upload
        uploaded_name = _upload_image_file(test_filename)
        
        if uploaded_name:
            print(f"✅ Image uploaded successfully: {uploaded_name}")
            return True
        else:
            print("❌ Image upload failed")
            return False
            
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return False
    finally:
        # Clean up test image
        if os.path.exists(test_filename):
            os.remove(test_filename)
            print(f"🗑️ Cleaned up test image: {test_filename}")

def test_incident_reporting():
    """Test incident reporting functionality."""
    print("\n🚨 Testing Incident Reporting")
    print("="*30)
    
    # Test payload structure
    test_payload = {
        "branch_id": BRANCH_ID,
        "incident_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reported_location": LOCATION,
        "incident_type": "unattended_customer",
        "status": "Open",
        "image": "test_image.jpg",
        "percentage": 85,
        "vehicle_template": "",
        "unknown_person": 0
    }
    
    print("📋 Test payload:")
    print(json.dumps(test_payload, indent=2))
    
    try:
        # Test incident reporting
        success = _report_incident_payload(
            image_name="test_image.jpg",
            incident_type="unattended_customer",
            branch_id=BRANCH_ID,
            location=LOCATION,
            confidence_percentage=0.85
        )
        
        if success:
            print("✅ Incident reported successfully")
            return True
        else:
            print("❌ Incident reporting failed")
            return False
            
    except Exception as e:
        print(f"❌ Incident reporting error: {e}")
        return False

def test_unattended_customer_alert():
    """Test complete unattended customer alert flow."""
    print("\n👤 Testing Unattended Customer Alert")
    print("="*35)
    
    # Create a test face region (simulating customer face)
    face_region = np.zeros((80, 80, 3), dtype=np.uint8)
    face_region[:, :, 2] = 200  # Blue channel
    cv2.putText(face_region, "FACE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print("✅ Created test face region")
    
    try:
        # Test complete alert flow
        success = report_unattended_customer(
            image_array=face_region,
            branch_id=BRANCH_ID,
            location=LOCATION,
            confidence_percentage=0.85
        )
        
        if success:
            print("✅ Unattended customer alert sent successfully")
            return True
        else:
            print("❌ Unattended customer alert failed")
            return False
            
    except Exception as e:
        print(f"❌ Unattended customer alert error: {e}")
        return False

def test_video_segment_alert():
    """Test video segment incident reporting."""
    print("\n🎥 Testing Video Segment Alert")
    print("="*30)
    
    # Create a test video file
    test_video = f"test_video_{int(time.time())}.mp4"
    
    # Create a simple video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, 25.0, (640, 480))
    
    for i in range(25):  # 1 second at 25fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = 100  # Green channel
        cv2.putText(frame, f"Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    
    if not os.path.exists(test_video):
        print("❌ Failed to create test video")
        return False
    
    print(f"✅ Created test video: {test_video}")
    
    try:
        # Test video segment incident reporting
        success = report_video_segment_incident(
            video_path=test_video,
            incident_type="unattended_customer",
            branch_id=BRANCH_ID,
            location=LOCATION
        )
        
        if success:
            print("✅ Video segment incident reported successfully")
            return True
        else:
            print("❌ Video segment incident reporting failed")
            return False
            
    except Exception as e:
        print(f"❌ Video segment incident error: {e}")
        return False
    finally:
        # Clean up test video
        if os.path.exists(test_video):
            os.remove(test_video)
            print(f"🗑️ Cleaned up test video: {test_video}")

def test_api_configuration():
    """Test API configuration."""
    print("\n⚙️ Testing API Configuration")
    print("="*30)
    
    print(f"📤 Upload URL: {API_URLS['upload_image']}")
    print(f"📋 Incident URL: {API_URLS['report_incident']}")
    print(f"🔑 API Key: {UPLOAD_API_KEY[:8]}..." if UPLOAD_API_KEY else "❌ No API Key")
    print(f"🏢 Branch ID: {BRANCH_ID}")
    print(f"📍 Location: {LOCATION}")
    
    # Check if URLs are properly configured
    if "your.api" in API_URLS['upload_image'] or "your.api" in API_URLS['report_incident']:
        print("⚠️ Warning: API URLs contain placeholder values")
        return False
    
    if not UPLOAD_API_KEY or UPLOAD_API_KEY == "your_api_key":
        print("⚠️ Warning: API key not properly configured")
        return False
    
    print("✅ API configuration looks good")
    return True

def run_all_api_tests():
    """Run all API tests."""
    print("🚀 API Alert System Test Suite")
    print("="*40)
    
    tests = [
        ("API Configuration", test_api_configuration),
        ("API Connectivity", test_api_connectivity),
        ("Image Upload", test_image_upload),
        ("Incident Reporting", test_incident_reporting),
        ("Unattended Customer Alert", test_unattended_customer_alert),
        ("Video Segment Alert", test_video_segment_alert)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "="*40)
    print(f"📊 API Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API tests passed! Alert system is working correctly.")
    else:
        print("⚠️ Some API tests failed. Check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_api_tests()
    exit(0 if success else 1)
