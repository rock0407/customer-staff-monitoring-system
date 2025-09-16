#!/usr/bin/env python3
"""
Simple runner script for ACSI project.
Checks configuration and provides helpful instructions.
"""

import sys
import os
import cv2
from config_loader import LINE_COORDS, HEADLESS, VIDEO_SOURCE

def check_config():
    """Check if the configuration is ready to run."""
    print("ACSI Project - Configuration Check")
    print("="*40)
    
    # Check video source
    is_stream = VIDEO_SOURCE.startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_stream:
        print(f"ℹ️  Detected stream source: {VIDEO_SOURCE}")
        # Try to open briefly to validate
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Fallback without explicit backend
            cap = cv2.VideoCapture(VIDEO_SOURCE)
        ok, _ = (cap.read() if cap.isOpened() else (False, None))
        if cap:
            cap.release()
        if not ok:
            print(f"❌ Cannot open stream: {VIDEO_SOURCE}")
            print("   Tips: verify credentials, network reachability, and that OpenCV has FFMPEG/GStreamer.")
            print("   If the camera supports it, try TCP transport (rtsp over tcp).")
            return False
        else:
            print(f"✅ Stream reachable: {VIDEO_SOURCE}")
    else:
        if not os.path.exists(VIDEO_SOURCE):
            print(f"❌ Video file not found: {VIDEO_SOURCE}")
            print("   Please update VIDEO_SOURCE in config.json")
            return False
        else:
            print(f"✅ Video file found: {VIDEO_SOURCE}")
    
    # Check line coordinates
    if LINE_COORDS is None:
        print("❌ Line coordinates not set")
        print("   Run: python setup_line.py")
        print("   Or set LINE_COORDS in config.json")
        return False
    else:
        print(f"✅ Line coordinates set: {LINE_COORDS}")
    
    # Check headless mode
    if HEADLESS:
        print("✅ Running in headless mode (server ready)")
    else:
        print("ℹ️  Running in GUI mode (for development)")
    
    return True

def main():
    if not check_config():
        print("\nPlease fix the configuration issues above.")
        sys.exit(1)
    
    print("\n🚀 Starting ACSI project...")
    print("Press Ctrl+C to stop")
    
    # Import and run main
    try:
        import main
        # The main.py will handle everything
    except KeyboardInterrupt:
        print("\n👋 ACSI stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 