# Customer-Staff Interaction (CSI) Video Analytics System

## Overview
This advanced video analytics system processes video streams or live RTSP feeds to:
- Detect and track people using YOLOv8
- Classify them as staff or customers based on a user-drawn separation line
- Monitor and log interactions between staff and customers in real-time
- Track unattended customers with configurable timers and alerts
- Automatically segment and upload video clips to API endpoints
- Provide comprehensive analytics and reporting

## Key Features
- **Advanced Person Detection**: YOLOv8-based detection with GPU acceleration support
- **Persistent Multi-Object Tracking**: ByteTrack implementation for robust ID maintenance across occlusions
- **Intelligent Classification**: User-defined line separation for staff/customer areas
- **Real-time Interaction Monitoring**: Distance-based interaction detection with configurable thresholds
- **Unattended Customer Alerts**: Timer-based system with confirmation periods
- **Automatic Video Segmentation**: 2-minute segments with API upload capabilities
- **Multi-Organization Support**: Configurable for different organizations and branches
- **Headless Server Mode**: Production-ready deployment without GUI
- **RTSP Stream Support**: Auto-reconnection and robust stream handling
- **Comprehensive Logging**: Detailed interaction logs and system monitoring

## Tracking Technology

### ByteTrack (Primary)
- **State-of-the-art tracking** - maintains consistent IDs even during complex occlusions
- Uses motion prediction and appearance features for robust re-identification
- Optimized for real-time performance with configurable parameters
- **Default and recommended** tracking solution

### Simple Tracker (Fallback)
- Lightweight fallback option when ByteTrack is unavailable
- Uses IoU-based matching with distance thresholds
- Suitable for basic tracking scenarios

Configuration is managed in `config.json` and loaded by `config_loader.py`.

## Quick Start

### 1. Setup Environment

#### Create Conda Environment
```bash
conda create -n bytetrack python=3.9 -y
conda activate bytetrack
```

#### Install PyTorch (Required for ByteTrack)
```bash
pip install torch==1.13.0 torchvision==0.14.0
```

#### Install Core Dependencies
```bash
pip install numpy==1.25.2 scipy==1.9.3 pandas==1.5.3
pip install matplotlib==3.9.4 seaborn==0.13.2
pip install opencv-python==4.10.0.84 pillow==11.3.0
```

#### Install YOLO and Tracking Dependencies
```bash
pip install ultralytics==8.0.196
pip install ultralytics-thop==2.0.15 thop==0.1.1.post2209072238
```

#### Install Additional Utilities
```bash
pip install pyyaml==6.0.2 tqdm==4.67.1 requests==2.31.0
pip install psutil==5.9.8 py-cpuinfo==9.0.0
pip install filterpy==1.4.5 lap==0.4.0
```

#### Install ByteTracker
```bash
pip install bytetracker==0.3.2
```

#### Alternative: Install from requirements.txt
```bash
pip install -r requirements.txt
```

### 2. Get Line Coordinates
```bash
python setup_line.py
```
- This opens your video and lets you draw a line
- Copy the coordinates to `config.json`

### 3. Configure
Edit `config.json` with your organization and camera settings:
```json
{
  "organizations": {
    "api_url": {
      "upload_image": "https://your-api.com/uploadFile",
      "report_incident": "https://your-api.com/api/Addprolificai_alerts",
      "report_footfall": "https://your-api.com/api/Addfootfall"
    },
    "upload_api_key": "your_api_key",
    "YourOrganization": {
      "name": "Your Organization Name",
      "description": "Organization description",
      "active": true,
      "ip_cameras": {
        "cam1": {
          "camera_config": {
            "protocol": "rtsp",
            "host": "192.168.1.100",
            "port": 554,
            "username": "admin",
            "password": "password",
            "path": "/cam/realmonitor"
          },
          "line_coords": [[x1, y1], [x2, y2]],
          "headless": false
        }
      },
      "interaction_settings": {
        "min_interaction_duration": 5.0,
        "interaction_threshold": 650,
        "unattended_threshold": 30.0,
        "unattended_confirmation_timer": 60.0
      },
      "tracking_settings": {
        "track_thresh": 0.3,
        "track_buffer": 30,
        "match_thresh": 0.5,
        "frame_rate": 30
      }
    }
  },
  "system_settings": {
    "current_organization": "YourOrganization",
    "current_branch": "YourBranch",
    "current_camera": "cam1"
  }
}
```

### 4. Run
```bash
python run.py
```

## Project Structure
```
customerstaffinteraction/
├── main.py                    # Main application entry point
├── run.py                     # Application runner with config validation
├── setup_line.py              # Interactive line coordinate setup tool
├── config.json                # Multi-organization configuration
├── config_loader.py           # Configuration loader and validator
├── detector.py                # YOLOv8 person detection
├── tracker_bytetrack.py       # ByteTrack multi-object tracking
├── tracker_simple.py          # Simple fallback tracker
├── line_drawer.py             # Line drawing utilities
├── line_calculating.py        # Line coordinate calculation
├── interaction.py             # Interaction detection & logging
├── video_segmenter.py         # Video segmentation & API upload
├── api_handler.py             # API communication handler
├── footfall.py                # Footfall analytics
├── test_system.py             # System testing utilities
├── requirements.txt           # Python dependencies
├── requirements-core.txt      # Core dependencies only
├── README.md                  # This documentation
├── Jenkinsfile               # CI/CD pipeline configuration
├── acsi.log                  # Application logs
├── interaction_log.txt       # Interaction event logs
├── segments/                 # Video segments directory
└── yolov8n.pt               # YOLOv8 model weights
```

## Usage Modes

### Development Mode (with GUI)
Set `headless` to `false` in your camera configuration within `config.json`, then run `python run.py`.

### Server Mode (Headless)
Set `headless` to `true` and provide `line_coords` in your camera configuration within `config.json`, then run `python run.py`.

## Configuration Options

### Video Source
- **Local file**: Configure `camera_config` with local file path
- **RTSP stream**: Configure `camera_config` with RTSP connection details:
  ```json
  "camera_config": {
    "protocol": "rtsp",
    "host": "192.168.1.100",
    "port": 554,
    "username": "admin",
    "password": "password",
    "path": "/cam/realmonitor"
  }
  ```

### Line Coordinates
- **Manual setup**: Run `python setup_line.py`
- **Direct config**: Set `line_coords` in camera configuration: `[[x1, y1], [x2, y2]]`

### Tracking Configuration
- **ByteTrack** (default): Configured in `tracking_settings`
- **Simple Tracker** (fallback): Automatically used if ByteTrack fails

### Interaction Settings
- `min_interaction_duration`: Minimum time for valid interaction (seconds)
- `interaction_threshold`: Distance threshold for interaction detection (pixels)
- `unattended_threshold`: Time before customer marked as unattended (seconds)
- `unattended_confirmation_timer`: Confirmation period for unattended alerts (seconds)

### API Endpoints
- Configure under `api_url` in `config.json` for different API operations

## Output Files

### Logs
- `acsi.log`: Application logs (errors, info, warnings)
- `interaction_log.txt`: Staff-customer interaction logs with timestamps and details

### Video Segments
- `segments/`: 2-minute video segments with interaction events
- `segments_unattended/`: Segments containing unattended customer alerts
- Automatically uploaded to configured API endpoints

## Server Deployment

### Systemd Service
Create `/etc/systemd/system/csi.service`:
```ini
[Unit]
Description=Customer-Staff Interaction Video Analytics
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/customerstaffinteraction
ExecStart=/path/to/conda/envs/bytetrack/bin/python run.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/path/to/customerstaffinteraction

[Install]
WantedBy=multi-user.target
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p segments segments_unattended logs

CMD ["python", "run.py"]
```

## Troubleshooting

### Common Issues
1. **"Cannot open video source"**: Check camera configuration in `config.json`
2. **"Line coordinates not set"**: Run `python setup_line.py` or set `line_coords` in config
3. **YOLOv8 model missing**: Download `yolov8n.pt` from Ultralytics or ensure it's in project directory
4. **GUI errors on server**: Set `headless: true` in camera configuration
5. **Tracking ID loss**: ByteTrack is used by default; check `tracking_settings` configuration
6. **ByteTrack import errors**: Ensure `ultralytics==8.0.196` and `bytetracker==0.3.2` are installed
7. **GPU not detected**: Check CUDA installation and set `use_gpu: true` in processing settings

### Logs
- Check `acsi.log` for detailed error messages and system status
- Check `interaction_log.txt` for interaction data and timing information
- Monitor system performance through built-in logging

### Performance Optimization
- Enable GPU acceleration in `processing_settings.use_gpu`
- Adjust `target_fps` and `frame_skip` for performance tuning
- Configure `max_cpu_usage` and `max_gpu_memory_gb` limits

## API Integration
The system automatically sends 2-minute video segments to configured API endpoints. Ensure your endpoints accept multipart form data with video files and proper authentication.

## Key Features Summary
- **Multi-Organization Support**: Configure multiple organizations and branches
- **Advanced Tracking**: ByteTrack for persistent ID maintenance across occlusions
- **Unattended Customer Monitoring**: Timer-based alerts with confirmation periods
- **Real-time Analytics**: Live interaction detection and logging
- **Automatic Video Segmentation**: Smart segmenting based on interaction events
- **Robust Deployment**: Headless mode with systemd service support
- **Comprehensive Logging**: Detailed logs for debugging and analytics 