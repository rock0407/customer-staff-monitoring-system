# ACSI: Staff-Customer Interaction Video Analytics

## Overview
This software processes video or live RTSP streams to:
- Detect and track people
- Classify them as staff or customers based on a user-drawn line
- Log and visualize interactions between staff and customers
- Save and send 10-minute video segments to an API

## Features
- Manual line drawing to separate staff and customers
- YOLOv8-based person detection
- **Persistent multi-object tracking** (DeepSORT/SORT) - maintains IDs even during occlusions
- Real-time interaction logging
- Live video with overlays
- Automatic video segmenting and API upload
- Headless mode for server deployment
- Auto-reconnect for RTSP streams

## Tracking Technology

### DeepSORT (Recommended)
- **Best for persistent tracking** - maintains IDs even when people are temporarily occluded
- Uses deep learning features for robust re-identification
- Handles complex scenarios with multiple people
- **Default choice** for most applications

### SORT (Alternative)
- Faster but less robust than DeepSORT
- Good for simpler scenarios
- Uses Kalman filtering for prediction
- Lighter computational requirements

Configuration is managed in `config.json` and loaded by `config_loader.py`.

## Quick Start

### 1. Setup Environment
```bash
conda activate acsi
pip install -r requirements.txt
```

### 2. Get Line Coordinates
```bash
python setup_line.py
```
- This opens your video and lets you draw a line
- Copy the coordinates to `config.py`

### 3. Configure
Edit `config.json` (examples shown in file):
```json
{
  "camera": {
    "video_source": "your_video.mp4",
    "line_coords": [[x1, y1], [x2, y2]],
    "headless": false
  }
}
```

### 4. Run
```bash
python run.py
```

## Project Structure
```
acsi/
├── main.py              # Main application
├── run.py               # Runner with config checks
├── setup_line.py        # Line coordinate setup tool
├── config.json          # Configuration
├── detector.py          # Person detection (YOLOv8)
├── tracker.py           # DeepSORT tracking
├── tracker_sort.py      # SORT tracking (alternative)
├── line_drawer.py       # Line drawing utilities
├── line_calculating.py  # Line coordinate calculation
├── interaction.py       # Interaction detection & logging
├── video_segmenter.py   # Video segmenting & API upload
├── requirements.txt     # Dependencies
├── README.md           # This file
├── acsi.log            # Application logs
├── interaction_log.txt # Interaction logs
├── segments/           # Video segments directory
└── yolov8n.pt         # YOLOv8 model (download separately)
```

## Usage Modes

### Development Mode (with GUI)
Set `camera.headless` to false in `config.json`, then run `python run.py`.

### Server Mode (Headless)
Set `camera.headless` to true and provide `line_coords` in `config.json`, then run `python run.py`.

## Configuration Options

### Video Source
- **Local file**: `VIDEO_SOURCE = 'video.mp4'`
- **RTSP stream**: `VIDEO_SOURCE = 'rtsp://user:pass@ip:port/stream'`

### Line Coordinates
- **Manual setup**: Run `python setup_line.py`
- **Direct config**: `LINE_COORDS = ((100, 200), (400, 200))`

### Tracker Selection
- **DeepSORT** (recommended): `TRACKER_TYPE = 'deepsort'`
- **SORT** (faster): `TRACKER_TYPE = 'sort'`

### API Endpoints
- Configure under `api.*` in `config.json`.

## Output Files

### Logs
- `acsi.log`: Application logs (errors, info, warnings)
- `interaction_log.txt`: Staff-customer interaction logs

### Video Segments
- `segments/`: 10-minute video segments
- Automatically sent to API endpoint

## Server Deployment

### Systemd Service
Create `/etc/systemd/system/acsi.service`:
```ini
[Unit]
Description=ACSI Video Analytics
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/acsi
ExecStart=/path/to/conda/envs/acsi/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker (Optional)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "run.py"]
```

## Troubleshooting

### Common Issues
1. **"Cannot open video source"**: Check `VIDEO_SOURCE` path/URL
2. **"Line coordinates not set"**: Run `python setup_line.py`
3. **YOLOv8 model missing**: Download `yolov8n.pt` from Ultralytics
4. **GUI errors on server**: Set `HEADLESS = True`
5. **Tracking ID loss**: Use DeepSORT (`TRACKER_TYPE = 'deepsort'`)

### Logs
- Check `acsi.log` for detailed error messages
- Check `interaction_log.txt` for interaction data

## API Integration
The system automatically sends 10-minute video segments to your API endpoint. Ensure your endpoint accepts multipart form data with video files.

## Notes
- The interaction threshold (distance) can be adjusted in `interaction.py`
- Video segments are saved locally before API upload
- The system auto-reconnects if RTSP stream drops
- All operations are logged for debugging
- **DeepSORT provides the best persistent tracking** for maintaining IDs across occlusions 