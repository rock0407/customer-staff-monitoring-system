# ACSI Quick Start Guide

## Complete Setup in 5 Minutes

### Step 1: Environment Setup
```bash
# Activate your conda environment
conda activate acsi

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Test System (Recommended)
```bash
# Run comprehensive system test
python test_system.py
```
This will verify all components work correctly before running the main system.

### Step 3: Get Line Coordinates
```bash
# Run the line setup tool
python setup_line.py
```
- A window will open showing your video's first frame
- Click two points to draw a line separating staff/customers
- Copy the coordinates that are printed

### Step 4: Configure the Project
Edit `config.py`:
```python
# Set your video source
VIDEO_SOURCE = 'test.mp4'  # or your video file

# Paste the coordinates from step 3
LINE_COORDS = ((x1, y1), (x2, y2))  # from setup_line.py

# For development (with GUI)
HEADLESS = False

# For server deployment (no GUI)
# HEADLESS = True

# Video upload settings (NEW!)
UPLOAD_ONLY_INTERACTION_VIDEOS = True  # Only upload segments with interactions
KEEP_ALL_VIDEOS_LOCALLY = True         # Always save all segments locally
```

### Step 5: Run the Project
```bash
# Use the runner script (recommended)
python run.py

# Or run directly
python main.py
```

## What You'll See

### Development Mode (HEADLESS = False)
- Live video window with tracking overlays
- Staff (red boxes) and customers (yellow boxes)
- Green line separating the areas
- **Green circles around people in active interactions**
- **Real-time interaction lines with duration counters**
- **Live statistics overlay** (frame count, time, segment info)
- Real-time interaction logging

### Server Mode (HEADLESS = True)
- No GUI windows
- All processing happens in background
- Logs saved to `acsi.log`
- Interactions saved to `interaction_log.txt`
- **Smart video uploading** (only interaction videos to API)

## New Smart Video Features 🎥

### **Selective API Uploads**
- ✅ **Only segments with interactions** are sent to API
- ✅ **All segments saved locally** with full visual processing
- ✅ **Timestamped filenames** for easy identification
- ✅ **Detailed logging** of upload decisions

### **Enhanced Visuals**
- 🟢 **Green circles**: Active interaction indicators
- 📏 **Green lines**: Real-time interaction connections
- ⏱️ **Duration counters**: Live interaction timing
- 📊 **Segment information**: Current segment status
- 🏷️ **Area labels**: Clear staff/customer zones

## Output Files

- **`acsi.log`**: Application logs with detailed processing info
- **`interaction_log.txt`**: Staff-customer interactions with timestamps
- **`segments/`**: 10-minute video segments (all saved locally, only interaction ones uploaded)

## Configuration Options

### **Video Settings**
```python
SEGMENT_DURATION = 600                   # 10 minutes per segment
UPLOAD_ONLY_INTERACTION_VIDEOS = True   # Smart uploading
KEEP_ALL_VIDEOS_LOCALLY = True          # Full local backup
```

### **Interaction Settings**
```python
MIN_INTERACTION_DURATION = 2.0  # Minimum 2 seconds (reduces false positives)
INTERACTION_THRESHOLD = 100     # Distance threshold in pixels
```

### **Tracking Settings**
```python
TRACKER_TYPE = 'deepsort'  # Best persistent tracking
# or 'sort' for faster processing
```

## System Testing 🧪

### **Before Running Main System**
```bash
python test_system.py
```

This comprehensive test will verify:
- ✅ All files present
- ✅ All imports working
- ✅ Configuration valid
- ✅ Detector functional
- ✅ Both trackers working
- ✅ Interaction logger ready
- ✅ Video segmenter operational

## Next Steps

1. **Test with your video**: Change `VIDEO_SOURCE` in `config.py`
2. **Adjust interaction threshold**: Edit `INTERACTION_THRESHOLD` for your space
3. **Set up API endpoint**: Configure `API_ENDPOINT` in `config.py`
4. **Deploy to server**: Set `HEADLESS = True` and use `systemd` or Docker

## Troubleshooting

- **System test fails**: Run `python test_system.py` to identify issues
- **"Video source not found"**: Check file path in `config.py`
- **"Line coordinates not set"**: Run `python setup_line.py`
- **GUI errors on server**: Set `HEADLESS = True`
- **Check logs**: Look at `acsi.log` for detailed errors
- **Tracking ID loss**: Ensure `TRACKER_TYPE = 'deepsort'`

## Ready for Production?

1. ✅ Run `python test_system.py` (all tests pass)
2. ✅ Set `HEADLESS = True`
3. ✅ Configure your `API_ENDPOINT`
4. ✅ Verify `UPLOAD_ONLY_INTERACTION_VIDEOS = True`
5. ✅ Use `systemd` service or Docker for auto-restart
6. ✅ Monitor logs and video segments

**Your ACSI system now intelligently uploads only interaction videos while keeping full local backups with professional visuals!** 🚀 