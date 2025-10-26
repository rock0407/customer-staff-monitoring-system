# 🔄 Video Analytics Integration Workflow

## 📋 **Complete System Workflow**

### **1. Video Processing & Detection**
```
📹 Live Video Stream
    ↓
🎯 Person Detection (YOLO)
    ↓
👥 Person Tracking (DeepSORT)
    ↓
🤝 Interaction Detection
    ↓
🚨 Unattended Customer Detection
```

### **2. Video Segment Creation**
```
📊 Event Detected (Interaction/Unattended)
    ↓
🎬 VideoSegmenter.create_segment()
    ↓
💾 Save to segments/ folder
    ↓
📝 Log segment details
    ↓
🔄 Trigger upload process
```

### **3. Video Upload & API Integration**
```
📤 Upload Video Segment
    ↓
🌐 POST to https://obs.prolificapp.in/uploadFile
    ↓
📋 Extract uploaded filename from response
    ↓
🔗 Generate video URL: https://s3-noi.aces3.ai/vizo361/{filename}
    ↓
🆔 Create incident ID: incident_{timestamp}
    ↓
📊 Create video_evidence object
```

### **4. Incident Reporting**
```
📤 Report Incident
    ↓
🌐 POST to https://api.prolificapp.in/api/Addprolificai_alerts
    ↓
📋 Include video_evidence in payload
    ↓
✅ Incident reported successfully
    ↓
🔗 Link video evidence to analytics
```

### **5. Analytics Integration**
```
📊 InteractionLogger.log_interaction()
    ↓
📈 MetricsCollector.add_interaction_time()
    ↓
🎥 Include video_evidence in analytics
    ↓
💾 Save to analytics_metrics.json
    ↓
🔗 Link to existing analytics entries
```

## 🔄 **Detailed Step-by-Step Workflow**

### **Phase 1: Event Detection**
1. **Live Video Processing**
   - Camera captures video stream
   - YOLO detects people in frames
   - DeepSORT tracks people across frames
   - Interaction detection algorithm runs
   - Unattended customer detection runs

2. **Event Classification**
   - **Interaction Event**: Staff + Customer proximity detected
   - **Unattended Event**: Customer alone for >30 seconds
   - **Queue Event**: Multiple customers waiting

### **Phase 2: Video Segment Creation**
1. **Segment Initialization**
   ```
   VideoSegmenter.__init__()
   ├── Set segment duration (120s)
   ├── Initialize video writer
   └── Set segment file path
   ```

2. **Frame Processing**
   ```
   For each frame:
   ├── Write frame to video segment
   ├── Track interaction events
   ├── Track unattended customers
   └── Update segment metadata
   ```

3. **Segment Finalization**
   ```
   Segment complete (time-based or event-based)
   ├── Close video writer
   ├── Save segment file
   ├── Determine segment type (interaction/unattended)
   └── Trigger upload process
   ```

### **Phase 3: Video Upload Process**
1. **Upload Preparation**
   ```
   _upload_video_file()
   ├── Check file exists
   ├── Calculate file size
   ├── Set timeout based on size
   └── Prepare headers (X-API-Key)
   ```

2. **Upload Execution**
   ```
   POST to upload API
   ├── Send video file
   ├── Handle retry logic (3 attempts)
   ├── Extract uploaded filename
   └── Return stored reference
   ```

3. **Response Processing**
   ```
   Extract from JSON response:
   ├── filename: "prolific_upload_123456.mp4"
   ├── url: "https://s3-noi.aces3.ai/vizo361/..."
   └── Generate video_evidence object
   ```

### **Phase 4: Incident Reporting**
1. **Incident Payload Creation**
   ```
   _report_incident_payload()
   ├── Basic incident data
   ├── Video filename reference
   ├── Timing details (for unattended)
   ├── Bounding box details
   └── Confidence percentage
   ```

2. **API Call**
   ```
   POST to incident API
   ├── Headers: X-API-Key
   ├── JSON payload
   ├── Handle response
   └── Return success/failure
   ```

### **Phase 5: Analytics Integration**
1. **Interaction Logging**
   ```
   InteractionLogger.log_interaction()
   ├── Log to file
   ├── Add to metrics collector
   ├── Include video_evidence
   └── Update daily summary
   ```

2. **Analytics Storage**
   ```
   MetricsCollector.add_interaction_time()
   ├── Create interaction record
   ├── Add video_evidence object
   ├── Update statistics
   └── Save to JSON file
   ```

3. **Video Evidence Linking**
   ```
   _link_video_evidence_to_analytics()
   ├── Find matching interaction
   ├── Add video_evidence object
   ├── Update analytics JSON
   └── Log success/failure
   ```

## 📊 **Data Flow Architecture**

### **Input Sources**
- 📹 **Live Video Stream**: RTSP camera feed
- 🎯 **Detection Results**: YOLO person detection
- 👥 **Tracking Data**: DeepSORT person tracking
- ⏰ **Timing Data**: Interaction/unattended durations

### **Processing Components**
- 🎬 **VideoSegmenter**: Creates video segments
- 📤 **API Handlers**: Upload and incident reporting
- 📊 **InteractionLogger**: Captures interaction data
- 📈 **MetricsCollector**: Stores analytics data

### **Output Destinations**
- 💾 **Local Storage**: Video segments in `segments/` folder
- ☁️ **S3 Storage**: Uploaded videos on AWS S3
- 📋 **Analytics JSON**: `analytics_metrics.json` with video links
- 🚨 **Incident System**: Reported incidents with video evidence

## 🔗 **Integration Points**

### **VideoSegmenter ↔ API Handlers**
```
VideoSegmenter._report_incident()
    ↓
api_handler.report_video_segment_incident()
    ↓
Returns: {success: true, video_evidence: {...}}
    ↓
VideoSegmenter._link_video_evidence_to_analytics()
```

### **InteractionLogger ↔ MetricsCollector**
```
InteractionLogger.log_interaction()
    ↓
MetricsCollector.add_interaction_time()
    ↓
Include video_evidence in analytics
    ↓
Save to analytics_metrics.json
```

### **API Handlers ↔ External Services**
```
Upload API: https://obs.prolificapp.in/uploadFile
    ↓
Incident API: https://api.prolificapp.in/api/Addprolificai_alerts
    ↓
S3 Storage: https://s3-noi.aces3.ai/vizo361/{filename}
```

## 📋 **Analytics JSON Structure**

### **Enhanced Structure with Video Evidence**
```json
{
  "metadata": {
    "version": "1.1"
  },
  "interaction_metrics": {
    "interaction_times": [
      {
        "staff_id": 100,
        "customer_id": 200,
        "interaction_duration": 30.0,
        "video_evidence": {
          "segment_filename": "segment_8_20251025_113052_interaction_start.mp4",
          "uploaded_filename": "prolific_upload_1642248025_abc123.mp4",
          "video_url": "https://s3-noi.aces3.ai/vizo361/prolific_upload_1642248025_abc123.mp4",
          "incident_id": "incident_12345",
          "upload_timestamp": "2025-10-25 11:31:30"
        }
      }
    ]
  }
}
```

## 🎯 **Key Benefits**

### **Complete Traceability**
- ✅ Every interaction has video evidence
- ✅ Every unattended event has video evidence
- ✅ Direct links from analytics to video files
- ✅ Incident IDs for tracking and compliance

### **Automatic Integration**
- ✅ No manual intervention required
- ✅ Seamless integration with existing workflow
- ✅ Automatic video upload and linking
- ✅ Real-time analytics updates

### **Compliance Support**
- ✅ Full audit trail for regulatory requirements
- ✅ Video evidence for every event
- ✅ Incident tracking and reporting
- ✅ Complete documentation trail

## 🚀 **Production Ready**

The system is now ready for your day-long testing with:
- **Complete video analytics integration**
- **Automatic video upload and linking**
- **Enhanced analytics JSON with video evidence**
- **Full traceability and compliance support**

Your system will automatically handle the entire workflow from video detection to analytics storage with complete video evidence integration!
