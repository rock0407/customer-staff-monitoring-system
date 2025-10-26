# 🔄 Video Analytics Integration - Visual Workflow

## 📊 **Complete System Workflow Diagram**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           🎥 VIDEO ANALYTICS INTEGRATION WORKFLOW              │
└─────────────────────────────────────────────────────────────────────────────────┘

📹 LIVE VIDEO STREAM
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  🎯 YOLO        │    │  👥 DeepSORT    │    │  🤝 Interaction │
│  Detection      │───▶│  Tracking       │───▶│  Detection      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │                                              │
    ▼                                              ▼
┌─────────────────┐                        ┌─────────────────┐
│  🚨 Unattended  │                        │  📊 Analytics   │
│  Detection      │                        │  Collection     │
└─────────────────┘                        └─────────────────┘
    │                                              │
    ▼                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🎬 VIDEO SEGMENT CREATION                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  💾 Save to     │    │  📝 Log Segment │    │  🔄 Trigger     │
│  segments/      │───▶│  Details        │───▶│  Upload         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    📤 VIDEO UPLOAD PROCESS                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  🌐 Upload to   │    │  📋 Extract     │    │  🔗 Generate    │
│  S3 Storage     │───▶│  Filename       │───▶│  Video URL      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  🆔 Create      │    │  📊 Create      │    │  ✅ Upload     │
│  Incident ID    │───▶│  Video Evidence│───▶│  Success       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🚨 INCIDENT REPORTING                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  📋 Create      │    │  🌐 POST to      │    │  ✅ Report      │
│  Payload        │───▶│  Incident API    │───▶│  Success        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    📊 ANALYTICS INTEGRATION                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  📊 Log         │    │  📈 Add to      │    │  🎥 Include     │
│  Interaction    │───▶│  Metrics       │───▶│  Video Evidence │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  🔗 Link to     │    │  💾 Save to     │    │  📋 Update      │
│  Analytics      │───▶│  JSON File      │───▶│  Statistics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ✅ COMPLETE AUDIT TRAIL                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  🎥 Video       │    │  📊 Analytics   │    │  🆔 Incident    │
│  Evidence       │    │  Data           │    │  Tracking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 **Data Flow Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  📹 Live Video  │    │  🎯 Detection   │    │  👥 Tracking    │
│  Stream         │    │  Results        │    │  Data           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING COMPONENTS                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  🎬 Video       │    │  📤 API         │    │  📊 Interaction │
│  Segmenter      │───▶│  Handlers       │───▶│  Logger         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │                                              │
    ▼                                              ▼
┌─────────────────┐                        ┌─────────────────┐
│  📈 Metrics     │                        │  🔗 Video       │
│  Collector      │                        │  Evidence       │
└─────────────────┘                        └─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT DESTINATIONS                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  💾 Local       │    │  ☁️ S3 Storage  │    │  📋 Analytics   │
│  Segments       │    │  (Video Files)  │    │  JSON           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
┌─────────────────┐
│  🚨 Incident    │
│  System         │
└─────────────────┘
```

## 📊 **Analytics JSON Structure Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYTICS JSON STRUCTURE                    │
└─────────────────────────────────────────────────────────────────┘

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
  },
  "unattended_metrics": {
    "unattended_times": [
      {
        "customer_id": 300,
        "unattended_duration": 60.0,
        "video_evidence": {
          "segment_filename": "segment_5_20251024_140203_unattended.mp4",
          "uploaded_filename": "prolific_upload_1642248123_ghi789.mp4",
          "video_url": "https://s3-noi.aces3.ai/vizo361/prolific_upload_1642248123_ghi789.mp4",
          "incident_id": "incident_12347",
          "upload_timestamp": "2025-10-24 14:03:15"
        }
      }
    ]
  }
}
```

## 🔗 **Integration Points**

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTEGRATION POINTS                      │
└─────────────────────────────────────────────────────────────────┘

VideoSegmenter ↔ API Handlers
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  📤 Upload      │    │  📋 Incident    │    │  🔗 Video       │
│  Video          │───▶│  Reporting     │───▶│  Evidence       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
InteractionLogger ↔ MetricsCollector
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  📊 Log         │    │  📈 Store       │    │  🎥 Include     │
│  Interaction    │───▶│  Analytics      │───▶│  Video Evidence│
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │
    ▼
API Handlers ↔ External Services
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  🌐 Upload API  │    │  🚨 Incident    │    │  ☁️ S3 Storage  │
│  (obs.prolific  │───▶│  API            │───▶│  (Video Files)  │
│  app.in)        │    │  (api.prolific  │    │                 │
└─────────────────┘    │  app.in)        │    └─────────────────┘
                       └─────────────────┘
```

## 🎯 **Key Benefits**

```
┌─────────────────────────────────────────────────────────────────┐
│                          KEY BENEFITS                          │
└─────────────────────────────────────────────────────────────────┘

✅ Complete Traceability
    ├── Every interaction has video evidence
    ├── Every unattended event has video evidence
    ├── Direct links from analytics to video files
    └── Incident IDs for tracking and compliance

✅ Automatic Integration
    ├── No manual intervention required
    ├── Seamless integration with existing workflow
    ├── Automatic video upload and linking
    └── Real-time analytics updates

✅ Compliance Support
    ├── Full audit trail for regulatory requirements
    ├── Video evidence for every event
    ├── Incident tracking and reporting
    └── Complete documentation trail
```

## 🚀 **Production Ready**

Your system is now ready for day-long testing with:
- **Complete video analytics integration**
- **Automatic video upload and linking**
- **Enhanced analytics JSON with video evidence**
- **Full traceability and compliance support**

The workflow is fully automated and will handle everything from video detection to analytics storage with complete video evidence integration!
