import json
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import re

class MetricsCollector:
    def __init__(self, metrics_file="analytics_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics_data = self._load_existing_metrics()
        
    def _load_existing_metrics(self):
        """Load existing metrics from JSON file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading metrics file: {e}")
        return self._create_empty_metrics()
    
    def _create_empty_metrics(self):
        """Create empty metrics structure"""
        return {
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.1"
            },
            "queue_metrics": {
                "queue_times": [],
                "avg_queue_time": 0,
                "max_queue_time": 0,
                "min_queue_time": 0,
                "total_queue_events": 0
            },
            "interaction_metrics": {
                "interaction_times": [],
                "min_interaction_time": 0,
                "max_interaction_time": 0,
                "avg_interaction_time": 0,
                "total_interactions": 0
            },
            "unattended_metrics": {
                "unattended_times": [],
                "min_unattended_time": 0,
                "max_unattended_time": 0,
                "avg_unattended_time": 0,
                "total_unattended_events": 0
            },
            "daily_summary": {},
            "hourly_breakdown": {}
        }
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        try:
            self.metrics_data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=2)
            logging.info(f"✅ Metrics saved to {self.metrics_file}")
        except Exception as e:
            logging.error(f"❌ Error saving metrics: {e}")
    
    # 1. QUEUE TIME METRICS
    def add_queue_time(self, customer_id, queue_start_time, queue_end_time):
        """Add queue time metric"""
        queue_duration = (queue_end_time - queue_start_time).total_seconds()
        
        queue_event = {
            "customer_id": customer_id,
            "queue_start": queue_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "queue_end": queue_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "queue_duration": queue_duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.metrics_data["queue_metrics"]["queue_times"].append(queue_event)
        self._update_queue_statistics()
        self.save_metrics()
        
        logging.info(f"📊 Queue time recorded: Customer {customer_id}, Duration: {queue_duration:.2f}s")
    
    def _update_queue_statistics(self):
        """Update queue statistics"""
        queue_times = [q["queue_duration"] for q in self.metrics_data["queue_metrics"]["queue_times"]]
        
        if queue_times:
            self.metrics_data["queue_metrics"]["avg_queue_time"] = round(sum(queue_times) / len(queue_times), 2)
            self.metrics_data["queue_metrics"]["max_queue_time"] = round(max(queue_times), 2)
            self.metrics_data["queue_metrics"]["min_queue_time"] = round(min(queue_times), 2)
            self.metrics_data["queue_metrics"]["total_queue_events"] = len(queue_times)
    
    # 2. INTERACTION TIME METRICS
    def add_interaction_time(self, staff_id, customer_id, interaction_start, interaction_end, video_evidence=None):
        """Add interaction time metric with optional video evidence"""
        interaction_duration = (interaction_end - interaction_start).total_seconds()
        
        interaction_event = {
            "staff_id": staff_id,
            "customer_id": customer_id,
            "interaction_start": interaction_start.strftime("%Y-%m-%d %H:%M:%S"),
            "interaction_end": interaction_end.strftime("%Y-%m-%d %H:%M:%S"),
            "interaction_duration": interaction_duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add video evidence if provided
        if video_evidence:
            interaction_event["video_evidence"] = video_evidence
            logging.info(f"🎥 Video evidence linked to interaction: Staff {staff_id} & Customer {customer_id}")
        
        self.metrics_data["interaction_metrics"]["interaction_times"].append(interaction_event)
        self._update_interaction_statistics()
        self.save_metrics()
        
        logging.info(f"🤝 Interaction time recorded: Staff {staff_id} & Customer {customer_id}, Duration: {interaction_duration:.2f}s")
    
    def _update_interaction_statistics(self):
        """Update interaction statistics"""
        interaction_times = [i["interaction_duration"] for i in self.metrics_data["interaction_metrics"]["interaction_times"]]
        
        if interaction_times:
            self.metrics_data["interaction_metrics"]["min_interaction_time"] = round(min(interaction_times), 2)
            self.metrics_data["interaction_metrics"]["max_interaction_time"] = round(max(interaction_times), 2)
            self.metrics_data["interaction_metrics"]["avg_interaction_time"] = round(sum(interaction_times) / len(interaction_times), 2)
            self.metrics_data["interaction_metrics"]["total_interactions"] = len(interaction_times)
    
    # 3. UNATTENDED TIME METRICS
    def add_unattended_time(self, customer_id, unattended_start, unattended_end, video_evidence=None):
        """Add unattended time metric with optional video evidence"""
        unattended_duration = (unattended_end - unattended_start).total_seconds()
        
        unattended_event = {
            "customer_id": customer_id,
            "unattended_start": unattended_start.strftime("%Y-%m-%d %H:%M:%S"),
            "unattended_end": unattended_end.strftime("%Y-%m-%d %H:%M:%S"),
            "unattended_duration": unattended_duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add video evidence if provided
        if video_evidence:
            unattended_event["video_evidence"] = video_evidence
            logging.info(f"🎥 Video evidence linked to unattended event: Customer {customer_id}")
        
        self.metrics_data["unattended_metrics"]["unattended_times"].append(unattended_event)
        self._update_unattended_statistics()
        self.save_metrics()
        
        logging.info(f"🚨 Unattended time recorded: Customer {customer_id}, Duration: {unattended_duration:.2f}s")
    
    def link_video_evidence_to_interaction(self, staff_id, customer_id, video_evidence):
        """Link video evidence to an existing interaction"""
        for interaction in self.metrics_data["interaction_metrics"]["interaction_times"]:
            if (interaction["staff_id"] == staff_id and 
                interaction["customer_id"] == customer_id and 
                "video_evidence" not in interaction):
                interaction["video_evidence"] = video_evidence
                logging.info(f"🔗 Video evidence linked to existing interaction: Staff {staff_id} & Customer {customer_id}")
                self.save_metrics()
                return True
        return False
    
    def link_video_evidence_to_unattended(self, customer_id, video_evidence):
        """Link video evidence to an existing unattended event"""
        for unattended in self.metrics_data["unattended_metrics"]["unattended_times"]:
            if (unattended["customer_id"] == customer_id and 
                "video_evidence" not in unattended):
                unattended["video_evidence"] = video_evidence
                logging.info(f"🔗 Video evidence linked to existing unattended event: Customer {customer_id}")
                self.save_metrics()
                return True
        return False
    
    def _update_unattended_statistics(self):
        """Update unattended statistics"""
        unattended_times = [u["unattended_duration"] for u in self.metrics_data["unattended_metrics"]["unattended_times"]]
        
        if unattended_times:
            self.metrics_data["unattended_metrics"]["min_unattended_time"] = round(min(unattended_times), 2)
            self.metrics_data["unattended_metrics"]["max_unattended_time"] = round(max(unattended_times), 2)
            self.metrics_data["unattended_metrics"]["avg_unattended_time"] = round(sum(unattended_times) / len(unattended_times), 2)
            self.metrics_data["unattended_metrics"]["total_unattended_events"] = len(unattended_times)
    
    # DAILY AND HOURLY BREAKDOWN
    def update_daily_summary(self):
        """Update daily summary metrics"""
        today = datetime.now().date().strftime("%Y-%m-%d")
        
        # Calculate today's metrics
        today_interactions = [i for i in self.metrics_data["interaction_metrics"]["interaction_times"] 
                            if i["timestamp"].startswith(today)]
        today_queue = [q for q in self.metrics_data["queue_metrics"]["queue_times"] 
                      if q["timestamp"].startswith(today)]
        today_unattended = [u for u in self.metrics_data["unattended_metrics"]["unattended_times"] 
                          if u["timestamp"].startswith(today)]
        
        self.metrics_data["daily_summary"][today] = {
            "interactions": {
                "count": len(today_interactions),
                "avg_duration": round(sum(i["interaction_duration"] for i in today_interactions) / len(today_interactions), 2) if today_interactions else 0,
                "min_duration": round(min(i["interaction_duration"] for i in today_interactions), 2) if today_interactions else 0,
                "max_duration": round(max(i["interaction_duration"] for i in today_interactions), 2) if today_interactions else 0
            },
            "queue_times": {
                "count": len(today_queue),
                "avg_duration": round(sum(q["queue_duration"] for q in today_queue) / len(today_queue), 2) if today_queue else 0,
                "min_duration": round(min(q["queue_duration"] for q in today_queue), 2) if today_queue else 0,
                "max_duration": round(max(q["queue_duration"] for q in today_queue), 2) if today_queue else 0
            },
            "unattended_times": {
                "count": len(today_unattended),
                "avg_duration": round(sum(u["unattended_duration"] for u in today_unattended) / len(today_unattended), 2) if today_unattended else 0,
                "min_duration": round(min(u["unattended_duration"] for u in today_unattended), 2) if today_unattended else 0,
                "max_duration": round(max(u["unattended_duration"] for u in today_unattended), 2) if today_unattended else 0
            }
        }
        
        # Update hourly breakdown for peak hours analysis
        self._update_hourly_breakdown()
        
        self.save_metrics()
    
    def _update_hourly_breakdown(self):
        """Update hourly breakdown for peak hours analysis"""
        today = datetime.now().date().strftime("%Y-%m-%d")
        
        # Initialize hourly breakdown for today
        if today not in self.metrics_data["hourly_breakdown"]:
            self.metrics_data["hourly_breakdown"][today] = {}
        
        # Get today's data
        today_interactions = [i for i in self.metrics_data["interaction_metrics"]["interaction_times"] 
                            if i["timestamp"].startswith(today)]
        today_queue = [q for q in self.metrics_data["queue_metrics"]["queue_times"] 
                      if q["timestamp"].startswith(today)]
        today_unattended = [u for u in self.metrics_data["unattended_metrics"]["unattended_times"] 
                          if u["timestamp"].startswith(today)]
        
        # Group by hour
        hourly_data = defaultdict(lambda: {
            "interactions": {"count": 0, "total_duration": 0},
            "queue_times": {"count": 0, "total_duration": 0},
            "unattended_times": {"count": 0, "total_duration": 0}
        })
        
        # Process interactions by hour
        for interaction in today_interactions:
            hour = interaction["timestamp"][:13] + ":00:00"  # Extract hour
            hourly_data[hour]["interactions"]["count"] += 1
            hourly_data[hour]["interactions"]["total_duration"] += interaction["interaction_duration"]
        
        # Process queue times by hour
        for queue in today_queue:
            hour = queue["timestamp"][:13] + ":00:00"  # Extract hour
            hourly_data[hour]["queue_times"]["count"] += 1
            hourly_data[hour]["queue_times"]["total_duration"] += queue["queue_duration"]
        
        # Process unattended times by hour
        for unattended in today_unattended:
            hour = unattended["timestamp"][:13] + ":00:00"  # Extract hour
            hourly_data[hour]["unattended_times"]["count"] += 1
            hourly_data[hour]["unattended_times"]["total_duration"] += unattended["unattended_duration"]
        
        # Calculate averages and store
        for hour, data in hourly_data.items():
            self.metrics_data["hourly_breakdown"][today][hour] = {
                "interactions": {
                    "count": data["interactions"]["count"],
                    "avg_duration": round(data["interactions"]["total_duration"] / data["interactions"]["count"], 2) if data["interactions"]["count"] > 0 else 0
                },
                "queue_times": {
                    "count": data["queue_times"]["count"],
                    "avg_duration": round(data["queue_times"]["total_duration"] / data["queue_times"]["count"], 2) if data["queue_times"]["count"] > 0 else 0
                },
                "unattended_times": {
                    "count": data["unattended_times"]["count"],
                    "avg_duration": round(data["unattended_times"]["total_duration"] / data["unattended_times"]["count"], 2) if data["unattended_times"]["count"] > 0 else 0
                }
            }
    
    def get_metrics_summary(self):
        """Get current metrics summary"""
        return {
            "queue_metrics": self.metrics_data["queue_metrics"],
            "interaction_metrics": self.metrics_data["interaction_metrics"],
            "unattended_metrics": self.metrics_data["unattended_metrics"],
            "last_updated": self.metrics_data["metadata"]["last_updated"]
        }
    
    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "queue_metrics": self.metrics_data["queue_metrics"],
            "interaction_metrics": self.metrics_data["interaction_metrics"],
            "unattended_metrics": self.metrics_data["unattended_metrics"],
            "daily_summary": self.metrics_data["daily_summary"],
            "hourly_breakdown": self.metrics_data["hourly_breakdown"],
            "peak_hours_analysis": self.get_peak_hours_analysis(),
            "summary": {
                "total_queue_events": self.metrics_data["queue_metrics"]["total_queue_events"],
                "total_interactions": self.metrics_data["interaction_metrics"]["total_interactions"],
                "total_unattended_events": self.metrics_data["unattended_metrics"]["total_unattended_events"],
                "last_updated": self.metrics_data["metadata"]["last_updated"]
            }
        }
    
    def get_peak_hours_analysis(self):
        """Get peak hours analysis based on hourly breakdown"""
        today = datetime.now().date().strftime("%Y-%m-%d")
        
        if today not in self.metrics_data["hourly_breakdown"]:
            return {"message": "No hourly data available yet"}
        
        hourly_data = self.metrics_data["hourly_breakdown"][today]
        
        # Find peak hours for each metric
        peak_interactions = self._find_peak_hours(hourly_data, "interactions")
        peak_queue = self._find_peak_hours(hourly_data, "queue_times")
        peak_unattended = self._find_peak_hours(hourly_data, "unattended_times")
        
        return {
            "peak_interaction_hours": peak_interactions,
            "peak_queue_hours": peak_queue,
            "peak_unattended_hours": peak_unattended,
            "analysis_date": today
        }
    
    def _find_peak_hours(self, hourly_data, metric_type):
        """Find peak hours for a specific metric type"""
        if not hourly_data:
            return []
        
        # Get all hours with data for this metric
        hours_with_data = []
        for hour, data in hourly_data.items():
            if data.get(metric_type, {}).get("count", 0) > 0:
                hours_with_data.append({
                    "hour": hour,
                    "count": data[metric_type]["count"],
                    "avg_duration": data[metric_type]["avg_duration"]
                })
        
        if not hours_with_data:
            return []
        
        # Sort by count (descending) to find peak hours
        hours_with_data.sort(key=lambda x: x["count"], reverse=True)
        
        # Return top 3 peak hours
        return hours_with_data[:3]
