"""
Hourly Interaction Data Aggregator for CSI System

This module aggregates interaction data over hourly periods and sends
meaningful analytics to the footfall API instead of individual interactions.
"""

import time
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from footfall import report_daily_interaction_summary

class HourlyInteractionAggregator:
    def __init__(self, aggregation_interval_hours: float = 1.0):
        """
        Initialize the hourly aggregator.
        
        Args:
            aggregation_interval_hours: How often to send aggregated data (default: 1 hour)
        """
        self.aggregation_interval = aggregation_interval_hours * 3600  # Convert to seconds
        self.last_aggregation_time = time.time()
        
        # Hourly data storage
        self.hourly_data = {
            'total_interactions': 0,
            'total_duration_seconds': 0.0,
            'interaction_count_by_staff': {},  # staff_id -> count
            'duration_by_staff': {},          # staff_id -> total_duration
            'interaction_count_by_customer': {},  # customer_id -> count
            'duration_by_customer': {},       # customer_id -> total_duration
            'interaction_timestamps': [],     # List of (timestamp, duration) tuples
            'peak_hour_data': {},            # hour -> interaction_count
            'start_time': time.time(),
            'current_hour': self._get_current_hour()
        }
        
        # Local storage file for persistence
        self.storage_file = "hourly_interaction_data.json"
        self._load_persistent_data()
        
        logging.info("🕐 Hourly Interaction Aggregator initialized")
        if aggregation_interval_hours < 1.0:
            interval_minutes = aggregation_interval_hours * 60
            logging.info(f"   - Aggregation interval: {interval_minutes:.1f} minutes")
        else:
            logging.info(f"   - Aggregation interval: {aggregation_interval_hours} hours")
        logging.info(f"   - Storage file: {self.storage_file}")
        logging.info(f"   - Current hour: {self.hourly_data['current_hour']}")

    def _get_current_hour(self) -> int:
        """Get current hour (0-23)."""
        return datetime.now().hour

    def _load_persistent_data(self):
        """Load persistent data from file if it exists."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    saved_data = json.load(f)
                    
                # Check if data is from the same hour
                if saved_data.get('current_hour') == self._get_current_hour():
                    self.hourly_data.update(saved_data)
                    logging.info(f"📂 Loaded persistent data for hour {self.hourly_data['current_hour']}")
                    logging.info(f"   - Total interactions: {self.hourly_data['total_interactions']}")
                    logging.info(f"   - Total duration: {self.hourly_data['total_duration_seconds']:.1f}s")
                else:
                    logging.info("🆕 New hour detected, starting fresh aggregation")
        except Exception as e:
            logging.warning(f"⚠️ Could not load persistent data: {e}")

    def _save_persistent_data(self):
        """Save current data to file for persistence."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.hourly_data, f, indent=2)
        except Exception as e:
            logging.error(f"❌ Could not save persistent data: {e}")

    def add_interaction(self, staff_id: str, customer_id: str, duration_seconds: float):
        """
        Add a completed interaction to the hourly aggregation.
        
        Args:
            staff_id: ID of the staff member
            customer_id: ID of the customer
            duration_seconds: Duration of the interaction
        """
        current_time = time.time()
        current_hour = self._get_current_hour()
        
        # Check if we've moved to a new hour
        if current_hour != self.hourly_data['current_hour']:
            self._finalize_current_hour()
            self._start_new_hour(current_hour)
        
        # Add interaction data
        self.hourly_data['total_interactions'] += 1
        self.hourly_data['total_duration_seconds'] += duration_seconds
        
        # Track by staff
        if staff_id not in self.hourly_data['interaction_count_by_staff']:
            self.hourly_data['interaction_count_by_staff'][staff_id] = 0
            self.hourly_data['duration_by_staff'][staff_id] = 0.0
        
        self.hourly_data['interaction_count_by_staff'][staff_id] += 1
        self.hourly_data['duration_by_staff'][staff_id] += duration_seconds
        
        # Track by customer
        if customer_id not in self.hourly_data['interaction_count_by_customer']:
            self.hourly_data['interaction_count_by_customer'][customer_id] = 0
            self.hourly_data['duration_by_customer'][customer_id] = 0.0
        
        self.hourly_data['interaction_count_by_customer'][customer_id] += 1
        self.hourly_data['duration_by_customer'][customer_id] += duration_seconds
        
        # Track timestamps for peak hour analysis
        self.hourly_data['interaction_timestamps'].append((current_time, duration_seconds))
        
        # Track peak hour data
        if current_hour not in self.hourly_data['peak_hour_data']:
            self.hourly_data['peak_hour_data'][current_hour] = 0
        self.hourly_data['peak_hour_data'][current_hour] += 1
        
        # Save persistent data
        self._save_persistent_data()
        
        logging.info("📊 Added interaction to hourly aggregation:")
        logging.info(f"   - Staff: {staff_id}, Customer: {customer_id}")
        logging.info(f"   - Duration: {duration_seconds:.1f}s")
        logging.info(f"   - Hourly total: {self.hourly_data['total_interactions']} interactions, {self.hourly_data['total_duration_seconds']:.1f}s")

    def _start_new_hour(self, new_hour: int):
        """Start aggregation for a new hour."""
        self.hourly_data = {
            'total_interactions': 0,
            'total_duration_seconds': 0.0,
            'interaction_count_by_staff': {},
            'duration_by_staff': {},
            'interaction_count_by_customer': {},
            'duration_by_customer': {},
            'interaction_timestamps': [],
            'peak_hour_data': {},
            'start_time': time.time(),
            'current_hour': new_hour
        }
        logging.info(f"🆕 Started new hour aggregation for hour {new_hour}")

    def _clear_hourly_data(self):
        """Clear current hour's data to optimize memory after successful reporting."""
        # Reset counters but keep the hour structure
        self.hourly_data['total_interactions'] = 0
        self.hourly_data['total_duration_seconds'] = 0.0
        self.hourly_data['interaction_count_by_staff'].clear()
        self.hourly_data['duration_by_staff'].clear()
        self.hourly_data['interaction_count_by_customer'].clear()
        self.hourly_data['duration_by_customer'].clear()
        self.hourly_data['interaction_timestamps'].clear()
        self.hourly_data['peak_hour_data'].clear()
        self.hourly_data['start_time'] = time.time()
        
        # Remove persistent data file to free up disk space
        try:
            if os.path.exists(self.storage_file):
                os.remove(self.storage_file)
                logging.info(f"🗑️ Removed persistent data file: {self.storage_file}")
        except Exception as e:
            logging.warning(f"⚠️ Could not remove persistent data file: {e}")
        
        logging.info("🧹 Cleared all hourly data for memory optimization")

    def _finalize_current_hour(self):
        """Finalize and send data for the current hour."""
        if self.hourly_data['total_interactions'] > 0:
            self._send_hourly_analytics()
        else:
            logging.info(f"⏭️ No interactions in hour {self.hourly_data['current_hour']}, skipping analytics")

    def _send_hourly_analytics(self):
        """Send hourly analytics to the footfall API."""
        try:
            # Calculate analytics
            total_interactions = self.hourly_data['total_interactions']
            total_duration = self.hourly_data['total_duration_seconds']
            avg_duration = total_duration / total_interactions if total_interactions > 0 else 0
            
            # Find peak hour
            peak_hour = max(self.hourly_data['peak_hour_data'].items(), key=lambda x: x[1])[0] if self.hourly_data['peak_hour_data'] else None
            
            # Calculate staff performance metrics
            staff_performance = self._calculate_staff_performance()
            
            # Send to footfall API
            success = report_daily_interaction_summary(
                total_interactions=total_interactions,
                total_duration=total_duration,
                avg_duration=avg_duration,
                peak_hour=peak_hour,
                staff_performance=staff_performance
            )
            
            if success:
                logging.info(f"✅ Sent analytics data for hour {self.hourly_data['current_hour']}:")
                logging.info(f"   - Total interactions: {total_interactions}")
                logging.info(f"   - Total duration: {total_duration/60:.1f} minutes")
                logging.info(f"   - Average duration: {avg_duration:.1f}s")
                logging.info(f"   - Peak hour: {peak_hour}")
                logging.info(f"   - Staff performance: {staff_performance}")
                
                # Clear data after successful reporting to optimize memory
                self._clear_hourly_data()
                logging.info("🧹 Cleared data after successful reporting")
            else:
                logging.error(f"❌ Failed to send analytics data for hour {self.hourly_data['current_hour']}")
                # Keep data for retry on next check
                
        except Exception as e:
            logging.error(f"❌ Error sending hourly analytics: {e}")

    def _calculate_staff_performance(self) -> Dict:
        """Calculate staff performance metrics."""
        performance = {}
        
        for staff_id, count in self.hourly_data['interaction_count_by_staff'].items():
            total_duration = self.hourly_data['duration_by_staff'][staff_id]
            avg_duration = total_duration / count if count > 0 else 0
            
            performance[staff_id] = {
                'interaction_count': count,
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'efficiency_score': count * avg_duration  # Simple efficiency metric
            }
        
        return performance

    def check_and_send_aggregated_data(self):
        """
        Check if it's time to send aggregated data and send if needed.
        This should be called periodically from the main loop.
        """
        current_time = time.time()
        
        # Check if we've moved to a new hour
        current_hour = self._get_current_hour()
        if current_hour != self.hourly_data['current_hour']:
            self._finalize_current_hour()
            self._start_new_hour(current_hour)
        
        # Check if aggregation interval has passed
        elif current_time - self.last_aggregation_time >= self.aggregation_interval:
            if self.hourly_data['total_interactions'] > 0:
                self._send_hourly_analytics()
                self.last_aggregation_time = current_time
            else:
                logging.info(f"⏭️ No interactions to aggregate in current hour {self.hourly_data['current_hour']}")

    def get_current_hour_stats(self) -> Dict:
        """Get current hour statistics."""
        return {
            'current_hour': self.hourly_data['current_hour'],
            'total_interactions': self.hourly_data['total_interactions'],
            'total_duration_seconds': self.hourly_data['total_duration_seconds'],
            'total_duration_minutes': self.hourly_data['total_duration_seconds'] / 60,
            'avg_duration': self.hourly_data['total_duration_seconds'] / max(1, self.hourly_data['total_interactions']),
            'staff_count': len(self.hourly_data['interaction_count_by_staff']),
            'customer_count': len(self.hourly_data['interaction_count_by_customer']),
            'peak_hour': max(self.hourly_data['peak_hour_data'].items(), key=lambda x: x[1])[0] if self.hourly_data['peak_hour_data'] else None
        }

    def force_send_current_data(self):
        """Force send current hour's data (useful for shutdown)."""
        if self.hourly_data['total_interactions'] > 0:
            logging.info("🔄 Force sending current hour's aggregated data...")
            self._send_hourly_analytics()
            # Data will be cleared automatically after successful sending
        else:
            logging.info("⏭️ No data to send in current hour")

    def cleanup_old_data(self, hours_to_keep: int = 24):
        """Clean up old data files (keep last 24 hours by default)."""
        try:
            if os.path.exists(self.storage_file):
                file_age_hours = (time.time() - os.path.getmtime(self.storage_file)) / 3600
                if file_age_hours > hours_to_keep:
                    os.remove(self.storage_file)
                    logging.info(f"🗑️ Cleaned up old aggregation data file (age: {file_age_hours:.1f} hours)")
        except Exception as e:
            logging.error(f"❌ Error cleaning up old data: {e}")

    def get_memory_usage_stats(self) -> Dict:
        """Get memory usage statistics for monitoring."""
        import sys
        
        # Calculate approximate memory usage
        total_interactions = self.hourly_data['total_interactions']
        staff_count = len(self.hourly_data['interaction_count_by_staff'])
        customer_count = len(self.hourly_data['interaction_count_by_customer'])
        timestamp_count = len(self.hourly_data['interaction_timestamps'])
        
        # Estimate memory usage (rough calculation)
        estimated_memory_kb = (
            total_interactions * 0.1 +  # Basic interaction data
            staff_count * 0.05 +        # Staff tracking
            customer_count * 0.05 +     # Customer tracking
            timestamp_count * 0.02      # Timestamp data
        )
        
        return {
            'total_interactions': total_interactions,
            'staff_count': staff_count,
            'customer_count': customer_count,
            'timestamp_count': timestamp_count,
            'estimated_memory_kb': round(estimated_memory_kb, 2),
            'data_file_exists': os.path.exists(self.storage_file),
            'data_file_size_kb': round(os.path.getsize(self.storage_file) / 1024, 2) if os.path.exists(self.storage_file) else 0
        }
