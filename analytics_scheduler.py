#!/usr/bin/env python3
"""
Analytics Scheduler - Updates analytics JSON every 10 minutes
This ensures continuous data collection and accumulation for analysis.
"""

import time
import threading
import logging
import signal
import sys
from datetime import datetime, timedelta
from metrics_collector import MetricsCollector
from interaction import InteractionLogger

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnalyticsScheduler:
    def __init__(self, update_interval_minutes=10):
        self.update_interval = update_interval_minutes * 60  # Convert to seconds
        self.running = False
        self.scheduler_thread = None
        self.metrics_collector = None  # Will be set by set_interaction_logger
        self.interaction_logger = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info(f"📊 Analytics Scheduler initialized (update interval: {update_interval_minutes} minutes)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info(f"🛑 Received signal {signum}, shutting down analytics scheduler...")
        self.stop()

    def set_interaction_logger(self, interaction_logger):
        """Set the interaction logger reference for data collection."""
        self.interaction_logger = interaction_logger
        # Use the same MetricsCollector instance from InteractionLogger
        self.metrics_collector = interaction_logger.metrics_collector
        logging.info("🔗 Interaction logger connected to analytics scheduler")

    def _update_analytics_data(self):
        """Update analytics data with current metrics."""
        try:
            if not self.metrics_collector:
                logging.error("❌ Metrics collector not available - analytics scheduler not properly initialized")
                return False
                
            current_time = datetime.now()
            logging.info(f"📊 Updating analytics data at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Update daily summary
            self.metrics_collector.update_daily_summary()
            
            # Save metrics to JSON file
            self.metrics_collector.save_metrics()
            
            # Get current metrics summary
            summary = self.metrics_collector.get_metrics_summary()
            
            # Extract metrics from nested structure
            interactions = summary.get('interaction_metrics', {}).get('total_interactions', 0)
            queue_events = summary.get('queue_metrics', {}).get('total_queue_events', 0)
            unattended_events = summary.get('unattended_metrics', {}).get('total_unattended_events', 0)
            
            logging.info(f"📈 Analytics updated - Interactions: {interactions}, "
                        f"Queue events: {queue_events}, "
                        f"Unattended events: {unattended_events}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error updating analytics data: {e}")
            return False

    def _scheduler_loop(self):
        """Main scheduler loop that runs every 10 minutes."""
        logging.info("🔄 Analytics scheduler started")
        
        while self.running:
            try:
                # Wait for the update interval
                time.sleep(self.update_interval)
                
                if self.running:  # Check if we should still be running
                    self._update_analytics_data()
                    
            except Exception as e:
                logging.error(f"❌ Error in scheduler loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def start(self):
        """Start the analytics scheduler."""
        if self.running:
            logging.warning("⚠️ Analytics scheduler is already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logging.info("🚀 Analytics scheduler started - will update every 10 minutes")

    def stop(self):
        """Stop the analytics scheduler."""
        if not self.running:
            logging.warning("⚠️ Analytics scheduler is not running")
            return
        
        self.running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # Final update before stopping
        self._update_analytics_data()
        
        logging.info("🛑 Analytics scheduler stopped")

    def force_update(self):
        """Force an immediate update of analytics data."""
        logging.info("🔄 Forcing immediate analytics update...")
        return self._update_analytics_data()

    def get_status(self):
        """Get current scheduler status."""
        return {
            "running": self.running,
            "update_interval_minutes": self.update_interval // 60,
            "next_update_in_seconds": self.update_interval if self.running else 0,
            "metrics_file": self.metrics_collector.metrics_file if self.metrics_collector else "Not initialized"
        }

def create_analytics_scheduler(update_interval_minutes=10):
    """Create and return an analytics scheduler instance."""
    return AnalyticsScheduler(update_interval_minutes)

def main():
    """Standalone analytics scheduler for testing."""
    print("🚀 Starting Standalone Analytics Scheduler")
    print("=" * 50)
    
    # Create scheduler
    scheduler = AnalyticsScheduler(update_interval_minutes=10)
    
    try:
        # Start scheduler
        scheduler.start()
        
        print("✅ Analytics scheduler started")
        print("📊 Will update analytics JSON every 10 minutes")
        print("🔄 Press Ctrl+C to stop")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping analytics scheduler...")
        scheduler.stop()
        print("✅ Analytics scheduler stopped")

if __name__ == "__main__":
    main()
