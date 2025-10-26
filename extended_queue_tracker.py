#!/usr/bin/env python3
"""
Extended Queue Tracker for Hours-Long Scenarios
Handles multiple customers waiting for extended periods with proper queue management
"""

import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import json

class ExtendedQueueTracker:
    def __init__(self, max_queue_capacity=50, cleanup_interval=300):
        """
        Initialize extended queue tracker for long-term scenarios
        
        Args:
            max_queue_capacity: Maximum number of customers in queue
            cleanup_interval: Interval for memory cleanup (seconds)
        """
        self.max_queue_capacity = max_queue_capacity
        self.cleanup_interval = cleanup_interval
        
        # Queue position tracking
        self.queue_positions = {}  # customer_id -> position
        self.queue_order = deque()  # Ordered queue of customer IDs
        self.queue_overflow = deque()  # Customers beyond capacity
        
        # Extended tracking data
        self.extended_queue_customers = {}  # customer_id -> extended data
        self.queue_start_times = {}  # customer_id -> queue start time
        self.queue_validation_times = {}  # customer_id -> validation start time
        
        # Performance tracking
        self.service_times = deque(maxlen=100)  # Last 100 service times
        self.queue_throughput = 0.0  # Customers served per hour
        self.last_cleanup = time.time()
        
        # Analytics
        self.queue_analytics = {
            "total_customers_served": 0,
            "total_wait_time": 0.0,
            "peak_queue_length": 0,
            "overflow_events": 0
        }
        
        logging.info(f"🎯 Extended Queue Tracker initialized (capacity: {max_queue_capacity})")
    
    def add_customer_to_queue(self, customer_id: str, position: Optional[int] = None) -> Dict:
        """
        Add customer to queue with position tracking
        
        Args:
            customer_id: Unique customer identifier
            position: Optional specific position in queue
            
        Returns:
            Dict with queue status and position
        """
        current_time = datetime.now()
        
        # Check if customer is already in queue
        if customer_id in self.queue_order:
            return {
                "status": "already_in_queue",
                "position": self.queue_positions[customer_id],
                "message": f"Customer {customer_id} already in queue at position {self.queue_positions[customer_id]}"
            }
        
        # Check queue capacity
        if len(self.queue_order) >= self.max_queue_capacity:
            # Queue overflow - add to overflow list
            self.queue_overflow.append(customer_id)
            self.queue_analytics["overflow_events"] += 1
            
            logging.warning(f"⚠️ Queue overflow: Customer {customer_id} added to overflow (capacity: {self.max_queue_capacity})")
            
            return {
                "status": "overflow",
                "position": len(self.queue_order) + len(self.queue_overflow),
                "message": f"Customer {customer_id} added to overflow queue"
            }
        else:
            # Normal queue position
            if position is None:
                position = len(self.queue_order) + 1
            
            self.queue_order.append(customer_id)
            self.queue_positions[customer_id] = position
            self.queue_start_times[customer_id] = current_time
            
            # Initialize extended tracking data
            self.extended_queue_customers[customer_id] = {
                "first_seen": current_time,
                "queue_start": current_time,
                "position_history": [position],
                "wait_time_segments": [],
                "last_activity": current_time
            }
            
            # Update peak queue length
            current_length = len(self.queue_order)
            if current_length > self.queue_analytics["peak_queue_length"]:
                self.queue_analytics["peak_queue_length"] = current_length
            
            logging.info(f"📊 Customer {customer_id} added to queue position {position} (queue length: {current_length})")
            
            return {
                "status": "added",
                "position": position,
                "message": f"Customer {customer_id} added to queue at position {position}"
            }
    
    def remove_customer_from_queue(self, customer_id: str) -> Dict:
        """
        Remove customer from queue and update positions
        
        Args:
            customer_id: Customer to remove from queue
            
        Returns:
            Dict with removal status and updated positions
        """
        if customer_id not in self.queue_order:
            # Check if in overflow
            if customer_id in self.queue_overflow:
                self.queue_overflow.remove(customer_id)
                logging.info(f"📤 Customer {customer_id} removed from overflow queue")
                return {"status": "removed_from_overflow", "message": f"Customer {customer_id} removed from overflow"}
            else:
                return {"status": "not_found", "message": f"Customer {customer_id} not in queue"}
        
        # Calculate wait time
        wait_time = 0.0
        if customer_id in self.queue_start_times:
            wait_time = (datetime.now() - self.queue_start_times[customer_id]).total_seconds()
            self.queue_analytics["total_wait_time"] += wait_time
            self.queue_analytics["total_customers_served"] += 1
        
        # Remove from queue
        self.queue_order.remove(customer_id)
        del self.queue_positions[customer_id]
        del self.queue_start_times[customer_id]
        
        # Clean up extended data
        if customer_id in self.extended_queue_customers:
            del self.extended_queue_customers[customer_id]
        
        # Update positions for remaining customers
        self._update_queue_positions()
        
        # Promote from overflow if space available
        promoted_customer = None
        if self.queue_overflow and len(self.queue_order) < self.max_queue_capacity:
            promoted_customer = self.queue_overflow.popleft()
            self.add_customer_to_queue(promoted_customer)
            logging.info(f"📈 Customer {promoted_customer} promoted from overflow to position {self.queue_positions[promoted_customer]}")
        
        logging.info(f"✅ Customer {customer_id} removed from queue (wait time: {wait_time:.2f}s)")
        
        return {
            "status": "removed",
            "wait_time": wait_time,
            "promoted_customer": promoted_customer,
            "message": f"Customer {customer_id} removed from queue after {wait_time:.2f}s wait"
        }
    
    def _update_queue_positions(self):
        """Update positions for all customers in queue"""
        for i, customer_id in enumerate(self.queue_order, 1):
            self.queue_positions[customer_id] = i
            
            # Update position history
            if customer_id in self.extended_queue_customers:
                self.extended_queue_customers[customer_id]["position_history"].append(i)
                # Keep only last 20 positions
                if len(self.extended_queue_customers[customer_id]["position_history"]) > 20:
                    self.extended_queue_customers[customer_id]["position_history"] = \
                        self.extended_queue_customers[customer_id]["position_history"][-20:]
    
    def get_queue_status(self) -> Dict:
        """Get current queue status and analytics"""
        current_time = datetime.now()
        
        # Calculate current wait times
        current_wait_times = {}
        for customer_id in self.queue_order:
            if customer_id in self.queue_start_times:
                wait_time = (current_time - self.queue_start_times[customer_id]).total_seconds()
                current_wait_times[customer_id] = wait_time
        
        # Calculate average service time
        avg_service_time = sum(self.service_times) / len(self.service_times) if self.service_times else 0
        
        return {
            "queue_length": len(self.queue_order),
            "overflow_count": len(self.queue_overflow),
            "capacity_utilization": len(self.queue_order) / self.max_queue_capacity,
            "current_positions": dict(self.queue_positions),
            "current_wait_times": current_wait_times,
            "average_wait_time": sum(current_wait_times.values()) / len(current_wait_times) if current_wait_times else 0,
            "longest_wait_time": max(current_wait_times.values()) if current_wait_times else 0,
            "estimated_wait_times": self._calculate_estimated_wait_times(),
            "queue_analytics": self.queue_analytics.copy(),
            "average_service_time": avg_service_time,
            "queue_throughput": self.queue_throughput
        }
    
    def _calculate_estimated_wait_times(self) -> Dict[str, float]:
        """Calculate estimated wait times for all customers in queue"""
        estimated_times = {}
        
        if not self.service_times:
            return estimated_times
        
        avg_service_time = sum(self.service_times) / len(self.service_times)
        
        for customer_id, position in self.queue_positions.items():
            estimated_time = (position - 1) * avg_service_time
            estimated_times[customer_id] = estimated_time
        
        return estimated_times
    
    def get_customer_queue_info(self, customer_id: str) -> Dict:
        """Get detailed information for a specific customer"""
        if customer_id not in self.queue_order:
            return {"status": "not_in_queue", "message": f"Customer {customer_id} not in queue"}
        
        current_time = datetime.now()
        position = self.queue_positions[customer_id]
        
        # Calculate wait time
        wait_time = 0.0
        if customer_id in self.queue_start_times:
            wait_time = (current_time - self.queue_start_times[customer_id]).total_seconds()
        
        # Get extended data
        extended_data = self.extended_queue_customers.get(customer_id, {})
        
        # Calculate estimated wait time
        estimated_wait = 0.0
        if self.service_times:
            avg_service_time = sum(self.service_times) / len(self.service_times)
            estimated_wait = (position - 1) * avg_service_time
        
        return {
            "customer_id": customer_id,
            "position": position,
            "wait_time": wait_time,
            "estimated_wait_time": estimated_wait,
            "queue_length": len(self.queue_order),
            "position_history": extended_data.get("position_history", []),
            "first_seen": extended_data.get("first_seen", None),
            "last_activity": extended_data.get("last_activity", None)
        }
    
    def cleanup_extended_data(self):
        """Clean up extended queue data to prevent memory issues"""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # Clean up old position history
        for customer_id in list(self.extended_queue_customers.keys()):
            customer_data = self.extended_queue_customers[customer_id]
            
            # Keep only last 20 positions
            if len(customer_data.get("position_history", [])) > 20:
                customer_data["position_history"] = customer_data["position_history"][-20:]
            
            # Archive very old data (1+ hours)
            if (current_time - customer_data["first_seen"].timestamp()) > 3600:
                self._archive_customer_data(customer_id)
        
        # Clean up old service times
        if len(self.service_times) > 100:
            # Keep only last 100 service times
            self.service_times = deque(list(self.service_times)[-100:], maxlen=100)
        
        self.last_cleanup = current_time
        logging.debug("🧹 Extended queue data cleaned up")
    
    def _archive_customer_data(self, customer_id: str):
        """Archive old customer data to prevent memory growth"""
        if customer_id in self.extended_queue_customers:
            # Archive to analytics instead of keeping in memory
            archived_data = self.extended_queue_customers[customer_id]
            self.queue_analytics[f"archived_{customer_id}"] = {
                "total_wait_time": (datetime.now() - archived_data["first_seen"]).total_seconds(),
                "position_changes": len(archived_data.get("position_history", [])),
                "archived_at": datetime.now().isoformat()
            }
            
            del self.extended_queue_customers[customer_id]
            logging.debug(f"📦 Archived data for customer {customer_id}")
    
    def get_extended_analytics(self) -> Dict:
        """Get comprehensive analytics for extended queue scenarios"""
        current_time = datetime.now()
        
        # Calculate queue efficiency metrics
        total_customers = self.queue_analytics["total_customers_served"]
        total_wait_time = self.queue_analytics["total_wait_time"]
        avg_wait_time = total_wait_time / total_customers if total_customers > 0 else 0
        
        # Calculate queue throughput
        if self.service_times:
            avg_service_time = sum(self.service_times) / len(self.service_times)
            self.queue_throughput = 3600 / avg_service_time if avg_service_time > 0 else 0
        
        return {
            "queue_performance": {
                "current_length": len(self.queue_order),
                "overflow_count": len(self.queue_overflow),
                "capacity_utilization": len(self.queue_order) / self.max_queue_capacity,
                "peak_length": self.queue_analytics["peak_queue_length"],
                "overflow_events": self.queue_analytics["overflow_events"]
            },
            "wait_time_metrics": {
                "average_wait_time": avg_wait_time,
                "total_customers_served": total_customers,
                "total_wait_time": total_wait_time,
                "current_wait_times": {
                    customer_id: (current_time - start_time).total_seconds()
                    for customer_id, start_time in self.queue_start_times.items()
                }
            },
            "service_metrics": {
                "average_service_time": sum(self.service_times) / len(self.service_times) if self.service_times else 0,
                "queue_throughput": self.queue_throughput,
                "service_efficiency": 1.0 / (sum(self.service_times) / len(self.service_times)) if self.service_times else 0
            },
            "queue_positions": dict(self.queue_positions),
            "estimated_wait_times": self._calculate_estimated_wait_times(),
            "timestamp": current_time.isoformat()
        }
    
    def record_service_time(self, service_duration: float):
        """Record service time for analytics"""
        self.service_times.append(service_duration)
        logging.debug(f"📊 Service time recorded: {service_duration:.2f}s")
    
    def export_analytics(self, filename: str = None) -> str:
        """Export comprehensive analytics to JSON file"""
        if filename is None:
            filename = f"extended_queue_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        analytics = self.get_extended_analytics()
        
        with open(filename, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)
        
        logging.info(f"📊 Extended queue analytics exported to {filename}")
        return filename
