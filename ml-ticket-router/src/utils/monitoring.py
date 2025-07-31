"""
Monitoring module for tracking system metrics and performance.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import numpy as np
import logging
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = datetime.now()
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Prediction metrics
        self.predictions = defaultdict(int)
        self.confidence_scores = deque(maxlen=10000)
        
        # Feedback metrics
        self.feedback_counts = {
            'positive': 0,
            'negative': 0,
            'correction': 0
        }
        
        # Performance metrics
        self.response_times = deque(maxlen=10000)
        self.error_count = 0
        
        # Time series data
        self.hourly_metrics = defaultdict(lambda: defaultdict(list))
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        logger.info("MetricsCollector initialized")
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.prediction_counter = Counter(
            'ticket_routing_predictions_total',
            'Total number of routing predictions',
            ['assigned_team']
        )
        
        self.feedback_counter = Counter(
            'ticket_routing_feedback_total',
            'Total feedback received',
            ['feedback_type']
        )
        
        self.error_counter = Counter(
            'ticket_routing_errors_total',
            'Total number of errors'
        )
        
        # Histograms
        self.response_time_histogram = Histogram(
            'ticket_routing_response_time_seconds',
            'Response time in seconds',
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.confidence_histogram = Histogram(
            'ticket_routing_confidence',
            'Prediction confidence scores',
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
        )
        
        # Gauges
        self.active_requests_gauge = Gauge(
            'ticket_routing_active_requests',
            'Number of active requests'
        )
        
        self.model_accuracy_gauge = Gauge(
            'ticket_routing_model_accuracy',
            'Current model accuracy'
        )
        
        # Summary
        self.processing_time_summary = Summary(
            'ticket_routing_processing_time_ms',
            'Processing time in milliseconds'
        )
        
    def record_prediction(self, assigned_team: str, confidence: float):
        """Record a routing prediction."""
        self.total_requests += 1
        self.predictions[assigned_team] += 1
        self.confidence_scores.append(confidence)
        
        # Update Prometheus metrics
        self.prediction_counter.labels(assigned_team=assigned_team).inc()
        self.confidence_histogram.observe(confidence)
        
        # Record hourly metrics
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.hourly_metrics[current_hour]['predictions'].append({
            'team': assigned_team,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
    def record_feedback(self, feedback_type: str):
        """Record feedback."""
        self.feedback_counts[feedback_type] += 1
        self.feedback_counter.labels(feedback_type=feedback_type).inc()
        
        # Record hourly metrics
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.hourly_metrics[current_hour]['feedback'].append({
            'type': feedback_type,
            'timestamp': datetime.now()
        })
        
    def record_response_time(self, response_time_ms: float):
        """Record API response time."""
        self.response_times.append(response_time_ms)
        self.response_time_histogram.observe(response_time_ms / 1000)  # Convert to seconds
        self.processing_time_summary.observe(response_time_ms)
        
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        self.error_counter.inc()
        
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
        
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
        
    def update_model_accuracy(self, accuracy: float):
        """Update model accuracy metric."""
        self.model_accuracy_gauge.set(accuracy)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0
        
        # Calculate accuracy from feedback
        total_feedback = sum(self.feedback_counts.values())
        if total_feedback > 0:
            accuracy = self.feedback_counts['positive'] / total_feedback
        else:
            accuracy = 0.0
            
        # Calculate average confidence
        avg_confidence = np.mean(self.confidence_scores) if self.confidence_scores else 0.0
        
        # Calculate response time statistics
        if self.response_times:
            response_time_stats = {
                'mean': np.mean(self.response_times),
                'median': np.median(self.response_times),
                'p95': np.percentile(self.response_times, 95),
                'p99': np.percentile(self.response_times, 99)
            }
        else:
            response_time_stats = {'mean': 0, 'median': 0, 'p95': 0, 'p99': 0}
            
        return {
            'total_predictions': self.total_requests,
            'accuracy': accuracy,
            'precision': accuracy,  # Simplified for demo
            'recall': accuracy,     # Simplified for demo
            'f1_score': accuracy,   # Simplified for demo
            'average_confidence': avg_confidence,
            'cache_hit_rate': cache_hit_rate,
            'error_rate': self.error_count / self.total_requests if self.total_requests > 0 else 0,
            'response_time_ms': response_time_stats,
            'feedback_counts': self.feedback_counts.copy()
        }
        
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics by team."""
        total = sum(self.predictions.values())
        if total == 0:
            return {}
            
        stats = {}
        for team, count in self.predictions.items():
            stats[team] = {
                'count': count,
                'percentage': (count / total) * 100,
                'avg_confidence': np.mean([
                    score for score, pred_team in zip(self.confidence_scores, self.predictions.keys())
                    if pred_team == team
                ]) if count > 0 else 0
            }
            
        return stats
        
    def get_trend(self, hours: Optional[int] = None, days: Optional[int] = None) -> Dict[str, Any]:
        """Get performance trend over time."""
        if days:
            cutoff_time = datetime.now() - timedelta(days=days)
        elif hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
        else:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
        trend_data = {
            'timestamps': [],
            'predictions': [],
            'feedback': [],
            'accuracy': []
        }
        
        for hour, metrics in sorted(self.hourly_metrics.items()):
            if hour >= cutoff_time:
                trend_data['timestamps'].append(hour.isoformat())
                trend_data['predictions'].append(len(metrics.get('predictions', [])))
                
                feedback = metrics.get('feedback', [])
                trend_data['feedback'].append(len(feedback))
                
                # Calculate hourly accuracy
                positive = sum(1 for f in feedback if f['type'] == 'positive')
                total = len(feedback)
                accuracy = positive / total if total > 0 else 0
                trend_data['accuracy'].append(accuracy)
                
        return trend_data
        
    def get_uptime_hours(self) -> float:
        """Get system uptime in hours."""
        uptime = datetime.now() - self.start_time
        return uptime.total_seconds() / 3600
        
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        metrics = self.get_metrics()
        
        # Define health thresholds
        health_status = {
            'status': 'healthy',
            'issues': []
        }
        
        # Check error rate
        if metrics['error_rate'] > 0.05:  # 5% error rate
            health_status['status'] = 'degraded'
            health_status['issues'].append(f"High error rate: {metrics['error_rate']:.1%}")
            
        # Check response time
        if metrics['response_time_ms']['p95'] > 1000:  # 1 second
            health_status['status'] = 'degraded'
            health_status['issues'].append(f"High response time: {metrics['response_time_ms']['p95']:.0f}ms")
            
        # Check accuracy
        if metrics['accuracy'] < 0.8:  # 80% accuracy
            health_status['status'] = 'warning'
            health_status['issues'].append(f"Low accuracy: {metrics['accuracy']:.1%}")
            
        return health_status
        
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        # This would typically be handled by prometheus_client
        # Here we provide a simple text format
        metrics = self.get_metrics()
        
        output = []
        output.append(f"# HELP ticket_routing_total_predictions Total predictions made")
        output.append(f"# TYPE ticket_routing_total_predictions counter")
        output.append(f"ticket_routing_total_predictions {self.total_requests}")
        
        output.append(f"# HELP ticket_routing_accuracy Current model accuracy")
        output.append(f"# TYPE ticket_routing_accuracy gauge")
        output.append(f"ticket_routing_accuracy {metrics['accuracy']}")
        
        output.append(f"# HELP ticket_routing_cache_hit_rate Cache hit rate")
        output.append(f"# TYPE ticket_routing_cache_hit_rate gauge")
        output.append(f"ticket_routing_cache_hit_rate {metrics['cache_hit_rate']}")
        
        return '\n'.join(output)
        
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.__init__()
        logger.info("Metrics reset")


class PerformanceTracker:
    """Track performance of individual operations."""
    
    def __init__(self, operation_name: str, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize performance tracker."""
        self.operation_name = operation_name
        self.metrics_collector = metrics_collector
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics."""
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            
            if self.metrics_collector:
                self.metrics_collector.record_response_time(elapsed_ms)
                
            logger.debug(f"{self.operation_name} completed in {elapsed_ms:.2f}ms")
            
        # Record error if exception occurred
        if exc_type and self.metrics_collector:
            self.metrics_collector.record_error()


def create_metrics_dashboard() -> Dict[str, Any]:
    """Create a metrics dashboard summary."""
    # This would typically integrate with Grafana or similar
    # Here we provide a simple structure
    return {
        'panels': [
            {
                'title': 'Routing Performance',
                'type': 'graph',
                'metrics': ['predictions_per_minute', 'average_confidence']
            },
            {
                'title': 'Model Accuracy',
                'type': 'gauge',
                'metric': 'current_accuracy'
            },
            {
                'title': 'Response Times',
                'type': 'heatmap',
                'metric': 'response_time_distribution'
            },
            {
                'title': 'Feedback Summary',
                'type': 'pie',
                'metric': 'feedback_types'
            }
        ]
    }