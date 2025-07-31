"""
Feedback loop module for continuous learning and model improvement.
Handles user feedback, performance monitoring, and model retraining.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import json
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and manages user feedback for model improvement."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feedback collector.
        
        Args:
            config: Feedback configuration
        """
        self.config = config.get('feedback', {})
        self.feedback_buffer = deque(maxlen=10000)
        self.performance_history = []
        self.feedback_stats = {
            'total_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'corrections': 0,
            'last_updated': datetime.now()
        }
        
        # Threading for async feedback processing
        self.lock = threading.Lock()
        self.is_running = True
        
        logger.info("FeedbackCollector initialized")
        
    def add_feedback(self, feedback: Dict[str, Any]):
        """
        Add user feedback for a routing decision.
        
        Args:
            feedback: Dictionary containing feedback data
                - ticket_id: Unique ticket identifier
                - predicted_class: Model's prediction
                - actual_class: Correct class (if correction)
                - confidence: Model's confidence
                - satisfaction_score: User satisfaction (1-5)
                - resolution_time: Time to resolve (hours)
                - feedback_type: 'positive', 'negative', 'correction'
                - timestamp: When feedback was given
                - features: Original feature vector (optional)
        """
        with self.lock:
            # Add timestamp if not provided
            if 'timestamp' not in feedback:
                feedback['timestamp'] = datetime.now()
                
            # Validate feedback
            if self._validate_feedback(feedback):
                self.feedback_buffer.append(feedback)
                self._update_stats(feedback)
                logger.debug(f"Added feedback for ticket {feedback.get('ticket_id')}")
            else:
                logger.warning(f"Invalid feedback rejected: {feedback}")
                
    def _validate_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Validate feedback data."""
        required_fields = ['ticket_id', 'predicted_class', 'feedback_type']
        return all(field in feedback for field in required_fields)
        
    def _update_stats(self, feedback: Dict[str, Any]):
        """Update feedback statistics."""
        self.feedback_stats['total_feedback'] += 1
        
        feedback_type = feedback.get('feedback_type')
        if feedback_type == 'positive':
            self.feedback_stats['positive_feedback'] += 1
        elif feedback_type == 'negative':
            self.feedback_stats['negative_feedback'] += 1
        elif feedback_type == 'correction':
            self.feedback_stats['corrections'] += 1
            
        self.feedback_stats['last_updated'] = datetime.now()
        
    def get_feedback_batch(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a batch of feedback for processing.
        
        Args:
            size: Number of feedback items to retrieve
            
        Returns:
            List of feedback dictionaries
        """
        with self.lock:
            if size is None:
                return list(self.feedback_buffer)
            else:
                return list(self.feedback_buffer)[-size:]
                
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        with self.lock:
            stats = self.feedback_stats.copy()
            
        # Calculate rates
        total = stats['total_feedback']
        if total > 0:
            stats['positive_rate'] = stats['positive_feedback'] / total
            stats['negative_rate'] = stats['negative_feedback'] / total
            stats['correction_rate'] = stats['corrections'] / total
        else:
            stats['positive_rate'] = 0
            stats['negative_rate'] = 0
            stats['correction_rate'] = 0
            
        return stats
        
    def clear_old_feedback(self, days: int = 30):
        """Remove feedback older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.lock:
            self.feedback_buffer = deque(
                [fb for fb in self.feedback_buffer 
                 if fb.get('timestamp', datetime.now()) > cutoff_date],
                maxlen=10000
            )
            
        logger.info(f"Cleared feedback older than {days} days")


class PerformanceMonitor:
    """Monitors model performance and triggers retraining."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config.get('feedback', {})
        self.metrics_history = []
        self.baseline_metrics = None
        self.alert_thresholds = self.config.get('retraining_trigger', {})
        
        logger.info("PerformanceMonitor initialized")
        
    def update_metrics(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """
        Update performance metrics.
        
        Args:
            metrics: Dictionary of metric values
            timestamp: Timestamp for metrics
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        metrics_entry = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Keep only recent history (last 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m['timestamp'] > cutoff_date
        ]
        
    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics.copy()
        logger.info(f"Baseline metrics set: {metrics}")
        
    def check_retraining_needed(self, current_metrics: Dict[str, float],
                               feedback_stats: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if model retraining is needed.
        
        Args:
            current_metrics: Current model performance metrics
            feedback_stats: Feedback statistics
            
        Returns:
            Tuple of (needs_retraining, reasons)
        """
        reasons = []
        
        # Check performance drop
        if self.baseline_metrics:
            perf_drop_threshold = self.alert_thresholds.get('performance_drop', 0.05)
            
            for metric, baseline_value in self.baseline_metrics.items():
                if metric in current_metrics:
                    current_value = current_metrics[metric]
                    if baseline_value - current_value > perf_drop_threshold:
                        reasons.append(
                            f"{metric} dropped by {(baseline_value - current_value):.2%}"
                        )
                        
        # Check feedback threshold
        feedback_threshold = self.alert_thresholds.get('feedback_threshold', 0.7)
        negative_rate = feedback_stats.get('negative_rate', 0)
        
        if negative_rate > (1 - feedback_threshold):
            reasons.append(
                f"High negative feedback rate: {negative_rate:.1%}"
            )
            
        # Check correction rate
        correction_rate = feedback_stats.get('correction_rate', 0)
        if correction_rate > 0.1:  # More than 10% corrections
            reasons.append(
                f"High correction rate: {correction_rate:.1%}"
            )
            
        return len(reasons) > 0, reasons
        
    def get_performance_trend(self, metric: str, days: int = 30) -> pd.DataFrame:
        """
        Get performance trend for a specific metric.
        
        Args:
            metric: Metric name
            days: Number of days to look back
            
        Returns:
            DataFrame with timestamp and metric values
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        trend_data = []
        for entry in self.metrics_history:
            if entry['timestamp'] > cutoff_date and metric in entry['metrics']:
                trend_data.append({
                    'timestamp': entry['timestamp'],
                    'value': entry['metrics'][metric]
                })
                
        return pd.DataFrame(trend_data)


class OnlineLearner:
    """Handles online learning and incremental model updates."""
    
    def __init__(self, base_model, config: Dict[str, Any]):
        """
        Initialize online learner.
        
        Args:
            base_model: Base classifier model
            config: Online learning configuration
        """
        self.base_model = base_model
        self.config = config.get('feedback', {}).get('online_learning', {})
        self.is_enabled = self.config.get('enabled', True)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.batch_size = self.config.get('batch_size', 10)
        
        # Buffer for mini-batch updates
        self.update_buffer = []
        
        logger.info(f"OnlineLearner initialized (enabled: {self.is_enabled})")
        
    def update(self, features: np.ndarray, true_label: int, predicted_label: int):
        """
        Update model with new example.
        
        Args:
            features: Feature vector
            true_label: Correct label
            predicted_label: Model's prediction
        """
        if not self.is_enabled:
            return
            
        # Add to buffer
        self.update_buffer.append({
            'features': features,
            'true_label': true_label,
            'predicted_label': predicted_label
        })
        
        # Perform mini-batch update if buffer is full
        if len(self.update_buffer) >= self.batch_size:
            self._perform_batch_update()
            
    def _perform_batch_update(self):
        """Perform mini-batch update on the model."""
        if not self.update_buffer:
            return
            
        # Extract batch data
        X_batch = np.array([item['features'] for item in self.update_buffer])
        y_batch = np.array([item['true_label'] for item in self.update_buffer])
        
        # Update model (implementation depends on model type)
        if hasattr(self.base_model, 'partial_fit'):
            # For models supporting incremental learning
            self.base_model.partial_fit(X_batch, y_batch)
        else:
            # For other models, we might need to retrain on combined data
            # This is a placeholder - actual implementation would be more sophisticated
            logger.warning("Model doesn't support partial_fit, skipping online update")
            
        # Clear buffer
        self.update_buffer = []
        
        logger.debug(f"Performed batch update with {len(X_batch)} examples")


class FeedbackLoop:
    """Main feedback loop coordinator."""
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize feedback loop.
        
        Args:
            model: Trained classifier model
            config: Full configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Initialize components
        self.feedback_collector = FeedbackCollector(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.online_learner = OnlineLearner(model, config)
        
        # State
        self.is_running = False
        self.update_thread = None
        self.last_retrain_time = datetime.now()
        
        logger.info("FeedbackLoop initialized")
        
    def start(self):
        """Start the feedback loop processing."""
        if self.is_running:
            logger.warning("Feedback loop already running")
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(target=self._process_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Feedback loop started")
        
    def stop(self):
        """Stop the feedback loop processing."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
            
        logger.info("Feedback loop stopped")
        
    def _process_loop(self):
        """Main processing loop for feedback."""
        update_frequency = self.config.get('feedback', {}).get('update_frequency', 'hourly')
        
        # Convert frequency to seconds
        frequency_map = {
            'realtime': 60,      # Every minute
            'hourly': 3600,      # Every hour
            'daily': 86400,      # Every day
            'weekly': 604800     # Every week
        }
        
        sleep_time = frequency_map.get(update_frequency, 3600)
        
        while self.is_running:
            try:
                self._process_feedback()
            except Exception as e:
                logger.error(f"Error in feedback loop: {e}")
                
            time.sleep(sleep_time)
            
    def _process_feedback(self):
        """Process accumulated feedback."""
        # Get feedback batch
        feedback_batch = self.feedback_collector.get_feedback_batch()
        
        if not feedback_batch:
            return
            
        # Process corrections for online learning
        corrections = [fb for fb in feedback_batch if fb['feedback_type'] == 'correction']
        
        for correction in corrections:
            if 'features' in correction and 'actual_class' in correction:
                self.online_learner.update(
                    correction['features'],
                    correction['actual_class'],
                    correction['predicted_class']
                )
                
        # Calculate current metrics from feedback
        current_metrics = self._calculate_metrics_from_feedback(feedback_batch)
        
        # Update performance monitor
        self.performance_monitor.update_metrics(current_metrics)
        
        # Check if retraining is needed
        feedback_stats = self.feedback_collector.get_stats()
        needs_retraining, reasons = self.performance_monitor.check_retraining_needed(
            current_metrics, feedback_stats
        )
        
        if needs_retraining:
            logger.warning(f"Retraining triggered: {', '.join(reasons)}")
            self._trigger_retraining(feedback_batch, reasons)
            
    def _calculate_metrics_from_feedback(self, feedback_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from feedback."""
        # Filter feedback with actual labels
        labeled_feedback = [
            fb for fb in feedback_batch 
            if 'actual_class' in fb or fb['feedback_type'] == 'positive'
        ]
        
        if not labeled_feedback:
            return {}
            
        # Calculate accuracy
        correct = sum(
            1 for fb in labeled_feedback
            if fb['feedback_type'] == 'positive' or 
            fb.get('actual_class') == fb['predicted_class']
        )
        
        accuracy = correct / len(labeled_feedback)
        
        # Calculate average satisfaction
        satisfaction_scores = [
            fb.get('satisfaction_score', 3) 
            for fb in feedback_batch 
            if 'satisfaction_score' in fb
        ]
        
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 3.0
        
        # Calculate average resolution time
        resolution_times = [
            fb.get('resolution_time', 24) 
            for fb in feedback_batch 
            if 'resolution_time' in fb
        ]
        
        avg_resolution_time = np.mean(resolution_times) if resolution_times else 24.0
        
        return {
            'accuracy': accuracy,
            'satisfaction_score': avg_satisfaction / 5.0,  # Normalize to 0-1
            'avg_resolution_time': avg_resolution_time,
            'feedback_count': len(feedback_batch)
        }
        
    def _trigger_retraining(self, feedback_batch: List[Dict[str, Any]], reasons: List[str]):
        """Trigger model retraining."""
        # Check minimum feedback count
        min_feedback = self.config.get('feedback', {}).get('min_feedback_count', 100)
        
        if len(feedback_batch) < min_feedback:
            logger.info(f"Insufficient feedback for retraining ({len(feedback_batch)} < {min_feedback})")
            return
            
        # Check time since last retrain
        hours_since_last = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
        if hours_since_last < 24:  # Don't retrain more than once per day
            logger.info(f"Too soon since last retraining ({hours_since_last:.1f} hours)")
            return
            
        # Prepare retraining data
        retrain_data = self._prepare_retrain_data(feedback_batch)
        
        # Log retraining event
        logger.info(f"Initiating retraining: {', '.join(reasons)}")
        logger.info(f"Retraining data size: {len(retrain_data)}")
        
        # In production, this would trigger an async retraining job
        # For now, we just log the event
        self.last_retrain_time = datetime.now()
        
    def _prepare_retrain_data(self, feedback_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for retraining from feedback."""
        retrain_data = []
        
        for feedback in feedback_batch:
            if 'features' in feedback:
                if feedback['feedback_type'] == 'correction' and 'actual_class' in feedback:
                    retrain_data.append({
                        'features': feedback['features'],
                        'label': feedback['actual_class'],
                        'weight': 2.0  # Higher weight for corrections
                    })
                elif feedback['feedback_type'] == 'positive':
                    retrain_data.append({
                        'features': feedback['features'],
                        'label': feedback['predicted_class'],
                        'weight': 1.0
                    })
                    
        return retrain_data
        
    def add_feedback(self, feedback: Dict[str, Any]):
        """
        Add feedback for a routing decision.
        
        Args:
            feedback: Feedback dictionary
        """
        self.feedback_collector.add_feedback(feedback)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the feedback loop."""
        return {
            'is_running': self.is_running,
            'feedback_stats': self.feedback_collector.get_stats(),
            'performance_metrics': self.performance_monitor.metrics_history[-1] 
                                  if self.performance_monitor.metrics_history else {},
            'last_retrain_time': self.last_retrain_time.isoformat(),
            'online_learning_enabled': self.online_learner.is_enabled
        }
        
    def save_state(self, path: str):
        """Save feedback loop state."""
        state = {
            'feedback_buffer': list(self.feedback_collector.feedback_buffer),
            'feedback_stats': self.feedback_collector.feedback_stats,
            'metrics_history': self.performance_monitor.metrics_history,
            'baseline_metrics': self.performance_monitor.baseline_metrics,
            'last_retrain_time': self.last_retrain_time.isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, default=str)
            
        logger.info(f"Feedback loop state saved to {path}")
        
    def load_state(self, path: str):
        """Load feedback loop state."""
        with open(path, 'r') as f:
            state = json.load(f)
            
        # Restore feedback collector state
        self.feedback_collector.feedback_buffer = deque(
            state['feedback_buffer'], 
            maxlen=10000
        )
        self.feedback_collector.feedback_stats = state['feedback_stats']
        
        # Restore performance monitor state
        self.performance_monitor.metrics_history = state['metrics_history']
        self.performance_monitor.baseline_metrics = state['baseline_metrics']
        
        # Restore last retrain time
        self.last_retrain_time = datetime.fromisoformat(state['last_retrain_time'])
        
        logger.info(f"Feedback loop state loaded from {path}")