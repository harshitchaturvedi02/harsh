"""
Comprehensive evaluation metrics for the ticket routing system
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RoutingMetrics:
    """Comprehensive metrics for ticket routing evaluation"""
    
    def __init__(self):
        self.metrics_history = []
        self.baseline_metrics = None
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None,
                                 class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle multi-class vs binary classification
        average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
        
        precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': len(y_true)
        }
        
        # Per-class metrics
        if class_names:
            class_report = classification_report(
                y_true, y_pred, 
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            metrics['per_class_metrics'] = class_report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Top-k accuracy (useful for routing where we might consider top suggestions)
        if y_prob is not None:
            for k in [1, 3, 5]:
                top_k_acc = self._calculate_top_k_accuracy(y_true, y_prob, k)
                metrics[f'top_{k}_accuracy'] = float(top_k_acc)
        
        # AUC-ROC for multi-class (if probabilities available)
        if y_prob is not None and len(np.unique(y_true)) > 2:
            try:
                auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                metrics['auc_roc'] = float(auc_score)
            except ValueError as e:
                logger.warning(f"Could not calculate AUC-ROC: {e}")
                metrics['auc_roc'] = None
        
        return metrics
    
    def _calculate_top_k_accuracy(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy"""
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)
    
    def evaluate_business_metrics(self, routing_decisions: List[Dict[str, Any]], 
                                feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate business-specific metrics"""
        
        if not feedback_data:
            return {'error': 'No feedback data available for business metrics'}
        
        df_feedback = pd.DataFrame(feedback_data)
        df_routing = pd.DataFrame(routing_decisions) if routing_decisions else pd.DataFrame()
        
        metrics = {}
        
        # Customer satisfaction metrics
        if 'rating' in df_feedback.columns:
            metrics['avg_customer_satisfaction'] = float(df_feedback['rating'].mean())
            metrics['satisfaction_std'] = float(df_feedback['rating'].std())
            metrics['satisfaction_distribution'] = df_feedback['rating'].value_counts().to_dict()
        
        # Routing accuracy from feedback
        if 'was_correctly_routed' in df_feedback.columns:
            metrics['routing_accuracy_feedback'] = float(df_feedback['was_correctly_routed'].mean())
        
        # Resolution quality
        if 'resolution_quality' in df_feedback.columns:
            metrics['avg_resolution_quality'] = float(df_feedback['resolution_quality'].mean())
        
        # Response time satisfaction
        if 'response_time_satisfaction' in df_feedback.columns:
            metrics['avg_response_time_satisfaction'] = float(df_feedback['response_time_satisfaction'].mean())
        
        # Confidence analysis
        if not df_routing.empty and 'confidence_score' in df_routing.columns:
            metrics['avg_confidence'] = float(df_routing['confidence_score'].mean())
            metrics['confidence_std'] = float(df_routing['confidence_score'].std())
            
            # Correlation between confidence and correctness
            if 'was_correctly_routed' in df_feedback.columns and len(df_routing) == len(df_feedback):
                correlation = np.corrcoef(df_routing['confidence_score'], df_feedback['was_correctly_routed'])[0, 1]
                metrics['confidence_correctness_correlation'] = float(correlation) if not np.isnan(correlation) else None
        
        # Department-wise performance
        if 'department' in df_feedback.columns:
            dept_metrics = {}
            for dept in df_feedback['department'].unique():
                dept_data = df_feedback[df_feedback['department'] == dept]
                dept_metrics[dept] = {
                    'count': len(dept_data),
                    'avg_rating': float(dept_data['rating'].mean()) if 'rating' in dept_data.columns else None,
                    'routing_accuracy': float(dept_data['was_correctly_routed'].mean()) if 'was_correctly_routed' in dept_data.columns else None
                }
            metrics['department_performance'] = dept_metrics
        
        return metrics
    
    def evaluate_operational_metrics(self, tickets_data: List[Dict[str, Any]], 
                                   user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate operational efficiency metrics"""
        
        if not tickets_data:
            return {'error': 'No ticket data available for operational metrics'}
        
        df_tickets = pd.DataFrame(tickets_data)
        df_users = pd.DataFrame(user_data) if user_data else pd.DataFrame()
        
        metrics = {}
        
        # Resolution time metrics
        if 'resolution_time_hours' in df_tickets.columns:
            resolution_times = df_tickets['resolution_time_hours'].dropna()
            if len(resolution_times) > 0:
                metrics['avg_resolution_time_hours'] = float(resolution_times.mean())
                metrics['median_resolution_time_hours'] = float(resolution_times.median())
                metrics['resolution_time_std'] = float(resolution_times.std())
                metrics['resolution_time_percentiles'] = {
                    '25th': float(resolution_times.quantile(0.25)),
                    '75th': float(resolution_times.quantile(0.75)),
                    '90th': float(resolution_times.quantile(0.90)),
                    '95th': float(resolution_times.quantile(0.95))
                }
        
        # Workload distribution
        if 'assignee_id' in df_tickets.columns:
            workload_dist = df_tickets['assignee_id'].value_counts()
            metrics['workload_distribution'] = workload_dist.to_dict()
            metrics['workload_balance_coefficient'] = float(workload_dist.std() / workload_dist.mean()) if workload_dist.mean() > 0 else None
        
        # Priority handling
        if 'priority' in df_tickets.columns:
            priority_dist = df_tickets['priority'].value_counts()
            metrics['priority_distribution'] = priority_dist.to_dict()
            
            # Priority vs resolution time
            if 'resolution_time_hours' in df_tickets.columns:
                priority_resolution = df_tickets.groupby('priority')['resolution_time_hours'].mean()
                metrics['priority_resolution_times'] = priority_resolution.to_dict()
        
        # Ticket volume trends
        if 'created_at' in df_tickets.columns:
            df_tickets['created_at'] = pd.to_datetime(df_tickets['created_at'])
            daily_volume = df_tickets.groupby(df_tickets['created_at'].dt.date).size()
            metrics['daily_volume_stats'] = {
                'avg_daily_volume': float(daily_volume.mean()),
                'max_daily_volume': int(daily_volume.max()),
                'min_daily_volume': int(daily_volume.min())
            }
        
        # User utilization
        if not df_users.empty and 'current_workload' in df_users.columns and 'workload_capacity' in df_users.columns:
            df_users['utilization'] = df_users['current_workload'] / df_users['workload_capacity']
            metrics['user_utilization'] = {
                'avg_utilization': float(df_users['utilization'].mean()),
                'max_utilization': float(df_users['utilization'].max()),
                'underutilized_users': len(df_users[df_users['utilization'] < 0.5]),
                'overutilized_users': len(df_users[df_users['utilization'] > 0.9])
            }
        
        return metrics
    
    def calculate_model_drift_metrics(self, current_performance: Dict[str, Any],
                                    historical_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics to detect model drift"""
        
        if not historical_performance:
            return {'status': 'no_historical_data'}
        
        # Get recent historical performance (last 30 days)
        recent_performance = historical_performance[-30:] if len(historical_performance) >= 30 else historical_performance
        
        drift_metrics = {}
        
        # Key metrics to monitor for drift
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'avg_customer_satisfaction', 'routing_accuracy_feedback']
        
        for metric in key_metrics:
            if metric in current_performance:
                current_value = current_performance[metric]
                
                # Calculate historical average
                historical_values = [perf.get(metric) for perf in recent_performance if perf.get(metric) is not None]
                
                if historical_values:
                    historical_avg = np.mean(historical_values)
                    historical_std = np.std(historical_values)
                    
                    # Calculate drift indicators
                    drift_score = abs(current_value - historical_avg) / (historical_std + 1e-8)
                    
                    drift_metrics[f'{metric}_drift'] = {
                        'current_value': float(current_value),
                        'historical_avg': float(historical_avg),
                        'historical_std': float(historical_std),
                        'drift_score': float(drift_score),
                        'is_significant': drift_score > 2.0  # 2 standard deviations
                    }
        
        # Overall drift assessment
        significant_drifts = sum(1 for metric_data in drift_metrics.values() 
                               if isinstance(metric_data, dict) and metric_data.get('is_significant', False))
        
        drift_metrics['overall_assessment'] = {
            'total_metrics_monitored': len(drift_metrics),
            'significant_drifts': significant_drifts,
            'drift_severity': 'high' if significant_drifts >= 3 else 'medium' if significant_drifts >= 1 else 'low',
            'recommendation': self._get_drift_recommendation(significant_drifts, len(drift_metrics))
        }
        
        return drift_metrics
    
    def _get_drift_recommendation(self, significant_drifts: int, total_metrics: int) -> str:
        """Get recommendation based on drift analysis"""
        if significant_drifts == 0:
            return "Model performance is stable. Continue monitoring."
        elif significant_drifts <= total_metrics * 0.3:
            return "Minor performance drift detected. Consider investigating data changes."
        elif significant_drifts <= total_metrics * 0.6:
            return "Moderate drift detected. Consider retraining with recent data."
        else:
            return "Significant drift detected. Immediate retraining recommended."
    
    def generate_performance_report(self, evaluation_results: Dict[str, Any],
                                  period_days: int = 30) -> str:
        """Generate a comprehensive performance report"""
        
        report = f"""
TICKET ROUTING SYSTEM PERFORMANCE REPORT
======================================
Evaluation Period: Last {period_days} days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MACHINE LEARNING METRICS:
-------------------------
"""
        
        # ML Metrics
        if 'ml_metrics' in evaluation_results:
            ml_metrics = evaluation_results['ml_metrics']
            report += f"Accuracy: {ml_metrics.get('accuracy', 'N/A'):.3f}\n"
            report += f"Precision: {ml_metrics.get('precision', 'N/A'):.3f}\n"
            report += f"Recall: {ml_metrics.get('recall', 'N/A'):.3f}\n"
            report += f"F1-Score: {ml_metrics.get('f1_score', 'N/A'):.3f}\n"
            
            if 'top_3_accuracy' in ml_metrics:
                report += f"Top-3 Accuracy: {ml_metrics['top_3_accuracy']:.3f}\n"
        
        # Business Metrics
        if 'business_metrics' in evaluation_results:
            business_metrics = evaluation_results['business_metrics']
            report += "\nBUSINESS METRICS:\n"
            report += "-----------------\n"
            
            if 'avg_customer_satisfaction' in business_metrics:
                report += f"Average Customer Satisfaction: {business_metrics['avg_customer_satisfaction']:.2f}/5.0\n"
            
            if 'routing_accuracy_feedback' in business_metrics:
                report += f"Routing Accuracy (from feedback): {business_metrics['routing_accuracy_feedback']:.1%}\n"
            
            if 'avg_resolution_quality' in business_metrics:
                report += f"Average Resolution Quality: {business_metrics['avg_resolution_quality']:.2f}/5.0\n"
        
        # Operational Metrics
        if 'operational_metrics' in evaluation_results:
            operational_metrics = evaluation_results['operational_metrics']
            report += "\nOPERATIONAL METRICS:\n"
            report += "--------------------\n"
            
            if 'avg_resolution_time_hours' in operational_metrics:
                report += f"Average Resolution Time: {operational_metrics['avg_resolution_time_hours']:.1f} hours\n"
            
            if 'workload_balance_coefficient' in operational_metrics:
                balance_coeff = operational_metrics['workload_balance_coefficient']
                if balance_coeff is not None:
                    balance_status = "Good" if balance_coeff < 0.3 else "Fair" if balance_coeff < 0.6 else "Poor"
                    report += f"Workload Balance: {balance_status} (coefficient: {balance_coeff:.3f})\n"
        
        # Drift Analysis
        if 'drift_metrics' in evaluation_results:
            drift_metrics = evaluation_results['drift_metrics']
            if 'overall_assessment' in drift_metrics:
                assessment = drift_metrics['overall_assessment']
                report += "\nMODEL DRIFT ANALYSIS:\n"
                report += "---------------------\n"
                report += f"Drift Severity: {assessment['drift_severity'].upper()}\n"
                report += f"Significant Drifts: {assessment['significant_drifts']}/{assessment['total_metrics_monitored']}\n"
                report += f"Recommendation: {assessment['recommendation']}\n"
        
        # Recommendations
        report += "\nRECOMMENDATIONS:\n"
        report += "----------------\n"
        report += self._generate_recommendations(evaluation_results)
        
        return report
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate actionable recommendations based on evaluation results"""
        recommendations = []
        
        # ML Performance recommendations
        if 'ml_metrics' in evaluation_results:
            ml_metrics = evaluation_results['ml_metrics']
            
            if ml_metrics.get('accuracy', 0) < 0.8:
                recommendations.append("• Model accuracy is below 80%. Consider retraining with more diverse data.")
            
            if ml_metrics.get('f1_score', 0) < 0.75:
                recommendations.append("• F1-score indicates room for improvement. Review feature engineering.")
        
        # Business metrics recommendations
        if 'business_metrics' in evaluation_results:
            business_metrics = evaluation_results['business_metrics']
            
            if business_metrics.get('avg_customer_satisfaction', 5) < 3.5:
                recommendations.append("• Customer satisfaction is low. Review routing decisions and agent training.")
            
            if business_metrics.get('routing_accuracy_feedback', 1) < 0.85:
                recommendations.append("• Routing accuracy needs improvement. Analyze misrouted tickets.")
        
        # Operational recommendations
        if 'operational_metrics' in evaluation_results:
            operational_metrics = evaluation_results['operational_metrics']
            
            if operational_metrics.get('avg_resolution_time_hours', 0) > 48:
                recommendations.append("• Resolution times are high. Consider workload redistribution.")
            
            user_util = operational_metrics.get('user_utilization', {})
            if user_util.get('overutilized_users', 0) > 0:
                recommendations.append("• Some users are overutilized. Balance workload distribution.")
        
        # Drift recommendations
        if 'drift_metrics' in evaluation_results:
            drift_assessment = evaluation_results['drift_metrics'].get('overall_assessment', {})
            if drift_assessment.get('drift_severity') in ['medium', 'high']:
                recommendations.append("• Model drift detected. Schedule retraining with recent data.")
        
        if not recommendations:
            recommendations.append("• System performance is satisfactory. Continue monitoring.")
        
        return "\n".join(recommendations)
    
    def save_metrics_history(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None):
        """Save metrics to history for trend analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics_entry = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Keep only last 90 days of history
        cutoff_date = datetime.now() - timedelta(days=90)
        self.metrics_history = [
            entry for entry in self.metrics_history 
            if entry['timestamp'] >= cutoff_date
        ]
    
    def get_metrics_trend(self, metric_name: str, days: int = 30) -> Dict[str, Any]:
        """Get trend analysis for a specific metric"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_metrics = [
            entry for entry in self.metrics_history 
            if entry['timestamp'] >= cutoff_date
        ]
        
        if not recent_metrics:
            return {'error': 'No historical data available'}
        
        # Extract metric values
        values = []
        timestamps = []
        
        for entry in recent_metrics:
            metric_value = self._extract_nested_metric(entry['metrics'], metric_name)
            if metric_value is not None:
                values.append(metric_value)
                timestamps.append(entry['timestamp'])
        
        if not values:
            return {'error': f'Metric {metric_name} not found in historical data'}
        
        # Calculate trend
        values_array = np.array(values)
        trend_slope = np.polyfit(range(len(values)), values_array, 1)[0] if len(values) > 1 else 0
        
        return {
            'metric_name': metric_name,
            'current_value': values[-1],
            'avg_value': float(np.mean(values)),
            'trend_slope': float(trend_slope),
            'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
            'data_points': len(values),
            'period_days': days
        }
    
    def _extract_nested_metric(self, metrics_dict: Dict[str, Any], metric_path: str) -> Optional[float]:
        """Extract metric value from nested dictionary using dot notation"""
        keys = metric_path.split('.')
        current = metrics_dict
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return float(current) if isinstance(current, (int, float)) else None


class CrossValidationEvaluator:
    """Cross-validation evaluation for model assessment"""
    
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
    
    def evaluate_with_cv(self, classifier, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation evaluation"""
        
        # Stratified K-Fold for better class distribution
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Scoring metrics
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(classifier, X, y, cv=skf, scoring=metric)
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'confidence_interval': [
                    float(scores.mean() - 1.96 * scores.std()),
                    float(scores.mean() + 1.96 * scores.std())
                ]
            }
        
        return cv_results