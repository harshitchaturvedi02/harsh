"""
Feedback loop system for continuous model improvement
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import asyncio
from sqlalchemy.orm import Session
from ..models.data_models import FeedbackDB, TicketDB, RoutingDecisionDB, UserDB
from ..models.ml_models import TicketClassifier, PerformanceTracker
from ..nlp.text_processor import TextProcessor, FeatureExtractor
import joblib
import os

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """Process and analyze feedback data"""
    
    def __init__(self):
        self.feedback_threshold = 0.7  # Minimum accuracy before retraining
        self.min_feedback_samples = 50  # Minimum feedback samples for retraining
        self.performance_weights = {
            'routing_accuracy': 0.4,
            'satisfaction_score': 0.3,
            'resolution_time': 0.3
        }
    
    def analyze_feedback(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback data to identify improvement opportunities"""
        if not feedback_data:
            return {'status': 'no_feedback', 'metrics': {}}
        
        df = pd.DataFrame(feedback_data)
        
        # Calculate key metrics
        metrics = {
            'total_feedback': len(df),
            'avg_rating': df['rating'].mean(),
            'routing_accuracy': df['was_correctly_routed'].mean(),
            'avg_resolution_quality': df['resolution_quality'].mean(),
            'avg_response_satisfaction': df['response_time_satisfaction'].mean(),
            'feedback_by_department': self._analyze_by_department(df),
            'feedback_by_assignee': self._analyze_by_assignee(df),
            'improvement_areas': self._identify_improvement_areas(df)
        }
        
        # Determine if retraining is needed
        needs_retraining = (
            metrics['routing_accuracy'] < self.feedback_threshold or 
            metrics['avg_rating'] < 3.0
        )
        
        return {
            'status': 'analyzed',
            'metrics': metrics,
            'needs_retraining': needs_retraining,
            'recommendation': self._generate_recommendations(metrics)
        }
    
    def _analyze_by_department(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feedback by department"""
        if 'department' not in df.columns:
            return {}
        
        dept_analysis = {}
        for dept in df['department'].unique():
            dept_data = df[df['department'] == dept]
            dept_analysis[dept] = {
                'count': len(dept_data),
                'avg_rating': dept_data['rating'].mean(),
                'routing_accuracy': dept_data['was_correctly_routed'].mean(),
                'avg_resolution_quality': dept_data['resolution_quality'].mean()
            }
        
        return dept_analysis
    
    def _analyze_by_assignee(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feedback by assignee"""
        if 'assignee_id' not in df.columns:
            return {}
        
        assignee_analysis = {}
        for assignee in df['assignee_id'].unique():
            assignee_data = df[df['assignee_id'] == assignee]
            assignee_analysis[assignee] = {
                'count': len(assignee_data),
                'avg_rating': assignee_data['rating'].mean(),
                'routing_accuracy': assignee_data['was_correctly_routed'].mean(),
                'avg_resolution_quality': assignee_data['resolution_quality'].mean()
            }
        
        return assignee_analysis
    
    def _identify_improvement_areas(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify specific areas for improvement"""
        improvements = []
        
        # Low routing accuracy
        if df['was_correctly_routed'].mean() < 0.8:
            improvements.append({
                'area': 'routing_accuracy',
                'current_score': df['was_correctly_routed'].mean(),
                'target_score': 0.85,
                'priority': 'high'
            })
        
        # Low satisfaction
        if df['rating'].mean() < 3.5:
            improvements.append({
                'area': 'overall_satisfaction',
                'current_score': df['rating'].mean(),
                'target_score': 4.0,
                'priority': 'medium'
            })
        
        # Poor resolution quality
        if df['resolution_quality'].mean() < 3.5:
            improvements.append({
                'area': 'resolution_quality',
                'current_score': df['resolution_quality'].mean(),
                'target_score': 4.0,
                'priority': 'medium'
            })
        
        return improvements
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if metrics['routing_accuracy'] < 0.8:
            recommendations.append("Consider retraining the routing model with recent data")
            recommendations.append("Review department keywords and classification rules")
        
        if metrics['avg_rating'] < 3.5:
            recommendations.append("Investigate assignee workload distribution")
            recommendations.append("Review team skills and expertise mapping")
        
        if metrics['avg_resolution_quality'] < 3.5:
            recommendations.append("Provide additional training to underperforming team members")
            recommendations.append("Review knowledge base and documentation")
        
        return recommendations


class ModelRetrainer:
    """Handle model retraining based on feedback"""
    
    def __init__(self, classifier: TicketClassifier, text_processor: TextProcessor):
        self.classifier = classifier
        self.text_processor = text_processor
        self.feature_extractor = FeatureExtractor(text_processor)
        self.retrain_threshold = 100  # Minimum new samples for retraining
        self.model_backup_dir = "models/backups"
        
        # Ensure backup directory exists
        os.makedirs(self.model_backup_dir, exist_ok=True)
    
    def should_retrain(self, feedback_analysis: Dict[str, Any], 
                      new_training_data_count: int) -> bool:
        """Determine if model should be retrained"""
        return (
            feedback_analysis.get('needs_retraining', False) and
            new_training_data_count >= self.retrain_threshold
        )
    
    def prepare_training_data(self, db_session: Session, 
                            since_date: Optional[datetime] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from database"""
        # Query tickets with feedback
        query = db_session.query(TicketDB).join(FeedbackDB)
        
        if since_date:
            query = query.filter(TicketDB.created_at >= since_date)
        
        tickets = query.all()
        
        if not tickets:
            raise ValueError("No training data available")
        
        # Extract features and labels
        X = []
        y = []
        
        for ticket in tickets:
            # Prepare ticket data
            ticket_data = {
                'title': ticket.title,
                'description': ticket.description,
                'priority': ticket.priority
            }
            
            # Extract features
            features = self.feature_extractor.extract_ticket_features(ticket_data)
            X.append(features)
            
            # Use assignee as label
            y.append(ticket.assignee_id)
        
        return np.array(X), np.array(y)
    
    def retrain_model(self, X: np.ndarray, y: np.ndarray, 
                     model_version: str = None) -> Dict[str, Any]:
        """Retrain the model with new data"""
        logger.info(f"Starting model retraining with {len(X)} samples")
        
        # Backup current model
        if self.classifier.is_fitted:
            backup_path = os.path.join(
                self.model_backup_dir, 
                f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            self.classifier.save_model(backup_path)
            logger.info(f"Current model backed up to {backup_path}")
        
        # Retrain model
        training_results = self.classifier.train(X, y)
        
        # Update feature names
        self.classifier.feature_names = self.feature_extractor.get_feature_names()
        
        # Save retrained model
        if model_version:
            model_path = f"models/ticket_classifier_{model_version}.joblib"
        else:
            model_path = f"models/ticket_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        self.classifier.save_model(model_path)
        
        logger.info(f"Model retrained and saved to {model_path}")
        
        return {
            'status': 'success',
            'model_path': model_path,
            'training_results': training_results,
            'samples_used': len(X),
            'retrain_timestamp': datetime.now().isoformat()
        }
    
    def incremental_update(self, new_X: np.ndarray, new_y: np.ndarray) -> Dict[str, Any]:
        """Perform incremental model update (for online learning)"""
        # For now, we'll do full retraining
        # In production, you might want to implement true incremental learning
        return self.retrain_model(new_X, new_y)


class FeedbackLoop:
    """Main feedback loop coordinator"""
    
    def __init__(self, classifier: TicketClassifier, text_processor: TextProcessor):
        self.classifier = classifier
        self.text_processor = text_processor
        self.feedback_processor = FeedbackProcessor()
        self.model_retrainer = ModelRetrainer(classifier, text_processor)
        self.performance_tracker = PerformanceTracker()
        
        # Configuration
        self.feedback_check_interval = timedelta(hours=24)  # Check feedback daily
        self.last_feedback_check = None
        self.auto_retrain = True
        
    async def process_feedback_batch(self, db_session: Session) -> Dict[str, Any]:
        """Process a batch of feedback and potentially retrain model"""
        try:
            # Get recent feedback
            cutoff_date = datetime.now() - timedelta(days=30)  # Last 30 days
            
            feedback_query = db_session.query(FeedbackDB).filter(
                FeedbackDB.created_at >= cutoff_date
            ).join(TicketDB)
            
            feedback_data = []
            for feedback in feedback_query.all():
                feedback_data.append({
                    'rating': feedback.rating,
                    'was_correctly_routed': feedback.was_correctly_routed,
                    'resolution_quality': feedback.resolution_quality,
                    'response_time_satisfaction': feedback.response_time_satisfaction,
                    'ticket_id': feedback.ticket_id,
                    'assignee_id': feedback.ticket.assignee_id,
                    'department': feedback.ticket.department,
                    'created_at': feedback.created_at
                })
            
            # Analyze feedback
            analysis = self.feedback_processor.analyze_feedback(feedback_data)
            
            # Check if retraining is needed
            if analysis.get('needs_retraining', False) and self.auto_retrain:
                # Prepare training data
                try:
                    X, y = self.model_retrainer.prepare_training_data(
                        db_session, 
                        since_date=cutoff_date
                    )
                    
                    # Retrain if we have enough data
                    if len(X) >= self.model_retrainer.retrain_threshold:
                        retrain_results = self.model_retrainer.retrain_model(X, y)
                        analysis['retrain_results'] = retrain_results
                    else:
                        analysis['retrain_results'] = {
                            'status': 'skipped',
                            'reason': 'insufficient_data',
                            'samples_available': len(X)
                        }
                        
                except Exception as e:
                    logger.error(f"Error during retraining: {e}")
                    analysis['retrain_results'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Update performance tracker
            for feedback in feedback_data:
                if 'resolution_time_hours' in feedback:
                    self.performance_tracker.update_metrics(
                        feedback['assignee_id'],
                        feedback.get('resolution_time_hours', 24),
                        feedback['rating'],
                        feedback['was_correctly_routed']
                    )
            
            self.last_feedback_check = datetime.now()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def update_routing_performance(self, ticket_id: int, assignee_id: int, 
                                 resolution_time_hours: float, satisfaction_score: float,
                                 was_correctly_routed: bool):
        """Update performance metrics for a specific routing decision"""
        self.performance_tracker.update_metrics(
            assignee_id,
            resolution_time_hours,
            satisfaction_score,
            was_correctly_routed
        )
        
        # Update classifier's performance tracker
        self.classifier.performance_tracker.update_metrics(
            assignee_id,
            resolution_time_hours,
            satisfaction_score,
            was_correctly_routed
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        return {
            'performance_metrics': self.performance_tracker.metrics,
            'last_feedback_check': self.last_feedback_check,
            'model_info': self.classifier.get_model_info(),
            'feedback_processor_config': {
                'feedback_threshold': self.feedback_processor.feedback_threshold,
                'min_feedback_samples': self.feedback_processor.min_feedback_samples
            }
        }
    
    def manual_retrain(self, db_session: Session, 
                      since_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Manually trigger model retraining"""
        try:
            X, y = self.model_retrainer.prepare_training_data(db_session, since_date)
            return self.model_retrainer.retrain_model(X, y)
        except Exception as e:
            logger.error(f"Manual retrain failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def configure_feedback_loop(self, config: Dict[str, Any]):
        """Configure feedback loop parameters"""
        if 'feedback_threshold' in config:
            self.feedback_processor.feedback_threshold = config['feedback_threshold']
        
        if 'min_feedback_samples' in config:
            self.feedback_processor.min_feedback_samples = config['min_feedback_samples']
        
        if 'retrain_threshold' in config:
            self.model_retrainer.retrain_threshold = config['retrain_threshold']
        
        if 'auto_retrain' in config:
            self.auto_retrain = config['auto_retrain']
        
        if 'feedback_check_interval_hours' in config:
            self.feedback_check_interval = timedelta(hours=config['feedback_check_interval_hours'])
        
        logger.info(f"Feedback loop configured with: {config}")


class AdaptiveLearning:
    """Adaptive learning system for continuous improvement"""
    
    def __init__(self, feedback_loop: FeedbackLoop):
        self.feedback_loop = feedback_loop
        self.learning_rate = 0.01
        self.adaptation_strategies = {
            'workload_balancing': self._adapt_workload_balancing,
            'department_keywords': self._adapt_department_keywords,
            'user_skills': self._adapt_user_skills
        }
    
    def adapt_system(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adapt system based on feedback patterns"""
        adaptations = {}
        
        for strategy_name, strategy_func in self.adaptation_strategies.items():
            try:
                adaptation_result = strategy_func(feedback_data)
                adaptations[strategy_name] = adaptation_result
            except Exception as e:
                logger.error(f"Error in adaptation strategy {strategy_name}: {e}")
                adaptations[strategy_name] = {'status': 'error', 'error': str(e)}
        
        return adaptations
    
    def _adapt_workload_balancing(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adapt workload balancing based on feedback"""
        # Analyze workload vs satisfaction correlation
        df = pd.DataFrame(feedback_data)
        
        if 'assignee_workload' in df.columns and 'rating' in df.columns:
            correlation = df['assignee_workload'].corr(df['rating'])
            
            if correlation < -0.3:  # Strong negative correlation
                # Increase workload penalty factor
                current_penalty = self.feedback_loop.classifier.workload_balancer.penalty_factor
                new_penalty = min(current_penalty * 1.1, 0.5)
                self.feedback_loop.classifier.workload_balancer.penalty_factor = new_penalty
                
                return {
                    'status': 'adapted',
                    'old_penalty_factor': current_penalty,
                    'new_penalty_factor': new_penalty,
                    'correlation': correlation
                }
        
        return {'status': 'no_adaptation_needed'}
    
    def _adapt_department_keywords(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adapt department keywords based on misrouting patterns"""
        # This would analyze frequently misrouted tickets and update keywords
        # For now, return placeholder
        return {'status': 'not_implemented'}
    
    def _adapt_user_skills(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adapt user skill profiles based on performance"""
        # This would update user skill weights based on resolution success
        # For now, return placeholder
        return {'status': 'not_implemented'}