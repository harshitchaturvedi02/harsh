"""
Feedback loop system for continuous model improvement.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger

from config.settings import settings
from database.database import get_db_context
from database.models import Ticket, RoutingFeedback, ModelPerformance
from models.ticket_classifier import classifier


class FeedbackLoop:
    """Feedback loop for continuous model improvement."""
    
    def __init__(self):
        self.retrain_threshold = settings.RETRAIN_THRESHOLD
        self.feedback_weight = settings.FEEDBACK_WEIGHT
        
    def collect_feedback_data(self, days_back: int = 30) -> pd.DataFrame:
        """Collect feedback data for model retraining."""
        with get_db_context() as db:
            # Get feedback from the last N days
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            feedback_data = db.query(RoutingFeedback).filter(
                RoutingFeedback.created_at >= cutoff_date
            ).all()
            
            # Get corresponding tickets
            ticket_ids = [f.ticket_id for f in feedback_data]
            tickets = db.query(Ticket).filter(Ticket.id.in_(ticket_ids)).all()
            
            # Create mapping
            ticket_map = {t.id: t for t in tickets}
            
            # Prepare data
            data = []
            for feedback in feedback_data:
                ticket = ticket_map.get(feedback.ticket_id)
                if ticket:
                    data.append({
                        'text': f"{ticket.title} {ticket.description}",
                        'actual_team': feedback.correct_team_id or ticket.predicted_team_id,
                        'predicted_team': ticket.predicted_team_id,
                        'was_correct': feedback.was_correct,
                        'user_satisfaction': feedback.user_satisfaction,
                        'resolution_time': feedback.resolution_time_hours,
                        'prediction_confidence': feedback.prediction_confidence,
                        'feedback_date': feedback.created_at
                    })
            
            return pd.DataFrame(data)
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """Analyze feedback trends and model performance."""
        with get_db_context() as db:
            # Get recent feedback
            cutoff_date = datetime.now() - timedelta(days=7)
            recent_feedback = db.query(RoutingFeedback).filter(
                RoutingFeedback.created_at >= cutoff_date
            ).all()
            
            if not recent_feedback:
                return {"error": "No recent feedback data"}
            
            # Calculate metrics
            total_feedback = len(recent_feedback)
            correct_routings = sum(1 for f in recent_feedback if f.was_correct)
            accuracy = correct_routings / total_feedback if total_feedback > 0 else 0
            
            # Average satisfaction
            satisfaction_scores = [f.user_satisfaction for f in recent_feedback if f.user_satisfaction]
            avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0
            
            # Average resolution time
            resolution_times = [f.resolution_time_hours for f in recent_feedback if f.resolution_time_hours]
            avg_resolution_time = np.mean(resolution_times) if resolution_times else 0
            
            # Team-wise accuracy
            team_accuracy = {}
            for feedback in recent_feedback:
                ticket = db.query(Ticket).filter(Ticket.id == feedback.ticket_id).first()
                if ticket:
                    team = ticket.predicted_team_id
                    if team not in team_accuracy:
                        team_accuracy[team] = {'correct': 0, 'total': 0}
                    team_accuracy[team]['total'] += 1
                    if feedback.was_correct:
                        team_accuracy[team]['correct'] += 1
            
            # Calculate team accuracy percentages
            for team in team_accuracy:
                team_accuracy[team]['accuracy'] = (
                    team_accuracy[team]['correct'] / team_accuracy[team]['total']
                )
            
            return {
                'total_feedback': total_feedback,
                'accuracy': accuracy,
                'average_satisfaction': avg_satisfaction,
                'average_resolution_time': avg_resolution_time,
                'team_accuracy': team_accuracy,
                'analysis_date': datetime.now().isoformat()
            }
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained based on feedback."""
        with get_db_context() as db:
            # Count recent feedback
            cutoff_date = datetime.now() - timedelta(days=7)
            recent_feedback_count = db.query(RoutingFeedback).filter(
                RoutingFeedback.created_at >= cutoff_date
            ).count()
            
            # Check accuracy trend
            if recent_feedback_count >= self.retrain_threshold:
                trends = self.analyze_feedback_trends()
                if 'accuracy' in trends and trends['accuracy'] < 0.8:  # Below 80% accuracy
                    logger.info(f"Retraining triggered: Low accuracy ({trends['accuracy']:.3f})")
                    return True
            
            return False
    
    def prepare_retraining_data(self) -> tuple:
        """Prepare data for model retraining."""
        # Get all historical data
        with get_db_context() as db:
            # Get all tickets with feedback
            feedback_tickets = db.query(Ticket).join(RoutingFeedback).all()
            
            texts = []
            labels = []
            weights = []
            
            for ticket in feedback_tickets:
                # Get the most recent feedback for this ticket
                feedback = db.query(RoutingFeedback).filter(
                    RoutingFeedback.ticket_id == ticket.id
                ).order_by(RoutingFeedback.created_at.desc()).first()
                
                if feedback:
                    text = f"{ticket.title} {ticket.description}"
                    texts.append(text)
                    
                    # Use correct team if feedback indicates wrong prediction
                    if feedback.was_correct:
                        labels.append(ticket.predicted_team_id)
                        weight = 1.0
                    else:
                        labels.append(feedback.correct_team_id or ticket.predicted_team_id)
                        # Give higher weight to corrected examples
                        weight = self.feedback_weight
                    
                    weights.append(weight)
            
            return texts, labels, weights
    
    def retrain_model(self) -> Dict[str, Any]:
        """Retrain the model with new feedback data."""
        logger.info("Starting model retraining with feedback data...")
        
        try:
            # Prepare retraining data
            texts, labels, weights = self.prepare_retraining_data()
            
            if len(texts) < 10:
                logger.warning("Insufficient data for retraining")
                return {"error": "Insufficient data for retraining"}
            
            # Retrain classifier
            model_scores = classifier.train(texts, labels, validation_split=0.2)
            
            # Save updated model
            model_filepath = f"{settings.MODEL_PATH}/ticket_classifier.pkl"
            classifier.save(model_filepath)
            
            # Log performance metrics
            self._log_performance_metrics(model_scores, len(texts))
            
            logger.info(f"Model retraining completed. New accuracy: {max(model_scores.values()):.4f}")
            
            return {
                'success': True,
                'model_scores': model_scores,
                'training_samples': len(texts),
                'retraining_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return {"error": str(e)}
    
    def _log_performance_metrics(self, model_scores: Dict[str, float], dataset_size: int):
        """Log performance metrics to database."""
        with get_db_context() as db:
            for model_name, accuracy in model_scores.items():
                performance = ModelPerformance(
                    model_version=classifier.model_version,
                    metric_name=f"{model_name}_accuracy",
                    metric_value=accuracy,
                    dataset_size=dataset_size,
                    training_duration_seconds=0,  # Could track actual training time
                    hyperparameters=json.dumps({}),  # Could track hyperparameters
                    feature_importance=json.dumps({})  # Could track feature importance
                )
                db.add(performance)
    
    def get_model_performance_history(self) -> List[Dict[str, Any]]:
        """Get historical model performance data."""
        with get_db_context() as db:
            performances = db.query(ModelPerformance).order_by(
                ModelPerformance.evaluation_date.desc()
            ).limit(100).all()
            
            return [
                {
                    'model_version': p.model_version,
                    'metric_name': p.metric_name,
                    'metric_value': p.metric_value,
                    'evaluation_date': p.evaluation_date.isoformat(),
                    'dataset_size': p.dataset_size
                }
                for p in performances
            ]
    
    def run_feedback_analysis(self) -> Dict[str, Any]:
        """Run comprehensive feedback analysis."""
        logger.info("Running feedback analysis...")
        
        # Analyze trends
        trends = self.analyze_feedback_trends()
        
        # Check if retraining is needed
        should_retrain = self.should_retrain()
        
        # Get performance history
        performance_history = self.get_model_performance_history()
        
        analysis = {
            'trends': trends,
            'should_retrain': should_retrain,
            'performance_history': performance_history,
            'analysis_date': datetime.now().isoformat()
        }
        
        if should_retrain:
            logger.info("Retraining recommended based on feedback analysis")
            retrain_result = self.retrain_model()
            analysis['retrain_result'] = retrain_result
        
        return analysis


# Global feedback loop instance
feedback_loop = FeedbackLoop()