"""
Explainability module for ticket routing decisions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import logging
import io
import base64
from ..models.ml_models import TicketClassifier
from ..nlp.text_processor import TextProcessor, FeatureExtractor

logger = logging.getLogger(__name__)


class RoutingExplainer:
    """Main explainer class for routing decisions"""
    
    def __init__(self, classifier: TicketClassifier, text_processor: TextProcessor):
        self.classifier = classifier
        self.text_processor = text_processor
        self.feature_extractor = FeatureExtractor(text_processor)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = []
        self.is_initialized = False
    
    def initialize_explainers(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """Initialize SHAP and LIME explainers with training data"""
        if not self.classifier.is_fitted:
            raise ValueError("Classifier must be trained before initializing explainers")
        
        self.feature_names = feature_names or self.feature_extractor.get_feature_names()
        
        # Initialize SHAP explainer
        try:
            # Use the random forest model for SHAP (most interpretable)
            rf_model = self.classifier.models.get('random_forest')
            if rf_model:
                self.shap_explainer = shap.TreeExplainer(rf_model['classifier'])
                logger.info("SHAP TreeExplainer initialized")
            else:
                # Fallback to KernelExplainer for other models
                def model_predict(X):
                    predictions, _ = self.classifier.predict(X)
                    return predictions
                
                # Use a sample of training data for KernelExplainer
                sample_size = min(100, len(X_train))
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                background_data = X_train[sample_indices]
                
                self.shap_explainer = shap.KernelExplainer(model_predict, background_data)
                logger.info("SHAP KernelExplainer initialized")
        
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
        
        # Initialize LIME explainer
        try:
            self.lime_explainer = LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                class_names=[str(cls) for cls in self.classifier.label_encoder.classes_],
                mode='classification',
                discretize_continuous=True
            )
            logger.info("LIME explainer initialized")
        
        except Exception as e:
            logger.warning(f"Failed to initialize LIME explainer: {e}")
            self.lime_explainer = None
        
        self.is_initialized = True
    
    def explain_prediction(self, ticket_data: Dict[str, Any], 
                         explanation_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate explanation for a routing prediction"""
        if not self.is_initialized:
            raise ValueError("Explainers must be initialized before generating explanations")
        
        # Extract features
        features = self.feature_extractor.extract_ticket_features(ticket_data)
        features = features.reshape(1, -1)
        
        # Get prediction
        predictions, probabilities = self.classifier.predict(features)
        predicted_assignee = self.classifier.label_encoder.inverse_transform(predictions)[0]
        
        explanation = {
            'ticket_data': ticket_data,
            'predicted_assignee': predicted_assignee,
            'confidence': float(probabilities[0][predictions[0]]),
            'all_probabilities': {
                str(cls): float(prob) 
                for cls, prob in zip(self.classifier.label_encoder.classes_, probabilities[0])
            }
        }
        
        # Add different types of explanations
        if explanation_type in ["comprehensive", "shap"] and self.shap_explainer:
            explanation['shap_explanation'] = self._get_shap_explanation(features)
        
        if explanation_type in ["comprehensive", "lime"] and self.lime_explainer:
            explanation['lime_explanation'] = self._get_lime_explanation(features[0])
        
        if explanation_type in ["comprehensive", "feature_importance"]:
            explanation['feature_importance'] = self._get_feature_importance_explanation(features)
        
        if explanation_type in ["comprehensive", "rules"]:
            explanation['rule_based_explanation'] = self._get_rule_based_explanation(ticket_data)
        
        return explanation
    
    def _get_shap_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP-based explanation"""
        try:
            shap_values = self.shap_explainer.shap_values(features)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # For multi-class, use the values for the predicted class
                predicted_class_idx = np.argmax(self.classifier.predict(features)[1][0])
                shap_values_for_prediction = shap_values[predicted_class_idx][0]
            else:
                shap_values_for_prediction = shap_values[0]
            
            # Get top contributing features
            feature_contributions = []
            for i, (feature_name, shap_value) in enumerate(zip(self.feature_names, shap_values_for_prediction)):
                if abs(shap_value) > 0.001:  # Only include significant contributions
                    feature_contributions.append({
                        'feature': feature_name,
                        'shap_value': float(shap_value),
                        'feature_value': float(features[0][i]) if i < len(features[0]) else 0,
                        'impact': 'positive' if shap_value > 0 else 'negative'
                    })
            
            # Sort by absolute SHAP value
            feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return {
                'method': 'SHAP',
                'feature_contributions': feature_contributions[:10],  # Top 10
                'base_value': float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else 0,
                'total_impact': float(sum(shap_values_for_prediction))
            }
        
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {'method': 'SHAP', 'error': str(e)}
    
    def _get_lime_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate LIME-based explanation"""
        try:
            # Define prediction function for LIME
            def predict_fn(X):
                _, probabilities = self.classifier.predict(X)
                return probabilities
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                features,
                predict_fn,
                num_features=min(10, len(features)),
                top_labels=3
            )
            
            # Extract feature contributions
            feature_contributions = []
            for feature_idx, contribution in explanation.as_list():
                if isinstance(feature_idx, str):
                    feature_name = feature_idx
                else:
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                
                feature_contributions.append({
                    'feature': feature_name,
                    'contribution': float(contribution),
                    'impact': 'positive' if contribution > 0 else 'negative'
                })
            
            return {
                'method': 'LIME',
                'feature_contributions': feature_contributions,
                'local_prediction_accuracy': float(explanation.score),
                'intercept': float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0
            }
        
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {'method': 'LIME', 'error': str(e)}
    
    def _get_feature_importance_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate feature importance-based explanation"""
        try:
            # Use Random Forest feature importances
            rf_model = self.classifier.models.get('random_forest')
            if rf_model and hasattr(rf_model['classifier'], 'feature_importances_'):
                importances = rf_model['classifier'].feature_importances_
                
                # Combine feature importance with feature values
                feature_contributions = []
                for i, (importance, value) in enumerate(zip(importances, features[0])):
                    if i < len(self.feature_names) and importance > 0.001:
                        feature_contributions.append({
                            'feature': self.feature_names[i],
                            'importance': float(importance),
                            'feature_value': float(value),
                            'contribution': float(importance * abs(value))
                        })
                
                # Sort by contribution
                feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
                
                return {
                    'method': 'Feature Importance',
                    'feature_contributions': feature_contributions[:10],
                    'total_importance': float(sum(importances))
                }
        
        except Exception as e:
            logger.error(f"Error generating feature importance explanation: {e}")
        
        return {'method': 'Feature Importance', 'error': 'Not available'}
    
    def _get_rule_based_explanation(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rule-based explanation using domain knowledge"""
        text = f"{ticket_data.get('title', '')} {ticket_data.get('description', '')}"
        text_features = self.text_processor.extract_features(text)
        
        rules_triggered = []
        
        # Department signal rules
        dept_signals = text_features.get('department_signals', {})
        max_dept_signal = max(dept_signals.items(), key=lambda x: x[1]) if dept_signals else (None, 0)
        
        if max_dept_signal[1] > 0.1:
            rules_triggered.append({
                'rule': f"High {max_dept_signal[0]} department signal",
                'confidence': float(max_dept_signal[1]),
                'explanation': f"Ticket contains keywords strongly associated with {max_dept_signal[0]} department"
            })
        
        # Urgency rules
        urgency_score = text_features.get('urgency_indicators', {}).get('urgency_score', 0)
        if urgency_score > 0.5:
            rules_triggered.append({
                'rule': "High urgency detected",
                'confidence': float(urgency_score),
                'explanation': "Ticket contains urgent language or indicators"
            })
        
        # Technical complexity rules
        tech_complexity = text_features.get('technical_complexity', 0)
        if tech_complexity > 0.3:
            rules_triggered.append({
                'rule': "High technical complexity",
                'confidence': float(tech_complexity),
                'explanation': "Ticket contains technical terms requiring specialized knowledge"
            })
        
        # Priority rules
        priority = ticket_data.get('priority', 'medium').lower()
        if priority in ['high', 'critical']:
            rules_triggered.append({
                'rule': f"Priority level: {priority}",
                'confidence': 1.0,
                'explanation': f"Ticket marked as {priority} priority"
            })
        
        return {
            'method': 'Rule-based',
            'rules_triggered': rules_triggered,
            'total_rules_evaluated': len(rules_triggered)
        }
    
    def generate_explanation_report(self, ticket_data: Dict[str, Any]) -> str:
        """Generate a human-readable explanation report"""
        explanation = self.explain_prediction(ticket_data, "comprehensive")
        
        report = f"""
TICKET ROUTING EXPLANATION REPORT
================================

Ticket: {ticket_data.get('title', 'N/A')}
Predicted Assignee: {explanation['predicted_assignee']}
Confidence: {explanation['confidence']:.2%}

REASONING:
---------
"""
        
        # Add rule-based explanations (most interpretable)
        if 'rule_based_explanation' in explanation:
            rules = explanation['rule_based_explanation'].get('rules_triggered', [])
            if rules:
                report += "\nKey Decision Factors:\n"
                for rule in rules[:3]:  # Top 3 rules
                    report += f"• {rule['explanation']} (confidence: {rule['confidence']:.2%})\n"
        
        # Add feature importance
        if 'feature_importance' in explanation:
            features = explanation['feature_importance'].get('feature_contributions', [])
            if features:
                report += "\nTop Contributing Features:\n"
                for feature in features[:5]:  # Top 5 features
                    report += f"• {feature['feature']}: {feature['feature_value']:.3f} (importance: {feature['importance']:.3f})\n"
        
        # Add alternative suggestions
        all_probs = explanation.get('all_probabilities', {})
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_probs) > 1:
            report += "\nAlternative Suggestions:\n"
            for assignee, prob in sorted_probs[1:4]:  # Next 3 best options
                report += f"• {assignee}: {prob:.2%}\n"
        
        return report
    
    def create_explanation_visualization(self, ticket_data: Dict[str, Any]) -> str:
        """Create visualization for explanation (returns base64 encoded image)"""
        try:
            explanation = self.explain_prediction(ticket_data, "comprehensive")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Routing Explanation: {ticket_data.get('title', 'Ticket')[:50]}...", fontsize=16)
            
            # 1. Confidence scores for all assignees
            all_probs = explanation.get('all_probabilities', {})
            if all_probs:
                assignees = list(all_probs.keys())[:10]  # Top 10
                probabilities = [all_probs[a] for a in assignees]
                
                axes[0, 0].barh(assignees, probabilities)
                axes[0, 0].set_title('Assignee Confidence Scores')
                axes[0, 0].set_xlabel('Confidence')
                axes[0, 0].set_xlim(0, 1)
            
            # 2. Feature importance
            if 'feature_importance' in explanation:
                features_data = explanation['feature_importance'].get('feature_contributions', [])[:8]
                if features_data:
                    feature_names = [f['feature'] for f in features_data]
                    importances = [f['importance'] for f in features_data]
                    
                    axes[0, 1].barh(feature_names, importances)
                    axes[0, 1].set_title('Feature Importance')
                    axes[0, 1].set_xlabel('Importance')
            
            # 3. SHAP values (if available)
            if 'shap_explanation' in explanation:
                shap_data = explanation['shap_explanation'].get('feature_contributions', [])[:8]
                if shap_data:
                    feature_names = [f['feature'] for f in shap_data]
                    shap_values = [f['shap_value'] for f in shap_data]
                    colors = ['red' if v < 0 else 'blue' for v in shap_values]
                    
                    axes[1, 0].barh(feature_names, shap_values, color=colors)
                    axes[1, 0].set_title('SHAP Feature Contributions')
                    axes[1, 0].set_xlabel('SHAP Value')
                    axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # 4. Rule-based explanation
            if 'rule_based_explanation' in explanation:
                rules_data = explanation['rule_based_explanation'].get('rules_triggered', [])
                if rules_data:
                    rule_names = [r['rule'] for r in rules_data[:5]]
                    confidences = [r['confidence'] for r in rules_data[:5]]
                    
                    axes[1, 1].barh(rule_names, confidences)
                    axes[1, 1].set_title('Triggered Rules')
                    axes[1, 1].set_xlabel('Confidence')
                    axes[1, 1].set_xlim(0, 1)
            
            # Adjust layout and save to base64
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Convert to base64
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            plt.close()
            
            return img_base64
        
        except Exception as e:
            logger.error(f"Error creating explanation visualization: {e}")
            return ""
    
    def batch_explain(self, tickets_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanations for multiple tickets"""
        explanations = []
        
        for ticket_data in tickets_data:
            try:
                explanation = self.explain_prediction(ticket_data)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Error explaining ticket {ticket_data.get('id', 'unknown')}: {e}")
                explanations.append({
                    'ticket_data': ticket_data,
                    'error': str(e)
                })
        
        return explanations
    
    def get_global_explanations(self) -> Dict[str, Any]:
        """Get global model explanations"""
        if not self.classifier.is_fitted:
            return {'error': 'Model not trained'}
        
        global_explanation = {
            'model_info': self.classifier.get_model_info(),
            'feature_importance': {},
            'class_distribution': {}
        }
        
        # Global feature importance from Random Forest
        rf_model = self.classifier.models.get('random_forest')
        if rf_model and hasattr(rf_model['classifier'], 'feature_importances_'):
            importances = rf_model['classifier'].feature_importances_
            feature_importance = {}
            
            for i, importance in enumerate(importances):
                if i < len(self.feature_names):
                    feature_importance[self.feature_names[i]] = float(importance)
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            global_explanation['feature_importance'] = dict(sorted_features[:20])  # Top 20
        
        # Class distribution
        classes = self.classifier.label_encoder.classes_
        global_explanation['available_assignees'] = list(classes)
        
        return global_explanation