"""
Explainability module for ticket routing decisions.
Provides interpretable explanations using SHAP and LIME.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import shap
import lime
import lime.lime_tabular
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TicketRoutingExplainer:
    """Provides explanations for ticket routing decisions."""
    
    def __init__(self, model, feature_names: List[str], config: Dict[str, Any]):
        """
        Initialize explainer with model and configuration.
        
        Args:
            model: Trained classifier model
            feature_names: List of feature names
            config: Explainability configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config.get('explainability', {})
        self.method = self.config.get('method', 'shap')
        
        self.shap_explainer = None
        self.lime_explainer = None
        self.background_data = None
        
        logger.info(f"Initialized explainer with method: {self.method}")
        
    def fit(self, X_train: np.ndarray, class_names: List[str]):
        """
        Fit the explainer on training data.
        
        Args:
            X_train: Training feature array
            class_names: List of class names
        """
        self.class_names = class_names
        self.background_data = X_train
        
        if self.method in ['shap', 'both']:
            self._init_shap_explainer(X_train)
            
        if self.method in ['lime', 'both']:
            self._init_lime_explainer(X_train)
            
        logger.info("Explainer fitted successfully")
        
    def _init_shap_explainer(self, X_train: np.ndarray):
        """Initialize SHAP explainer based on model type."""
        shap_config = self.config.get('shap', {})
        explainer_type = shap_config.get('explainer_type', 'tree')
        
        # Sample background data if too large
        max_samples = shap_config.get('sample_size', 100)
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            background = X_train[indices]
        else:
            background = X_train
            
        # Create appropriate explainer
        if hasattr(self.model, 'predict_proba'):
            if explainer_type == 'tree' and hasattr(self.model, 'model'):
                # For tree-based models (Random Forest, XGBoost)
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model.model)
                except:
                    # Fallback to kernel explainer
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        background
                    )
            else:
                # For other models (Neural Networks, etc.)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background
                )
        else:
            raise ValueError("Model must have predict_proba method")
            
    def _init_lime_explainer(self, X_train: np.ndarray):
        """Initialize LIME explainer."""
        lime_config = self.config.get('lime', {})
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
        
    def explain_instance(self, instance: np.ndarray, 
                        prediction: Optional[int] = None,
                        top_features: int = 10) -> Dict[str, Any]:
        """
        Explain a single routing decision.
        
        Args:
            instance: Feature array for single ticket (1D or 2D)
            prediction: Optional predicted class
            top_features: Number of top features to show
            
        Returns:
            Dictionary containing explanation details
        """
        # Ensure instance is 2D
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
            
        # Get prediction if not provided
        if prediction is None:
            prediction = self.model.predict(instance)[0]
            
        # Get prediction probabilities
        probabilities = self.model.predict_proba(instance)[0]
        
        explanation = {
            'prediction': int(prediction),
            'predicted_class': self.class_names[prediction],
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        # Get feature explanations
        if self.method in ['shap', 'both']:
            explanation['shap'] = self._get_shap_explanation(
                instance, prediction, top_features
            )
            
        if self.method in ['lime', 'both']:
            explanation['lime'] = self._get_lime_explanation(
                instance[0], prediction, top_features
            )
            
        # Add natural language explanation
        explanation['summary'] = self._generate_summary(explanation)
        
        return explanation
        
    def _get_shap_explanation(self, instance: np.ndarray, 
                             prediction: int, 
                             top_features: int) -> Dict[str, Any]:
        """Get SHAP-based explanation."""
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(instance)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # Multi-class: select values for predicted class
            class_shap_values = shap_values[prediction][0]
        else:
            # Binary or single array output
            if shap_values.ndim == 3:
                class_shap_values = shap_values[0, :, prediction]
            else:
                class_shap_values = shap_values[0]
                
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': class_shap_values,
            'abs_shap_value': np.abs(class_shap_values),
            'feature_value': instance[0]
        }).sort_values('abs_shap_value', ascending=False)
        
        # Get top features
        top_positive = feature_importance[feature_importance['shap_value'] > 0].head(top_features // 2)
        top_negative = feature_importance[feature_importance['shap_value'] < 0].head(top_features // 2)
        top_features_df = pd.concat([top_positive, top_negative]).sort_values('abs_shap_value', ascending=False)
        
        return {
            'feature_importance': top_features_df.to_dict('records'),
            'base_value': float(self.shap_explainer.expected_value[prediction] 
                               if isinstance(self.shap_explainer.expected_value, np.ndarray)
                               else self.shap_explainer.expected_value),
            'prediction_value': float(np.sum(class_shap_values) + 
                                    (self.shap_explainer.expected_value[prediction]
                                     if isinstance(self.shap_explainer.expected_value, np.ndarray)
                                     else self.shap_explainer.expected_value))
        }
        
    def _get_lime_explanation(self, instance: np.ndarray, 
                             prediction: int, 
                             top_features: int) -> Dict[str, Any]:
        """Get LIME-based explanation."""
        lime_config = self.config.get('lime', {})
        
        # Generate explanation
        exp = self.lime_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=top_features,
            num_samples=lime_config.get('num_samples', 5000)
        )
        
        # Extract feature importance
        feature_importance = []
        for feature_idx, importance in exp.as_list():
            feature_importance.append({
                'feature': feature_idx,
                'importance': importance,
                'abs_importance': abs(importance)
            })
            
        feature_importance_df = pd.DataFrame(feature_importance).sort_values(
            'abs_importance', ascending=False
        )
        
        return {
            'feature_importance': feature_importance_df.to_dict('records'),
            'local_prediction': exp.local_pred[0] if hasattr(exp, 'local_pred') else None,
            'intercept': exp.intercept[prediction] if hasattr(exp, 'intercept') else None
        }
        
    def _generate_summary(self, explanation: Dict[str, Any]) -> str:
        """Generate natural language summary of the explanation."""
        predicted_class = explanation['predicted_class']
        confidence = explanation['confidence']
        
        summary = f"The ticket has been routed to '{predicted_class}' with {confidence:.1%} confidence. "
        
        # Add top contributing factors
        if 'shap' in explanation:
            top_features = explanation['shap']['feature_importance'][:3]
            if top_features:
                summary += "Key factors: "
                factors = []
                for feat in top_features:
                    direction = "increases" if feat['shap_value'] > 0 else "decreases"
                    factors.append(f"{feat['feature']} (value: {feat['feature_value']:.2f}) {direction} likelihood")
                summary += "; ".join(factors) + ". "
                
        # Add alternative predictions if close
        probs = explanation['probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_probs) > 1 and sorted_probs[1][1] > 0.3:
            summary += f"Alternative routing to '{sorted_probs[1][0]}' was also considered ({sorted_probs[1][1]:.1%} confidence)."
            
        return summary
        
    def explain_model_global(self, X_sample: Optional[np.ndarray] = None, 
                           top_features: int = 20) -> Dict[str, Any]:
        """
        Generate global model explanations.
        
        Args:
            X_sample: Sample of data for explanation (uses background data if None)
            top_features: Number of top features to show
            
        Returns:
            Dictionary containing global explanation
        """
        if X_sample is None:
            X_sample = self.background_data[:1000]  # Use subset of background data
            
        global_explanation = {}
        
        if self.method in ['shap', 'both']:
            # Calculate SHAP values for sample
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                # Multi-class: average across classes
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
                
            # Create feature importance dataframe
            global_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False).head(top_features)
            
            global_explanation['feature_importance'] = global_importance.to_dict('records')
            global_explanation['shap_values'] = shap_values
            
        return global_explanation
        
    def plot_explanation(self, instance: np.ndarray, 
                        prediction: Optional[int] = None,
                        save_path: Optional[str] = None):
        """
        Create visualization of the explanation.
        
        Args:
            instance: Feature array for single ticket
            prediction: Optional predicted class
            save_path: Optional path to save the plot
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
            
        if prediction is None:
            prediction = self.model.predict(instance)[0]
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Prediction probabilities
        ax = axes[0, 0]
        probs = self.model.predict_proba(instance)[0]
        y_pos = np.arange(len(self.class_names))
        colors = ['green' if i == prediction else 'gray' for i in range(len(self.class_names))]
        ax.barh(y_pos, probs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_xlim(0, 1)
        
        # Plot 2: SHAP waterfall
        if self.method in ['shap', 'both']:
            ax = axes[0, 1]
            explanation = self.explain_instance(instance, prediction)
            shap_data = explanation['shap']['feature_importance'][:10]
            
            features = [d['feature'] for d in shap_data]
            values = [d['shap_value'] for d in shap_data]
            colors = ['red' if v < 0 else 'green' for v in values]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('SHAP Value')
            ax.set_title(f'Feature Contributions for {self.class_names[prediction]}')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
        # Plot 3: Feature values
        ax = axes[1, 0]
        feature_data = pd.DataFrame({
            'feature': self.feature_names,
            'value': instance[0]
        }).sort_values('value', ascending=False).head(15)
        
        y_pos = np.arange(len(feature_data))
        ax.barh(y_pos, feature_data['value'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_data['feature'])
        ax.set_xlabel('Feature Value')
        ax.set_title('Top Feature Values')
        
        # Plot 4: Global feature importance
        ax = axes[1, 1]
        if hasattr(self.model, 'get_feature_importance'):
            importance_df = self.model.get_feature_importance(top_n=15)
            y_pos = np.arange(len(importance_df))
            ax.barh(y_pos, importance_df['importance'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Global Feature Importance')
        else:
            ax.text(0.5, 0.5, 'Global importance not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Global Feature Importance')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def generate_report(self, instance: np.ndarray, 
                       ticket_id: str,
                       prediction: Optional[int] = None) -> str:
        """
        Generate a detailed explanation report.
        
        Args:
            instance: Feature array for single ticket
            ticket_id: Ticket identifier
            prediction: Optional predicted class
            
        Returns:
            Formatted report string
        """
        explanation = self.explain_instance(instance, prediction)
        
        report = f"""
TICKET ROUTING EXPLANATION REPORT
================================

Ticket ID: {ticket_id}
Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ROUTING DECISION
---------------
Assigned Team: {explanation['predicted_class']}
Confidence: {explanation['confidence']:.1%}

PREDICTION PROBABILITIES
-----------------------
"""
        for class_name, prob in sorted(explanation['probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True):
            report += f"{class_name:.<30} {prob:.1%}\n"
            
        report += """
KEY CONTRIBUTING FACTORS
-----------------------
"""
        if 'shap' in explanation:
            for i, feat in enumerate(explanation['shap']['feature_importance'][:10], 1):
                direction = "↑" if feat['shap_value'] > 0 else "↓"
                report += f"{i:2d}. {feat['feature']:<25} {direction} (value: {feat['feature_value']:.3f}, impact: {feat['shap_value']:.3f})\n"
                
        report += f"""
EXPLANATION SUMMARY
------------------
{explanation['summary']}

RECOMMENDATION
-------------
"""
        # Add recommendations based on confidence
        if explanation['confidence'] < 0.6:
            report += "Low confidence routing. Consider manual review or additional information gathering.\n"
        elif explanation['confidence'] < 0.8:
            report += "Moderate confidence routing. Monitor resolution time and user feedback.\n"
        else:
            report += "High confidence routing. Proceed with automated assignment.\n"
            
        report += """
================================
Generated by ML Ticket Router
"""
        
        return report
        
    def save(self, path: str):
        """Save explainer state."""
        import joblib
        joblib.dump({
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'background_data': self.background_data
        }, path)
        logger.info(f"Explainer saved to {path}")
        
    @classmethod
    def load(cls, path: str, model) -> 'TicketRoutingExplainer':
        """Load explainer from saved state."""
        import joblib
        state = joblib.load(path)
        
        explainer = cls(model, state['feature_names'], {'explainability': state['config']})
        explainer.class_names = state['class_names']
        explainer.background_data = state['background_data']
        explainer.fit(state['background_data'], state['class_names'])
        
        logger.info(f"Explainer loaded from {path}")
        return explainer