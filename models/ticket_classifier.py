"""
Machine learning classifier for ticket routing.
"""
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import mlflow
import mlflow.sklearn
from loguru import logger

from config.settings import settings
from utils.text_processor import text_processor


class TicketClassifier:
    """Ensemble classifier for ticket routing."""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.explainer = None
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def prepare_features(self, texts: List[str], labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training/prediction."""
        # Text features
        tfidf_features = text_processor.create_tfidf_features(texts)
        self.vectorizer = text_processor.tfidf_vectorizer
        
        # Metadata features
        metadata_features = []
        for text in texts:
            metadata = text_processor.extract_metadata_features(text)
            metadata_features.append([
                metadata['text_length'],
                metadata['word_count'],
                metadata['sentence_count'],
                float(metadata['has_question_mark']),
                float(metadata['has_exclamation_mark']),
                float(metadata['has_urgent_words']),
                float(metadata['has_error_words']),
                float(metadata['has_technical_words']),
                float(metadata['has_billing_words'])
            ])
        
        metadata_features = np.array(metadata_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, metadata_features])
        
        # Store feature names for explainability
        tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
        metadata_feature_names = [
            'text_length', 'word_count', 'sentence_count', 'has_question_mark',
            'has_exclamation_mark', 'has_urgent_words', 'has_error_words',
            'has_technical_words', 'has_billing_words'
        ]
        self.feature_names = tfidf_feature_names + metadata_feature_names
        
        # Scale features
        if labels is not None:
            # Training mode - fit scaler
            combined_features = self.scaler.fit_transform(combined_features)
            labels_encoded = self.label_encoder.fit_transform(labels)
            return combined_features, labels_encoded
        else:
            # Prediction mode - transform only
            combined_features = self.scaler.transform(combined_features)
            return combined_features, None
    
    def train(self, texts: List[str], labels: List[str], validation_split: float = 0.2):
        """Train the ensemble classifier."""
        logger.info(f"Training classifier with {len(texts)} samples")
        
        # Prepare features
        X, y = self.prepare_features(texts, labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=settings.RANDOM_STATE, stratify=y
        )
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=settings.RANDOM_STATE
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=settings.RANDOM_STATE
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=settings.RANDOM_STATE
            )
        }
        
        # Train models
        model_scores = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            model_scores[name] = accuracy
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        # Create SHAP explainer for the best model
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = self.models[best_model_name]
        
        # Create explainer (using a subset for performance)
        explainer_data = X_val[:100] if len(X_val) > 100 else X_val
        self.explainer = shap.TreeExplainer(best_model) if hasattr(best_model, 'tree_') else shap.LinearExplainer(best_model, explainer_data)
        
        # Log to MLflow
        self._log_to_mlflow(model_scores, X_train.shape[0], X_val.shape[0])
        
        logger.info(f"Training completed. Best model: {best_model_name} ({model_scores[best_model_name]:.4f})")
        return model_scores
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict team assignment for a ticket."""
        # Prepare features
        X, _ = self.prepare_features([text])
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                pred_class = model.predict(X)[0]
                confidence = np.max(proba)
            else:
                pred_class = model.predict(X)[0]
                confidence = 1.0  # Default confidence for models without proba
            
            predictions[name] = pred_class
            confidences[name] = confidence
        
        # Ensemble prediction (majority vote with confidence weighting)
        ensemble_pred = self._ensemble_predict(predictions, confidences)
        
        # Get explanation
        explanation = self._get_explanation(X[0], ensemble_pred)
        
        # Decode prediction
        predicted_team = self.label_encoder.inverse_transform([ensemble_pred])[0]
        
        return {
            'predicted_team': predicted_team,
            'confidence': np.mean(list(confidences.values())),
            'model_predictions': predictions,
            'model_confidences': confidences,
            'explanation': explanation,
            'feature_importance': self._get_feature_importance(X[0])
        }
    
    def _ensemble_predict(self, predictions: Dict[str, int], confidences: Dict[str, float]) -> int:
        """Combine predictions from multiple models."""
        # Weighted voting based on confidence
        vote_counts = {}
        for model_name, pred in predictions.items():
            confidence = confidences[model_name]
            if pred not in vote_counts:
                vote_counts[pred] = 0
            vote_counts[pred] += confidence
        
        # Return prediction with highest weighted votes
        return max(vote_counts, key=vote_counts.get)
    
    def _get_explanation(self, features: np.ndarray, prediction: int) -> Dict[str, Any]:
        """Get SHAP explanation for prediction."""
        if self.explainer is None:
            return {"error": "Explainer not available"}
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(features.reshape(1, -1))
            
            # Get top features
            feature_importance = np.abs(shap_values[0])
            top_indices = np.argsort(feature_importance)[-10:]  # Top 10 features
            
            explanation = {
                'top_features': [
                    {
                        'feature': self.feature_names[i],
                        'importance': float(feature_importance[i]),
                        'value': float(features[i])
                    }
                    for i in top_indices
                ],
                'prediction_strength': float(np.sum(feature_importance))
            }
            
            return explanation
        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
            return {"error": str(e)}
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for the prediction."""
        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            importance_dict[feature_name] = float(features[i])
        return importance_dict
    
    def _log_to_mlflow(self, model_scores: Dict[str, float], train_size: int, val_size: int):
        """Log training results to MLflow."""
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("model_version", self.model_version)
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("val_size", val_size)
                mlflow.log_param("feature_count", len(self.feature_names))
                
                # Log metrics
                for model_name, score in model_scores.items():
                    mlflow.log_metric(f"{model_name}_accuracy", score)
                
                # Log models
                for model_name, model in self.models.items():
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
                
                # Log feature names
                mlflow.log_param("feature_names", json.dumps(self.feature_names))
                
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")
    
    def save(self, filepath: str):
        """Save the trained classifier."""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'explainer': self.explainer,
            'model_version': self.model_version
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained classifier."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.explainer = model_data['explainer']
        self.model_version = model_data['model_version']
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Evaluate model performance."""
        X, y = self.prepare_features(texts, labels)
        
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X)
            
            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'classification_report': classification_report(y, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist()
            }
        
        return results


# Global classifier instance
classifier = TicketClassifier()