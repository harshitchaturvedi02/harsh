"""
Machine Learning models for ticket routing
"""
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class WorkloadBalancer:
    """Balance workload across team members"""
    
    def __init__(self, penalty_factor: float = 0.1):
        self.penalty_factor = penalty_factor
    
    def calculate_workload_penalty(self, user_workload: int, user_capacity: int) -> float:
        """Calculate penalty based on current workload"""
        if user_capacity == 0:
            return float('inf')
        
        workload_ratio = user_workload / user_capacity
        if workload_ratio >= 1.0:
            return float('inf')  # User is at capacity
        
        # Exponential penalty as workload approaches capacity
        penalty = np.exp(workload_ratio * 3) - 1
        return penalty * self.penalty_factor


class PerformanceTracker:
    """Track user performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def update_metrics(self, user_id: int, resolution_time: float, 
                      satisfaction_score: float, was_correctly_routed: bool):
        """Update performance metrics for a user"""
        if user_id not in self.metrics:
            self.metrics[user_id] = {
                'avg_resolution_time': [],
                'satisfaction_scores': [],
                'correct_routing_rate': [],
                'total_tickets': 0
            }
        
        metrics = self.metrics[user_id]
        metrics['avg_resolution_time'].append(resolution_time)
        metrics['satisfaction_scores'].append(satisfaction_score)
        metrics['correct_routing_rate'].append(1.0 if was_correctly_routed else 0.0)
        metrics['total_tickets'] += 1
    
    def get_performance_score(self, user_id: int) -> float:
        """Calculate overall performance score for a user"""
        if user_id not in self.metrics:
            return 0.5  # Default neutral score
        
        metrics = self.metrics[user_id]
        if not metrics['satisfaction_scores']:
            return 0.5
        
        # Weighted combination of metrics
        avg_satisfaction = np.mean(metrics['satisfaction_scores']) / 5.0  # Normalize to 0-1
        routing_accuracy = np.mean(metrics['correct_routing_rate'])
        
        # Inverse of resolution time (faster is better)
        if metrics['avg_resolution_time']:
            avg_resolution = np.mean(metrics['avg_resolution_time'])
            resolution_score = 1.0 / (1.0 + avg_resolution / 24.0)  # Normalize by 24 hours
        else:
            resolution_score = 0.5
        
        # Weighted average
        performance_score = (
            0.4 * avg_satisfaction +
            0.3 * routing_accuracy +
            0.3 * resolution_score
        )
        
        return performance_score


class TicketClassifier:
    """Main classifier for ticket routing"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.workload_balancer = WorkloadBalancer()
        self.performance_tracker = PerformanceTracker()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                ))
            ]),
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ))
            ]),
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
        }
    
    def _create_neural_network(self, input_dim: int, num_classes: int) -> keras.Model:
        """Create a neural network model"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, user_features: Optional[Dict] = None) -> Dict[str, Any]:
        """Train the models"""
        logger.info(f"Training models on {len(X)} samples with {X.shape[1]} features")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        results = {}
        
        # Train traditional ML models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results[name] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"{name} - Test Accuracy: {test_score:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Train neural network
        if X.shape[1] > 50:  # Only use NN for high-dimensional data
            try:
                logger.info("Training neural network...")
                
                nn_model = self._create_neural_network(X.shape[1], len(np.unique(y_encoded)))
                
                # Train with early stopping
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                history = nn_model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                self.models['neural_network'] = nn_model
                
                # Evaluate
                train_loss, train_acc = nn_model.evaluate(X_train, y_train, verbose=0)
                test_loss, test_acc = nn_model.evaluate(X_test, y_test, verbose=0)
                
                results['neural_network'] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                
                logger.info(f"Neural Network - Test Accuracy: {test_acc:.4f}")
                
            except Exception as e:
                logger.error(f"Error training neural network: {e}")
                results['neural_network'] = {'error': str(e)}
        
        self.is_fitted = True
        return results
    
    def predict(self, X: np.ndarray, user_data: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict assignee for tickets"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and hasattr(model, 'predict'):
                    probs = model.predict(X, verbose=0)
                    preds = np.argmax(probs, axis=1)
                else:
                    preds = model.predict(X)
                    probs = model.predict_proba(X)
                
                predictions[name] = preds
                probabilities[name] = probs
                
            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {e}")
        
        # Ensemble prediction (majority voting with confidence weighting)
        if self.model_type == "ensemble" and len(predictions) > 1:
            final_predictions, final_probabilities = self._ensemble_predict(predictions, probabilities)
        else:
            # Use best performing model (random forest as default)
            best_model = 'random_forest'
            final_predictions = predictions.get(best_model, list(predictions.values())[0])
            final_probabilities = probabilities.get(best_model, list(probabilities.values())[0])
        
        # Apply workload balancing if user data is provided
        if user_data:
            final_predictions, final_probabilities = self._apply_workload_balancing(
                final_predictions, final_probabilities, user_data
            )
        
        return final_predictions, final_probabilities
    
    def _ensemble_predict(self, predictions: Dict, probabilities: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Combine predictions from multiple models"""
        # Weight models based on their cross-validation performance
        model_weights = {
            'random_forest': 0.3,
            'gradient_boosting': 0.25,
            'logistic_regression': 0.2,
            'svm': 0.15,
            'neural_network': 0.1
        }
        
        # Average probabilities with weights
        weighted_probs = None
        total_weight = 0
        
        for name, probs in probabilities.items():
            weight = model_weights.get(name, 0.1)
            if weighted_probs is None:
                weighted_probs = probs * weight
            else:
                weighted_probs += probs * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_probs /= total_weight
        
        # Get final predictions
        final_predictions = np.argmax(weighted_probs, axis=1)
        
        return final_predictions, weighted_probs
    
    def _apply_workload_balancing(self, predictions: np.ndarray, probabilities: np.ndarray, 
                                 user_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply workload balancing to predictions"""
        balanced_predictions = predictions.copy()
        balanced_probabilities = probabilities.copy()
        
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Get top 3 candidates
            top_indices = np.argsort(probs)[-3:][::-1]
            
            best_user_id = None
            best_score = -1
            
            for idx in top_indices:
                user_id = self.label_encoder.inverse_transform([idx])[0]
                
                if user_id in user_data:
                    user_info = user_data[user_id]
                    
                    # Calculate combined score
                    prediction_confidence = probs[idx]
                    workload_penalty = self.workload_balancer.calculate_workload_penalty(
                        user_info.get('current_workload', 0),
                        user_info.get('workload_capacity', 10)
                    )
                    performance_score = self.performance_tracker.get_performance_score(user_id)
                    
                    # Combined score (higher is better)
                    if workload_penalty != float('inf'):
                        combined_score = (
                            prediction_confidence * 0.5 +
                            performance_score * 0.3 +
                            (1.0 - workload_penalty) * 0.2
                        )
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_user_id = idx
            
            if best_user_id is not None:
                balanced_predictions[i] = best_user_id
        
        return balanced_predictions, balanced_probabilities
    
    def explain_prediction(self, X: np.ndarray, prediction_idx: int) -> Dict[str, Any]:
        """Explain a specific prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before explanation")
        
        # Get feature importance from random forest
        rf_model = self.models['random_forest']['classifier']
        feature_importance = rf_model.feature_importances_
        
        # Get the specific sample
        sample = X[prediction_idx].reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.models['random_forest'].predict(sample)[0]
        probabilities = self.models['random_forest'].predict_proba(sample)[0]
        
        # Create explanation
        explanation = {
            'predicted_assignee': self.label_encoder.inverse_transform([prediction])[0],
            'confidence': probabilities[prediction],
            'top_features': [],
            'alternative_suggestions': []
        }
        
        # Top contributing features
        if len(self.feature_names) == len(feature_importance):
            feature_contributions = list(zip(self.feature_names, feature_importance))
            feature_contributions.sort(key=lambda x: x[1], reverse=True)
            explanation['top_features'] = feature_contributions[:10]
        
        # Alternative suggestions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_3_indices:
            if idx != prediction:
                explanation['alternative_suggestions'].append({
                    'assignee': self.label_encoder.inverse_transform([idx])[0],
                    'confidence': probabilities[idx]
                })
        
        return explanation
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'models': {},
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'model_type': self.model_type
        }
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'neural_network':
                model_data['models'][name] = model
        
        joblib.dump(model_data, filepath)
        
        # Save neural network separately if it exists
        if 'neural_network' in self.models:
            nn_path = filepath.replace('.joblib', '_nn.h5')
            self.models['neural_network'].save(nn_path)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        self.model_type = model_data['model_type']
        
        # Load neural network if it exists
        nn_path = filepath.replace('.joblib', '_nn.h5')
        if os.path.exists(nn_path):
            self.models['neural_network'] = keras.models.load_model(nn_path)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        return {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'num_classes': len(self.label_encoder.classes_) if self.is_fitted else 0,
            'classes': list(self.label_encoder.classes_) if self.is_fitted else [],
            'num_features': len(self.feature_names),
            'available_models': list(self.models.keys())
        }