"""
Machine learning classifiers for ticket routing.
Includes Random Forest, XGBoost, Neural Network, and Ensemble models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BaseTicketClassifier:
    """Base class for ticket classifiers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize classifier with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        self.classes_ = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseTicketClassifier':
        """Fit the classifier."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        raise NotImplementedError
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature array
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, average='weighted'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y, y_pred, average=None
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y, y_pred),
            'per_class_metrics': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1_score': per_class_f1.tolist()
            },
            'prediction_confidence': {
                'mean': np.mean(np.max(y_proba, axis=1)),
                'std': np.std(np.max(y_proba, axis=1)),
                'min': np.min(np.max(y_proba, axis=1))
            }
        }
        
        return metrics


class RandomForestTicketClassifier(BaseTicketClassifier):
    """Random Forest classifier for ticket routing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        rf_config = config.get('random_forest', {})
        
        self.model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 20),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            class_weight=rf_config.get('class_weight', 'balanced'),
            n_jobs=rf_config.get('n_jobs', -1),
            random_state=config.get('random_state', 42)
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'RandomForestTicketClassifier':
        """
        Fit the Random Forest classifier.
        
        Args:
            X: Feature array
            y: Target labels
            feature_names: Optional feature names for importance analysis
            
        Returns:
            Self
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.feature_names = feature_names
        self.is_fitted = True
        
        logger.info(f"Random Forest trained with {self.model.n_estimators} trees")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


class XGBoostTicketClassifier(BaseTicketClassifier):
    """XGBoost classifier for ticket routing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        xgb_config = config.get('xgboost', {})
        
        self.model_params = {
            'n_estimators': xgb_config.get('n_estimators', 150),
            'max_depth': xgb_config.get('max_depth', 10),
            'learning_rate': xgb_config.get('learning_rate', 0.1),
            'subsample': xgb_config.get('subsample', 0.8),
            'colsample_bytree': xgb_config.get('colsample_bytree', 0.8),
            'scale_pos_weight': xgb_config.get('scale_pos_weight', 1),
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': config.get('random_state', 42)
        }
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'XGBoostTicketClassifier':
        """
        Fit the XGBoost classifier.
        
        Args:
            X: Feature array
            y: Target labels
            feature_names: Optional feature names
            eval_set: Optional validation set for early stopping
            
        Returns:
            Self
        """
        # Get number of classes
        self.classes_ = np.unique(y)
        self.model_params['num_class'] = len(self.classes_)
        
        # Create model
        self.model = xgb.XGBClassifier(**self.model_params)
        
        # Fit model
        if eval_set:
            self.model.fit(
                X, y,
                eval_set=[eval_set],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X, y)
            
        self.feature_names = feature_names
        self.is_fitted = True
        
        logger.info(f"XGBoost trained with {self.model.n_estimators} trees")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
        
    def get_feature_importance(self, importance_type: str = 'gain', top_n: int = 20) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Map feature indices to names
        feature_importance = []
        for feat_idx, importance in importance_dict.items():
            feat_name = self.feature_names[int(feat_idx[1:])] if self.feature_names else feat_idx
            feature_importance.append({'feature': feat_name, 'importance': importance})
            
        importance_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)
        return importance_df.head(top_n)


class NeuralNetworkTicketClassifier(BaseTicketClassifier):
    """Neural Network classifier for ticket routing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nn_config = config.get('neural_network', {})
        self.model = None
        self.scaler = None
        
    def _build_model(self, input_dim: int, output_dim: int) -> keras.Model:
        """Build the neural network architecture."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.nn_config.get('hidden_layers', [512, 256, 128])[0],
            activation=self.nn_config.get('activation', 'relu'),
            input_shape=(input_dim,)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.nn_config.get('dropout_rate', 0.3)))
        
        # Hidden layers
        for units in self.nn_config.get('hidden_layers', [512, 256, 128])[1:]:
            model.add(layers.Dense(units, activation=self.nn_config.get('activation', 'relu')))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.nn_config.get('dropout_rate', 0.3)))
        
        # Output layer
        model.add(layers.Dense(output_dim, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.nn_config.get('learning_rate', 0.001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'NeuralNetworkTicketClassifier':
        """
        Fit the Neural Network classifier.
        
        Args:
            X: Feature array
            y: Target labels
            validation_data: Optional validation set
            
        Returns:
            Self
        """
        # Store classes
        self.classes_ = np.unique(y)
        
        # Build model
        self.model = self._build_model(X.shape[1], len(self.classes_))
        
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.nn_config.get('early_stopping_patience', 5),
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Fit model
        history = self.model.fit(
            X, y,
            batch_size=self.nn_config.get('batch_size', 32),
            epochs=self.nn_config.get('epochs', 50),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info(f"Neural Network trained for {len(history.history['loss'])} epochs")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X, verbose=0)


class EnsembleTicketClassifier(BaseTicketClassifier):
    """Ensemble classifier combining multiple models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ensemble_config = config
        self.models = {}
        self.ensemble_model = None
        
        # Initialize individual models
        self._init_models()
        
    def _init_models(self):
        """Initialize component models."""
        model_types = self.ensemble_config.get('ensemble_models', ['random_forest', 'xgboost'])
        
        if 'random_forest' in model_types:
            self.models['random_forest'] = RandomForestTicketClassifier(self.ensemble_config)
            
        if 'xgboost' in model_types:
            self.models['xgboost'] = XGBoostTicketClassifier(self.ensemble_config)
            
        if 'neural_network' in model_types:
            self.models['neural_network'] = NeuralNetworkTicketClassifier(self.ensemble_config)
            
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'EnsembleTicketClassifier':
        """
        Fit the ensemble classifier.
        
        Args:
            X: Feature array
            y: Target labels
            feature_names: Optional feature names
            validation_data: Optional validation set
            
        Returns:
            Self
        """
        self.classes_ = np.unique(y)
        self.feature_names = feature_names
        
        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            if name == 'neural_network' and validation_data:
                model.fit(X, y, validation_data=validation_data)
            elif name == 'xgboost' and validation_data:
                model.fit(X, y, feature_names=feature_names, eval_set=validation_data)
            else:
                model.fit(X, y, feature_names=feature_names)
                
        # Create voting classifier
        estimators = [(name, model.model) for name, model in self.models.items()]
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=self.ensemble_config.get('weights', None)
        )
        
        # Fit ensemble
        self.ensemble_model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"Ensemble trained with {len(self.models)} models")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.ensemble_model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.ensemble_model.predict_proba(X)
        
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each model in the ensemble."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)
        return predictions
        
    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Evaluate each model in the ensemble."""
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            results[name] = model.evaluate(X, y)
            
        # Evaluate ensemble
        results['ensemble'] = self.evaluate(X, y)
        
        return results


def create_classifier(config: Dict[str, Any]) -> BaseTicketClassifier:
    """
    Factory function to create appropriate classifier.
    
    Args:
        config: Model configuration
        
    Returns:
        Classifier instance
    """
    model_type = config.get('type', 'random_forest')
    
    if model_type == 'random_forest':
        return RandomForestTicketClassifier(config)
    elif model_type == 'xgboost':
        return XGBoostTicketClassifier(config)
    elif model_type == 'neural_network':
        return NeuralNetworkTicketClassifier(config)
    elif model_type == 'ensemble':
        return EnsembleTicketClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class CalibratedTicketClassifier:
    """Wrapper for probability calibration."""
    
    def __init__(self, base_classifier: BaseTicketClassifier, method: str = 'sigmoid'):
        """
        Initialize calibrated classifier.
        
        Args:
            base_classifier: Base classifier to calibrate
            method: Calibration method ('sigmoid' or 'isotonic')
        """
        self.base_classifier = base_classifier
        self.calibrated_clf = CalibratedClassifierCV(
            base_classifier.model,
            method=method,
            cv=3
        )
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CalibratedTicketClassifier':
        """Fit the calibrated classifier."""
        self.calibrated_clf.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.calibrated_clf.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.calibrated_clf.predict_proba(X)