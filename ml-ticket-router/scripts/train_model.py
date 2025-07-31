#!/usr/bin/env python3
"""
Training script for ML ticket routing models.
Handles data preparation, model training, evaluation, and saving.
"""

import os
import sys
import argparse
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import joblib
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.models.classifiers import create_classifier, CalibratedTicketClassifier
from src.models.explainability import TicketRoutingExplainer
from src.models.feedback_loop import FeedbackLoop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles the complete model training pipeline."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.text_preprocessor = None
        self.feature_engineer = None
        self.classifier = None
        self.explainer = None
        
        # Set up MLflow
        if self.config['monitoring']['mlflow']['enabled']:
            mlflow.set_tracking_uri(self.config['monitoring']['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['monitoring']['mlflow']['experiment_name'])
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load ticket data from CSV file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            DataFrame with ticket data
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Basic validation
        required_columns = ['ticket_id', 'description', 'assigned_to']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Add synthetic columns if missing (for demo)
        if 'created_at' not in df.columns:
            df['created_at'] = pd.date_range(
                start='2023-01-01', 
                periods=len(df), 
                freq='H'
            )
            
        if 'priority' not in df.columns:
            df['priority'] = np.random.choice(
                ['low', 'medium', 'high', 'critical'], 
                size=len(df),
                p=[0.2, 0.5, 0.25, 0.05]
            )
            
        if 'user_id' not in df.columns:
            df['user_id'] = np.random.randint(1000, 5000, size=len(df))
            
        logger.info(f"Loaded {len(df)} tickets with {len(df.columns)} features")
        return df
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Prepare features for training.
        
        Args:
            df: Raw ticket data
            
        Returns:
            Tuple of (features, labels, feature_dataframe)
        """
        logger.info("Preparing features...")
        
        # Initialize preprocessors
        self.text_preprocessor = TextPreprocessor(self.config['preprocessing'])
        self.feature_engineer = FeatureEngineer(self.config['preprocessing'])
        
        # Extract text features
        texts = df['description'].tolist()
        tfidf_matrix, text_features_df = self.text_preprocessor.fit_transform(texts)
        
        # Combine with original data
        combined_df = pd.concat([df, text_features_df], axis=1)
        
        # Feature engineering
        X_engineered = self.feature_engineer.fit_transform(combined_df)
        
        # Get embeddings
        embeddings = self.text_preprocessor.get_embeddings(texts)
        
        # Combine all features
        X = np.hstack([
            X_engineered,
            tfidf_matrix.toarray(),
            embeddings
        ])
        
        # Extract labels
        y = df['assigned_to']
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X, y, combined_df
        
    def train_model(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        """
        Train the classification model.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Training results dictionary
        """
        logger.info("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config['training']['validation_size'],
            random_state=self.config['training']['random_state'],
            stratify=y_train
        )
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'model_type': self.config['model']['type'],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            # Create classifier
            self.classifier = create_classifier(self.config['model'])
            
            # Train model
            if hasattr(self.classifier, 'fit'):
                self.classifier.fit(
                    X_train, y_train,
                    feature_names=self.feature_engineer.get_feature_names(),
                    validation_data=(X_val, y_val) if self.config['model']['type'] in ['xgboost', 'neural_network'] else None
                )
            
            # Evaluate on test set
            test_metrics = self.classifier.evaluate(X_test, y_test)
            
            # Cross-validation
            if self.config['training']['cross_validation']['enabled']:
                cv_scores = cross_val_score(
                    self.classifier.model,
                    X_train, y_train,
                    cv=self.config['training']['cross_validation']['n_folds'],
                    scoring=self.config['training']['cross_validation']['scoring']
                )
                test_metrics['cv_score_mean'] = cv_scores.mean()
                test_metrics['cv_score_std'] = cv_scores.std()
                
            # Log metrics
            mlflow.log_metrics({
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1_score']
            })
            
            # Generate classification report
            y_pred = self.classifier.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save artifacts
            mlflow.sklearn.log_model(self.classifier.model, "model")
            
            logger.info(f"Model trained with accuracy: {test_metrics['accuracy']:.3f}")
            
            return {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': test_metrics,
                'classification_report': report,
                'feature_names': self.feature_engineer.get_feature_names()
            }
            
    def setup_explainability(self, X_train: np.ndarray, y_train: pd.Series):
        """Set up model explainability."""
        logger.info("Setting up explainability...")
        
        self.explainer = TicketRoutingExplainer(
            self.classifier,
            self.feature_engineer.get_feature_names(),
            self.config
        )
        
        # Fit explainer
        class_names = sorted(y_train.unique())
        self.explainer.fit(X_train[:1000], class_names)  # Use subset for efficiency
        
        logger.info("Explainability configured")
        
    def save_model(self, output_dir: str, results: Dict[str, Any]):
        """
        Save trained model and artifacts.
        
        Args:
            output_dir: Directory to save model
            results: Training results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "classifier.pkl"
        joblib.dump(self.classifier, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save preprocessors
        text_prep_path = output_path / "text_preprocessor.pkl"
        self.text_preprocessor.save(text_prep_path)
        
        feature_eng_path = output_path / "feature_engineer.pkl"
        self.feature_engineer.save(feature_eng_path)
        
        # Save explainer
        explainer_path = output_path / "explainer.pkl"
        self.explainer.save(explainer_path)
        
        # Save class names
        class_names = sorted(self.classifier.classes_)
        with open(output_path / "class_names.json", 'w') as f:
            json.dump(class_names, f)
            
        # Save sample data for explainer
        sample_data = {
            'X': results.get('X_sample', np.array([])),
            'feature_names': results['feature_names']
        }
        joblib.dump(sample_data, output_path / "sample_data.pkl")
        
        # Save training results
        with open(output_path / "training_results.json", 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'metrics': results['metrics'],
                'classification_report': results['classification_report']
            }, f, indent=2)
            
        # Create model card
        self._create_model_card(output_path, results)
        
        logger.info(f"All artifacts saved to {output_path}")
        
    def _create_model_card(self, output_path: Path, results: Dict[str, Any]):
        """Create model documentation card."""
        model_card = f"""# ML Ticket Router Model Card

## Model Details
- **Model Type**: {self.config['model']['type']}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Version**: 1.0.0

## Training Data
- **Total Samples**: {results['train_samples'] + results['test_samples']}
- **Training Samples**: {results['train_samples']}
- **Test Samples**: {results['test_samples']}
- **Number of Classes**: {len(self.classifier.classes_)}
- **Features**: {len(results['feature_names'])}

## Performance
- **Accuracy**: {results['metrics']['accuracy']:.3f}
- **Precision**: {results['metrics']['precision']:.3f}
- **Recall**: {results['metrics']['recall']:.3f}
- **F1 Score**: {results['metrics']['f1_score']:.3f}

## Features Used
- Text features (TF-IDF, embeddings, sentiment)
- Temporal features (time of day, day of week)
- User features (history, account age)
- Engineered features (urgency score, complexity)

## Limitations
- Model performance may degrade with new ticket categories
- Requires retraining when team structure changes
- Performance depends on quality of ticket descriptions

## Ethical Considerations
- Ensure fair treatment across different user groups
- Monitor for bias in routing decisions
- Maintain transparency in routing logic
"""
        
        with open(output_path / "MODEL_CARD.md", 'w') as f:
            f.write(model_card)


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic ticket data for demo purposes."""
    logger.info(f"Generating {n_samples} synthetic tickets...")
    
    # Define categories and teams
    teams = [
        'Technical Support',
        'Billing',
        'Account Management', 
        'Product Feedback',
        'Security',
        'Sales'
    ]
    
    # Sample ticket templates
    templates = {
        'Technical Support': [
            "I can't login to my account, it says invalid password",
            "The application crashes when I try to upload files",
            "Getting error 500 when accessing the dashboard",
            "My data is not syncing across devices",
            "The export feature is not working properly"
        ],
        'Billing': [
            "I was charged twice for my subscription",
            "Need to update my payment method",
            "Request for invoice for last month",
            "Want to cancel my subscription",
            "Upgrade my plan to premium"
        ],
        'Account Management': [
            "Need to change my email address",
            "Request to merge two accounts",
            "Delete my account and all data",
            "Add team members to my account",
            "Change account ownership"
        ],
        'Product Feedback': [
            "Feature request: dark mode",
            "Suggestion for improving the UI",
            "The new update is great!",
            "Missing feature from competitor",
            "Feedback on recent changes"
        ],
        'Security': [
            "Suspicious login attempt on my account",
            "Report potential security vulnerability",
            "Enable two-factor authentication",
            "Account may have been compromised",
            "Security audit request"
        ],
        'Sales': [
            "Interested in enterprise pricing",
            "Request for product demo",
            "Questions about features",
            "Custom solution inquiry",
            "Partnership opportunity"
        ]
    }
    
    # Generate tickets
    tickets = []
    for i in range(n_samples):
        team = np.random.choice(teams, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
        template = np.random.choice(templates[team])
        
        # Add variations
        variations = [
            f"{template}",
            f"URGENT: {template}",
            f"{template}. This is really important!",
            f"Hi, {template}. Thanks!",
            f"{template}. I've tried everything but nothing works."
        ]
        
        description = np.random.choice(variations)
        
        tickets.append({
            'ticket_id': f'TICK-{i+1:05d}',
            'description': description,
            'assigned_to': team,
            'priority': np.random.choice(['low', 'medium', 'high', 'critical'], p=[0.2, 0.5, 0.25, 0.05]),
            'created_at': pd.Timestamp.now() - pd.Timedelta(hours=np.random.randint(0, 720))
        })
        
    df = pd.DataFrame(tickets)
    logger.info(f"Generated {len(df)} synthetic tickets")
    return df


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML ticket routing model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='data/models/latest',
                       help='Output directory for model artifacts')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for demo')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(args.config)
    
    # Load or generate data
    if args.synthetic or args.data is None:
        df = generate_synthetic_data(args.n_samples)
        # Save synthetic data
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/synthetic_tickets.csv', index=False)
    else:
        df = trainer.load_data(args.data)
        
    # Prepare features
    X, y, feature_df = trainer.prepare_features(df)
    
    # Train model
    results = trainer.train_model(X, y)
    results['X_sample'] = X[:100]  # Save sample for explainer
    
    # Setup explainability
    trainer.setup_explainability(X, y)
    
    # Save model
    trainer.save_model(args.output, results)
    
    logger.info("Training completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model Type: {trainer.config['model']['type']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
    print(f"Model saved to: {args.output}")
    print("="*50)


if __name__ == "__main__":
    main()