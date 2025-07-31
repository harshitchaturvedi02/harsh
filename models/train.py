"""
Model training script for the ticket routing system.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import joblib

from config.settings import settings
from models.ticket_classifier import classifier
from data.generate_sample_data import create_training_dataset
from database.database import init_db


def load_training_data() -> tuple:
    """Load training data from CSV or generate if not exists."""
    csv_path = os.path.join(settings.TRAINING_DATA_PATH, 'training_data.csv')
    
    if not os.path.exists(csv_path):
        logger.info("Training data not found. Generating sample data...")
        df = create_training_dataset()
    else:
        logger.info("Loading existing training data...")
        df = pd.read_csv(csv_path)
    
    # Prepare text and labels
    texts = df['text'].tolist()
    labels = df['team'].tolist()
    
    logger.info(f"Loaded {len(texts)} training samples")
    logger.info(f"Label distribution: {df['team'].value_counts().to_dict()}")
    
    return texts, labels


def train_model():
    """Train the ticket routing model."""
    logger.info("Starting model training...")
    
    # Load training data
    texts, labels = load_training_data()
    
    # Train the classifier
    model_scores = classifier.train(texts, labels, validation_split=settings.VALIDATION_SPLIT)
    
    # Create model directory
    os.makedirs(settings.MODEL_PATH, exist_ok=True)
    
    # Save the trained model
    model_filepath = os.path.join(settings.MODEL_PATH, 'ticket_classifier.pkl')
    classifier.save(model_filepath)
    
    # Save model metadata
    metadata = {
        'model_version': classifier.model_version,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(texts),
        'model_scores': model_scores,
        'feature_count': len(classifier.feature_names),
        'teams': list(set(labels))
    }
    
    metadata_path = os.path.join(settings.MODEL_PATH, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model training completed!")
    logger.info(f"Model saved to: {model_filepath}")
    logger.info(f"Model scores: {model_scores}")
    
    return model_scores


def evaluate_model():
    """Evaluate the trained model."""
    logger.info("Evaluating model...")
    
    # Load test data (use a portion of training data for now)
    texts, labels = load_training_data()
    
    # Split for evaluation
    test_size = int(len(texts) * settings.TEST_SPLIT)
    test_texts = texts[-test_size:]
    test_labels = labels[-test_size:]
    
    # Evaluate
    results = classifier.evaluate(test_texts, test_labels)
    
    # Print results
    for model_name, result in results.items():
        logger.info(f"\n{model_name.upper()} Results:")
        logger.info(f"Accuracy: {result['accuracy']:.4f}")
        
        # Print classification report
        report = result['classification_report']
        for team, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                logger.info(f"{team}: Precision={metrics['precision']:.3f}, "
                          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    return results


def test_predictions():
    """Test model predictions with sample tickets."""
    logger.info("Testing model predictions...")
    
    test_tickets = [
        {
            "title": "Server Down Emergency",
            "description": "Our production server is completely down and not responding. This is affecting all our customers. Need immediate assistance."
        },
        {
            "title": "Billing Question",
            "description": "I received an invoice for $200 but I think there's an error. Can you review my billing statement?"
        },
        {
            "title": "How to Use New Feature",
            "description": "I see there's a new collaboration feature. Can you show me how to use it and invite team members?"
        },
        {
            "title": "Login Bug Report",
            "description": "The login page is broken. Users cannot sign in and are getting error messages."
        },
        {
            "title": "Feature Request - Dark Mode",
            "description": "Please add a dark mode option to the application. Many users would appreciate this feature."
        }
    ]
    
    for i, ticket in enumerate(test_tickets, 1):
        text = f"{ticket['title']} {ticket['description']}"
        prediction = classifier.predict(text)
        
        logger.info(f"\nTest Ticket {i}:")
        logger.info(f"Title: {ticket['title']}")
        logger.info(f"Predicted Team: {prediction['predicted_team']}")
        logger.info(f"Confidence: {prediction['confidence']:.3f}")
        
        # Show top features
        if 'explanation' in prediction and 'top_features' in prediction['explanation']:
            logger.info("Top features:")
            for feature in prediction['explanation']['top_features'][:3]:
                logger.info(f"  - {feature['feature']}: {feature['importance']:.3f}")


def main():
    """Main training pipeline."""
    logger.info("Starting ticket routing model training pipeline...")
    
    # Initialize database
    init_db()
    
    # Train model
    model_scores = train_model()
    
    # Evaluate model
    evaluation_results = evaluate_model()
    
    # Test predictions
    test_predictions()
    
    logger.info("Training pipeline completed successfully!")
    
    return {
        'model_scores': model_scores,
        'evaluation_results': evaluation_results
    }


if __name__ == "__main__":
    main()