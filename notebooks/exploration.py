"""
Data exploration and analysis script for the ML ticket routing system.
"""
import sys
import os
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from config.settings import settings
from data.generate_sample_data import create_training_dataset
from models.ticket_classifier import classifier
from utils.text_processor import text_processor


def analyze_data():
    """Analyze the training data."""
    print("=== DATA ANALYSIS ===")
    
    # Generate sample data
    df = create_training_dataset()
    print(f"Generated {len(df)} sample tickets")
    
    # Analyze team distribution
    team_counts = df['team'].value_counts()
    print("\nTeam Distribution:")
    print(team_counts)
    
    # Analyze priority distribution
    priority_counts = df['priority'].value_counts()
    print("\nPriority Distribution:")
    print(priority_counts)
    
    # Analyze text characteristics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print("\nText Statistics:")
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    print(f"Average word count: {df['word_count'].mean():.1f} words")
    print(f"Min text length: {df['text_length'].min()} characters")
    print(f"Max text length: {df['text_length'].max()} characters")
    
    return df


def analyze_features():
    """Analyze feature engineering."""
    print("\n=== FEATURE ANALYSIS ===")
    
    # Extract features for a sample of tickets
    df = create_training_dataset()
    sample_texts = df['text'].head(10).tolist()
    
    print("Feature extraction example:")
    features = text_processor.get_feature_vector(sample_texts[0])
    print(json.dumps(features, indent=2, default=str))
    
    # Analyze metadata features
    metadata_features = []
    for text in df['text'].head(100):  # Sample for performance
        metadata = text_processor.extract_metadata_features(text)
        metadata_features.append(metadata)
    
    metadata_df = pd.DataFrame(metadata_features)
    print("\nMetadata Features Summary:")
    print(metadata_df.describe())
    
    # Analyze keyword patterns
    all_keywords = []
    for text in df['text'].head(50):  # Sample for performance
        keywords = text_processor.extract_keywords(text, top_k=5)
        all_keywords.extend(keywords)
    
    from collections import Counter
    keyword_counts = Counter(all_keywords)
    print("\nTop 20 Keywords:")
    for keyword, count in keyword_counts.most_common(20):
        print(f"{keyword}: {count}")


def train_and_evaluate_model():
    """Train and evaluate the model."""
    print("\n=== MODEL TRAINING AND EVALUATION ===")
    
    # Generate data
    df = create_training_dataset()
    texts = df['text'].tolist()
    labels = df['team'].tolist()
    
    print("Training model...")
    model_scores = classifier.train(texts, labels)
    print("\nModel Scores:")
    for model_name, score in model_scores.items():
        print(f"{model_name}: {score:.4f}")
    
    # Evaluate model performance
    evaluation_results = classifier.evaluate(texts, labels)
    
    print("\nModel Evaluation Results:")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        # Print per-team metrics
        report = results['classification_report']
        for team, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"  {team}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    return model_scores, evaluation_results


def test_predictions():
    """Test predictions on sample tickets."""
    print("\n=== PREDICTION TESTING ===")
    
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
    
    print("Sample Predictions:")
    confidences = []
    
    for i, ticket in enumerate(test_tickets, 1):
        text = f"{ticket['title']} {ticket['description']}"
        prediction = classifier.predict(text)
        confidences.append(prediction['confidence'])
        
        print(f"\n{i}. {ticket['title']}")
        print(f"   Predicted Team: {prediction['predicted_team']}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        
        # Show top features
        if 'explanation' in prediction and 'top_features' in prediction['explanation']:
            print("   Top Features:")
            for feature in prediction['explanation']['top_features'][:3]:
                print(f"     - {feature['feature']}: {feature['importance']:.3f}")
    
    print(f"\nAverage confidence: {np.mean(confidences):.3f}")
    print(f"Confidence std: {np.std(confidences):.3f}")


def analyze_feature_importance():
    """Analyze feature importance."""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    test_tickets = [
        {"title": "Server Down", "description": "Server is down and not responding"},
        {"title": "Billing Issue", "description": "I was charged twice for the same service"},
        {"title": "Feature Help", "description": "How do I use the new collaboration feature?"},
        {"title": "Bug Report", "description": "The login page is broken and not working"},
        {"title": "Feature Request", "description": "Please add dark mode to the application"}
    ]
    
    feature_importance_data = []
    
    for ticket in test_tickets:
        text = f"{ticket['title']} {ticket['description']}"
        prediction = classifier.predict(text)
        
        if 'explanation' in prediction and 'top_features' in prediction['explanation']:
            for feature in prediction['explanation']['top_features']:
                feature_importance_data.append({
                    'feature': feature['feature'],
                    'importance': feature['importance'],
                    'ticket_title': ticket['title']
                })
    
    feature_df = pd.DataFrame(feature_importance_data)
    
    # Plot top features
    top_features = feature_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
    
    print("Top 10 Most Important Features:")
    for feature, importance in top_features.items():
        print(f"  {feature}: {importance:.3f}")


def performance_summary(model_scores):
    """Generate performance summary."""
    print("\n=== PERFORMANCE SUMMARY ===")
    
    df = create_training_dataset()
    texts = df['text'].tolist()
    labels = df['team'].tolist()
    
    # Test predictions for confidence analysis
    test_tickets = [
        {"title": "Test", "description": "Test description"},
        {"title": "Another Test", "description": "Another test description"}
    ]
    
    confidences = []
    for ticket in test_tickets:
        text = f"{ticket['title']} {ticket['description']}"
        prediction = classifier.predict(text)
        confidences.append(prediction['confidence'])
    
    print(f"Training samples: {len(texts)}")
    print(f"Number of teams: {len(set(labels))}")
    print(f"Feature count: {len(classifier.feature_names)}")
    print(f"\nModel Accuracies:")
    for model_name, score in model_scores.items():
        print(f"  {model_name}: {score:.4f}")
    print(f"\nBest model: {max(model_scores, key=model_scores.get)} "
          f"({max(model_scores.values()):.4f})")
    print(f"\nAverage prediction confidence: {np.mean(confidences):.3f}")
    print(f"Model version: {classifier.model_version}")


def main():
    """Run the complete analysis."""
    print("ML Ticket Routing System - Data Exploration")
    print("=" * 50)
    
    # Run all analyses
    df = analyze_data()
    analyze_features()
    model_scores, evaluation_results = train_and_evaluate_model()
    test_predictions()
    analyze_feature_importance()
    performance_summary(model_scores)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()