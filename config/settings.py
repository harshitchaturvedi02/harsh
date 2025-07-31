"""
Configuration settings for the ML Ticket Routing System.
"""
import os
from typing import List, Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    DATABASE_URL: str = "sqlite:///./ticket_routing.db"
    
    # ML Model Settings
    MODEL_PATH: str = "models/saved/"
    VECTORIZER_PATH: str = "models/saved/vectorizer.pkl"
    CLASSIFIER_PATH: str = "models/saved/classifier.pkl"
    EXPLAINER_PATH: str = "models/saved/explainer.pkl"
    
    # NLP Settings
    SPACY_MODEL: str = "en_core_web_sm"
    MAX_TEXT_LENGTH: int = 1000
    MIN_CONFIDENCE_THRESHOLD: float = 0.7
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "ticket-routing"
    
    # Weights & Biases
    WANDB_PROJECT: str = "ticket-routing"
    WANDB_ENTITY: Optional[str] = None
    
    # Teams and Departments
    SUPPORT_TEAMS: List[str] = [
        "technical_support",
        "billing_support", 
        "product_support",
        "general_inquiries",
        "bug_reports",
        "feature_requests"
    ]
    
    # Model Training
    TRAINING_DATA_PATH: str = "data/training/"
    VALIDATION_SPLIT: float = 0.2
    TEST_SPLIT: float = 0.1
    RANDOM_STATE: int = 42
    
    # Feature Engineering
    USE_TFIDF: bool = True
    USE_WORD2VEC: bool = True
    USE_BERT: bool = False  # Set to True for BERT embeddings
    MAX_FEATURES: int = 5000
    
    # Feedback Loop
    FEEDBACK_WEIGHT: float = 0.1
    RETRAIN_THRESHOLD: int = 1000  # Retrain after N new feedback samples
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()