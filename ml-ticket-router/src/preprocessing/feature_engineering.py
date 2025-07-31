"""
Feature engineering module for ticket routing system.
Handles creation of derived features and feature transformations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for ticket data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.categorical_features = config.get('categorical_features', [])
        self.numerical_features = config.get('numerical_features', [])
        self.engineered_features = config.get('feature_engineering', [])
        
        self.label_encoders = {}
        self.preprocessor = None
        self.feature_names = []
        
        logger.info("FeatureEngineer initialized")
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp columns.
        
        Args:
            df: DataFrame with timestamp columns
            
        Returns:
            DataFrame with additional time features
        """
        # Assume 'created_at' column exists
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Extract time components
            df['hour'] = df['created_at'].dt.hour
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['day_of_month'] = df['created_at'].dt.day
            df['month'] = df['created_at'].dt.month
            df['quarter'] = df['created_at'].dt.quarter
            df['is_weekend'] = df['created_at'].dt.dayofweek.isin([5, 6]).astype(int)
            df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
            
            # Cyclical encoding for time features
            if 'time_of_day_sin' in self.engineered_features:
                df['time_of_day_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            if 'time_of_day_cos' in self.engineered_features:
                df['time_of_day_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            if 'day_of_week_sin' in self.engineered_features:
                df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            if 'day_of_week_cos' in self.engineered_features:
                df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Time since epoch (for trend analysis)
            df['timestamp_numeric'] = df['created_at'].astype(np.int64) // 10**9
            
        return df
    
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-based features.
        
        Args:
            df: DataFrame with user information
            
        Returns:
            DataFrame with additional user features
        """
        # User activity features
        if 'user_id' in df.columns:
            # Calculate user statistics (would be joined from historical data)
            user_stats = df.groupby('user_id').agg({
                'ticket_id': 'count',
                'priority': lambda x: (x == 'high').sum() / len(x) if len(x) > 0 else 0
            }).rename(columns={
                'ticket_id': 'user_ticket_count',
                'priority': 'user_high_priority_ratio'
            })
            
            df = df.merge(user_stats, on='user_id', how='left')
        
        # Account age features
        if 'account_created_at' in df.columns and 'created_at' in df.columns:
            df['account_created_at'] = pd.to_datetime(df['account_created_at'])
            df['account_age_days'] = (df['created_at'] - df['account_created_at']).dt.days
            df['is_new_user'] = (df['account_age_days'] < 30).astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different columns.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        # Priority and sentiment interaction
        if 'priority' in df.columns and 'sentiment_compound' in df.columns:
            df['priority_sentiment_interaction'] = (
                df['priority'].map({'low': 1, 'medium': 2, 'high': 3, 'critical': 4}) * 
                df['sentiment_compound']
            )
        
        # Urgency score combining multiple factors
        if all(col in df.columns for col in ['urgency_keywords_count', 'exclamation_count', 'caps_ratio']):
            df['urgency_score'] = (
                df['urgency_keywords_count'] * 2 + 
                df['exclamation_count'] + 
                df['caps_ratio'] * 5
            )
        
        # Complexity score
        if all(col in df.columns for col in ['word_count', 'entity_count', 'has_code']):
            df['complexity_score'] = (
                np.log1p(df['word_count']) + 
                df['entity_count'] * 0.5 + 
                df['has_code'].astype(int) * 2
            )
        
        return df
    
    def create_historical_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features based on historical ticket data.
        
        Args:
            df: Current DataFrame
            historical_df: Historical ticket data
            
        Returns:
            DataFrame with historical features
        """
        if historical_df is None:
            # Create placeholder features
            df['avg_resolution_time_by_category'] = 24.0  # Default 24 hours
            df['category_volume_last_7days'] = 100
            df['assignee_workload'] = 10
            return df
        
        # Calculate category-based statistics
        category_stats = historical_df.groupby('category').agg({
            'resolution_time_hours': ['mean', 'median', 'std'],
            'satisfaction_score': 'mean',
            'ticket_id': 'count'
        })
        
        # Flatten column names
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
        category_stats = category_stats.reset_index()
        
        # Merge with current data
        df = df.merge(category_stats, on='category', how='left')
        
        # Calculate assignee workload
        if 'assignee' in historical_df.columns:
            recent_date = historical_df['created_at'].max() - pd.Timedelta(days=7)
            recent_tickets = historical_df[historical_df['created_at'] > recent_date]
            
            assignee_workload = recent_tickets.groupby('assignee').size().reset_index(name='assignee_workload')
            df = df.merge(assignee_workload, on='assignee', how='left')
        
        return df
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineering pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target variable (optional)
            
        Returns:
            Self
        """
        # Create all engineered features
        X_engineered = self._engineer_features(X.copy())
        
        # Identify numeric and categorical columns
        numeric_features = list(set(self.numerical_features + [
            col for col in X_engineered.columns 
            if X_engineered[col].dtype in ['int64', 'float64'] and 
            col not in self.categorical_features
        ]))
        
        categorical_features = list(set(self.categorical_features + [
            col for col in X_engineered.columns 
            if X_engineered[col].dtype == 'object'
        ]))
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(X_engineered)
        
        # Store feature names for later use
        self._store_feature_names(numeric_features, categorical_features, X_engineered)
        
        logger.info(f"FeatureEngineer fitted with {len(self.feature_names)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature array
        """
        # Create engineered features
        X_engineered = self._engineer_features(X.copy())
        
        # Apply preprocessing
        X_transformed = self.preprocessor.transform(X_engineered)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit and transform features.
        
        Args:
            X: Feature DataFrame
            y: Target variable (optional)
            
        Returns:
            Transformed feature array
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create time features
        df = self.create_time_features(df)
        
        # Create user features
        df = self.create_user_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create historical features
        df = self.create_historical_features(df)
        
        # Custom feature engineering based on config
        for feature in self.engineered_features:
            if feature not in df.columns:
                if feature == 'ticket_length' and 'description' in df.columns:
                    df['ticket_length'] = df['description'].str.len()
                elif feature == 'title_length' and 'title' in df.columns:
                    df['title_length'] = df['title'].str.len()
                # Add more custom features as needed
        
        return df
    
    def _store_feature_names(self, numeric_features: List[str], 
                           categorical_features: List[str], 
                           X: pd.DataFrame):
        """Store feature names after transformation."""
        feature_names = []
        
        # Numeric features keep their names
        feature_names.extend(numeric_features)
        
        # Get categorical feature names after one-hot encoding
        if categorical_features:
            cat_transformer = self.preprocessor.named_transformers_['cat']
            ohe = cat_transformer.named_steps['onehot']
            cat_feature_names = []
            
            for i, cat_feature in enumerate(categorical_features):
                categories = ohe.categories_[i][1:]  # Skip first due to drop='first'
                cat_feature_names.extend([f"{cat_feature}_{cat}" for cat in categories])
            
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        return self.feature_names
    
    def get_feature_importance(self, model, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            raise ValueError("Model doesn't have feature importance information")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def save(self, path: str):
        """Save feature engineer state."""
        import joblib
        joblib.dump({
            'config': self.config,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders
        }, path)
        logger.info(f"FeatureEngineer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load feature engineer from saved state."""
        import joblib
        state = joblib.load(path)
        
        engineer = cls(state['config'])
        engineer.preprocessor = state['preprocessor']
        engineer.feature_names = state['feature_names']
        engineer.label_encoders = state['label_encoders']
        
        logger.info(f"FeatureEngineer loaded from {path}")
        return engineer