"""
Text preprocessing module for ticket routing system.
Handles NLP tasks including tokenization, feature extraction, and text analysis.
"""

import re
import string
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
from functools import lru_cache

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Advanced text preprocessing for ticket routing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.text_config = config.get('text_features', {})
        
        # Initialize NLP models
        self._init_nlp_models()
        
        # Initialize feature extractors
        self._init_feature_extractors()
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info("TextPreprocessor initialized successfully")
    
    def _init_nlp_models(self):
        """Initialize NLP models for text processing."""
        # SpaCy model for NER and POS tagging
        try:
            self.nlp = spacy.load(
                self.text_config.get('named_entities', {}).get('model', 'en_core_web_sm')
            )
        except OSError:
            logger.warning("SpaCy model not found. Downloading...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Sentence transformer for embeddings
        embedding_model = self.text_config.get('word_embeddings', {}).get(
            'model', 'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.sentence_transformer = SentenceTransformer(embedding_model)
        
        # Sentiment analyzer
        if self.text_config.get('sentiment', {}).get('enabled', True):
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
    def _init_feature_extractors(self):
        """Initialize feature extraction components."""
        # TF-IDF vectorizer
        tfidf_config = self.text_config.get('tfidf', {})
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 5000),
            ngram_range=tuple(tfidf_config.get('ngram_range', [1, 3])),
            min_df=tfidf_config.get('min_df', 2),
            max_df=tfidf_config.get('max_df', 0.95),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Urgency keywords
        self.urgency_keywords = {
            'urgent', 'asap', 'immediately', 'critical', 'emergency',
            'quickly', 'fast', 'now', 'today', 'priority', 'important',
            'severe', 'blocker', 'showstopper', 'breaking', 'down'
        }
        
    def _compile_patterns(self):
        """Compile regex patterns for text analysis."""
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
            'ticket_ref': re.compile(r'#?\b[A-Z]{2,}-\d+\b|\b\d{5,}\b'),
            'code_block': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            'special_chars': re.compile(r'[^a-zA-Z0-9\s]'),
            'multiple_spaces': re.compile(r'\s+'),
            'caps_words': re.compile(r'\b[A-Z]{2,}\b')
        }
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Preprocess a single text document.
        
        Args:
            text: Raw ticket text
            
        Returns:
            Dictionary containing preprocessed text and features
        """
        if not text or not isinstance(text, str):
            return self._empty_features()
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract basic features
        basic_features = self._extract_basic_features(text, cleaned_text)
        
        # Extract NLP features
        nlp_features = self._extract_nlp_features(cleaned_text)
        
        # Extract sentiment features
        sentiment_features = self._extract_sentiment_features(cleaned_text)
        
        # Combine all features
        features = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            **basic_features,
            **nlp_features,
            **sentiment_features
        }
        
        return features
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.patterns['url'].sub(' URL ', text)
        
        # Remove email addresses
        text = self.patterns['email'].sub(' EMAIL ', text)
        
        # Remove phone numbers
        text = self.patterns['phone'].sub(' PHONE ', text)
        
        # Remove ticket references
        text = self.patterns['ticket_ref'].sub(' TICKET_REF ', text)
        
        # Remove code blocks
        text = self.patterns['code_block'].sub(' CODE_BLOCK ', text)
        
        # Remove extra whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        return text.strip()
    
    def _extract_basic_features(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Extract basic text features."""
        features = {
            'ticket_length': len(original_text),
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text),
            'avg_word_length': np.mean([len(word) for word in cleaned_text.split()]) if cleaned_text else 0,
            'caps_ratio': len(self.patterns['caps_words'].findall(original_text)) / max(len(original_text.split()), 1),
            'punctuation_ratio': len([c for c in original_text if c in string.punctuation]) / max(len(original_text), 1),
            'special_char_ratio': len(self.patterns['special_chars'].findall(original_text)) / max(len(original_text), 1),
            'urgency_keywords_count': sum(1 for word in cleaned_text.split() if word in self.urgency_keywords),
            'has_url': bool(self.patterns['url'].search(original_text)),
            'has_email': bool(self.patterns['email'].search(original_text)),
            'has_phone': bool(self.patterns['phone'].search(original_text)),
            'has_ticket_ref': bool(self.patterns['ticket_ref'].search(original_text)),
            'has_code': bool(self.patterns['code_block'].search(original_text)),
            'exclamation_count': original_text.count('!'),
            'question_count': original_text.count('?'),
            'newline_count': original_text.count('\n')
        }
        
        return features
    
    @lru_cache(maxsize=1000)
    def _extract_nlp_features(self, text: str) -> Dict[str, Any]:
        """Extract NLP-based features using spaCy."""
        doc = self.nlp(text)
        
        # Named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_counts = {}
        for _, label in entities:
            entity_counts[f'entity_{label}'] = entity_counts.get(f'entity_{label}', 0) + 1
        
        # POS tags
        pos_counts = {}
        for token in doc:
            if not token.is_stop and not token.is_punct:
                pos_counts[f'pos_{token.pos_}'] = pos_counts.get(f'pos_{token.pos_}', 0) + 1
        
        # Dependency parsing features
        dep_counts = {}
        for token in doc:
            dep_counts[f'dep_{token.dep_}'] = dep_counts.get(f'dep_{token.dep_}', 0) + 1
        
        features = {
            'entity_count': len(entities),
            'unique_entity_count': len(set(entities)),
            **entity_counts,
            **pos_counts,
            **dep_counts,
            'noun_chunks_count': len(list(doc.noun_chunks)),
            'sentence_count': len(list(doc.sents))
        }
        
        return features
    
    def _extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract sentiment features."""
        if not self.sentiment_analyzer:
            return {}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        return {
            'sentiment_compound': scores['compound'],
            'sentiment_positive': scores['pos'],
            'sentiment_negative': scores['neg'],
            'sentiment_neutral': scores['neu'],
            'sentiment_label': self._get_sentiment_label(scores['compound'])
        }
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound sentiment score to label."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary."""
        return {
            'original_text': '',
            'cleaned_text': '',
            'ticket_length': 0,
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'caps_ratio': 0,
            'punctuation_ratio': 0,
            'special_char_ratio': 0,
            'urgency_keywords_count': 0,
            'has_url': False,
            'has_email': False,
            'has_phone': False,
            'has_ticket_ref': False,
            'has_code': False,
            'exclamation_count': 0,
            'question_count': 0,
            'newline_count': 0,
            'entity_count': 0,
            'unique_entity_count': 0,
            'noun_chunks_count': 0,
            'sentence_count': 0,
            'sentiment_compound': 0,
            'sentiment_positive': 0,
            'sentiment_negative': 0,
            'sentiment_neutral': 1,
            'sentiment_label': 'neutral'
        }
    
    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fit preprocessing pipeline and transform texts.
        
        Args:
            texts: List of ticket texts
            
        Returns:
            Tuple of (TF-IDF matrix, feature dataframe)
        """
        # Extract features for all texts
        all_features = []
        cleaned_texts = []
        
        for text in texts:
            features = self.preprocess_text(text)
            all_features.append(features)
            cleaned_texts.append(features['cleaned_text'])
        
        # Fit and transform TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        
        # Create feature dataframe
        feature_df = pd.DataFrame(all_features)
        
        # Add embeddings if enabled
        if self.text_config.get('word_embeddings', {}).get('enabled', True):
            embeddings = self.get_embeddings(cleaned_texts)
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
            )
            feature_df = pd.concat([feature_df, embedding_df], axis=1)
        
        return tfidf_matrix, feature_df
    
    def transform(self, texts: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transform texts using fitted preprocessing pipeline.
        
        Args:
            texts: List of ticket texts
            
        Returns:
            Tuple of (TF-IDF matrix, feature dataframe)
        """
        # Extract features for all texts
        all_features = []
        cleaned_texts = []
        
        for text in texts:
            features = self.preprocess_text(text)
            all_features.append(features)
            cleaned_texts.append(features['cleaned_text'])
        
        # Transform using fitted TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.transform(cleaned_texts)
        
        # Create feature dataframe
        feature_df = pd.DataFrame(all_features)
        
        # Add embeddings if enabled
        if self.text_config.get('word_embeddings', {}).get('enabled', True):
            embeddings = self.get_embeddings(cleaned_texts)
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
            )
            feature_df = pd.concat([feature_df, embedding_df], axis=1)
        
        return tfidf_matrix, feature_df
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get sentence embeddings for texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.sentence_transformer.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings
    
    def save(self, path: str):
        """Save preprocessor state."""
        import joblib
        joblib.dump({
            'config': self.config,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'urgency_keywords': self.urgency_keywords
        }, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TextPreprocessor':
        """Load preprocessor from saved state."""
        import joblib
        state = joblib.load(path)
        
        preprocessor = cls(state['config'])
        preprocessor.tfidf_vectorizer = state['tfidf_vectorizer']
        preprocessor.urgency_keywords = state['urgency_keywords']
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor