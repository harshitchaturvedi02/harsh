"""
Text preprocessing utilities for NLP analysis.
"""
import re
import string
import json
from typing import List, Dict, Any, Optional
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from config.settings import settings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load(settings.SPACY_MODEL)
except OSError:
    print(f"Downloading spaCy model: {settings.SPACY_MODEL}")
    spacy.cli.download(settings.SPACY_MODEL)
    nlp = spacy.load(settings.SPACY_MODEL)


class TextProcessor:
    """Text preprocessing and feature extraction."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text."""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.lower() not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def extract_spacy_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features using spaCy."""
        doc = nlp(text)
        
        features = {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'key_phrases': [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']],
            'sentiment_score': self._calculate_sentiment(doc),
            'complexity_score': self._calculate_complexity(doc)
        }
        
        return features
    
    def _calculate_sentiment(self, doc) -> float:
        """Calculate simple sentiment score based on positive/negative words."""
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'like'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'problem', 'issue', 'error'}
        
        positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate text complexity score."""
        if len(doc) == 0:
            return 0.0
        
        # Average word length
        avg_word_length = np.mean([len(token.text) for token in doc])
        
        # Percentage of long words (>6 characters)
        long_words = sum(1 for token in doc if len(token.text) > 6)
        long_word_ratio = long_words / len(doc)
        
        # Sentence complexity (average tokens per sentence)
        sentences = list(doc.sents)
        avg_sentence_length = np.mean([len(sent) for sent in sentences]) if sentences else 0
        
        # Normalize and combine
        complexity = (avg_word_length / 10) + long_word_ratio + (avg_sentence_length / 20)
        return min(complexity, 1.0)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text."""
        doc = nlp(text)
        
        # Get noun phrases and named entities
        keywords = []
        
        # Add noun chunks
        keywords.extend([chunk.text.lower() for chunk in doc.noun_chunks])
        
        # Add named entities
        keywords.extend([ent.text.lower() for ent in doc.ents])
        
        # Add important words (nouns, verbs, adjectives)
        keywords.extend([
            token.text.lower() 
            for token in doc 
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and len(token.text) > 3
        ])
        
        # Count frequencies and return top k
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(top_k)]
    
    def create_tfidf_features(self, texts: List[str], max_features: int = None) -> np.ndarray:
        """Create TF-IDF features from text list."""
        if max_features is None:
            max_features = settings.MAX_FEATURES
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Fit and transform
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        return tfidf_features.toarray()
    
    def extract_metadata_features(self, text: str) -> Dict[str, Any]:
        """Extract metadata features from text."""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'has_question_mark': '?' in text,
            'has_exclamation_mark': '!' in text,
            'has_urgent_words': self._has_urgent_words(text),
            'has_error_words': self._has_error_words(text),
            'has_technical_words': self._has_technical_words(text),
            'has_billing_words': self._has_billing_words(text)
        }
        
        return features
    
    def _has_urgent_words(self, text: str) -> bool:
        """Check if text contains urgent words."""
        urgent_words = {'urgent', 'emergency', 'critical', 'asap', 'immediately', 'broken', 'down'}
        return any(word in text.lower() for word in urgent_words)
    
    def _has_error_words(self, text: str) -> bool:
        """Check if text contains error-related words."""
        error_words = {'error', 'bug', 'crash', 'fail', 'broken', 'not working', 'issue', 'problem'}
        return any(word in text.lower() for word in error_words)
    
    def _has_technical_words(self, text: str) -> bool:
        """Check if text contains technical words."""
        technical_words = {'api', 'database', 'server', 'code', 'programming', 'technical', 'system'}
        return any(word in text.lower() for word in technical_words)
    
    def _has_billing_words(self, text: str) -> bool:
        """Check if text contains billing-related words."""
        billing_words = {'billing', 'payment', 'invoice', 'charge', 'subscription', 'refund', 'money'}
        return any(word in text.lower() for word in billing_words)
    
    def get_feature_vector(self, text: str) -> Dict[str, Any]:
        """Get comprehensive feature vector for a text."""
        cleaned_text = self.clean_text(text)
        
        features = {
            'text': cleaned_text,
            'tokens': self.tokenize_and_lemmatize(cleaned_text),
            'keywords': self.extract_keywords(text),
            'metadata': self.extract_metadata_features(text),
            'spacy_features': self.extract_spacy_features(text)
        }
        
        return features


# Global text processor instance
text_processor = TextProcessor()