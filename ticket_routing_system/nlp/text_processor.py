"""
Natural Language Processing module for ticket content analysis
"""
import re
import spacy
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Advanced text processing for ticket routing"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the text processor with spaCy model"""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Installing...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Initialize BERT tokenizer and model for embeddings
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        
        # Topic modeling
        self.topic_model = LatentDirichletAllocation(
            n_components=20,
            random_state=42,
            max_iter=10
        )
        
        # Technical keywords for different departments
        self.department_keywords = {
            'technical_support': [
                'bug', 'error', 'crash', 'not working', 'broken', 'issue', 'problem',
                'server', 'database', 'api', 'connection', 'timeout', 'performance',
                'installation', 'configuration', 'setup', 'integration'
            ],
            'billing': [
                'payment', 'invoice', 'billing', 'charge', 'refund', 'subscription',
                'credit card', 'account', 'plan', 'upgrade', 'downgrade', 'pricing',
                'cost', 'fee', 'transaction', 'receipt'
            ],
            'sales': [
                'demo', 'trial', 'purchase', 'buy', 'quote', 'pricing', 'features',
                'comparison', 'competitor', 'discount', 'offer', 'proposal',
                'contract', 'license', 'enterprise'
            ],
            'product': [
                'feature request', 'enhancement', 'improvement', 'suggestion',
                'feedback', 'usability', 'ui', 'ux', 'design', 'workflow',
                'functionality', 'capability', 'roadmap'
            ],
            'security': [
                'security', 'vulnerability', 'breach', 'hack', 'password', 'login',
                'authentication', 'authorization', 'permissions', 'access',
                'encryption', 'privacy', 'gdpr', 'compliance'
            ]
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features from text"""
        preprocessed_text = self.preprocess_text(text)
        doc = self.nlp(preprocessed_text)
        
        features = {
            'text_length': len(text),
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': np.mean([len(token.text) for token in doc if not token.is_space]),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'entities': self._extract_entities(doc),
            'keywords': self._extract_keywords(doc),
            'sentiment': self._analyze_sentiment(doc),
            'urgency_indicators': self._detect_urgency(text),
            'technical_complexity': self._assess_technical_complexity(doc),
            'department_signals': self._detect_department_signals(preprocessed_text)
        }
        
        return features
    
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        return entities
    
    def _extract_keywords(self, doc) -> List[str]:
        """Extract important keywords using POS tagging and frequency"""
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Return unique keywords
        return list(set(keywords))
    
    def _analyze_sentiment(self, doc) -> Dict[str, float]:
        """Analyze sentiment using spaCy's sentiment analysis"""
        # Simple rule-based sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'broken', 'frustrated']
        
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop]
        
        positive_score = sum(1 for word in tokens if word in positive_words)
        negative_score = sum(1 for word in tokens if word in negative_words)
        
        total_score = positive_score + negative_score
        if total_score == 0:
            return {'polarity': 0.0, 'confidence': 0.0}
        
        polarity = (positive_score - negative_score) / len(tokens) if tokens else 0
        confidence = total_score / len(tokens) if tokens else 0
        
        return {'polarity': polarity, 'confidence': min(confidence, 1.0)}
    
    def _detect_urgency(self, text: str) -> Dict[str, Any]:
        """Detect urgency indicators in text"""
        urgency_keywords = [
            'urgent', 'asap', 'immediately', 'critical', 'emergency', 'high priority',
            'deadline', 'time sensitive', 'right away', 'quickly', 'fast'
        ]
        
        text_lower = text.lower()
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in text_lower)
        
        # Check for multiple exclamation marks
        multiple_exclamations = len(re.findall(r'!{2,}', text))
        
        # Check for ALL CAPS words
        caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
        
        urgency_score = (urgency_count * 0.4 + multiple_exclamations * 0.3 + caps_words * 0.3)
        
        return {
            'urgency_score': min(urgency_score, 1.0),
            'urgency_keywords_found': urgency_count,
            'caps_words': caps_words,
            'multiple_exclamations': multiple_exclamations
        }
    
    def _assess_technical_complexity(self, doc) -> float:
        """Assess technical complexity of the text"""
        technical_terms = [
            'api', 'database', 'server', 'code', 'function', 'algorithm', 'sql',
            'json', 'xml', 'http', 'ssl', 'encryption', 'authentication',
            'configuration', 'deployment', 'integration', 'middleware'
        ]
        
        tokens = [token.lemma_.lower() for token in doc]
        technical_count = sum(1 for token in tokens if token in technical_terms)
        
        # Normalize by text length
        complexity_score = technical_count / len(tokens) if tokens else 0
        
        return min(complexity_score * 10, 1.0)  # Scale to 0-1
    
    def _detect_department_signals(self, text: str) -> Dict[str, float]:
        """Detect signals for different departments"""
        department_scores = {}
        
        for department, keywords in self.department_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            # Normalize by number of keywords for the department
            department_scores[department] = score / len(keywords)
        
        return department_scores
    
    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text"""
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.bert_tokenizer(
            preprocessed_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings.flatten()
    
    def fit_tfidf(self, texts: List[str]) -> None:
        """Fit TF-IDF vectorizer on training texts"""
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        self.tfidf_vectorizer.fit(preprocessed_texts)
        logger.info(f"TF-IDF vectorizer fitted on {len(texts)} documents")
    
    def get_tfidf_features(self, text: str) -> np.ndarray:
        """Get TF-IDF features for text"""
        preprocessed_text = self.preprocess_text(text)
        return self.tfidf_vectorizer.transform([preprocessed_text]).toarray().flatten()
    
    def fit_topic_model(self, texts: List[str]) -> None:
        """Fit topic model on training texts"""
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        tfidf_features = self.tfidf_vectorizer.transform(preprocessed_texts)
        self.topic_model.fit(tfidf_features)
        logger.info(f"Topic model fitted with {self.topic_model.n_components} topics")
    
    def get_topic_features(self, text: str) -> np.ndarray:
        """Get topic distribution for text"""
        preprocessed_text = self.preprocess_text(text)
        tfidf_features = self.tfidf_vectorizer.transform([preprocessed_text])
        topic_distribution = self.topic_model.transform(tfidf_features)
        return topic_distribution.flatten()
    
    def get_comprehensive_features(self, text: str) -> Dict[str, Any]:
        """Get all features for a text"""
        # Basic NLP features
        nlp_features = self.extract_features(text)
        
        # TF-IDF features
        try:
            tfidf_features = self.get_tfidf_features(text)
            nlp_features['tfidf_features'] = tfidf_features
        except Exception as e:
            logger.warning(f"Could not extract TF-IDF features: {e}")
            nlp_features['tfidf_features'] = np.array([])
        
        # Topic features
        try:
            topic_features = self.get_topic_features(text)
            nlp_features['topic_features'] = topic_features
        except Exception as e:
            logger.warning(f"Could not extract topic features: {e}")
            nlp_features['topic_features'] = np.array([])
        
        # BERT embeddings
        try:
            bert_embeddings = self.get_bert_embeddings(text)
            nlp_features['bert_embeddings'] = bert_embeddings
        except Exception as e:
            logger.warning(f"Could not extract BERT embeddings: {e}")
            nlp_features['bert_embeddings'] = np.array([])
        
        return nlp_features


class FeatureExtractor:
    """Feature extraction for machine learning models"""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
    
    def extract_ticket_features(self, ticket_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from ticket data for ML models"""
        # Combine title and description
        text = f"{ticket_data.get('title', '')} {ticket_data.get('description', '')}"
        
        # Get comprehensive text features
        text_features = self.text_processor.get_comprehensive_features(text)
        
        # Extract numerical features
        numerical_features = [
            text_features['text_length'],
            text_features['word_count'],
            text_features['sentence_count'],
            text_features['avg_word_length'],
            text_features['exclamation_count'],
            text_features['question_count'],
            text_features['uppercase_ratio'],
            text_features['sentiment']['polarity'],
            text_features['sentiment']['confidence'],
            text_features['urgency_indicators']['urgency_score'],
            text_features['technical_complexity'],
        ]
        
        # Add department signal scores
        for dept, score in text_features['department_signals'].items():
            numerical_features.append(score)
        
        # Add priority encoding (if available)
        priority_mapping = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        priority_score = priority_mapping.get(ticket_data.get('priority', 'medium').lower(), 2)
        numerical_features.append(priority_score)
        
        # Convert to numpy array
        feature_vector = np.array(numerical_features)
        
        # Add TF-IDF features if available
        if len(text_features.get('tfidf_features', [])) > 0:
            feature_vector = np.concatenate([feature_vector, text_features['tfidf_features']])
        
        # Add topic features if available
        if len(text_features.get('topic_features', [])) > 0:
            feature_vector = np.concatenate([feature_vector, text_features['topic_features']])
        
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        base_features = [
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'uppercase_ratio',
            'sentiment_polarity', 'sentiment_confidence', 'urgency_score',
            'technical_complexity'
        ]
        
        # Add department signals
        for dept in self.text_processor.department_keywords.keys():
            base_features.append(f'dept_signal_{dept}')
        
        base_features.append('priority_score')
        
        # TF-IDF and topic features would be added dynamically
        return base_features