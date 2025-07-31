"""
FastAPI application for ML-based ticket routing system.
Provides REST API endpoints for ticket routing, feedback, and model management.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import asyncio
from functools import lru_cache
import redis
import json
import hashlib
import os
import yaml

# Import our modules
from ..preprocessing.text_preprocessor import TextPreprocessor
from ..preprocessing.feature_engineering import FeatureEngineer
from ..models.classifiers import create_classifier
from ..models.explainability import TicketRoutingExplainer
from ..models.feedback_loop import FeedbackLoop
from ..utils.monitoring import MetricsCollector
from ..utils.auth import verify_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Ticket Router API",
    description="Intelligent ticket routing system with ML-based classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and components
text_preprocessor = None
feature_engineer = None
classifier = None
explainer = None
feedback_loop = None
metrics_collector = None
redis_client = None
config = None


# Pydantic models for request/response
class TicketRequest(BaseModel):
    ticket_id: str
    title: Optional[str] = None
    description: str
    priority: Optional[str] = "medium"
    category: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ['low', 'medium', 'high', 'critical']
        if v not in valid_priorities:
            raise ValueError(f'Priority must be one of {valid_priorities}')
        return v


class BatchTicketRequest(BaseModel):
    tickets: List[TicketRequest]
    
    @validator('tickets')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 tickets')
        return v


class RoutingResponse(BaseModel):
    ticket_id: str
    assigned_to: str
    confidence: float
    alternative_assignments: Dict[str, float]
    explanation: Optional[str] = None
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    ticket_id: str
    predicted_class: str
    feedback_type: str  # 'positive', 'negative', 'correction'
    actual_class: Optional[str] = None
    satisfaction_score: Optional[int] = Field(None, ge=1, le=5)
    resolution_time: Optional[float] = None
    comments: Optional[str] = None
    
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        valid_types = ['positive', 'negative', 'correction']
        if v not in valid_types:
            raise ValueError(f'Feedback type must be one of {valid_types}')
        return v
        
    @validator('actual_class')
    def validate_actual_class(cls, v, values):
        if values.get('feedback_type') == 'correction' and v is None:
            raise ValueError('actual_class is required for correction feedback')
        return v


class ModelPerformanceResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    feedback_stats: Dict[str, Any]
    last_updated: datetime


class ExplanationRequest(BaseModel):
    ticket_id: str
    include_visualization: bool = False


@lru_cache()
def load_config():
    """Load configuration from file."""
    config_path = os.getenv('CONFIG_PATH', 'config/model_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def initialize_components():
    """Initialize all ML components."""
    global text_preprocessor, feature_engineer, classifier, explainer, feedback_loop
    global metrics_collector, redis_client, config
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded")
        
        # Initialize Redis for caching
        if config.get('api', {}).get('caching', {}).get('enabled', True):
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True
            )
            logger.info("Redis client initialized")
        
        # Initialize preprocessors
        text_preprocessor = TextPreprocessor(config['preprocessing'])
        feature_engineer = FeatureEngineer(config['preprocessing'])
        logger.info("Preprocessors initialized")
        
        # Load trained models
        model_path = os.getenv('MODEL_PATH', 'data/models/latest')
        
        # Load preprocessor states
        text_preprocessor = TextPreprocessor.load(f"{model_path}/text_preprocessor.pkl")
        feature_engineer = FeatureEngineer.load(f"{model_path}/feature_engineer.pkl")
        
        # Load classifier
        import joblib
        classifier = joblib.load(f"{model_path}/classifier.pkl")
        logger.info("Models loaded")
        
        # Initialize explainer
        explainer = TicketRoutingExplainer(
            classifier,
            feature_engineer.get_feature_names(),
            config
        )
        
        # Load class names
        with open(f"{model_path}/class_names.json", 'r') as f:
            class_names = json.load(f)
        
        # Fit explainer with sample data
        sample_data = joblib.load(f"{model_path}/sample_data.pkl")
        explainer.fit(sample_data['X'], class_names)
        logger.info("Explainer initialized")
        
        # Initialize feedback loop
        feedback_loop = FeedbackLoop(classifier, config)
        feedback_loop.start()
        logger.info("Feedback loop started")
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector()
        logger.info("Metrics collector initialized")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await initialize_components()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if feedback_loop:
        feedback_loop.stop()
    if redis_client:
        redis_client.close()


def get_cache_key(ticket: TicketRequest) -> str:
    """Generate cache key for a ticket."""
    ticket_str = f"{ticket.ticket_id}:{ticket.description}:{ticket.priority}"
    return hashlib.md5(ticket_str.encode()).hexdigest()


async def process_single_ticket(ticket: TicketRequest, 
                              include_explanation: bool = False) -> RoutingResponse:
    """Process a single ticket for routing."""
    start_time = asyncio.get_event_loop().time()
    
    # Check cache first
    cache_key = get_cache_key(ticket)
    if redis_client:
        cached_result = redis_client.get(f"routing:{cache_key}")
        if cached_result:
            result = json.loads(cached_result)
            result['processing_time_ms'] = (asyncio.get_event_loop().time() - start_time) * 1000
            return RoutingResponse(**result)
    
    try:
        # Prepare ticket data
        ticket_data = pd.DataFrame([{
            'ticket_id': ticket.ticket_id,
            'title': ticket.title or '',
            'description': ticket.description,
            'priority': ticket.priority,
            'category': ticket.category,
            'user_id': ticket.user_id,
            'created_at': ticket.created_at or datetime.now(),
            **ticket.metadata
        }])
        
        # Text preprocessing
        text_features = []
        for _, row in ticket_data.iterrows():
            features = text_preprocessor.preprocess_text(row['description'])
            text_features.append(features)
        
        text_df = pd.DataFrame(text_features)
        
        # Combine with ticket data
        combined_df = pd.concat([ticket_data, text_df], axis=1)
        
        # Feature engineering
        X = feature_engineer.transform(combined_df)
        
        # Get embeddings
        embeddings = text_preprocessor.get_embeddings([ticket.description])
        
        # Combine all features
        X_final = np.hstack([X, embeddings])
        
        # Make prediction
        prediction = classifier.predict(X_final)[0]
        probabilities = classifier.predict_proba(X_final)[0]
        
        # Get class names
        class_names = explainer.class_names
        
        # Create response
        assigned_to = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        # Get alternative assignments
        sorted_indices = np.argsort(probabilities)[::-1]
        alternatives = {
            class_names[idx]: float(probabilities[idx])
            for idx in sorted_indices[:3]
            if idx != prediction
        }
        
        # Get explanation if requested
        explanation = None
        if include_explanation:
            explanation_data = explainer.explain_instance(X_final[0], prediction)
            explanation = explanation_data['summary']
        
        # Create response
        response = RoutingResponse(
            ticket_id=ticket.ticket_id,
            assigned_to=assigned_to,
            confidence=confidence,
            alternative_assignments=alternatives,
            explanation=explanation,
            processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
        )
        
        # Cache result
        if redis_client:
            redis_client.setex(
                f"routing:{cache_key}",
                config.get('api', {}).get('caching', {}).get('ttl', 3600),
                json.dumps(response.dict())
            )
        
        # Update metrics
        metrics_collector.record_prediction(assigned_to, confidence)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing ticket {ticket.ticket_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "ML Ticket Router API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check component status
        components_status = {
            "text_preprocessor": text_preprocessor is not None,
            "feature_engineer": feature_engineer is not None,
            "classifier": classifier is not None,
            "explainer": explainer is not None,
            "feedback_loop": feedback_loop is not None and feedback_loop.is_running,
            "redis": redis_client.ping() if redis_client else False
        }
        
        all_healthy = all(components_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/api/v1/route-ticket", 
         response_model=RoutingResponse,
         tags=["Routing"])
async def route_ticket(
    ticket: TicketRequest,
    include_explanation: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """Route a single ticket to the appropriate team."""
    return await process_single_ticket(ticket, include_explanation)


@app.post("/api/v1/route-batch",
         response_model=List[RoutingResponse],
         tags=["Routing"])
async def route_batch(
    request: BatchTicketRequest,
    include_explanation: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(verify_api_key)
):
    """Route multiple tickets in batch."""
    # Process tickets concurrently
    tasks = [
        process_single_ticket(ticket, include_explanation)
        for ticket in request.tickets
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and log them
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing ticket {request.tickets[i].ticket_id}: {result}")
        else:
            valid_results.append(result)
    
    return valid_results


@app.post("/api/v1/feedback", tags=["Feedback"])
async def submit_feedback(
    feedback: FeedbackRequest,
    api_key: str = Depends(verify_api_key)
):
    """Submit feedback for a routing decision."""
    try:
        # Prepare feedback data
        feedback_data = {
            'ticket_id': feedback.ticket_id,
            'predicted_class': feedback.predicted_class,
            'feedback_type': feedback.feedback_type,
            'timestamp': datetime.now()
        }
        
        if feedback.actual_class:
            feedback_data['actual_class'] = feedback.actual_class
        if feedback.satisfaction_score:
            feedback_data['satisfaction_score'] = feedback.satisfaction_score
        if feedback.resolution_time:
            feedback_data['resolution_time'] = feedback.resolution_time
            
        # Add to feedback loop
        feedback_loop.add_feedback(feedback_data)
        
        # Update metrics
        metrics_collector.record_feedback(feedback.feedback_type)
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "ticket_id": feedback.ticket_id
        }
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/performance",
        response_model=ModelPerformanceResponse,
        tags=["Model Management"])
async def get_model_performance(api_key: str = Depends(verify_api_key)):
    """Get current model performance metrics."""
    try:
        # Get feedback stats
        feedback_stats = feedback_loop.get_status()
        
        # Get current metrics
        metrics = metrics_collector.get_metrics()
        
        return ModelPerformanceResponse(
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1_score=metrics.get('f1_score', 0.0),
            total_predictions=metrics.get('total_predictions', 0),
            feedback_stats=feedback_stats['feedback_stats'],
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/explain/{ticket_id}", tags=["Explainability"])
async def explain_routing(
    ticket_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed explanation for a routing decision."""
    try:
        # For demo purposes, we'll create a sample explanation
        # In production, this would retrieve the actual features and prediction
        
        return {
            "ticket_id": ticket_id,
            "explanation": {
                "predicted_class": "Technical Support",
                "confidence": 0.85,
                "key_factors": [
                    {
                        "feature": "urgency_keywords_count",
                        "value": 3,
                        "impact": "positive",
                        "description": "High urgency detected in ticket"
                    },
                    {
                        "feature": "sentiment_negative",
                        "value": 0.7,
                        "impact": "positive",
                        "description": "Negative sentiment indicates issue"
                    },
                    {
                        "feature": "technical_terms_count",
                        "value": 5,
                        "impact": "positive",
                        "description": "Technical terminology present"
                    }
                ],
                "summary": "Ticket routed to Technical Support due to high urgency, negative sentiment, and technical content."
            }
        }
        
    except Exception as e:
        logger.error(f"Error explaining routing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/model/retrain", tags=["Model Management"])
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Manually trigger model retraining."""
    try:
        # In production, this would trigger an async retraining job
        background_tasks.add_task(retrain_model)
        
        return {
            "status": "success",
            "message": "Retraining job scheduled",
            "job_id": f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        logger.error(f"Error triggering retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def retrain_model():
    """Background task for model retraining."""
    logger.info("Starting model retraining...")
    # Placeholder for actual retraining logic
    await asyncio.sleep(5)  # Simulate retraining
    logger.info("Model retraining completed")


@app.get("/api/v1/stats/summary", tags=["Statistics"])
async def get_stats_summary(api_key: str = Depends(verify_api_key)):
    """Get summary statistics."""
    try:
        stats = {
            "routing_stats": metrics_collector.get_routing_stats(),
            "feedback_stats": feedback_loop.feedback_collector.get_stats(),
            "performance_trend": {
                "last_24h": metrics_collector.get_trend(hours=24),
                "last_7d": metrics_collector.get_trend(days=7),
                "last_30d": metrics_collector.get_trend(days=30)
            },
            "system_info": {
                "uptime_hours": metrics_collector.get_uptime_hours(),
                "total_requests": metrics_collector.total_requests,
                "cache_hit_rate": metrics_collector.get_cache_hit_rate()
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)