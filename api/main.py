"""
FastAPI application for the ML ticket routing system.
"""
import time
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from loguru import logger

from config.settings import settings
from database.database import get_db, init_db
from database.models import Ticket, Team, RoutingFeedback, ModelPerformance
from models.ticket_classifier import classifier
from api.models import (
    TicketRequest, TicketResponse, FeedbackRequest, FeedbackResponse,
    ModelStatus, AnalyticsResponse, HealthCheck, ErrorResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ticket routing API...")
    
    # Initialize database
    init_db()
    
    # Load model if exists
    model_path = os.path.join(settings.MODEL_PATH, 'ticket_classifier.pkl')
    if os.path.exists(model_path):
        try:
            classifier.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning("No trained model found. Please train the model first.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ticket routing API...")


# Create FastAPI app
app = FastAPI(
    title="ML Ticket Routing System",
    description="A machine learning system for automatically routing support tickets to the most appropriate team",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "ML Ticket Routing System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database connection
        db.execute("SELECT 1")
        database_connected = True
    except Exception:
        database_connected = False
    
    # Check model status
    model_loaded = len(classifier.models) > 0
    
    return HealthCheck(
        status="healthy" if database_connected and model_loaded else "degraded",
        timestamp=datetime.now(),
        version="1.0.0",
        database_connected=database_connected,
        model_loaded=model_loaded
    )


@app.post("/predict", response_model=TicketResponse)
async def predict_ticket(
    request: TicketRequest,
    db: Session = Depends(get_db)
):
    """Predict team assignment for a new ticket."""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if not classifier.models:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train the model first."
            )
        
        # Prepare text for prediction
        text = f"{request.title} {request.description}"
        
        # Get prediction
        prediction = classifier.predict(text)
        
        # Create ticket in database
        ticket = Ticket(
            title=request.title,
            description=request.description,
            priority=request.priority,
            user_email=request.user_email,
            category=request.category,
            tags=json.dumps(request.tags) if request.tags else None,
            predicted_team_id=prediction['predicted_team'],
            predicted_confidence=prediction['confidence'],
            predicted_features=json.dumps(prediction['feature_importance'])
        )
        
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Ticket {ticket.id} routed to {prediction['predicted_team']} "
                   f"(confidence: {prediction['confidence']:.3f})")
        
        return TicketResponse(
            ticket_id=ticket.id,
            predicted_team=prediction['predicted_team'],
            confidence=prediction['confidence'],
            explanation=prediction['explanation'],
            feature_importance=prediction['feature_importance'],
            model_version=classifier.model_version,
            processing_time_ms=processing_time,
            created_at=ticket.created_at
        )
        
    except Exception as e:
        logger.error(f"Error predicting ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """Submit feedback on routing decision."""
    try:
        # Get the ticket
        ticket = db.query(Ticket).filter(Ticket.id == request.ticket_id).first()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Create feedback record
        feedback = RoutingFeedback(
            ticket_id=request.ticket_id,
            was_correct=request.was_correct,
            correct_team_id=request.correct_team_id,
            user_satisfaction=request.user_satisfaction,
            resolution_time_hours=request.resolution_time_hours,
            feedback_notes=request.feedback_notes,
            model_version=classifier.model_version,
            prediction_confidence=ticket.predicted_confidence
        )
        
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        
        logger.info(f"Feedback submitted for ticket {request.ticket_id}: "
                   f"correct={request.was_correct}")
        
        return FeedbackResponse(
            feedback_id=feedback.id,
            ticket_id=feedback.ticket_id,
            was_correct=feedback.was_correct,
            model_version=feedback.model_version,
            created_at=feedback.created_at,
            message="Feedback submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status", response_model=ModelStatus)
async def get_model_status():
    """Get model status and information."""
    try:
        # Get model file size
        model_path = os.path.join(settings.MODEL_PATH, 'ticket_classifier.pkl')
        model_size_mb = None
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Get metadata
        metadata_path = os.path.join(settings.MODEL_PATH, 'model_metadata.json')
        last_training_date = None
        accuracy = None
        teams = []
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                last_training_date = datetime.fromisoformat(metadata['training_date'])
                accuracy = max(metadata['model_scores'].values()) if metadata['model_scores'] else None
                teams = metadata.get('teams', [])
        
        return ModelStatus(
            model_version=classifier.model_version,
            is_loaded=len(classifier.models) > 0,
            last_training_date=last_training_date,
            accuracy=accuracy,
            feature_count=len(classifier.feature_names),
            teams=teams,
            model_size_mb=model_size_mb
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(db: Session = Depends(get_db)):
    """Get routing analytics and statistics."""
    try:
        # Get ticket statistics
        total_tickets = db.query(Ticket).count()
        
        # Get team distribution
        team_distribution = {}
        teams = db.query(Team).all()
        for team in teams:
            count = db.query(Ticket).filter(Ticket.predicted_team_id == team.name).count()
            team_distribution[team.name] = count
        
        # Get feedback statistics
        total_feedback = db.query(RoutingFeedback).count()
        correct_routings = db.query(RoutingFeedback).filter(RoutingFeedback.was_correct == True).count()
        routing_accuracy = correct_routings / total_feedback if total_feedback > 0 else 0
        
        # Get average confidence
        avg_confidence = db.query(Ticket.predicted_confidence).filter(
            Ticket.predicted_confidence.isnot(None)
        ).scalar()
        avg_confidence = float(avg_confidence) if avg_confidence else 0
        
        # Get recent predictions
        recent_tickets = db.query(Ticket).order_by(Ticket.created_at.desc()).limit(10).all()
        recent_predictions = []
        for ticket in recent_tickets:
            recent_predictions.append({
                'ticket_id': ticket.id,
                'title': ticket.title,
                'predicted_team': ticket.predicted_team_id,
                'confidence': ticket.predicted_confidence,
                'created_at': ticket.created_at.isoformat()
            })
        
        # Feedback stats
        feedback_stats = {
            'total_feedback': total_feedback,
            'correct_routings': correct_routings,
            'incorrect_routings': total_feedback - correct_routings,
            'average_satisfaction': db.query(RoutingFeedback.user_satisfaction).filter(
                RoutingFeedback.user_satisfaction.isnot(None)
            ).scalar() or 0
        }
        
        return AnalyticsResponse(
            total_tickets=total_tickets,
            routing_accuracy=routing_accuracy,
            average_confidence=avg_confidence,
            team_distribution=team_distribution,
            feedback_stats=feedback_stats,
            recent_predictions=recent_predictions
        )
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/teams", response_model=List[Dict[str, Any]])
async def get_teams(db: Session = Depends(get_db)):
    """Get all support teams."""
    try:
        teams = db.query(Team).all()
        return [
            {
                'id': team.id,
                'name': team.name,
                'description': team.description,
                'keywords': json.loads(team.keywords) if team.keywords else [],
                'member_count': len(team.members)
            }
            for team in teams
        ]
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickets", response_model=List[Dict[str, Any]])
async def get_tickets(
    limit: int = 50,
    offset: int = 0,
    team: str = None,
    db: Session = Depends(get_db)
):
    """Get tickets with optional filtering."""
    try:
        query = db.query(Ticket)
        
        if team:
            query = query.filter(Ticket.predicted_team_id == team)
        
        tickets = query.order_by(Ticket.created_at.desc()).offset(offset).limit(limit).all()
        
        return [
            {
                'id': ticket.id,
                'title': ticket.title,
                'description': ticket.description,
                'predicted_team': ticket.predicted_team_id,
                'confidence': ticket.predicted_confidence,
                'priority': ticket.priority,
                'status': ticket.status,
                'created_at': ticket.created_at.isoformat()
            }
            for ticket in tickets
        ]
    except Exception as e:
        logger.error(f"Error getting tickets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )