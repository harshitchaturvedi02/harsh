"""
FastAPI application for the ticket routing system
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime, timedelta
import os
import json
from sqlalchemy import func, Float

from ..models.data_models import (
    TicketCreate, TicketUpdate, Ticket, UserCreate, User, 
    FeedbackCreate, Feedback, RoutingPrediction
)
from ..models.ml_models import TicketClassifier
from ..nlp.text_processor import TextProcessor, FeatureExtractor
from ..feedback.feedback_loop import FeedbackLoop
from ..explainability.explainer import RoutingExplainer
from .database import get_db, engine, Base
from .dependencies import get_current_user, get_routing_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Ticket Routing System",
    description="ML-powered ticket routing with explainable AI and continuous learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for ML components
routing_service = None
feedback_loop = None
explainer = None


class RoutingService:
    """Main service class for ticket routing"""
    
    def __init__(self):
        self.classifier = TicketClassifier()
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor(self.text_processor)
        self.feedback_loop = None
        self.explainer = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the routing service"""
        try:
            # Try to load existing model
            model_path = "models/ticket_classifier_latest.joblib"
            if os.path.exists(model_path):
                self.classifier.load_model(model_path)
                logger.info("Loaded existing model")
            else:
                logger.warning("No existing model found. Please train a model first.")
            
            # Initialize feedback loop
            self.feedback_loop = FeedbackLoop(self.classifier, self.text_processor)
            
            # Initialize explainer
            self.explainer = RoutingExplainer(self.classifier, self.text_processor)
            
            self.is_initialized = True
            logger.info("Routing service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize routing service: {e}")
            raise
    
    def route_ticket(self, ticket_data: Dict[str, Any], 
                    user_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Route a ticket to the best assignee"""
        if not self.classifier.is_fitted:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Extract features
        features = self.feature_extractor.extract_ticket_features(ticket_data)
        features = features.reshape(1, -1)
        
        # Get prediction
        predictions, probabilities = self.classifier.predict(features, user_data)
        
        # Get assignee
        predicted_assignee = self.classifier.label_encoder.inverse_transform(predictions)[0]
        confidence = float(probabilities[0][predictions[0]])
        
        # Get alternative suggestions
        top_3_indices = probabilities[0].argsort()[-3:][::-1]
        alternatives = []
        for idx in top_3_indices:
            if idx != predictions[0]:
                assignee = self.classifier.label_encoder.inverse_transform([idx])[0]
                alternatives.append({
                    'assignee_id': assignee,
                    'confidence': float(probabilities[0][idx])
                })
        
        return {
            'predicted_assignee_id': predicted_assignee,
            'confidence': confidence,
            'alternatives': alternatives,
            'model_version': 'latest',
            'timestamp': datetime.now().isoformat()
        }


# Initialize routing service on startup
@app.on_event("startup")
async def startup_event():
    global routing_service
    routing_service = RoutingService()
    await routing_service.initialize()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ticket-routing-system",
        "version": "1.0.0"
    }


# Ticket endpoints
@app.post("/tickets", response_model=Dict[str, Any])
async def create_ticket(
    ticket: TicketCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new ticket and get routing recommendation"""
    try:
        # Create ticket in database
        from ..models.data_models import TicketDB
        db_ticket = TicketDB(**ticket.dict())
        db.add(db_ticket)
        db.commit()
        db.refresh(db_ticket)
        
        # Get routing recommendation
        ticket_data = {
            'title': ticket.title,
            'description': ticket.description,
            'priority': ticket.priority
        }
        
        # Get user data for workload balancing
        from ..models.data_models import UserDB
        users = db.query(UserDB).filter(UserDB.is_active == True).all()
        user_data = {
            user.id: {
                'current_workload': user.current_workload,
                'workload_capacity': user.workload_capacity,
                'skills': user.skills or [],
                'department': user.department
            }
            for user in users
        }
        
        routing_result = routing_service.route_ticket(ticket_data, user_data)
        
        # Store routing decision
        from ..models.data_models import RoutingDecisionDB
        routing_decision = RoutingDecisionDB(
            ticket_id=db_ticket.id,
            model_version=routing_result['model_version'],
            predicted_assignee_id=routing_result['predicted_assignee_id'],
            confidence_score=routing_result['confidence'],
            reasoning={'alternatives': routing_result['alternatives']},
            features_used={'ticket_data': ticket_data}
        )
        db.add(routing_decision)
        
        # Update ticket with predicted assignee
        db_ticket.assignee_id = routing_result['predicted_assignee_id']
        db.commit()
        
        return {
            'ticket_id': db_ticket.id,
            'routing_recommendation': routing_result,
            'status': 'created'
        }
        
    except Exception as e:
        logger.error(f"Error creating ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    """Get ticket by ID"""
    from ..models.data_models import TicketDB
    ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.put("/tickets/{ticket_id}", response_model=Ticket)
async def update_ticket(
    ticket_id: int,
    ticket_update: TicketUpdate,
    db: Session = Depends(get_db)
):
    """Update ticket"""
    from ..models.data_models import TicketDB
    ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Update fields
    for field, value in ticket_update.dict(exclude_unset=True).items():
        setattr(ticket, field, value)
    
    ticket.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(ticket)
    
    return ticket


@app.get("/tickets", response_model=List[Ticket])
async def list_tickets(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    assignee_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List tickets with optional filtering"""
    from ..models.data_models import TicketDB
    query = db.query(TicketDB)
    
    if status:
        query = query.filter(TicketDB.status == status)
    if assignee_id:
        query = query.filter(TicketDB.assignee_id == assignee_id)
    
    tickets = query.offset(skip).limit(limit).all()
    return tickets


# Routing endpoints
@app.post("/route", response_model=RoutingPrediction)
async def route_ticket_endpoint(
    ticket_data: Dict[str, Any],
    include_explanation: bool = False,
    db: Session = Depends(get_db)
):
    """Route a ticket without creating it in the database"""
    try:
        # Get user data for workload balancing
        from ..models.data_models import UserDB
        users = db.query(UserDB).filter(UserDB.is_active == True).all()
        user_data = {
            user.id: {
                'current_workload': user.current_workload,
                'workload_capacity': user.workload_capacity,
                'skills': user.skills or [],
                'department': user.department
            }
            for user in users
        }
        
        routing_result = routing_service.route_ticket(ticket_data, user_data)
        
        result = {
            'ticket_id': 0,  # Placeholder for non-persisted routing
            'predicted_assignee_id': routing_result['predicted_assignee_id'],
            'confidence_score': routing_result['confidence'],
            'reasoning': {'alternatives': routing_result['alternatives']},
            'alternative_suggestions': routing_result['alternatives'],
            'model_version': routing_result['model_version']
        }
        
        # Add explanation if requested
        if include_explanation and routing_service.explainer.is_initialized:
            explanation = routing_service.explainer.explain_prediction(ticket_data)
            result['explanation'] = explanation
        
        return result
        
    except Exception as e:
        logger.error(f"Error routing ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/{ticket_id}")
async def explain_routing_decision(
    ticket_id: int,
    explanation_type: str = "comprehensive",
    db: Session = Depends(get_db)
):
    """Explain a routing decision for a specific ticket"""
    try:
        from ..models.data_models import TicketDB
        ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        if not routing_service.explainer.is_initialized:
            raise HTTPException(status_code=503, detail="Explainer not initialized")
        
        ticket_data = {
            'title': ticket.title,
            'description': ticket.description,
            'priority': ticket.priority
        }
        
        explanation = routing_service.explainer.explain_prediction(
            ticket_data, explanation_type
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error explaining routing decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{ticket_id}/report")
async def get_explanation_report(ticket_id: int, db: Session = Depends(get_db)):
    """Get human-readable explanation report"""
    try:
        from ..models.data_models import TicketDB
        ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        ticket_data = {
            'title': ticket.title,
            'description': ticket.description,
            'priority': ticket.priority
        }
        
        report = routing_service.explainer.generate_explanation_report(ticket_data)
        
        return {"report": report}
        
    except Exception as e:
        logger.error(f"Error generating explanation report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# User management endpoints
@app.post("/users", response_model=User)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    from ..models.data_models import UserDB
    db_user = UserDB(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/users", response_model=List[User])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    department: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List users with optional filtering"""
    from ..models.data_models import UserDB
    query = db.query(UserDB)
    
    if is_active is not None:
        query = query.filter(UserDB.is_active == is_active)
    if department:
        query = query.filter(UserDB.department == department)
    
    users = query.offset(skip).limit(limit).all()
    return users


@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID"""
    from ..models.data_models import UserDB
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# Feedback endpoints
@app.post("/feedback", response_model=Feedback)
async def submit_feedback(
    feedback: FeedbackCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Submit feedback for a ticket"""
    from ..models.data_models import FeedbackDB, TicketDB
    
    # Verify ticket exists
    ticket = db.query(TicketDB).filter(TicketDB.id == feedback.ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Create feedback
    db_feedback = FeedbackDB(**feedback.dict())
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    
    # Update performance metrics in background
    if ticket.assignee_id and ticket.resolution_time_hours:
        background_tasks.add_task(
            update_performance_metrics,
            ticket.id,
            ticket.assignee_id,
            ticket.resolution_time_hours,
            feedback.rating,
            feedback.was_correctly_routed
        )
    
    return db_feedback


async def update_performance_metrics(
    ticket_id: int,
    assignee_id: int,
    resolution_time_hours: float,
    satisfaction_score: int,
    was_correctly_routed: bool
):
    """Background task to update performance metrics"""
    try:
        routing_service.feedback_loop.update_routing_performance(
            ticket_id,
            assignee_id,
            resolution_time_hours,
            satisfaction_score,
            was_correctly_routed
        )
        logger.info(f"Updated performance metrics for assignee {assignee_id}")
    except Exception as e:
        logger.error(f"Error updating performance metrics: {e}")


@app.get("/feedback/analyze")
async def analyze_feedback(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Analyze recent feedback"""
    try:
        analysis = await routing_service.feedback_loop.process_feedback_batch(db)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@app.post("/model/train")
async def train_model(
    background_tasks: BackgroundTasks,
    retrain_days: int = 90,
    db: Session = Depends(get_db)
):
    """Train or retrain the routing model"""
    try:
        since_date = datetime.now() - timedelta(days=retrain_days)
        
        # Run training in background
        background_tasks.add_task(
            run_model_training,
            db,
            since_date
        )
        
        return {
            "status": "training_started",
            "message": f"Model training started with data from last {retrain_days} days"
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_model_training(db: Session, since_date: datetime):
    """Background task for model training"""
    try:
        result = routing_service.feedback_loop.manual_retrain(db, since_date)
        logger.info(f"Model training completed: {result}")
    except Exception as e:
        logger.error(f"Model training failed: {e}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    try:
        info = routing_service.classifier.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        performance = routing_service.feedback_loop.get_performance_summary()
        return performance
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoints
@app.get("/analytics/dashboard")
async def get_dashboard_data(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get dashboard analytics data"""
    try:
        from ..models.data_models import TicketDB, FeedbackDB
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Ticket statistics
        total_tickets = db.query(TicketDB).filter(TicketDB.created_at >= cutoff_date).count()
        resolved_tickets = db.query(TicketDB).filter(
            TicketDB.created_at >= cutoff_date,
            TicketDB.status == 'resolved'
        ).count()
        
        # Feedback statistics
        feedback_query = db.query(FeedbackDB).join(TicketDB).filter(
            TicketDB.created_at >= cutoff_date
        )
        
        avg_rating = db.query(func.avg(FeedbackDB.rating)).join(TicketDB).filter(
            TicketDB.created_at >= cutoff_date
        ).scalar() or 0
        
        routing_accuracy = db.query(func.avg(FeedbackDB.was_correctly_routed.cast(Float))).join(TicketDB).filter(
            TicketDB.created_at >= cutoff_date
        ).scalar() or 0
        
        return {
            'total_tickets': total_tickets,
            'resolved_tickets': resolved_tickets,
            'resolution_rate': resolved_tickets / total_tickets if total_tickets > 0 else 0,
            'avg_satisfaction': avg_rating,
            'routing_accuracy': routing_accuracy,
            'period_days': days
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.post("/config/feedback-loop")
async def configure_feedback_loop(config: Dict[str, Any]):
    """Configure feedback loop parameters"""
    try:
        routing_service.feedback_loop.configure_feedback_loop(config)
        return {"status": "configured", "config": config}
    except Exception as e:
        logger.error(f"Error configuring feedback loop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# File upload endpoint for batch processing
@app.post("/tickets/batch")
async def upload_tickets_batch(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Upload and process tickets in batch"""
    try:
        # Read uploaded file
        content = await file.read()
        tickets_data = json.loads(content)
        
        results = []
        for ticket_data in tickets_data:
            try:
                # Create ticket
                ticket_create = TicketCreate(**ticket_data)
                
                # This would normally call create_ticket, but for simplicity:
                routing_result = routing_service.route_ticket({
                    'title': ticket_create.title,
                    'description': ticket_create.description,
                    'priority': ticket_create.priority
                })
                
                results.append({
                    'ticket_data': ticket_data,
                    'routing_result': routing_result,
                    'status': 'processed'
                })
                
            except Exception as e:
                results.append({
                    'ticket_data': ticket_data,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return {
            'total_processed': len(results),
            'successful': len([r for r in results if r['status'] == 'processed']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error processing batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)