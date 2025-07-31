"""
Pydantic models for API requests and responses.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    """Request model for ticket routing."""
    title: str = Field(..., description="Ticket title", min_length=1, max_length=255)
    description: str = Field(..., description="Ticket description", min_length=1)
    user_email: Optional[str] = Field(None, description="User email address")
    priority: Optional[str] = Field("medium", description="Ticket priority (low, medium, high, urgent)")
    category: Optional[str] = Field(None, description="Ticket category")
    tags: Optional[List[str]] = Field(None, description="Ticket tags")


class TicketResponse(BaseModel):
    """Response model for ticket routing."""
    ticket_id: int
    predicted_team: str
    confidence: float
    explanation: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_version: str
    processing_time_ms: float
    created_at: datetime


class FeedbackRequest(BaseModel):
    """Request model for routing feedback."""
    ticket_id: int = Field(..., description="Ticket ID")
    was_correct: bool = Field(..., description="Whether the routing was correct")
    correct_team_id: Optional[int] = Field(None, description="Correct team ID if routing was wrong")
    user_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="User satisfaction rating (1-5)")
    resolution_time_hours: Optional[float] = Field(None, description="Time to resolution in hours")
    feedback_notes: Optional[str] = Field(None, description="Additional feedback notes")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_id: int
    ticket_id: int
    was_correct: bool
    model_version: str
    created_at: datetime
    message: str


class ModelStatus(BaseModel):
    """Model status information."""
    model_version: str
    is_loaded: bool
    last_training_date: Optional[datetime]
    accuracy: Optional[float]
    feature_count: int
    teams: List[str]
    model_size_mb: Optional[float]


class AnalyticsResponse(BaseModel):
    """Analytics response model."""
    total_tickets: int
    routing_accuracy: float
    average_confidence: float
    team_distribution: Dict[str, int]
    feedback_stats: Dict[str, Any]
    recent_predictions: List[Dict[str, Any]]


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    database_connected: bool
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime