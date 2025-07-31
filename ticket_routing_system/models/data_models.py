"""
Data models for the ticket routing system
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Department(str, Enum):
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    SALES = "sales"
    PRODUCT = "product"
    SECURITY = "security"
    GENERAL = "general"


# SQLAlchemy Models
class TicketDB(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(20), default=TicketStatus.OPEN)
    priority = Column(String(20), default=TicketPriority.MEDIUM)
    department = Column(String(50))
    assignee_id = Column(Integer, ForeignKey("users.id"))
    creator_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime)
    resolution_time_hours = Column(Float)
    tags = Column(JSON)
    metadata = Column(JSON)
    
    # Relationships
    assignee = relationship("UserDB", foreign_keys=[assignee_id], back_populates="assigned_tickets")
    creator = relationship("UserDB", foreign_keys=[creator_id], back_populates="created_tickets")
    feedback = relationship("FeedbackDB", back_populates="ticket")
    routing_decisions = relationship("RoutingDecisionDB", back_populates="ticket")


class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True)
    email = Column(String(255), unique=True, index=True)
    full_name = Column(String(200))
    department = Column(String(50))
    skills = Column(JSON)  # List of skills/expertise areas
    workload_capacity = Column(Integer, default=10)
    current_workload = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    performance_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assigned_tickets = relationship("TicketDB", foreign_keys=[TicketDB.assignee_id], back_populates="assignee")
    created_tickets = relationship("TicketDB", foreign_keys=[TicketDB.creator_id], back_populates="creator")


class FeedbackDB(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"))
    rating = Column(Integer)  # 1-5 scale
    was_correctly_routed = Column(Boolean)
    resolution_quality = Column(Integer)  # 1-5 scale
    response_time_satisfaction = Column(Integer)  # 1-5 scale
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ticket = relationship("TicketDB", back_populates="feedback")


class RoutingDecisionDB(Base):
    __tablename__ = "routing_decisions"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"))
    model_version = Column(String(50))
    predicted_assignee_id = Column(Integer, ForeignKey("users.id"))
    confidence_score = Column(Float)
    reasoning = Column(JSON)  # Explanation of the decision
    features_used = Column(JSON)
    alternative_suggestions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ticket = relationship("TicketDB", back_populates="routing_decisions")
    predicted_assignee = relationship("UserDB")


# Pydantic Models for API
class TicketBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=10)
    priority: TicketPriority = TicketPriority.MEDIUM
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}


class TicketCreate(TicketBase):
    creator_id: int


class TicketUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TicketStatus] = None
    priority: Optional[TicketPriority] = None
    assignee_id: Optional[int] = None
    tags: Optional[List[str]] = None


class Ticket(TicketBase):
    id: int
    status: TicketStatus
    department: Optional[str]
    assignee_id: Optional[int]
    creator_id: int
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    resolution_time_hours: Optional[float]
    
    class Config:
        from_attributes = True


class UserBase(BaseModel):
    username: str
    email: str
    full_name: str
    department: Optional[str]
    skills: Optional[List[str]] = []
    workload_capacity: int = 10


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    current_workload: int
    is_active: bool
    performance_metrics: Optional[Dict[str, Any]] = {}
    created_at: datetime
    
    class Config:
        from_attributes = True


class FeedbackCreate(BaseModel):
    ticket_id: int
    rating: int = Field(..., ge=1, le=5)
    was_correctly_routed: bool
    resolution_quality: int = Field(..., ge=1, le=5)
    response_time_satisfaction: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None


class Feedback(FeedbackCreate):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class RoutingPrediction(BaseModel):
    ticket_id: int
    predicted_assignee_id: int
    confidence_score: float
    reasoning: Dict[str, Any]
    alternative_suggestions: List[Dict[str, Any]]
    model_version: str


class RoutingDecision(RoutingPrediction):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True