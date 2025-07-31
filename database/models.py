"""
Database models for the ticket routing system.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Team(Base):
    """Support team model."""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    description = Column(Text)
    keywords = Column(Text)  # JSON string of keywords
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    tickets = relationship("Ticket", back_populates="assigned_team")
    members = relationship("TeamMember", back_populates="team")


class TeamMember(Base):
    """Team member model."""
    __tablename__ = "team_members"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True)
    email = Column(String(255), unique=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="members")
    assigned_tickets = relationship("Ticket", back_populates="assigned_member")


class Ticket(Base):
    """Support ticket model."""
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    description = Column(Text)
    priority = Column(String(20), default="medium")  # low, medium, high, urgent
    status = Column(String(20), default="open")  # open, in_progress, resolved, closed
    
    # ML Prediction fields
    predicted_team_id = Column(Integer, ForeignKey("teams.id"))
    predicted_confidence = Column(Float)
    predicted_features = Column(Text)  # JSON string of feature importance
    
    # Assignment fields
    assigned_team_id = Column(Integer, ForeignKey("teams.id"))
    assigned_member_id = Column(Integer, ForeignKey("team_members.id"))
    
    # Metadata
    user_email = Column(String(255), index=True)
    category = Column(String(100), index=True)
    tags = Column(Text)  # JSON string of tags
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    # Relationships
    assigned_team = relationship("Team", back_populates="tickets")
    assigned_member = relationship("TeamMember", back_populates="assigned_tickets")
    feedback = relationship("RoutingFeedback", back_populates="ticket")


class RoutingFeedback(Base):
    """Feedback on routing decisions."""
    __tablename__ = "routing_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"))
    
    # Feedback data
    was_correct = Column(Boolean)
    correct_team_id = Column(Integer, ForeignKey("teams.id"))
    user_satisfaction = Column(Integer)  # 1-5 scale
    resolution_time_hours = Column(Float)
    feedback_notes = Column(Text)
    
    # ML model info
    model_version = Column(String(50))
    prediction_confidence = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ticket = relationship("Ticket", back_populates="feedback")
    correct_team = relationship("Team")


class ModelPerformance(Base):
    """Model performance tracking."""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50), index=True)
    metric_name = Column(String(50), index=True)
    metric_value = Column(Float)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    dataset_size = Column(Integer)
    training_duration_seconds = Column(Float)
    
    # Additional metadata
    hyperparameters = Column(Text)  # JSON string
    feature_importance = Column(Text)  # JSON string