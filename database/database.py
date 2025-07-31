"""
Database connection and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from typing import Generator

from config.settings import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import models to ensure they're registered
from database.models import Base

# Create tables
def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Initialize database
def init_db():
    """Initialize the database with default data."""
    create_tables()
    
    from database.models import Team, TeamMember
    
    with get_db_context() as db:
        # Check if teams already exist
        if db.query(Team).count() == 0:
            # Create default teams
            teams_data = [
                {
                    "name": "technical_support",
                    "description": "Handles technical issues, bugs, and system problems",
                    "keywords": '["error", "bug", "crash", "technical", "system", "login", "performance"]'
                },
                {
                    "name": "billing_support", 
                    "description": "Handles billing, payments, and subscription issues",
                    "keywords": '["billing", "payment", "invoice", "subscription", "refund", "charge"]'
                },
                {
                    "name": "product_support",
                    "description": "Handles product usage questions and feature requests",
                    "keywords": '["feature", "how to", "usage", "product", "guide", "tutorial"]'
                },
                {
                    "name": "general_inquiries",
                    "description": "Handles general questions and account management",
                    "keywords": '["account", "general", "question", "help", "information"]'
                },
                {
                    "name": "bug_reports",
                    "description": "Handles bug reports and technical issues",
                    "keywords": '["bug", "error", "issue", "problem", "broken", "not working"]'
                },
                {
                    "name": "feature_requests",
                    "description": "Handles feature requests and product suggestions",
                    "keywords": '["feature", "request", "suggestion", "improvement", "new", "enhancement"]'
                }
            ]
            
            for team_data in teams_data:
                team = Team(**team_data)
                db.add(team)
            
            db.commit()
            
            # Create some sample team members
            members_data = [
                {"name": "John Tech", "email": "john.tech@company.com", "team_id": 1},
                {"name": "Sarah Billing", "email": "sarah.billing@company.com", "team_id": 2},
                {"name": "Mike Product", "email": "mike.product@company.com", "team_id": 3},
                {"name": "Lisa General", "email": "lisa.general@company.com", "team_id": 4},
                {"name": "Alex Bug", "email": "alex.bug@company.com", "team_id": 5},
                {"name": "Emma Feature", "email": "emma.feature@company.com", "team_id": 6},
            ]
            
            for member_data in members_data:
                member = TeamMember(**member_data)
                db.add(member)
            
            db.commit()