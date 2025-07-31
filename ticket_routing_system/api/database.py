"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL - can be configured via environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://user:password@localhost/ticket_routing_db"
)

# For SQLite (development/testing)
if os.getenv("ENVIRONMENT") == "development":
    DATABASE_URL = "sqlite:///./ticket_routing.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "false").lower() == "true"  # Enable SQL logging if needed
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    # Import all models to ensure they are registered
    from ..models.data_models import Base as ModelsBase
    ModelsBase.metadata.create_all(bind=engine)