"""
Database setup with SQLAlchemy ORM (PostgreSQL)
"""

import os
from pathlib import Path
from dotenv import load_dotenv # type: ignore
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Get database URL from environment or use SQLite fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./shl_assessment.db"  # Fallback for local development
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
    pool_pre_ping=True,  # Verify connections are alive before using
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============================================================================
# Database Models
# ============================================================================

class Conversation(Base):
    """Track multi-turn conversations"""
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.utcnow)
    turns = Column(Integer, default=0)
    user_intent = Column(Text)
    constraints = Column(JSON, default={})  # Extracted constraints
    is_completed = Column(Boolean, default=False)
    
    # Relationships
    messages = relationship("MessageRecord", back_populates="conversation", cascade="all, delete-orphan")
    recommendations = relationship("RecommendationRecord", back_populates="conversation", cascade="all, delete-orphan")
    access_logs = relationship("AccessLog", back_populates="conversation", cascade="all, delete-orphan")


class MessageRecord(Base):
    """Store conversation messages"""
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class RecommendationRecord(Base):
    """Store all recommendations made during conversation"""
    __tablename__ = "recommendations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    assessment_name = Column(String)
    assessment_url = Column(String)
    test_type = Column(String)
    rank = Column(Integer)
    turn_number = Column(Integer)  # Which turn was this recommended
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="recommendations")


class AccessLog(Base):
    """Log all API accesses for monitoring"""
    __tablename__ = "access_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    endpoint = Column(String)  # "/health" or "/chat"
    status_code = Column(Integer)
    response_time_ms = Column(Integer)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="access_logs")


# ============================================================================
# Database Initialization
# ============================================================================

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("✓ Database initialized successfully")
