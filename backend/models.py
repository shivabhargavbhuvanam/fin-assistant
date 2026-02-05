"""
SQLAlchemy ORM models for Smart Financial Coach.

Includes:
    - Session (with clerk_user_id for multi-user support)
    - Transaction, Category, Anomaly, RecurringCharge
    - Delta, Insight, Goal
    - Conversation, Message

Author: Smart Financial Coach Team
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Date, DateTime,
    ForeignKey, Text, JSON, Index
)
from sqlalchemy.orm import relationship
from database import Base


class Session(Base):
    """
    Upload session tracking.
    
    Each upload creates a new session. Users can have multiple sessions.
    All data is scoped by clerk_user_id for multi-user isolation.
    """
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    
    # User ownership (required for multi-user support)
    clerk_user_id = Column(String, nullable=False, index=True)
    
    # Session metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    filename = Column(String)
    row_count = Column(Integer)
    status = Column(String, default="processing")  # processing|ready|error
    
    # Sample data flag
    is_sample = Column(Boolean, default=False)
    
    # Display name for session switcher
    name = Column(String)

    # Relationships
    transactions = relationship("Transaction", back_populates="session", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="session", cascade="all, delete-orphan")
    recurring_charges = relationship("RecurringCharge", back_populates="session", cascade="all, delete-orphan")
    deltas = relationship("Delta", back_populates="session", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="session", cascade="all, delete-orphan")
    goals = relationship("Goal", back_populates="session", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")
    
    # Index for faster user queries
    __table_args__ = (
        Index('ix_sessions_clerk_user_id_created', 'clerk_user_id', 'created_at'),
    )


class Category(Base):
    """Reference table for transaction categories."""
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    icon = Column(String)
    color = Column(String)
    is_essential = Column(Boolean, default=False)

    transactions = relationship("Transaction", back_populates="category")
    recurring_charges = relationship("RecurringCharge", back_populates="category")
    deltas = relationship("Delta", back_populates="category")


class Transaction(Base):
    """Core transaction data."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    date = Column(Date, nullable=False)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"))
    category_confidence = Column(Float)
    category_source = Column(String)  # 'rule'|'ai'|'user'
    raw_description = Column(String)

    session = relationship("Session", back_populates="transactions")
    category = relationship("Category", back_populates="transactions")
    anomalies = relationship("Anomaly", back_populates="transaction")


class Anomaly(Base):
    """Detailed anomaly detection results."""
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    anomaly_type = Column(String)  # 'amount'|'frequency'|'merchant'
    severity = Column(String)  # 'low'|'medium'|'high'
    expected_value = Column(Float)
    actual_value = Column(Float)
    z_score = Column(Float)
    explanation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="anomalies")
    transaction = relationship("Transaction", back_populates="anomalies")


class RecurringCharge(Base):
    """Subscription and recurring charge detection."""
    __tablename__ = "recurring_charges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    description_pattern = Column(String)
    category_id = Column(Integer, ForeignKey("categories.id"))
    average_amount = Column(Float)
    frequency_days = Column(Integer)  # 30=monthly, 7=weekly
    occurrence_count = Column(Integer)
    first_seen = Column(Date)
    last_seen = Column(Date)
    is_gray_charge = Column(Boolean, default=False)
    confidence = Column(Float)

    session = relationship("Session", back_populates="recurring_charges")
    category = relationship("Category", back_populates="recurring_charges")


class Delta(Base):
    """Month-over-month spending changes."""
    __tablename__ = "deltas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"))
    current_month = Column(String)
    previous_month = Column(String)
    current_amount = Column(Float)
    previous_amount = Column(Float)
    change_amount = Column(Float)
    change_percent = Column(Float)

    session = relationship("Session", back_populates="deltas")
    category = relationship("Category", back_populates="deltas")


class Insight(Base):
    """AI-generated insights."""
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    type = Column(String, nullable=False)  # 'spending'|'anomaly'|'subscription'|'savings'|'goal'
    priority = Column(Integer)  # 1=high, 2=medium, 3=low
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    action = Column(String)
    reasoning = Column(Text)
    confidence = Column(Float)
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="insights")


class Goal(Base):
    """User savings goals."""
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    target_amount = Column(Float, nullable=False)
    suggested_cuts = Column(JSON)
    achievable = Column(Boolean)
    gap_amount = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="goals")


class Conversation(Base):
    """Chat conversation tracking per session."""
    __tablename__ = "conversations"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Conversation context - cached financial summary for quick access
    context_summary = Column(JSON)  # Cached spending summary, key metrics
    
    session = relationship("Session", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")


class Message(Base):
    """Individual chat messages."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user'|'assistant'|'system'|'tool'
    content = Column(Text, nullable=False)
    
    # For tool calls
    tool_calls = Column(JSON)  # If assistant made tool calls
    tool_call_id = Column(String)  # If this is a tool response
    tool_name = Column(String)  # Name of the tool called
    
    # Metadata
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
