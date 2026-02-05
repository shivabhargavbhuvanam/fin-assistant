"""SQLite database setup with 8 production-grade tables."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./financial_coach.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database with all tables and seed data."""
    from models import (
        Session, Category, Transaction, Anomaly,
        RecurringCharge, Delta, Insight, Goal,
        Conversation, Message
    )

    Base.metadata.create_all(bind=engine)

    # Seed default categories
    db = SessionLocal()
    try:
        existing = db.query(Category).count()
        if existing == 0:
            default_categories = [
                Category(name="Income", icon="💰", color="#22c55e", is_essential=False),
                Category(name="Housing", icon="🏠", color="#3b82f6", is_essential=True),
                Category(name="Utilities", icon="💡", color="#6366f1", is_essential=True),
                Category(name="Groceries", icon="🛒", color="#10b981", is_essential=True),
                Category(name="Dining", icon="🍽️", color="#f59e0b", is_essential=False),
                Category(name="Transportation", icon="🚗", color="#8b5cf6", is_essential=True),
                Category(name="Healthcare", icon="🏥", color="#ef4444", is_essential=True),
                Category(name="Subscriptions", icon="📺", color="#ec4899", is_essential=False),
                Category(name="Shopping", icon="🛍️", color="#f97316", is_essential=False),
                Category(name="Entertainment", icon="🎮", color="#a855f7", is_essential=False),
                Category(name="Transfer", icon="↔️", color="#64748b", is_essential=False),
                Category(name="Other", icon="📝", color="#94a3b8", is_essential=False),
            ]
            db.add_all(default_categories)
            db.commit()
    finally:
        db.close()
