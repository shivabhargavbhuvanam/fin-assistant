"""
Module: main.py
Description: FastAPI application entry point with all API routes for the Smart Financial Coach.

This module provides REST API endpoints for:
    - CSV file upload and processing
    - Financial transaction analysis (categorization, anomaly detection, recurring charges)
    - AI-powered insight generation
    - Dashboard data consolidation
    - Goal-based savings recommendations

Author: Smart Financial Coach Team
Created: 2025-01-31

Dependencies:
    - FastAPI for REST API framework
    - SQLAlchemy for database operations
    - OpenAI for AI-powered features

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from auth import get_current_user, get_optional_user, is_auth_configured
from services.observability import (
    logger, metrics, timed, timed_block,
    log_analysis_start, log_analysis_complete,
    log_chat_request
)
from services import (
    AIService, CSVProcessor, Categorizer,
    AnomalyDetector, RecurringDetector, InsightGenerator,
    ChatService, PatternAnalyzer, GoalForecaster,
    FortuneGenerator, build_financial_stats
)
from schemas import (
    UploadResponse, AnalyzeResponse, DashboardResponse,
    GoalRequest, GoalResponse, HealthResponse,
    TransactionOut, AnomalyOut, RecurringChargeOut,
    InsightOut, DeltaOut, SpendingSummary, GoalCut, CategoryOut,
    ChatRequest, ChatResponse, SuggestedPromptsResponse,
    SessionOut, SessionListResponse, UserInfoResponse,
    ConversationHistoryResponse, ChatMessage, FortuneResponse
)
from models import (
    Session, Transaction, Category, Anomaly,
    RecurringCharge, Delta, Insight, Goal,
    Conversation, Message
)
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession

from database import get_db, init_db, SessionLocal


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter to prevent API abuse.

    Limits requests per session to prevent:
        - OpenAI budget drain from chat spam
        - Denial of service attacks
        - Runaway clients

    Uses sliding window algorithm with configurable limits.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for given identifier.

        Args:
            identifier: Session ID or IP address.

        Returns:
            True if request is allowed, False if rate limited.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[identifier] = [
            t for t in self.requests[identifier]
            if t > window_start
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Record request
        self.requests[identifier].append(now)
        return True

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        current_requests = [
            t for t in self.requests.get(identifier, [])
            if t > window_start
        ]
        return max(0, self.max_requests - len(current_requests))

    def get_reset_time(self, identifier: str) -> float:
        """Get seconds until rate limit resets."""
        if identifier not in self.requests or not self.requests[identifier]:
            return 0
        oldest = min(self.requests[identifier])
        return max(0, oldest + self.window_seconds - time.time())


# Global rate limiter instances
# 30 chat requests per minute
chat_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
# 100 API requests per minute
api_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


# =============================================================================
# Application Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.

    On startup:
        - Initialize database tables
        - Seed default categories

    On shutdown:
        - Cleanup resources if needed
    """
    # Startup
    print("🚀 Starting Smart Financial Coach API...")
    init_db()
    print("✅ Database initialized with default categories")

    yield

    # Shutdown
    print("👋 Shutting down Smart Financial Coach API...")


# =============================================================================
# FastAPI Application Configuration
# =============================================================================

app = FastAPI(
    title="Smart Financial Coach API",
    description="""
    AI-powered financial analysis API that helps users understand their spending patterns,
    detect anomalies, identify recurring charges, and receive personalized insights.
    
    ## Features
    - 📤 CSV Upload & Processing
    - 🏷️ Hybrid AI + Rule-based Transaction Categorization
    - 🚨 Statistical Anomaly Detection
    - 🔄 Recurring Charge & Subscription Detection
    - 💡 AI-Generated Personalized Insights
    - 🎯 Goal-Based Savings Recommendations
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration - Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://35.235.97.147:5173",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Dependency Injection
# =============================================================================

def get_ai_service() -> AIService:
    """
    Dependency: Provide AIService instance.

    Returns:
        AIService: Configured OpenAI wrapper service.
    """
    return AIService()


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
    description="Check the health status of the API, database, and OpenAI connection."
)
async def health_check(
    db: DBSession = Depends(get_db),
    ai_service: AIService = Depends(get_ai_service)
) -> HealthResponse:
    """
    Perform health check on all system components.

    Args:
        db: Database session from dependency injection.
        ai_service: AI service from dependency injection.

    Returns:
        HealthResponse: Status of API, database, and OpenAI connection.

    Example:
        GET /health
        Response: {"status": "healthy", "database": "connected", "openai": "connected"}
    """
    # Check database connection
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    # Check OpenAI connection
    try:
        openai_connected = await ai_service.check_connection()
        openai_status = "connected" if openai_connected else "disconnected"
    except Exception as e:
        openai_status = f"error: {str(e)}"

    overall_status = "healthy" if db_status == "connected" else "degraded"

    return HealthResponse(
        status=overall_status,
        database=db_status,
        openai=openai_status
    )


@app.get(
    "/metrics",
    tags=["System"],
    summary="Get application metrics",
    description="Returns metrics including request counts, timing data, and session statistics."
)
async def get_metrics():
    """
    Get application metrics for monitoring and debugging.

    Returns:
        Dict with counters, timings, and session metrics.

    Example:
        GET /metrics
        Response: {
            "uptime_seconds": 3600,
            "counters": {"analysis.completed": 15, "chat.requests": 42},
            "timings": {"anomaly_detection": {"avg_ms": 120.5, ...}}
        }
    """
    return metrics.get_summary()


# =============================================================================
# Session Ownership Helper
# =============================================================================

def verify_session_ownership(
    db: DBSession,
    session_id: str,
    user_id: str
) -> Session:
    """
    Verify that a session exists and belongs to the given user.

    Args:
        db: Database session.
        session_id: Session ID to verify.
        user_id: Clerk user ID (owner).

    Returns:
        Session object if valid.

    Raises:
        HTTPException: 404 if session not found or doesn't belong to user.
    """
    session = (
        db.query(Session)
        .filter(Session.id == session_id)
        .filter(Session.clerk_user_id == user_id)
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found or access denied."
        )

    return session


# =============================================================================
# User & Session Management Endpoints
# =============================================================================

@app.get(
    "/me",
    response_model=UserInfoResponse,
    tags=["User"],
    summary="Get current user info and sessions",
)
async def get_current_user_info(
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> UserInfoResponse:
    """
    Get current user's info and list of sessions.

    Returns:
        UserInfoResponse with user ID and list of sessions.
    """
    sessions = (
        db.query(Session)
        .filter(Session.clerk_user_id == user_id)
        .order_by(Session.created_at.desc())
        .all()
    )

    has_sample = any(s.is_sample for s in sessions)

    return UserInfoResponse(
        user_id=user_id,
        sessions=[
            SessionOut(
                id=s.id,
                name=s.name or s.filename or (
                    "Sample Data" if s.is_sample else "Uploaded Data"),
                filename=s.filename,
                row_count=s.row_count,
                status=s.status,
                is_sample=s.is_sample,
                created_at=s.created_at,
            )
            for s in sessions
        ],
        has_sample_session=has_sample,
        active_session_id=sessions[0].id if sessions else None,
    )


@app.get(
    "/sessions",
    response_model=SessionListResponse,
    tags=["User"],
    summary="List user's sessions",
)
async def list_sessions(
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> SessionListResponse:
    """
    List all sessions for the current user.

    Returns:
        SessionListResponse with list of sessions.
    """
    sessions = (
        db.query(Session)
        .filter(Session.clerk_user_id == user_id)
        .order_by(Session.created_at.desc())
        .all()
    )

    return SessionListResponse(
        sessions=[
            SessionOut(
                id=s.id,
                name=s.name or s.filename or (
                    "Sample Data" if s.is_sample else "Uploaded Data"),
                filename=s.filename,
                row_count=s.row_count,
                status=s.status,
                is_sample=s.is_sample,
                created_at=s.created_at,
            )
            for s in sessions
        ],
        active_session_id=sessions[0].id if sessions else None,
    )


@app.post(
    "/sessions/sample",
    response_model=UploadResponse,
    tags=["User"],
    summary="Create sample session for user",
)
async def create_sample_session(
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> UploadResponse:
    """
    Create a sample data session for the current user.

    If user already has a sample session, returns the existing one.
    Otherwise, creates a new sample session with synthetic data.

    Returns:
        UploadResponse with session ID and transaction count.
    """
    # Check if user already has a sample session
    existing_sample = (
        db.query(Session)
        .filter(Session.clerk_user_id == user_id)
        .filter(Session.is_sample == True)
        .first()
    )

    if existing_sample:
        return UploadResponse(
            session_id=existing_sample.id,
            filename=existing_sample.filename,
            row_count=existing_sample.row_count,
            status=existing_sample.status,
        )

    # Create new sample session with synthetic data
    from synthetic_data import SyntheticDataGenerator

    generator = SyntheticDataGenerator()
    transactions = generator.generate()

    processor = CSVProcessor(db)

    # Create session with user ownership
    import uuid
    session_id = str(uuid.uuid4())

    session = Session(
        id=session_id,
        clerk_user_id=user_id,
        filename="sample_data.csv",
        row_count=len(transactions),
        status="processing",
        is_sample=True,
        name="Sample Data",
    )
    db.add(session)

    # Add transactions
    for txn in transactions:
        transaction = Transaction(
            session_id=session_id,
            date=txn["date"],
            description=txn["description"],
            amount=txn["amount"],
            raw_description=txn["description"],
        )
        db.add(transaction)

    db.commit()

    return UploadResponse(
        session_id=session_id,
        filename="sample_data.csv",
        row_count=len(transactions),
        status="processing",
    )


@app.delete(
    "/sessions/{session_id}",
    tags=["User"],
    summary="Delete a session",
)
async def delete_session(
    session_id: str,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Delete a session and all its associated data.

    Args:
        session_id: Session ID to delete.

    Returns:
        Success message.
    """
    session = verify_session_ownership(db, session_id, user_id)

    db.delete(session)
    db.commit()

    return {"message": "Session deleted successfully"}


# =============================================================================
# Upload Endpoints
# =============================================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Upload transaction CSV file",
    description="Upload a CSV file containing financial transactions for analysis."
)
async def upload_csv(
    file: UploadFile = File(...,
                            description="CSV file with columns: date, description, amount"),
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> UploadResponse:
    """
    Upload and process a CSV file containing financial transactions.

    The CSV file should contain at least these columns:
        - date: Transaction date (various formats supported)
        - description: Transaction description/merchant name
        - amount: Transaction amount (positive for income, negative for expenses)

    Args:
        file: The uploaded CSV file.
        db: Database session from dependency injection.
        user_id: Authenticated user ID from Clerk.

    Returns:
        UploadResponse: Session ID and upload metadata.

    Raises:
        HTTPException: If file is not a CSV or validation fails.

    Example:
        POST /upload
        Content-Type: multipart/form-data
        file: transactions.csv
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported. Please upload a .csv file."
        )

    try:
        processor = CSVProcessor(db)
        session_id, row_count = await processor.process(file, clerk_user_id=user_id)

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            row_count=row_count,
            status="processing"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process CSV: {str(e)}"
        )


@app.post(
    "/sample",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Use sample transaction data",
    description="Load pre-generated sample transaction data for demonstration."
)
async def use_sample_data(
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> UploadResponse:
    """
    Load synthetic sample data for demonstration purposes.

    This endpoint generates realistic financial transactions including:
        - Regular patterns (rent, subscriptions, groceries)
        - Anomalies (unusual large purchases)
        - Gray charges (small unknown recurring charges)

    Args:
        db: Database session from dependency injection.
        user_id: Authenticated user ID from Clerk.

    Returns:
        UploadResponse: Session ID and metadata for the sample data.

    Example:
        POST /sample
        Response: {"session_id": "uuid", "filename": "sample_transactions.csv", ...}
    """
    try:
        # Import synthetic data generator
        from synthetic_data import generate_synthetic_transactions

        transactions = generate_synthetic_transactions()
        processor = CSVProcessor(db)
        session_id, row_count = processor.process_synthetic(
            transactions, clerk_user_id=user_id)

        return UploadResponse(
            session_id=session_id,
            filename="sample_transactions.csv",
            row_count=row_count,
            status="processing"
        )

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Sample data generator not yet implemented."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sample data: {str(e)}"
        )


# =============================================================================
# Analysis Endpoint
# =============================================================================

@app.post(
    "/analyze/{session_id}",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Run full financial analysis",
    description="Execute all analysis services: categorization, anomaly detection, recurring detection, and insight generation."
)
async def analyze_session(
    session_id: str,
    db: DBSession = Depends(get_db),
    ai_service: AIService = Depends(get_ai_service),
    user_id: str = Depends(get_current_user)
) -> AnalyzeResponse:
    """
    Run comprehensive financial analysis on uploaded transactions.

    This endpoint orchestrates all analysis services:
        1. Transaction categorization (hybrid rule-based + AI)
        2. Anomaly detection (statistical z-score analysis)
        3. Recurring charge detection (subscriptions, gray charges)
        4. Month-over-month delta calculation
        5. AI insight generation

    Args:
        session_id: The session ID from the upload endpoint.
        db: Database session from dependency injection.
        ai_service: AI service from dependency injection.
        user_id: Authenticated user ID from Clerk.

    Returns:
        AnalyzeResponse: Summary of analysis results.

    Raises:
        HTTPException: If session not found or analysis fails.

    Example:
        POST /analyze/abc-123-def
        Response: {"session_id": "abc-123-def", "status": "ready", ...}
    """
    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)

    try:
        # Get transaction count for logging
        txn_count = db.query(Transaction).filter(
            Transaction.session_id == session_id).count()
        log_analysis_start(session_id, txn_count)

        # Step 1: Categorize transactions
        with timed_block("categorization"):
            categorizer = Categorizer(db, ai_service)
            categorized_count = await categorizer.categorize_all(session_id)
            metrics.gauge("last_categorized_count", categorized_count)

        # Step 2: Detect anomalies
        with timed_block("anomaly_detection"):
            anomaly_detector = AnomalyDetector(db)
            anomalies_count = anomaly_detector.detect(session_id)
            metrics.gauge("last_anomalies_count", anomalies_count)

        # Step 3: Detect recurring charges
        with timed_block("recurring_detection"):
            recurring_detector = RecurringDetector(db)
            recurring_count = recurring_detector.detect(session_id)

        # Step 4: Calculate deltas (month-over-month changes)
        insight_gen = InsightGenerator(db, ai_service)
        insight_gen.calculate_deltas(session_id)

        # Step 5: Detect spending patterns (NEW - proactive)
        pattern_analyzer = PatternAnalyzer(db, session_id)
        patterns = pattern_analyzer.detect_all()

        # Step 6: Run goal forecasting (NEW - proactive)
        goal_forecaster = GoalForecaster(db, session_id)
        forecast = goal_forecaster.analyze()

        # Step 7: Generate AI insights
        insights_count = await insight_gen.generate(session_id)

        # Step 8: Add pattern-based proactive insights
        pattern_insights_count = _add_pattern_insights(
            db, session_id, patterns, forecast)

        # Update session status
        session.status = "ready"
        db.commit()

        # Log completion
        results = {
            "transactions": categorized_count,
            "anomalies": anomalies_count,
            "recurring": recurring_count,
            "insights": insights_count + pattern_insights_count
        }
        log_analysis_complete(session_id, results)

        return AnalyzeResponse(
            session_id=session_id,
            status="ready",
            transactions_categorized=categorized_count,
            anomalies_detected=anomalies_count,
            recurring_charges_found=recurring_count,
            insights_generated=insights_count + pattern_insights_count
        )

    except Exception as e:
        # Log error
        logger.error("Analysis failed", error=str(e),
                     session_id=session_id[:8])
        metrics.increment("analysis.failed")

        # Update session status to error
        session.status = "error"
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# =============================================================================
# Dashboard Endpoint
# =============================================================================

@app.get(
    "/dashboard/{session_id}",
    response_model=DashboardResponse,
    tags=["Dashboard"],
    summary="Get consolidated dashboard data",
    description="Retrieve all analysis results in a single response for dashboard rendering."
)
async def get_dashboard(
    session_id: str,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> DashboardResponse:
    """
    Get consolidated dashboard data for a session.

    This endpoint returns all analyzed data in a single response:
        - Spending summary (totals, by category)
        - AI-generated insights
        - Detected anomalies
        - Recurring charges
        - Month-over-month deltas

    Args:
        session_id: The session ID from the upload endpoint.
        db: Database session from dependency injection.
        user_id: Authenticated user ID from Clerk.

    Returns:
        DashboardResponse: Complete dashboard data.

    Raises:
        HTTPException: If session not found.

    Example:
        GET /dashboard/abc-123-def
    """
    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)

    # Build category lookup
    categories = {c.id: c for c in db.query(Category).all()}

    # Get transactions and calculate summary
    transactions = db.query(Transaction).filter(
        Transaction.session_id == session_id
    ).all()

    summary = _calculate_spending_summary(transactions, categories)

    # Get insights (ordered by priority)
    insights = db.query(Insight).filter(
        Insight.session_id == session_id
    ).order_by(Insight.priority).all()

    insights_out = [
        InsightOut(
            id=i.id,
            type=i.type,
            priority=i.priority,
            title=i.title,
            description=i.description,
            action=i.action,
            reasoning=i.reasoning or "",
            confidence=i.confidence or 0.8,
            data=i.data
        )
        for i in insights
    ]

    # Get anomalies
    anomalies = db.query(Anomaly).filter(
        Anomaly.session_id == session_id
    ).all()

    anomalies_out = []
    for a in anomalies:
        txn = db.query(Transaction).filter(
            Transaction.id == a.transaction_id).first()
        if txn:
            cat = categories.get(txn.category_id)
            anomalies_out.append(
                AnomalyOut(
                    id=a.id,
                    transaction_id=txn.id,
                    transaction_description=txn.description,
                    transaction_amount=txn.amount,
                    transaction_date=txn.date,
                    category_name=cat.name if cat else "Unknown",
                    anomaly_type=a.anomaly_type or "amount",
                    severity=a.severity or "medium",
                    expected=a.expected_value or 0,
                    actual=a.actual_value or 0,
                    z_score=a.z_score or 0,
                    explanation=a.explanation or ""
                )
            )

    # Get recurring charges
    recurring = db.query(RecurringCharge).filter(
        RecurringCharge.session_id == session_id
    ).all()

    recurring_out = []
    for r in recurring:
        cat = categories.get(r.category_id)
        freq_str = "monthly" if r.frequency_days >= 25 else "weekly"
        recurring_out.append(
            RecurringChargeOut(
                id=r.id,
                description=r.description_pattern or "",
                category=CategoryOut(
                    id=cat.id if cat else 0,
                    name=cat.name if cat else "Other",
                    icon=cat.icon if cat else "📝",
                    color=cat.color if cat else "#94a3b8",
                    is_essential=cat.is_essential if cat else False
                ),
                amount=r.average_amount or 0,
                frequency=freq_str,
                frequency_days=r.frequency_days or 30,
                occurrences=r.occurrence_count or 0,
                is_gray_charge=r.is_gray_charge or False,
                confidence=r.confidence or 0.8
            )
        )

    # Get deltas
    deltas = db.query(Delta).filter(
        Delta.session_id == session_id
    ).all()

    deltas_out = []
    for d in deltas:
        cat = categories.get(d.category_id)
        if cat:
            deltas_out.append(
                DeltaOut(
                    category_id=d.category_id,
                    category_name=cat.name,
                    category_icon=cat.icon,
                    current_month=d.current_month or "",
                    previous_month=d.previous_month or "",
                    current_amount=d.current_amount or 0,
                    previous_amount=d.previous_amount or 0,
                    change_amount=d.change_amount or 0,
                    change_percent=d.change_percent or 0
                )
            )

    return DashboardResponse(
        session_id=session_id,
        status=session.status,
        summary=summary,
        insights=insights_out,
        anomalies=anomalies_out,
        recurring_charges=recurring_out,
        deltas=deltas_out
    )


def _add_pattern_insights(
    db: DBSession,
    session_id: str,
    patterns: list[dict],
    forecast: dict
) -> int:
    """
    Add proactive pattern-based insights to the database.

    These insights are generated from pattern analysis and goal forecasting,
    providing actionable recommendations without user prompting.

    Args:
        db: Database session.
        session_id: Session ID.
        patterns: Detected spending patterns.
        forecast: Goal forecast results.

    Returns:
        Number of insights added.
    """
    insights_added = 0

    # Add top pattern insights (limit to 3)
    for i, pattern in enumerate(patterns[:3]):
        pattern_type = pattern.get('type', 'pattern')
        yearly_savings = pattern.get('yearly_savings', 0)

        if yearly_savings < 200:  # Skip small savings
            continue

        # Determine priority based on savings amount
        if yearly_savings >= 1000:
            priority = 1
        elif yearly_savings >= 500:
            priority = 2
        else:
            priority = 3

        # Create insight based on pattern type
        if pattern_type == 'merchant_habit':
            merchant = pattern.get('merchant', 'Unknown')
            title = f"{merchant}: ${pattern.get('monthly_cost', 0):,.0f}/month"
            description = pattern.get(
                'suggestion', f"Reduce visits to save ${yearly_savings:,.0f}/year")
            insight_type = 'savings'
        elif pattern_type == 'category_pattern':
            category = pattern.get('category', 'Unknown')
            title = f"{category} spending: ${pattern.get('monthly_spend', 0):,.0f}/month"
            description = pattern.get(
                'suggestion', f"Reduce spending to save ${yearly_savings:,.0f}/year")
            insight_type = 'spending'
        elif pattern_type == 'weekend_splurge':
            title = "Weekend spending spike detected"
            description = pattern.get(
                'suggestion', f"Plan ahead to save ${yearly_savings:,.0f}/year")
            insight_type = 'spending'
        elif pattern_type == 'payday_impulse':
            title = "Post-payday spending surge"
            description = pattern.get(
                'suggestion', f"Wait 48 hours before purchases")
            insight_type = 'spending'
        else:
            continue

        insight = Insight(
            session_id=session_id,
            type=insight_type,
            priority=priority,
            title=title,
            description=description,
            action=f"Save ${yearly_savings:,.0f}/year",
            reasoning=f"Based on {pattern.get('transaction_count', 0) or pattern.get('monthly_visits', 0) or 'your'} transactions",
            confidence=0.85,
            data=pattern
        )
        db.add(insight)
        insights_added += 1

    # Add savings capacity insight from forecast
    forecast_insights = forecast.get('insights', [])
    for fi in forecast_insights[:2]:  # Top 2 forecast insights
        if fi.get('type') == 'warning':
            priority = 1
            insight_type = 'anomaly'
        elif fi.get('type') == 'savings_potential':
            priority = 1
            insight_type = 'savings'
        else:
            priority = 2
            insight_type = 'savings'

        insight = Insight(
            session_id=session_id,
            type=insight_type,
            priority=priority,
            title=fi.get('title', 'Savings opportunity'),
            description=fi.get('description', ''),
            action=fi.get('action', ''),
            reasoning="Based on your income and spending patterns",
            confidence=0.9,
            data=fi.get('data', {})
        )
        db.add(insight)
        insights_added += 1

    db.commit()
    return insights_added


def _calculate_spending_summary(
    transactions: list[Transaction],
    categories: dict[int, Category]
) -> SpendingSummary:
    """
    Calculate spending summary from transactions.

    Args:
        transactions: List of transaction objects.
        categories: Dictionary mapping category ID to Category object.

    Returns:
        SpendingSummary: Aggregated spending data.
    """
    total_income = sum(t.amount for t in transactions if t.amount > 0)
    total_spending = sum(t.amount for t in transactions if t.amount < 0)

    by_category = {}
    for t in transactions:
        if t.category_id and t.category_id in categories:
            cat = categories[t.category_id]
            if cat.name not in by_category:
                by_category[cat.name] = {
                    "amount": 0,
                    "count": 0,
                    "icon": cat.icon,
                    "color": cat.color,
                    "is_essential": cat.is_essential
                }
            by_category[cat.name]["amount"] += t.amount
            by_category[cat.name]["count"] += 1

    return SpendingSummary(
        total_income=total_income,
        total_spending=total_spending,
        net=total_income + total_spending,
        by_category=by_category
    )


# =============================================================================
# Goal Endpoint (Stretch Feature)
# =============================================================================

@app.post(
    "/goal/{session_id}",
    response_model=GoalResponse,
    tags=["Goals"],
    summary="Get AI-powered savings goal recommendations",
    description="Provide a savings target and receive AI-powered suggestions on how to achieve it."
)
async def get_goal_recommendations(
    session_id: str,
    goal: GoalRequest,
    db: DBSession = Depends(get_db),
    ai_service: AIService = Depends(get_ai_service)
) -> GoalResponse:
    """
    Get AI-powered recommendations to achieve a savings goal.

    The AI analyzes current spending patterns and suggests specific
    category cuts to help achieve the target savings amount.

    Args:
        session_id: The session ID from the upload endpoint.
        goal: GoalRequest with target_amount.
        db: Database session from dependency injection.
        ai_service: AI service from dependency injection.

    Returns:
        GoalResponse: Achievability assessment and suggested cuts.

    Raises:
        HTTPException: If session not found or goal calculation fails.

    Example:
        POST /goal/abc-123-def
        Body: {"target_amount": 500}
    """
    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)

    # Build category lookup
    categories = {c.id: c for c in db.query(Category).all()}

    # Get transactions
    transactions = db.query(Transaction).filter(
        Transaction.session_id == session_id
    ).all()

    if not transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No transactions found. Please run analysis first."
        )

    # Calculate current discretionary spending
    discretionary_spending = 0
    category_spending = {}

    for t in transactions:
        if t.amount < 0 and t.category_id:
            cat = categories.get(t.category_id)
            if cat and not cat.is_essential:
                discretionary_spending += abs(t.amount)

                if cat.name not in category_spending:
                    category_spending[cat.name] = {
                        "amount": 0,
                        "icon": cat.icon,
                        "is_essential": cat.is_essential
                    }
                category_spending[cat.name]["amount"] += abs(t.amount)

    # Estimate monthly average (assuming 3 months of data)
    monthly_discretionary = discretionary_spending / 3

    # Determine achievability
    target = goal.target_amount
    achievable = monthly_discretionary >= target
    gap = 0 if achievable else target - monthly_discretionary

    # Generate suggested cuts (prioritize non-essential, larger categories)
    suggested_cuts = []
    sorted_categories = sorted(
        category_spending.items(),
        key=lambda x: x[1]["amount"],
        reverse=True
    )

    remaining_target = target
    for cat_name, data in sorted_categories:
        if remaining_target <= 0:
            break

        monthly_amount = data["amount"] / 3

        # Suggest cutting 20-50% depending on category type
        if cat_name in ["Entertainment", "Shopping", "Dining"]:
            cut_percent = 0.30  # 30% cut
            difficulty = "easy"
        elif cat_name == "Subscriptions":
            cut_percent = 0.50  # 50% cut (cancel unused)
            difficulty = "easy"
        else:
            cut_percent = 0.20  # 20% cut
            difficulty = "moderate"

        savings = monthly_amount * cut_percent
        remaining_target -= savings

        suggested_cuts.append(
            GoalCut(
                category=cat_name,
                category_icon=data["icon"],
                current_amount=monthly_amount,
                suggested_amount=monthly_amount - savings,
                savings=savings,
                difficulty=difficulty,
                is_essential=data["is_essential"]
            )
        )

    total_potential_savings = sum(c.savings for c in suggested_cuts)

    # Get AI advice
    context = {
        "monthly_discretionary": monthly_discretionary,
        "target": target,
        "achievable": achievable,
        "categories": category_spending
    }

    ai_advice = await ai_service.generate_goal_advice(
        context, target, [c.model_dump() for c in suggested_cuts]
    )

    # Store goal in database
    goal_record = Goal(
        session_id=session_id,
        target_amount=target,
        suggested_cuts=[c.model_dump() for c in suggested_cuts],
        achievable=achievable,
        gap_amount=gap if gap > 0 else None
    )
    db.add(goal_record)
    db.commit()

    return GoalResponse(
        target_amount=target,
        current_discretionary=monthly_discretionary,
        achievable=achievable,
        suggested_cuts=suggested_cuts,
        total_potential_savings=total_potential_savings,
        gap_amount=gap if gap > 0 else None,
        ai_advice=ai_advice
    )


# =============================================================================
# Transaction Endpoints (Utility)
# =============================================================================

@app.get(
    "/transactions/{session_id}",
    response_model=list[TransactionOut],
    tags=["Transactions"],
    summary="Get all transactions for a session",
    description="Retrieve all categorized transactions for a session."
)
async def get_transactions(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> list[TransactionOut]:
    """
    Get paginated list of transactions for a session.

    Args:
        session_id: The session ID.
        limit: Maximum number of transactions to return (default 100).
        offset: Number of transactions to skip (default 0).
        db: Database session from dependency injection.

    Returns:
        List of TransactionOut objects.
    """
    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)

    transactions = (
        db.query(Transaction)
        .filter(Transaction.session_id == session_id)
        .order_by(Transaction.date.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    categories = {c.id: c for c in db.query(Category).all()}

    return [
        TransactionOut(
            id=t.id,
            date=t.date,
            description=t.description,
            amount=t.amount,
            category=CategoryOut(
                id=categories[t.category_id].id,
                name=categories[t.category_id].name,
                icon=categories[t.category_id].icon,
                color=categories[t.category_id].color,
                is_essential=categories[t.category_id].is_essential
            ) if t.category_id and t.category_id in categories else None,
            confidence=t.category_confidence,
            source=t.category_source
        )
        for t in transactions
    ]


@app.get(
    "/categories",
    response_model=list[CategoryOut],
    tags=["Categories"],
    summary="Get all categories",
    description="Retrieve all available transaction categories."
)
async def get_categories(
    db: DBSession = Depends(get_db)
) -> list[CategoryOut]:
    """
    Get all available categories.

    Args:
        db: Database session from dependency injection.

    Returns:
        List of CategoryOut objects.
    """
    categories = db.query(Category).all()

    return [
        CategoryOut(
            id=c.id,
            name=c.name,
            icon=c.icon,
            color=c.color,
            is_essential=c.is_essential
        )
        for c in categories
    ]


# =============================================================================
# Chat Endpoints
# =============================================================================

@app.post(
    "/chat/{session_id}",
    tags=["Chat"],
    summary="Chat with the financial coach",
    description="Send a message to the AI financial coach and receive a streaming response."
)
async def chat(
    session_id: str,
    request: ChatRequest,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Chat with the AI financial coach.

    The AI has access to your financial data and can answer questions about:
    - Spending patterns and summaries
    - Anomalies and unusual transactions
    - Recurring charges and subscriptions
    - Personalized recommendations

    Rate limited to 30 requests per minute per session to prevent abuse.

    Args:
        session_id: The session ID from upload.
        request: ChatRequest with message and optional conversation_id.
        db: Database session.

    Returns:
        StreamingResponse with the AI's reply.
    """
    # Rate limiting check
    if not chat_rate_limiter.is_allowed(session_id):
        remaining = chat_rate_limiter.get_remaining(session_id)
        reset_time = chat_rate_limiter.get_reset_time(session_id)
        metrics.increment("chat.rate_limited")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {reset_time:.0f} seconds.",
            headers={
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(reset_time)),
            }
        )

    # Log chat request
    log_chat_request(session_id, len(request.message))

    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)

    chat_service = ChatService(db)

    async def generate():
        """Generate streaming response."""
        async for chunk in chat_service.chat(
            session_id,
            request.message,
            request.conversation_id
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.post(
    "/chat/{session_id}/sync",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat with the financial coach (non-streaming)",
    description="Send a message and receive the complete response at once."
)
async def chat_sync(
    session_id: str,
    request: ChatRequest,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> ChatResponse:
    """
    Chat with the AI financial coach (non-streaming version).

    Same as /chat but returns the complete response at once
    instead of streaming. Useful for clients that don't support streaming.

    Rate limited to 30 requests per minute per session to prevent abuse.

    Args:
        session_id: The session ID from upload.
        request: ChatRequest with message and optional conversation_id.
        db: Database session.

    Raises:
        HTTPException: 429 if rate limit exceeded.

    Returns:
        ChatResponse with the complete message.
    """
    # Rate limiting check
    if not chat_rate_limiter.is_allowed(session_id):
        remaining = chat_rate_limiter.get_remaining(session_id)
        reset_time = chat_rate_limiter.get_reset_time(session_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {reset_time:.0f} seconds.",
            headers={
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(reset_time)),
            }
        )

    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)

    chat_service = ChatService(db)

    # Collect all chunks into a single response
    full_response = ""
    conversation_id = request.conversation_id

    async for chunk in chat_service.chat(
        session_id,
        request.message,
        conversation_id
    ):
        full_response += chunk

    # Get the conversation ID from the most recent conversation
    if not conversation_id:
        conv = db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).order_by(Conversation.created_at.desc()).first()
        if conv:
            conversation_id = conv.id

    # Get suggested prompts
    suggested_prompts = chat_service.get_suggested_prompts(session_id)

    return ChatResponse(
        conversation_id=conversation_id or "",
        message=full_response,
        suggested_prompts=suggested_prompts
    )


@app.get(
    "/chat/{session_id}/prompts",
    response_model=SuggestedPromptsResponse,
    tags=["Chat"],
    summary="Get suggested prompts",
    description="Get contextual suggested prompts based on the financial data."
)
async def get_suggested_prompts(
    session_id: str,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> SuggestedPromptsResponse:
    """
    Get suggested prompts for the chat interface.

    Returns contextual prompts based on what's in the user's data,
    like "Tell me about the 3 anomalies" if anomalies exist.

    Args:
        session_id: The session ID.
        db: Database session.

    Returns:
        SuggestedPromptsResponse with list of prompts.
    """
    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)

    chat_service = ChatService(db)
    prompts = chat_service.get_suggested_prompts(session_id)

    return SuggestedPromptsResponse(prompts=prompts)


@app.get(
    "/chat/{session_id}/history/{conversation_id}",
    response_model=ConversationHistoryResponse,
    tags=["Chat"],
    summary="Get conversation history",
    description="Retrieve the message history for a conversation."
)
async def get_conversation_history(
    session_id: str,
    conversation_id: str,
    db: DBSession = Depends(get_db)
) -> ConversationHistoryResponse:
    """
    Get the message history for a conversation.

    Args:
        session_id: The session ID.
        conversation_id: The conversation ID.
        db: Database session.

    Returns:
        ConversationHistoryResponse with messages.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.session_id == session_id
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found."
        )

    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
            timestamp=m.created_at
        )
        for m in conversation.messages
        if m.role in ["user", "assistant"]
    ]

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
        created_at=conversation.created_at
    )


# =============================================================================
# Fortune Cookie Endpoint
# =============================================================================

@app.get(
    "/fortune/{session_id}",
    response_model=FortuneResponse,
    tags=["Fortune"],
    summary="Generate a mystical financial fortune",
    description="Get a cryptic but actionable fortune cookie message based on your spending patterns."
)
async def generate_fortune(
    session_id: str,
    db: DBSession = Depends(get_db),
    user_id: str = Depends(get_current_user)
) -> FortuneResponse:
    """
    Generate a mystical financial fortune based on user's spending patterns.
    
    The fortune is:
    - Cryptic yet actionable (fortune cookie style)
    - References actual financial data
    - Color-coded by sentiment (positive/neutral/warning)
    - Optionally includes a "lucky number" as a savings amount
    
    Args:
        session_id: The session ID from upload.
        db: Database session.
        user_id: Authenticated user ID.
        
    Returns:
        FortuneResponse with fortune text, sentiment, and optional lucky number.
    """
    # Rate limiting (reuse chat limiter, but more generous)
    if not api_rate_limiter.is_allowed(f"fortune:{session_id}"):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many fortune requests. Try again in a moment."
        )
    
    # Verify session exists and belongs to user
    session = verify_session_ownership(db, session_id, user_id)
    
    # Build category lookup
    categories = {c.id: c for c in db.query(Category).all()}
    
    # Get transactions for summary
    transactions = db.query(Transaction).filter(
        Transaction.session_id == session_id
    ).all()
    
    if not transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No transactions found. Please analyze your data first."
        )
    
    # Calculate summary
    summary = _calculate_spending_summary(transactions, categories)
    
    # Get anomaly count
    anomaly_count = db.query(Anomaly).filter(
        Anomaly.session_id == session_id
    ).count()
    
    # Get recurring charges
    recurring = db.query(RecurringCharge).filter(
        RecurringCharge.session_id == session_id
    ).all()
    
    recurring_list = [
        {"is_gray_charge": r.is_gray_charge}
        for r in recurring
    ]
    
    # Build financial stats for fortune generation
    stats = build_financial_stats(
        summary=summary.model_dump(),
        anomaly_count=anomaly_count,
        recurring_charges=recurring_list,
        insights=[]
    )
    
    # Generate fortune
    generator = FortuneGenerator()
    fortune = await generator.generate(stats)
    
    return FortuneResponse(
        fortune=fortune.text,
        sentiment=fortune.sentiment.value,
        lucky_number=fortune.lucky_number
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
