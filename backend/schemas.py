"""Pydantic request/response schemas for type safety."""

from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional, Literal


# Request schemas
class GoalRequest(BaseModel):
    target_amount: float = Field(gt=0, description="Monthly savings target")


# Response schemas
class CategoryOut(BaseModel):
    id: int
    name: str
    icon: Optional[str] = None
    color: Optional[str] = None
    is_essential: bool = False

    class Config:
        from_attributes = True


class TransactionOut(BaseModel):
    id: int
    date: date
    description: str
    amount: float
    category: Optional[CategoryOut] = None
    confidence: Optional[float] = None
    source: Optional[str] = None

    class Config:
        from_attributes = True


class AnomalyOut(BaseModel):
    id: int
    transaction_id: int
    transaction_description: str
    transaction_amount: float
    transaction_date: date
    category_name: str
    anomaly_type: str
    severity: str
    expected: float
    actual: float
    z_score: float
    explanation: str

    class Config:
        from_attributes = True


class RecurringChargeOut(BaseModel):
    id: int
    description: str
    category: CategoryOut
    amount: float
    frequency: str
    frequency_days: int
    occurrences: int
    is_gray_charge: bool
    confidence: float

    class Config:
        from_attributes = True


class DeltaOut(BaseModel):
    category_id: int
    category_name: str
    category_icon: Optional[str] = None
    current_month: str
    previous_month: str
    current_amount: float
    previous_amount: float
    change_amount: float
    change_percent: float

    class Config:
        from_attributes = True


class InsightOut(BaseModel):
    id: int
    type: str
    priority: int
    title: str
    description: str
    action: Optional[str] = None
    reasoning: str
    confidence: float
    data: Optional[dict] = None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    row_count: int
    status: str


class AnalyzeResponse(BaseModel):
    session_id: str
    status: str
    transactions_categorized: int
    anomalies_detected: int
    recurring_charges_found: int
    insights_generated: int


class SpendingSummary(BaseModel):
    total_income: float
    total_spending: float
    net: float
    by_category: dict[str, dict]


class DashboardResponse(BaseModel):
    session_id: str
    status: str
    summary: SpendingSummary
    insights: list[InsightOut]
    anomalies: list[AnomalyOut]
    recurring_charges: list[RecurringChargeOut]
    deltas: list[DeltaOut]


class GoalCut(BaseModel):
    category: str
    category_icon: Optional[str] = None
    current_amount: float
    suggested_amount: float
    savings: float
    difficulty: str
    is_essential: bool


class GoalResponse(BaseModel):
    target_amount: float
    current_discretionary: float
    achievable: bool
    suggested_cuts: list[GoalCut]
    total_potential_savings: float
    gap_amount: Optional[float] = None
    ai_advice: str


class HealthResponse(BaseModel):
    status: str
    database: str
    openai: str


# =============================================================================
# Chat Schemas
# =============================================================================

class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID for context")


class ChatMessage(BaseModel):
    """Individual chat message."""
    role: str  # 'user' | 'assistant'
    content: str
    timestamp: Optional[datetime] = None


class ChatResponse(BaseModel):
    """Response schema for chat endpoint (non-streaming)."""
    conversation_id: str
    message: str
    suggested_prompts: list[str] = []


class SuggestedPromptsResponse(BaseModel):
    """Response schema for suggested prompts."""
    prompts: list[str]


class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history."""
    conversation_id: str
    messages: list[ChatMessage]
    created_at: Optional[datetime] = None


# =============================================================================
# Session Management Schemas
# =============================================================================

class SessionOut(BaseModel):
    """Session response schema for session listing."""
    id: str
    name: Optional[str] = None
    filename: Optional[str] = None
    row_count: Optional[int] = None
    status: str
    is_sample: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """Response schema for listing user sessions."""
    sessions: list[SessionOut]
    active_session_id: Optional[str] = None


class UserInfoResponse(BaseModel):
    """Response schema for current user info."""
    user_id: str
    sessions: list[SessionOut]
    has_sample_session: bool
    active_session_id: Optional[str] = None


# =============================================================================
# Fortune Cookie Schemas
# =============================================================================

class FortuneResponse(BaseModel):
    """Response schema for fortune cookie generation."""
    fortune: str = Field(..., description="The mystical fortune text")
    sentiment: Literal["positive", "neutral", "warning"] = Field(
        ..., description="Fortune sentiment (affects cookie color)"
    )
    lucky_number: Optional[str] = Field(
        None, description="Lucky number as dollar amount (e.g., '$450')"
    )
