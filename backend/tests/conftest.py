"""
Pytest configuration and shared fixtures for Smart Financial Coach tests.

This file is automatically loaded by pytest and provides:
    - Database fixtures (mock and real)
    - Sample data fixtures
    - Common test utilities

Author: Smart Financial Coach Team
"""

import pytest
import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import MagicMock, Mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []
    db.query.return_value.filter.return_value.first.return_value = None
    return db


@pytest.fixture
def mock_category():
    """Create a mock category."""
    category = Mock()
    category.id = 1
    category.name = "Dining"
    category.is_essential = False
    return category


# =============================================================================
# Transaction Fixtures
# =============================================================================

class MockTransaction:
    """Mock transaction object for testing."""
    
    def __init__(self, id, amount, category_id, date_val, description):
        self.id = id
        self.amount = amount
        self.category_id = category_id
        self.date = date_val
        self.description = description


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    base_date = date.today() - timedelta(days=30)
    
    return [
        MockTransaction(1, -25.00, 1, base_date, "COFFEE SHOP"),
        MockTransaction(2, -50.00, 1, base_date + timedelta(days=1), "RESTAURANT"),
        MockTransaction(3, -30.00, 1, base_date + timedelta(days=2), "COFFEE SHOP"),
        MockTransaction(4, -45.00, 1, base_date + timedelta(days=3), "LUNCH"),
        MockTransaction(5, -35.00, 1, base_date + timedelta(days=4), "DINNER"),
        MockTransaction(6, -500.00, 1, base_date + timedelta(days=5), "EXPENSIVE DINNER"),
        MockTransaction(7, -28.00, 1, base_date + timedelta(days=6), "COFFEE SHOP"),
        MockTransaction(8, -42.00, 1, base_date + timedelta(days=7), "RESTAURANT"),
    ]


@pytest.fixture
def large_transaction_set():
    """Create a larger set of transactions (60+) for ML testing."""
    base_date = date.today() - timedelta(days=90)
    
    transactions = []
    for i in range(60):
        amount = -25.00 - (i % 10) * 5  # Varying amounts
        transactions.append(
            MockTransaction(
                id=i + 1,
                amount=amount,
                category_id=(i % 5) + 1,
                date_val=base_date + timedelta(days=i),
                description=f"MERCHANT_{i % 10}"
            )
        )
    
    # Add some anomalies
    transactions.append(MockTransaction(61, -1000.00, 1, base_date + timedelta(days=61), "LUXURY STORE"))
    transactions.append(MockTransaction(62, -800.00, 2, base_date + timedelta(days=62), "JEWELRY"))
    
    return transactions


@pytest.fixture
def recurring_transactions():
    """Create transactions with recurring patterns."""
    base_date = date.today() - timedelta(days=180)
    
    transactions = []
    
    # Monthly subscription - Netflix
    for month in range(6):
        transactions.append(
            MockTransaction(
                id=len(transactions) + 1,
                amount=-15.99,
                category_id=6,  # Subscriptions
                date_val=base_date + timedelta(days=month * 30),
                description="NETFLIX SUBSCRIPTION"
            )
        )
    
    # Weekly purchase - Coffee
    for week in range(12):
        transactions.append(
            MockTransaction(
                id=len(transactions) + 1,
                amount=-5.50,
                category_id=1,
                date_val=base_date + timedelta(days=week * 7),
                description="STARBUCKS"
            )
        )
    
    return transactions


# =============================================================================
# AI Service Fixtures
# =============================================================================

@pytest.fixture
def mock_ai_service():
    """Create a mock AI service."""
    service = MagicMock()
    service.client = None  # Simulate no API key
    
    async def mock_categorize(*args, **kwargs):
        return []
    
    async def mock_insights(*args, **kwargs):
        return []
    
    service.categorize_transactions = mock_categorize
    service.generate_insights = mock_insights
    
    return service


# =============================================================================
# Test Utilities
# =============================================================================

def assert_valid_severity(severity: str) -> None:
    """Assert that severity is valid."""
    assert severity in ["low", "medium", "high"], f"Invalid severity: {severity}"


def assert_valid_confidence(confidence: float) -> None:
    """Assert that confidence is in valid range."""
    assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
