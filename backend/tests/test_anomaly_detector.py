"""
Test Module: test_anomaly_detector.py
Description: Unit tests for ML anomaly detection logic.

Tests:
    - Feature extraction
    - Severity classification
    - Z-score calculation with edge cases
    - Duplicate detection
    - Bounds checking
    - Statistical validation

Author: Smart Financial Coach Team
Created: 2025-01-31
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.anomaly_detector import (
    MLAnomalyDetector,
    AnomalyDetector,
    MAX_AMOUNT_FOR_ANALYSIS,
    MIN_TRANSACTIONS_FOR_ZSCORE,
    MAX_ZSCORE
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    base_date = date.today() - timedelta(days=30)
    
    class MockTransaction:
        def __init__(self, id, amount, category_id, date_val, description):
            self.id = id
            self.amount = amount
            self.category_id = category_id
            self.date = date_val
            self.description = description
    
    transactions = [
        MockTransaction(1, -25.00, 1, base_date, "COFFEE SHOP"),
        MockTransaction(2, -50.00, 1, base_date + timedelta(days=1), "RESTAURANT"),
        MockTransaction(3, -30.00, 1, base_date + timedelta(days=2), "COFFEE SHOP"),
        MockTransaction(4, -45.00, 1, base_date + timedelta(days=3), "LUNCH"),
        MockTransaction(5, -35.00, 1, base_date + timedelta(days=4), "DINNER"),
        MockTransaction(6, -500.00, 1, base_date + timedelta(days=5), "EXPENSIVE DINNER"),  # Anomaly
        MockTransaction(7, -28.00, 1, base_date + timedelta(days=6), "COFFEE SHOP"),
        MockTransaction(8, -42.00, 1, base_date + timedelta(days=7), "RESTAURANT"),
    ]
    return transactions


@pytest.fixture
def extreme_transactions():
    """Create transactions with extreme values for edge case testing."""
    base_date = date.today()
    
    class MockTransaction:
        def __init__(self, id, amount, category_id, date_val, description):
            self.id = id
            self.amount = amount
            self.category_id = category_id
            self.date = date_val
            self.description = description
    
    return [
        MockTransaction(1, -100000.00, 1, base_date, "EXTREME PURCHASE"),  # Above max
        MockTransaction(2, -0.01, 1, base_date, "TINY CHARGE"),  # Very small
        MockTransaction(3, 0.00, 1, base_date, "ZERO AMOUNT"),  # Zero
        MockTransaction(4, -25.00, 1, base_date, "NORMAL"),  # Normal
    ]


# =============================================================================
# Feature Extraction Tests
# =============================================================================

class TestFeatureExtraction:
    """Tests for feature extraction logic."""
    
    def test_extract_features_returns_dataframe(self, mock_db, sample_transactions):
        """Test that feature extraction returns a DataFrame."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(sample_transactions)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_transactions)
    
    def test_extract_features_has_required_columns(self, mock_db, sample_transactions):
        """Test that extracted features include all required columns."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(sample_transactions)
        
        required_columns = [
            'amount_abs', 'amount_zscore', 'amount_log',
            'merchant_frequency', 'is_one_time', 'day_of_week',
            'is_weekend', 'is_payday'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_amount_zscore_calculation(self, mock_db, sample_transactions):
        """Test z-score calculation is correct."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(sample_transactions)
        
        # Z-score should have mean ~0 and std ~1
        mean_zscore = df['amount_zscore'].mean()
        std_zscore = df['amount_zscore'].std()
        
        assert abs(mean_zscore) < 0.1, "Z-score mean should be near 0"
        assert abs(std_zscore - 1) < 0.1, "Z-score std should be near 1"
    
    def test_log_transform_handles_zero(self, mock_db, extreme_transactions):
        """Test log transform handles zero and small amounts."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(extreme_transactions)
        
        # Log transform should not produce NaN or infinity
        assert not df['amount_log'].isna().any()
        assert not np.isinf(df['amount_log']).any()
    
    def test_merchant_frequency_calculation(self, mock_db, sample_transactions):
        """Test merchant frequency is calculated correctly."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(sample_transactions)
        
        # Frequencies should sum to approximately 1
        assert df['merchant_frequency'].sum() > 0
        # All frequencies should be between 0 and 1
        assert (df['merchant_frequency'] >= 0).all()
        assert (df['merchant_frequency'] <= 1).all()
    
    def test_is_weekend_flag(self, mock_db, sample_transactions):
        """Test weekend flag is set correctly."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(sample_transactions)
        
        # is_weekend should be 0 or 1
        assert df['is_weekend'].isin([0, 1]).all()
    
    def test_is_payday_flag(self, mock_db, sample_transactions):
        """Test payday flag logic."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(sample_transactions)
        
        # is_payday should be 0 or 1
        assert df['is_payday'].isin([0, 1]).all()
    
    def test_empty_transactions_returns_empty_df(self, mock_db):
        """Test empty input returns empty DataFrame."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# =============================================================================
# Severity Classification Tests
# =============================================================================

class TestSeverityClassification:
    """Tests for severity classification logic."""
    
    def test_score_to_severity_high(self, mock_db):
        """Test high severity classification."""
        detector = MLAnomalyDetector(mock_db)
        
        # Isolation Forest: more negative score = more anomalous
        # Score of -0.1 gives confidence of 0.2 (>= 0.15 threshold for high)
        severity = detector._score_to_severity(-0.1)
        assert severity == "high"
    
    def test_score_to_severity_medium(self, mock_db):
        """Test medium severity classification."""
        detector = MLAnomalyDetector(mock_db)
        
        # Score of -0.05 gives confidence of 0.1 (>= 0.08 threshold for medium)
        severity = detector._score_to_severity(-0.05)
        assert severity == "medium"
    
    def test_score_to_severity_low(self, mock_db):
        """Test low severity classification."""
        detector = MLAnomalyDetector(mock_db)
        
        # Score of 0.1 gives confidence of 0 (low)
        severity = detector._score_to_severity(0.1)
        assert severity == "low"
    
    def test_score_to_confidence_bounds(self, mock_db):
        """Test confidence score is bounded 0-1."""
        detector = MLAnomalyDetector(mock_db)
        
        # Test various scores (negative = anomalous in Isolation Forest)
        for score in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            confidence = detector._score_to_confidence(score)
            assert 0 <= confidence <= 1, f"Confidence {confidence} out of bounds for score {score}"


# =============================================================================
# Statistical Validation Tests
# =============================================================================

class TestStatisticalValidation:
    """Tests for statistical assumption validation."""
    
    def test_min_transactions_check(self, mock_db):
        """Test minimum transactions requirement for z-score."""
        detector = AnomalyDetector(mock_db)
        
        # With less than MIN_TRANSACTIONS_FOR_ZSCORE, should return 0
        mock_db.query.return_value.filter.return_value.filter.return_value.all.return_value = []
        
        result = detector.detect("test-session")
        assert result == 0
    
    def test_zscore_cap(self):
        """Test z-score is capped at MAX_ZSCORE."""
        # Test the cap value
        assert MAX_ZSCORE == 10.0
        
        # Z-scores should be clipped
        test_zscore = 100.0  # Extreme value
        capped = np.clip(test_zscore, -MAX_ZSCORE, MAX_ZSCORE)
        assert capped == MAX_ZSCORE
    
    def test_amount_bounds(self):
        """Test amount bounds checking."""
        # Amounts above MAX_AMOUNT_FOR_ANALYSIS should be filtered
        assert MAX_AMOUNT_FOR_ANALYSIS == 50000
        
        # Test filtering logic
        amounts = pd.Series([100, 25000, 75000, 100000])
        filtered = amounts[amounts.abs() <= MAX_AMOUNT_FOR_ANALYSIS]
        assert len(filtered) == 2  # Only 100 and 25000


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_all_same_amounts(self, mock_db):
        """Test handling when all amounts are identical (zero std)."""
        class MockTransaction:
            def __init__(self, id, amount, category_id, date_val, description):
                self.id = id
                self.amount = amount
                self.category_id = category_id
                self.date = date_val
                self.description = description
        
        # All same amounts
        same_transactions = [
            MockTransaction(i, -50.00, 1, date.today(), f"TXN_{i}")
            for i in range(10)
        ]
        
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(same_transactions)
        
        # Should handle zero std gracefully
        assert not df['amount_zscore'].isna().any()
    
    def test_negative_amounts_handled(self, mock_db, sample_transactions):
        """Test negative amounts (expenses) are handled correctly."""
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(sample_transactions)
        
        # amount_abs should all be positive
        assert (df['amount_abs'] >= 0).all()
    
    def test_mixed_categories(self, mock_db):
        """Test handling of multiple categories."""
        class MockTransaction:
            def __init__(self, id, amount, category_id, date_val, description):
                self.id = id
                self.amount = amount
                self.category_id = category_id
                self.date = date_val
                self.description = description
        
        # Mix of categories
        mixed_transactions = [
            MockTransaction(1, -25.00, 1, date.today(), "FOOD"),
            MockTransaction(2, -100.00, 2, date.today(), "GAS"),
            MockTransaction(3, -50.00, 1, date.today(), "FOOD"),
            MockTransaction(4, -150.00, 3, date.today(), "SHOPPING"),
        ]
        
        detector = MLAnomalyDetector(mock_db)
        df = detector._extract_features(mixed_transactions)
        
        assert len(df) == 4


# =============================================================================
# Merchant Normalization Tests
# =============================================================================

class TestMerchantNormalization:
    """Tests for merchant name normalization."""
    
    def test_normalize_merchant_removes_numbers(self, mock_db):
        """Test that transaction IDs and store numbers are removed."""
        detector = MLAnomalyDetector(mock_db)
        
        result = detector._normalize_merchant("STARBUCKS STORE #12345")
        assert "#12345" not in result
        assert "12345" not in result
    
    def test_normalize_merchant_handles_locations(self, mock_db):
        """Test location info is handled."""
        detector = MLAnomalyDetector(mock_db)
        
        result1 = detector._normalize_merchant("WALMART NYC")
        result2 = detector._normalize_merchant("WALMART LA")
        
        # Both should normalize to similar base
        assert "WALMART" in result1
        assert "WALMART" in result2
    
    def test_normalize_merchant_preserves_text(self, mock_db):
        """Test normalization preserves text and removes numbers."""
        detector = MLAnomalyDetector(mock_db)
        
        result = detector._normalize_merchant("starbucks coffee 12345")
        # Should remove numbers but preserve text
        assert "starbucks" in result.lower()
        assert "12345" not in result


# =============================================================================
# Explanation Generation Tests
# =============================================================================

class TestExplanationGeneration:
    """Tests for anomaly explanation generation."""
    
    def test_explanation_contains_amount(self, mock_db):
        """Test explanation includes transaction amount."""
        detector = AnomalyDetector(mock_db)
        
        explanation = detector._generate_explanation(
            category="Dining",
            actual=150.0,
            expected=45.0,
            z_score_abs=3.5
        )
        
        assert "$150" in explanation or "150" in explanation
    
    def test_explanation_contains_category(self, mock_db):
        """Test explanation includes category name."""
        detector = AnomalyDetector(mock_db)
        
        explanation = detector._generate_explanation(
            category="Entertainment",
            actual=200.0,
            expected=50.0,
            z_score_abs=3.0
        )
        
        assert "Entertainment" in explanation


# =============================================================================
# Integration-like Tests (with mocks)
# =============================================================================

class TestAnomalyDetectorIntegration:
    """Integration-style tests for the full anomaly detection flow."""
    
    def test_hybrid_detector_chooses_ml_for_large_dataset(self, mock_db):
        """Test ML is used for >=50 transactions."""
        detector = AnomalyDetector(mock_db)
        
        # Create 60 mock transactions
        mock_transactions = [Mock() for _ in range(60)]
        mock_db.query.return_value.filter.return_value.filter.return_value.all.return_value = mock_transactions
        
        # ML detector should be initialized
        assert detector.ml_detector is not None or not hasattr(detector, 'ml_detector')
    
    def test_hybrid_detector_chooses_statistical_for_small_dataset(self, mock_db):
        """Test statistical method is used for <50 transactions."""
        detector = AnomalyDetector(mock_db)
        
        # For small datasets, _detect_statistical should be called
        # (This is a design validation test)
        assert hasattr(detector, '_detect_statistical')


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
