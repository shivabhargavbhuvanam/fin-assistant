"""
Subscription and recurring charge detection using rule-based pattern matching.

This module uses deterministic rules rather than ML because:
1. Subscriptions follow predictable patterns (same amount, regular intervals)
2. Rules are transparent, fast, and don't require training data
3. ML would just learn the same rules with extra complexity

Detection Strategies:
    - Interval analysis (monthly ~30 days, weekly ~7 days)
    - Amount consistency (low variance = subscription)
    - Known merchant patterns (Netflix, Spotify, etc.)
    - Gray charge detection (small, forgotten subscriptions)

Author: Smart Financial Coach Team
"""

import re
from collections import defaultdict
from datetime import date
from typing import Optional
from sqlalchemy.orm import Session as DBSession

from models import Transaction, RecurringCharge, Category


class RecurringDetector:
    """
    Identify subscriptions and recurring charges using rule-based detection.
    
    Why rules over ML:
        - Subscriptions are deterministic: same merchant, same amount, regular intervals
        - Rules are transparent and explainable
        - No training data needed
        - Works on first upload (no cold start problem)
    """
    
    # Known subscription keywords (case-insensitive matching)
    SUBSCRIPTION_KEYWORDS = {
        # Streaming
        'netflix', 'spotify', 'hulu', 'disney', 'hbo', 'amazon prime',
        'apple music', 'youtube premium', 'peacock', 'paramount',
        'audible', 'kindle unlimited', 'crunchyroll',
        
        # Software/Cloud
        'adobe', 'microsoft 365', 'office 365', 'dropbox', 'icloud',
        'google storage', 'github', 'slack', 'zoom', 'notion',
        
        # Fitness
        'gym', 'fitness', 'peloton', 'planet fitness', 'orangetheory',
        
        # News/Media
        'nytimes', 'wsj', 'washington post', 'economist', 'medium',
        
        # Services
        'linkedin premium', 'grammarly', 'lastpass', '1password',
        'nordvpn', 'expressvpn', 'headspace', 'calm',
    }
    
    # Patterns that indicate subscriptions (regex)
    SUBSCRIPTION_PATTERNS = [
        r'subscription',
        r'recurring',
        r'monthly',
        r'membership',
        r'premium',
        r'plus\s*$',  # ends with "plus"
    ]
    
    # Patterns that are NOT subscriptions (utilities, rent, etc.)
    NON_SUBSCRIPTION_PATTERNS = [
        r'electric', r'power company', r'utility',
        r'water', r'sewer', r'gas company',
        r'rent', r'landlord', r'lease',
        r'insurance', r'paycheck', r'salary',
    ]

    def __init__(self, db: DBSession):
        self.db = db

    def detect(self, session_id: str) -> int:
        """Detect recurring charges in transactions."""
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.amount < 0)  # Spending only
            .order_by(Transaction.date)
            .all()
        )

        if len(transactions) < 2:
            return 0

        # Group by normalized description
        grouped = self._group_similar_transactions(transactions)

        recurring_count = 0
        category_map = {c.id: c for c in self.db.query(Category).all()}

        for pattern, txns in grouped.items():
            if len(txns) < 2:
                continue

            # Check for regular intervals
            intervals = self._calculate_intervals(txns)
            if not intervals:
                continue

            avg_interval = sum(intervals) / len(intervals)
            interval_variance = self._variance(intervals)

            # Monthly (28-31 days) or weekly (6-8 days) patterns
            is_monthly = 25 <= avg_interval <= 35 and interval_variance < 25
            is_weekly = 6 <= avg_interval <= 8 and interval_variance < 4

            if is_monthly or is_weekly:
                avg_amount = sum(t.amount for t in txns) / len(txns)
                category_id = txns[0].category_id

                # Determine frequency string
                if is_weekly:
                    frequency_days = 7
                else:
                    frequency_days = 30

                # Gray charge detection: small, possibly unknown, recurring
                category_name = (
                    category_map[category_id].name if category_id else "Other"
                )
                is_gray = abs(avg_amount) < 15 and category_name in [
                    "Other",
                    "Subscriptions",
                ]

                # Check for obviously small forgotten charges
                if abs(avg_amount) < 5 and len(pattern) < 20:
                    is_gray = True

                # Calculate confidence based on regularity
                confidence = 1.0 - min(interval_variance / 20, 0.5)

                # Determine if this is a known subscription type
                is_subscription = self._is_known_subscription(pattern)
                is_utility = self._is_utility_or_bill(pattern)
                
                # Boost confidence for known subscription patterns
                if is_subscription:
                    confidence = min(confidence + 0.1, 1.0)
                
                recurring = RecurringCharge(
                    session_id=session_id,
                    description_pattern=pattern,
                    category_id=category_id,
                    average_amount=avg_amount,
                    frequency_days=frequency_days,
                    occurrence_count=len(txns),
                    first_seen=min(t.date for t in txns),
                    last_seen=max(t.date for t in txns),
                    is_gray_charge=is_gray,
                    confidence=confidence,
                )
                self.db.add(recurring)
                recurring_count += 1

        self.db.commit()
        return recurring_count
    
    def _is_known_subscription(self, description: str) -> bool:
        """
        Check if description matches known subscription patterns.
        
        Uses keyword matching and regex patterns - transparent and fast.
        
        Args:
            description: Transaction description (already normalized).
            
        Returns:
            True if matches subscription pattern.
        """
        desc_lower = description.lower()
        
        # Check known subscription keywords
        for keyword in self.SUBSCRIPTION_KEYWORDS:
            if keyword in desc_lower:
                return True
        
        # Check regex patterns
        for pattern in self.SUBSCRIPTION_PATTERNS:
            if re.search(pattern, desc_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _is_utility_or_bill(self, description: str) -> bool:
        """
        Check if description matches utility/bill patterns (not subscriptions).
        
        These are recurring but not discretionary subscriptions.
        
        Args:
            description: Transaction description.
            
        Returns:
            True if matches utility/bill pattern.
        """
        desc_lower = description.lower()
        
        for pattern in self.NON_SUBSCRIPTION_PATTERNS:
            if re.search(pattern, desc_lower, re.IGNORECASE):
                return True
        
        return False

    def _group_similar_transactions(
        self, transactions: list[Transaction]
    ) -> dict[str, list[Transaction]]:
        """Group transactions by similar description patterns."""
        groups = defaultdict(list)

        for txn in transactions:
            # Normalize description for grouping
            pattern = self._normalize_for_matching(txn.description)
            groups[pattern].append(txn)

        return groups

    def _normalize_for_matching(self, description: str) -> str:
        """Normalize description for pattern matching."""
        # Convert to uppercase
        s = description.upper()

        # Remove common transaction noise
        s = re.sub(r"\d{4,}", "", s)  # Remove long numbers (transaction IDs)
        s = re.sub(r"#\d+", "", s)  # Remove order numbers
        s = re.sub(r"\*+", "", s)  # Remove asterisks
        s = re.sub(r"\s+", " ", s)  # Normalize whitespace
        s = s.strip()

        # Take first 30 chars for matching
        return s[:30] if len(s) > 30 else s

    def _calculate_intervals(self, transactions: list[Transaction]) -> list[int]:
        """Calculate day intervals between transactions."""
        dates = sorted([t.date for t in transactions])
        intervals = []

        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i - 1]).days
            intervals.append(interval)

        return intervals

    def _variance(self, values: list[float]) -> float:
        """Calculate variance of a list of numbers."""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
