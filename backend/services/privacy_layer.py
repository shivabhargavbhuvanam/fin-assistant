"""
Module: privacy_layer.py
Description: Anonymizes transaction data before sending to external AI services

Privacy Principles:
    - Never send raw merchant names to OpenAI
    - Aggregate data where possible
    - Hash identifiers for pattern matching
    - Keep reverse mapping locally only

Author: Smart Financial Coach Team
Created: 2025-01-31

Usage:
    privacy = PrivacyLayer()
    safe_context = privacy.anonymize_for_ai(transactions, anomalies)
"""

import re
import hashlib
from typing import Optional
from collections import defaultdict


class PrivacyLayer:
    """
    Anonymizes financial data for AI processing.
    
    Ensures no PII (Personally Identifiable Information) or 
    merchant-specific data is sent to external AI services.
    """

    def __init__(self):
        """Initialize privacy layer with empty mappings."""
        # Local mapping for reverse lookup (never sent to AI)
        self._hash_to_merchant: dict[str, str] = {}
        self._merchant_to_hash: dict[str, str] = {}

    def anonymize_transactions(self, transactions: list[dict]) -> list[dict]:
        """
        Anonymize a list of transactions for AI processing.
        
        Args:
            transactions: List of transaction dicts with descriptions.
            
        Returns:
            Anonymized transactions with hashed merchant IDs.
        """
        anonymized = []
        
        for t in transactions:
            anon = {
                'amount': t.get('amount', 0),
                'category': t.get('category', 'Unknown'),
                'date_week': self._get_week_of_month(t.get('date')),
                'is_weekend': self._is_weekend(t.get('date')),
            }
            
            # Hash the merchant name
            if 'description' in t:
                merchant_id = self._hash_merchant(t['description'])
                anon['merchant_id'] = merchant_id
            
            anonymized.append(anon)
        
        return anonymized

    def aggregate_for_categorization(self, transactions: list) -> list[dict]:
        """
        Aggregate transactions by merchant pattern for AI categorization.
        
        Instead of sending individual transactions, we send:
        - Merchant pattern hash
        - Count of transactions
        - Average amount
        - Amount range
        
        Args:
            transactions: List of Transaction ORM objects.
            
        Returns:
            Aggregated merchant patterns for categorization.
        """
        merchant_groups = defaultdict(lambda: {
            'count': 0,
            'total': 0,
            'amounts': [],
            'sample_words': set()
        })
        
        for t in transactions:
            if not t.description:
                continue
                
            # Normalize and hash
            normalized = self._normalize_merchant(t.description)
            merchant_id = self._hash_merchant(normalized)
            
            merchant_groups[merchant_id]['count'] += 1
            merchant_groups[merchant_id]['total'] += abs(t.amount)
            merchant_groups[merchant_id]['amounts'].append(abs(t.amount))
            
            # Extract safe category hints (no PII)
            hints = self._extract_category_hints(t.description)
            merchant_groups[merchant_id]['sample_words'].update(hints)
        
        # Build aggregated list
        aggregated = []
        for merchant_id, data in merchant_groups.items():
            amounts = data['amounts']
            aggregated.append({
                'merchant_id': merchant_id,
                'transaction_count': data['count'],
                'avg_amount': round(data['total'] / data['count'], 2) if data['count'] > 0 else 0,
                'min_amount': round(min(amounts), 2) if amounts else 0,
                'max_amount': round(max(amounts), 2) if amounts else 0,
                'category_hints': list(data['sample_words'])[:5],  # Limit hints
            })
        
        return aggregated

    def anonymize_for_insights(self, context: dict) -> dict:
        """
        Prepare context for AI insight generation.
        
        Removes all merchant-specific information, keeps only:
        - Category aggregates
        - Percentages
        - Counts
        - Severity levels
        
        Args:
            context: Full context dict with spending data.
            
        Returns:
            Privacy-safe context for AI.
        """
        safe_context = {}
        
        # Spending summary is already safe (category-level)
        if 'spending_summary' in context:
            safe_context['spending_summary'] = context['spending_summary']
        
        # Anonymize anomalies
        if 'anomalies' in context:
            safe_context['anomalies'] = [
                {
                    'category': a.get('category', 'Unknown'),
                    'amount': a.get('amount', 0),
                    'expected_range': f"{a.get('expected', 0) * 0.8:.0f}-{a.get('expected', 0) * 1.2:.0f}",
                    'severity': a.get('severity', 'low'),
                    'type': a.get('type', 'amount'),
                    # NO merchant names
                }
                for a in context.get('anomalies', [])
            ]
        
        # Anonymize recurring charges
        if 'recurring_charges' in context:
            safe_context['recurring_charges'] = [
                {
                    'category': r.get('category', 'Unknown'),
                    'monthly_amount': r.get('amount', 0),
                    'frequency': r.get('frequency', 'monthly'),
                    'is_gray_charge': r.get('is_gray_charge', False),
                    # NO merchant names
                }
                for r in context.get('recurring_charges', [])
            ]
        
        # Anonymize patterns
        if 'patterns' in context:
            safe_context['patterns'] = [
                {
                    'type': p.get('type', 'unknown'),
                    'category': p.get('category', 'Unknown'),
                    'monthly_spend': p.get('monthly_spend', 0),
                    'yearly_savings': p.get('yearly_savings', 0),
                    'suggestion_type': self._get_suggestion_type(p),
                    # NO merchant names
                }
                for p in context.get('patterns', [])
            ]
        
        # Keep safe metadata
        safe_context['total_transactions'] = context.get('total_transactions', 0)
        safe_context['date_range'] = context.get('date_range', {})
        safe_context['months_of_data'] = context.get('months_of_data', 1)
        
        return safe_context

    def deanonymize_result(self, result: dict) -> dict:
        """
        Reverse anonymization for display purposes.
        
        Uses local mapping to restore merchant names where needed.
        
        Args:
            result: AI result with anonymized references.
            
        Returns:
            Result with restored merchant names.
        """
        # This would use self._hash_to_merchant to restore names
        # For now, we don't need this since we avoid sending merchant refs
        return result

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _hash_merchant(self, description: str) -> str:
        """Create a consistent hash for a merchant."""
        if not description:
            return "UNKNOWN"
            
        normalized = self._normalize_merchant(description)
        hash_value = hashlib.md5(normalized.lower().encode()).hexdigest()[:8]
        merchant_id = f"M_{hash_value}"
        
        # Store mapping for potential reverse lookup
        self._hash_to_merchant[merchant_id] = normalized
        self._merchant_to_hash[normalized] = merchant_id
        
        return merchant_id

    def _normalize_merchant(self, description: str) -> str:
        """Normalize merchant name for consistent hashing."""
        if not description:
            return ""
            
        text = description.upper().strip()
        
        # Remove store numbers, locations, zip codes
        text = re.sub(r'\s*#\d+.*$', '', text)
        text = re.sub(r'\s+\d{5}(-\d{4})?$', '', text)
        text = re.sub(r'\s+[A-Z]{2}\s*$', '', text)  # State codes
        text = re.sub(r'\s+USA\s*$', '', text)
        text = re.sub(r'\s+\d+\.\d{2}\s*$', '', text)
        text = re.sub(r'\s+\*+\d+$', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Take first 2-3 words
        words = text.split()[:3]
        return ' '.join(words)

    def _extract_category_hints(self, description: str) -> set[str]:
        """Extract safe category hints from description."""
        hints = set()
        desc_lower = description.lower()
        
        # Generic category keywords (not merchant-specific)
        if any(w in desc_lower for w in ['coffee', 'cafe', 'espresso']):
            hints.add('coffee_shop')
        if any(w in desc_lower for w in ['restaurant', 'grill', 'kitchen', 'diner']):
            hints.add('restaurant')
        if any(w in desc_lower for w in ['grocery', 'market', 'foods']):
            hints.add('grocery')
        if any(w in desc_lower for w in ['gas', 'fuel', 'petrol', 'shell', 'chevron', 'exxon']):
            hints.add('gas_station')
        if any(w in desc_lower for w in ['pharmacy', 'drug', 'cvs', 'walgreen']):
            hints.add('pharmacy')
        if any(w in desc_lower for w in ['amazon', 'online', 'shop']):
            hints.add('online_shopping')
        if any(w in desc_lower for w in ['uber', 'lyft', 'taxi', 'ride']):
            hints.add('transportation')
        if any(w in desc_lower for w in ['netflix', 'spotify', 'hulu', 'disney', 'subscription']):
            hints.add('subscription')
        if any(w in desc_lower for w in ['gym', 'fitness', 'yoga']):
            hints.add('fitness')
        
        return hints

    def _get_week_of_month(self, date) -> Optional[int]:
        """Get week of month (1-5) from date."""
        if not date:
            return None
        try:
            day = date.day if hasattr(date, 'day') else int(date.split('-')[2])
            return (day - 1) // 7 + 1
        except:
            return None

    def _is_weekend(self, date) -> Optional[bool]:
        """Check if date is weekend."""
        if not date:
            return None
        try:
            if hasattr(date, 'weekday'):
                return date.weekday() >= 5
            return None
        except:
            return None

    def _get_suggestion_type(self, pattern: dict) -> str:
        """Get generic suggestion type from pattern."""
        pattern_type = pattern.get('type', '')
        
        if 'coffee' in str(pattern_type).lower():
            return 'reduce_coffee'
        elif 'dining' in str(pattern_type).lower():
            return 'cook_more'
        elif 'weekend' in str(pattern_type).lower():
            return 'plan_weekends'
        elif 'payday' in str(pattern_type).lower():
            return 'delay_purchases'
        else:
            return 'reduce_spending'


# Singleton instance for convenience
_privacy_layer: Optional[PrivacyLayer] = None


def get_privacy_layer() -> PrivacyLayer:
    """Get or create singleton privacy layer instance."""
    global _privacy_layer
    if _privacy_layer is None:
        _privacy_layer = PrivacyLayer()
    return _privacy_layer
