"""
Module: pattern_analyzer.py
Description: Detects spending patterns for proactive insights.

Pattern Types:
    1. Category-Specific Patterns (dining, coffee, entertainment)
    2. Merchant Frequency Patterns (habitual spending at specific places)
    3. Temporal Patterns (weekend splurges, payday impulses)

Author: Smart Financial Coach Team
Created: 2025-01-31

Usage:
    analyzer = PatternAnalyzer(db, session_id)
    patterns = analyzer.detect_all()
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict
from sqlalchemy.orm import Session as DBSession

from models import Transaction, Category


class PatternAnalyzer:
    """
    Detects actionable spending patterns from transaction history.
    
    Focuses on patterns that lead to concrete savings suggestions,
    not just statistical observations.
    """

    # Merchant patterns for habit detection
    COFFEE_MERCHANTS = [
        'starbucks', 'dunkin', 'coffee', 'cafe', 'peet', 'caribou',
        'dutch bros', 'tim hortons', 'espresso', 'latte', 'brew'
    ]
    
    DINING_MERCHANTS = [
        'restaurant', 'grill', 'pizza', 'burger', 'sushi', 'thai',
        'chinese', 'mexican', 'indian', 'kitchen', 'eatery', 'diner',
        'bistro', 'tavern', 'pub', 'bar', 'doordash', 'uber eats',
        'grubhub', 'postmates', 'seamless'
    ]
    
    FAST_FOOD = [
        'mcdonald', 'wendy', 'burger king', 'taco bell', 'chick-fil-a',
        'chipotle', 'subway', 'kfc', 'popeyes', 'five guys', 'shake shack'
    ]

    # Savings percentages for replacement suggestions
    REPLACEMENT_SAVINGS = {
        'coffee': 0.70,      # Brew at home saves 70%
        'dining': 0.50,      # Cook at home saves 50%
        'fast_food': 0.60,   # Pack lunch saves 60%
        'delivery': 0.40,    # Pickup instead saves 40%
        'entertainment': 0.30,
        'shopping': 0.25,
    }

    def __init__(self, db: DBSession, session_id: str):
        """
        Initialize pattern analyzer.
        
        Args:
            db: Database session.
            session_id: Session ID to analyze.
        """
        self.db = db
        self.session_id = session_id
        self._transactions: Optional[list] = None
        self._categories: Optional[dict] = None

    @property
    def transactions(self) -> list:
        """Lazy load transactions."""
        if self._transactions is None:
            self._transactions = self.db.query(Transaction).filter(
                Transaction.session_id == self.session_id,
                Transaction.amount < 0  # Only expenses
            ).all()
        return self._transactions

    @property
    def categories(self) -> dict:
        """Lazy load category mapping."""
        if self._categories is None:
            self._categories = {c.id: c for c in self.db.query(Category).all()}
        return self._categories

    def detect_all(self) -> list[dict]:
        """
        Run all pattern detections.
        
        Returns:
            List of detected patterns with savings suggestions.
        """
        patterns = []
        
        # 1. Category patterns
        patterns.extend(self._detect_category_patterns())
        
        # 2. Merchant frequency patterns
        patterns.extend(self._detect_merchant_patterns())
        
        # 3. Temporal patterns
        patterns.extend(self._detect_temporal_patterns())
        
        # Sort by potential savings (highest first)
        patterns.sort(key=lambda p: p.get('yearly_savings', 0), reverse=True)
        
        return patterns

    # =========================================================================
    # Category-Specific Patterns
    # =========================================================================

    def _detect_category_patterns(self) -> list[dict]:
        """Detect high-impact category spending patterns."""
        patterns = []
        
        # Calculate months of data
        if not self.transactions:
            return patterns
            
        dates = [t.date for t in self.transactions if t.date]
        if not dates:
            return patterns
            
        min_date = min(dates)
        max_date = max(dates)
        months = max(1, (max_date - min_date).days / 30)
        
        # Aggregate by category
        category_spending = defaultdict(lambda: {'amount': 0, 'count': 0})
        for t in self.transactions:
            if t.category_id and t.category_id in self.categories:
                cat = self.categories[t.category_id]
                category_spending[cat.name]['amount'] += abs(t.amount)
                category_spending[cat.name]['count'] += 1
                category_spending[cat.name]['is_essential'] = cat.is_essential
        
        # Analyze non-essential categories
        for cat_name, data in category_spending.items():
            if data.get('is_essential', False):
                continue
                
            monthly = data['amount'] / months
            yearly = monthly * 12
            
            # Only flag high-impact patterns (>$100/month or >$1200/year)
            if yearly < 500:
                continue
            
            # Calculate potential savings based on category
            savings_rate = self._get_category_savings_rate(cat_name)
            potential_savings = yearly * savings_rate
            
            if potential_savings >= 200:  # Only suggest if meaningful
                patterns.append({
                    'type': 'category_pattern',
                    'category': cat_name,
                    'monthly_spend': round(monthly, 2),
                    'yearly_projection': round(yearly, 2),
                    'yearly_savings': round(potential_savings, 2),
                    'savings_rate': savings_rate,
                    'transaction_count': data['count'],
                    'suggestion': self._get_category_suggestion(cat_name, monthly, potential_savings),
                    'priority': 1 if potential_savings >= 1000 else 2
                })
        
        return patterns

    def _get_category_savings_rate(self, category: str) -> float:
        """Get realistic savings rate for a category."""
        cat_lower = category.lower()
        
        if 'dining' in cat_lower or 'restaurant' in cat_lower:
            return self.REPLACEMENT_SAVINGS['dining']
        elif 'entertainment' in cat_lower:
            return self.REPLACEMENT_SAVINGS['entertainment']
        elif 'shopping' in cat_lower:
            return self.REPLACEMENT_SAVINGS['shopping']
        else:
            return 0.25  # Default 25% reduction possible

    def _get_category_suggestion(self, category: str, monthly: float, yearly_savings: float) -> str:
        """Generate actionable suggestion for category."""
        cat_lower = category.lower()
        
        if 'dining' in cat_lower:
            return f"Cook 3 extra meals per week at home → Save ${yearly_savings:,.0f}/year"
        elif 'entertainment' in cat_lower:
            return f"Try free alternatives for some activities → Save ${yearly_savings:,.0f}/year"
        elif 'shopping' in cat_lower:
            return f"Apply the 24-hour rule before purchases → Save ${yearly_savings:,.0f}/year"
        else:
            return f"Reduce by 25% → Save ${yearly_savings:,.0f}/year"

    # =========================================================================
    # Merchant Frequency Patterns
    # =========================================================================

    def _detect_merchant_patterns(self) -> list[dict]:
        """Detect habitual spending at specific merchants."""
        patterns = []
        
        if not self.transactions:
            return patterns
        
        # Calculate months of data
        dates = [t.date for t in self.transactions if t.date]
        if not dates:
            return patterns
            
        min_date = min(dates)
        max_date = max(dates)
        months = max(1, (max_date - min_date).days / 30)
        
        # Group by normalized merchant
        merchant_data = defaultdict(lambda: {
            'count': 0, 
            'total': 0, 
            'amounts': [],
            'pattern_type': None
        })
        
        for t in self.transactions:
            normalized = self._normalize_merchant(t.description)
            if not normalized:
                continue
                
            merchant_data[normalized]['count'] += 1
            merchant_data[normalized]['total'] += abs(t.amount)
            merchant_data[normalized]['amounts'].append(abs(t.amount))
            
            # Detect pattern type
            if not merchant_data[normalized]['pattern_type']:
                merchant_data[normalized]['pattern_type'] = self._detect_merchant_type(t.description)
        
        # Analyze frequent merchants
        for merchant, data in merchant_data.items():
            monthly_visits = data['count'] / months
            
            # Only flag habitual spending (8+ visits per month)
            if monthly_visits < 4:
                continue
            
            monthly_cost = data['total'] / months
            yearly_cost = monthly_cost * 12
            avg_per_visit = data['total'] / data['count'] if data['count'] > 0 else 0
            
            # Get savings based on type
            pattern_type = data['pattern_type'] or 'other'
            savings_rate = self.REPLACEMENT_SAVINGS.get(pattern_type, 0.25)
            yearly_savings = yearly_cost * savings_rate
            
            if yearly_savings >= 200:  # Only meaningful savings
                patterns.append({
                    'type': 'merchant_habit',
                    'merchant': merchant,
                    'pattern_type': pattern_type,
                    'monthly_visits': round(monthly_visits, 1),
                    'monthly_cost': round(monthly_cost, 2),
                    'yearly_cost': round(yearly_cost, 2),
                    'yearly_savings': round(yearly_savings, 2),
                    'avg_per_visit': round(avg_per_visit, 2),
                    'is_habit': monthly_visits >= 8,
                    'suggestion': self._get_merchant_suggestion(merchant, pattern_type, yearly_savings),
                    'priority': 1 if yearly_savings >= 1000 else 2
                })
        
        return patterns

    def _normalize_merchant(self, description: str) -> str:
        """Normalize merchant name for grouping."""
        if not description:
            return ""
            
        # Uppercase and clean
        text = description.upper().strip()
        
        # Remove common suffixes/patterns
        text = re.sub(r'\s*#\d+.*$', '', text)  # Remove store numbers
        text = re.sub(r'\s+\d{5}(-\d{4})?$', '', text)  # Remove zip codes
        text = re.sub(r'\s+(CA|NY|TX|FL|WA|IL|PA|OH|GA|NC|MI|NJ|VA|AZ|MA|TN|IN|MO|MD|WI|CO|MN|SC|AL|LA|KY|OR|OK|CT|IA|UT|NV|AR|MS|KS|NM|NE|WV|ID|HI|NH|ME|MT|RI|DE|SD|ND|AK|DC|VT|WY)\s*$', '', text)
        text = re.sub(r'\s+USA\s*$', '', text)
        text = re.sub(r'\s+\d+\.\d{2}\s*$', '', text)  # Remove amounts
        text = re.sub(r'\s+\*+\d+$', '', text)  # Remove card refs
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        
        # Take first 2-3 words for common merchants
        words = text.split()[:3]
        return ' '.join(words)

    def _detect_merchant_type(self, description: str) -> Optional[str]:
        """Detect the type of merchant."""
        desc_lower = description.lower()
        
        if any(m in desc_lower for m in self.COFFEE_MERCHANTS):
            return 'coffee'
        elif any(m in desc_lower for m in self.FAST_FOOD):
            return 'fast_food'
        elif any(m in desc_lower for m in self.DINING_MERCHANTS):
            return 'dining'
        elif 'delivery' in desc_lower or 'doordash' in desc_lower or 'uber eats' in desc_lower:
            return 'delivery'
        
        return None

    def _get_merchant_suggestion(self, merchant: str, pattern_type: str, yearly_savings: float) -> str:
        """Generate suggestion for merchant habit."""
        if pattern_type == 'coffee':
            return f"Brewing at home 3 days/week saves ${yearly_savings:,.0f}/year"
        elif pattern_type == 'fast_food':
            return f"Pack lunch 3 days/week saves ${yearly_savings:,.0f}/year"
        elif pattern_type == 'dining':
            return f"Cook at home more often saves ${yearly_savings:,.0f}/year"
        elif pattern_type == 'delivery':
            return f"Pick up orders or cook saves ${yearly_savings:,.0f}/year"
        else:
            return f"Reduce visits by 50% saves ${yearly_savings/2:,.0f}/year"

    # =========================================================================
    # Temporal Patterns
    # =========================================================================

    def _detect_temporal_patterns(self) -> list[dict]:
        """Detect time-based spending patterns."""
        patterns = []
        
        if not self.transactions:
            return patterns
        
        # Group by day of week (0=Monday, 6=Sunday)
        by_dow = defaultdict(lambda: {'total': 0, 'count': 0})
        by_month_day = defaultdict(lambda: {'total': 0, 'count': 0})
        
        for t in self.transactions:
            if not t.date:
                continue
                
            dow = t.date.weekday()
            day = t.date.day
            
            by_dow[dow]['total'] += abs(t.amount)
            by_dow[dow]['count'] += 1
            by_month_day[day]['total'] += abs(t.amount)
            by_month_day[day]['count'] += 1
        
        # Weekend vs weekday analysis
        weekday_total = sum(by_dow[d]['total'] for d in range(5))
        weekend_total = sum(by_dow[d]['total'] for d in [5, 6])
        weekday_days = 5
        weekend_days = 2
        
        weekday_avg = weekday_total / weekday_days if weekday_days > 0 else 0
        weekend_avg = weekend_total / weekend_days if weekend_days > 0 else 0
        
        if weekend_avg > weekday_avg * 1.5 and weekend_total > 200:
            excess = (weekend_avg - weekday_avg) * weekend_days * 4  # Monthly excess
            yearly_savings = excess * 12 * 0.3  # 30% reducible
            
            if yearly_savings >= 200:
                patterns.append({
                    'type': 'weekend_splurge',
                    'weekday_avg': round(weekday_avg, 2),
                    'weekend_avg': round(weekend_avg, 2),
                    'excess_ratio': round(weekend_avg / weekday_avg, 2) if weekday_avg > 0 else 0,
                    'monthly_excess': round(excess, 2),
                    'yearly_savings': round(yearly_savings, 2),
                    'suggestion': f"Plan weekend activities ahead → Save ${yearly_savings:,.0f}/year",
                    'priority': 2
                })
        
        # Payday impulse (spending spike in first week)
        week1_total = sum(by_month_day[d]['total'] for d in range(1, 8))
        week3_total = sum(by_month_day[d]['total'] for d in range(15, 22))
        
        if week1_total > week3_total * 2 and week1_total > 500:
            excess = (week1_total - week3_total) * 0.5  # Half is impulse
            yearly_savings = excess * 12 * 0.4  # 40% avoidable
            
            if yearly_savings >= 200:
                patterns.append({
                    'type': 'payday_impulse',
                    'week1_spending': round(week1_total, 2),
                    'week3_spending': round(week3_total, 2),
                    'impulse_ratio': round(week1_total / week3_total, 2) if week3_total > 0 else 0,
                    'yearly_savings': round(yearly_savings, 2),
                    'suggestion': f"Wait 48 hours before post-payday purchases → Save ${yearly_savings:,.0f}/year",
                    'priority': 2
                })
        
        return patterns

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def get_top_insights(self, limit: int = 3) -> list[dict]:
        """Get top actionable insights from all patterns."""
        patterns = self.detect_all()
        
        # Filter to high-impact patterns only
        high_impact = [p for p in patterns if p.get('yearly_savings', 0) >= 300]
        
        return high_impact[:limit]

    def get_total_savings_potential(self) -> dict:
        """Calculate total potential savings across all patterns."""
        patterns = self.detect_all()
        
        total = sum(p.get('yearly_savings', 0) for p in patterns)
        by_type = defaultdict(float)
        
        for p in patterns:
            by_type[p['type']] += p.get('yearly_savings', 0)
        
        return {
            'total_yearly': round(total, 2),
            'total_monthly': round(total / 12, 2),
            'by_pattern_type': dict(by_type),
            'pattern_count': len(patterns)
        }
