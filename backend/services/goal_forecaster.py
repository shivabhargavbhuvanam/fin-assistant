"""
Module: goal_forecaster.py
Description: Proactive goal and savings forecasting.

Unlike the reactive /goal endpoint, this module:
    - Auto-calculates savings capacity
    - Projects achievable goals
    - Suggests specific category cuts
    - Runs during /analyze (not on user request)

Author: Smart Financial Coach Team
Created: 2025-01-31

Usage:
    forecaster = GoalForecaster(db, session_id)
    forecast = forecaster.analyze()
"""

from collections import defaultdict
from sqlalchemy.orm import Session as DBSession

from models import Transaction, Category, RecurringCharge


class GoalForecaster:
    """
    Proactive financial goal forecasting.
    
    Analyzes current spending to determine:
    - How much can be saved realistically
    - Which categories to cut
    - Timeline for common goals
    """

    # Replaceability scores (how easy to reduce spending)
    # Higher = easier to cut
    REPLACEABILITY = {
        'Dining': 0.8,
        'Coffee': 0.9,
        'Entertainment': 0.6,
        'Shopping': 0.5,
        'Subscriptions': 0.7,
        'Travel': 0.4,
        'Personal Care': 0.3,
        # Essential categories have low replaceability
        'Housing': 0.05,
        'Utilities': 0.1,
        'Groceries': 0.15,
        'Healthcare': 0.05,
        'Transportation': 0.2,
        'Insurance': 0.0,
    }

    # Common savings goals with typical amounts
    COMMON_GOALS = [
        {'name': 'Emergency Fund (1 month)', 'months': 3, 'amount_factor': 1.0},  # 1 month expenses in 3 months
        {'name': 'Emergency Fund (3 months)', 'months': 12, 'amount_factor': 3.0},
        {'name': 'Vacation Fund', 'months': 6, 'amount_factor': 0.5},
        {'name': 'New Car Down Payment', 'months': 18, 'amount_factor': 2.0},
    ]

    def __init__(self, db: DBSession, session_id: str):
        """
        Initialize goal forecaster.
        
        Args:
            db: Database session.
            session_id: Session ID to analyze.
        """
        self.db = db
        self.session_id = session_id

    def analyze(self) -> dict:
        """
        Run full proactive analysis.
        
        Returns:
            Comprehensive forecast with savings capacity, cuts, and timelines.
        """
        # Get transactions and categories
        transactions = self.db.query(Transaction).filter(
            Transaction.session_id == self.session_id
        ).all()
        
        categories = {c.id: c for c in self.db.query(Category).all()}
        
        if not transactions:
            return self._empty_forecast()
        
        # Calculate date range
        dates = [t.date for t in transactions if t.date]
        if not dates:
            return self._empty_forecast()
            
        months = max(1, (max(dates) - min(dates)).days / 30)
        
        # Calculate monthly income and expenses
        monthly_income = sum(t.amount for t in transactions if t.amount > 0) / months
        monthly_expenses = abs(sum(t.amount for t in transactions if t.amount < 0)) / months
        
        # Break down by category
        by_category = defaultdict(lambda: {'amount': 0, 'is_essential': False, 'name': ''})
        for t in transactions:
            if t.amount >= 0:
                continue
            if t.category_id and t.category_id in categories:
                cat = categories[t.category_id]
                by_category[cat.id]['amount'] += abs(t.amount)
                by_category[cat.id]['is_essential'] = cat.is_essential
                by_category[cat.id]['name'] = cat.name
                by_category[cat.id]['icon'] = cat.icon
        
        # Calculate monthly amounts
        for cat_id in by_category:
            by_category[cat_id]['monthly'] = by_category[cat_id]['amount'] / months
        
        # Calculate savings capacity
        essential = sum(d['monthly'] for d in by_category.values() if d['is_essential'])
        discretionary = sum(d['monthly'] for d in by_category.values() if not d['is_essential'])
        
        # Realistic savings: 30% of discretionary
        base_savings_capacity = discretionary * 0.30
        
        # Calculate specific cuts
        suggested_cuts = self._calculate_cuts(by_category, categories)
        
        # Total from cuts
        total_from_cuts = sum(c['monthly_savings'] for c in suggested_cuts)
        
        # Effective savings capacity (use higher of base or cuts)
        savings_capacity = max(base_savings_capacity, total_from_cuts * 0.7)
        
        # Current net (income - expenses)
        current_net = monthly_income - monthly_expenses
        
        # Project common goals
        goal_projections = self._project_goals(monthly_expenses, savings_capacity)
        
        # Get recurring charges for context
        recurring = self.db.query(RecurringCharge).filter(
            RecurringCharge.session_id == self.session_id,
            RecurringCharge.is_gray_charge == True
        ).all()
        
        gray_charge_savings = sum(abs(r.average_amount) for r in recurring)
        
        return {
            'summary': {
                'monthly_income': round(monthly_income, 2),
                'monthly_expenses': round(monthly_expenses, 2),
                'current_net': round(current_net, 2),
                'essential_spending': round(essential, 2),
                'discretionary_spending': round(discretionary, 2),
                'savings_capacity': round(savings_capacity, 2),
                'savings_capacity_yearly': round(savings_capacity * 12, 2),
            },
            'suggested_cuts': suggested_cuts[:5],  # Top 5 cuts
            'gray_charge_savings': round(gray_charge_savings, 2),
            'goal_projections': goal_projections,
            'health_score': self._calculate_health_score(
                monthly_income, monthly_expenses, savings_capacity, discretionary
            ),
            'insights': self._generate_proactive_insights(
                monthly_income, monthly_expenses, savings_capacity, 
                suggested_cuts, gray_charge_savings
            )
        }

    def _calculate_cuts(self, by_category: dict, categories: dict) -> list[dict]:
        """Calculate specific category cuts ranked by impact."""
        cuts = []
        
        for cat_id, data in by_category.items():
            if data['is_essential']:
                continue
                
            monthly = data['monthly']
            if monthly < 20:  # Skip tiny categories
                continue
            
            # Get replaceability score
            cat_name = data['name']
            replaceability = self._get_replaceability(cat_name)
            
            # Calculate potential savings (replaceability * amount)
            potential = monthly * replaceability
            
            if potential < 10:  # Skip if savings too small
                continue
            
            cuts.append({
                'category': cat_name,
                'category_icon': data.get('icon', ''),
                'current_monthly': round(monthly, 2),
                'suggested_reduction': replaceability,
                'monthly_savings': round(potential, 2),
                'yearly_savings': round(potential * 12, 2),
                'difficulty': self._get_difficulty(replaceability),
                'action': self._get_cut_action(cat_name, potential),
            })
        
        # Sort by yearly savings descending
        cuts.sort(key=lambda c: c['yearly_savings'], reverse=True)
        
        return cuts

    def _get_replaceability(self, category_name: str) -> float:
        """Get replaceability score for a category."""
        for key, score in self.REPLACEABILITY.items():
            if key.lower() in category_name.lower():
                return score
        return 0.3  # Default 30% reducible

    def _get_difficulty(self, replaceability: float) -> str:
        """Convert replaceability to difficulty label."""
        if replaceability >= 0.7:
            return 'easy'
        elif replaceability >= 0.4:
            return 'moderate'
        else:
            return 'hard'

    def _get_cut_action(self, category: str, monthly_savings: float) -> str:
        """Generate actionable suggestion for category cut."""
        cat_lower = category.lower()
        
        if 'dining' in cat_lower or 'restaurant' in cat_lower:
            return f"Cook 3 extra meals per week → Save ${monthly_savings:.0f}/month"
        elif 'coffee' in cat_lower:
            return f"Brew at home 3 days/week → Save ${monthly_savings:.0f}/month"
        elif 'entertainment' in cat_lower:
            return f"Find free alternatives for some activities → Save ${monthly_savings:.0f}/month"
        elif 'subscription' in cat_lower:
            return f"Cancel unused subscriptions → Save ${monthly_savings:.0f}/month"
        elif 'shopping' in cat_lower:
            return f"Apply 24-hour rule before purchases → Save ${monthly_savings:.0f}/month"
        else:
            return f"Reduce spending by {int(self._get_replaceability(category) * 100)}% → Save ${monthly_savings:.0f}/month"

    def _project_goals(self, monthly_expenses: float, savings_capacity: float) -> list[dict]:
        """Project timelines for common goals."""
        projections = []
        
        for goal in self.COMMON_GOALS:
            target_amount = monthly_expenses * goal['amount_factor']
            
            if savings_capacity > 0:
                months_needed = target_amount / savings_capacity
                achievable = months_needed <= goal['months'] * 1.5  # Allow 50% buffer
            else:
                months_needed = float('inf')
                achievable = False
            
            projections.append({
                'goal': goal['name'],
                'target_amount': round(target_amount, 2),
                'target_months': goal['months'],
                'projected_months': round(months_needed, 1) if months_needed != float('inf') else None,
                'achievable': achievable,
                'monthly_required': round(target_amount / goal['months'], 2),
            })
        
        return projections

    def _calculate_health_score(
        self, 
        income: float, 
        expenses: float, 
        savings_capacity: float,
        discretionary: float
    ) -> dict:
        """Calculate overall financial health score."""
        
        # Savings rate (ideal: 20%+)
        savings_rate = savings_capacity / income if income > 0 else 0
        savings_score = min(100, (savings_rate / 0.20) * 100)
        
        # Expense ratio (ideal: <80% of income)
        expense_ratio = expenses / income if income > 0 else 1
        expense_score = max(0, (1 - expense_ratio) * 100 + 20)
        
        # Discretionary control (ideal: <40% of expenses)
        discretionary_ratio = discretionary / expenses if expenses > 0 else 0
        discretionary_score = max(0, (0.5 - discretionary_ratio) * 200)
        
        # Overall score
        overall = (savings_score * 0.4 + expense_score * 0.4 + discretionary_score * 0.2)
        
        return {
            'overall': round(min(100, max(0, overall)), 0),
            'savings_rate': round(savings_rate * 100, 1),
            'expense_ratio': round(expense_ratio * 100, 1),
            'discretionary_ratio': round(discretionary_ratio * 100, 1),
            'rating': self._get_rating(overall),
        }

    def _get_rating(self, score: float) -> str:
        """Convert score to rating."""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        else:
            return 'needs_attention'

    def _generate_proactive_insights(
        self,
        income: float,
        expenses: float,
        savings_capacity: float,
        cuts: list[dict],
        gray_savings: float
    ) -> list[dict]:
        """Generate proactive insights from forecast."""
        insights = []
        
        # Main savings insight
        if savings_capacity > 0:
            yearly = savings_capacity * 12
            insights.append({
                'type': 'savings_potential',
                'priority': 1,
                'title': f"You could save ${savings_capacity:,.0f}/month",
                'description': f"Based on your spending patterns, you have the potential to save ${yearly:,.0f} per year by making small adjustments.",
                'action': 'See suggested cuts below',
                'data': {'monthly': savings_capacity, 'yearly': yearly}
            })
        
        # Top cut insight
        if cuts:
            top_cut = cuts[0]
            insights.append({
                'type': 'top_cut',
                'priority': 1,
                'title': f"{top_cut['category']}: ${top_cut['current_monthly']:,.0f}/month",
                'description': top_cut['action'],
                'action': f"Save ${top_cut['yearly_savings']:,.0f}/year",
                'data': top_cut
            })
        
        # Gray charges insight
        if gray_savings > 0:
            insights.append({
                'type': 'gray_charges',
                'priority': 2,
                'title': f"${gray_savings:,.0f}/month in forgotten subscriptions",
                'description': "These small recurring charges add up. Consider canceling unused ones.",
                'action': f"Cancel to save ${gray_savings * 12:,.0f}/year",
                'data': {'monthly': gray_savings}
            })
        
        # Living beyond means warning
        if expenses > income:
            gap = expenses - income
            insights.append({
                'type': 'warning',
                'priority': 1,
                'title': "Spending exceeds income",
                'description': f"You're spending ${gap:,.0f}/month more than you earn. This needs immediate attention.",
                'action': "Review expenses and find cuts",
                'data': {'gap': gap}
            })
        
        return insights

    def _empty_forecast(self) -> dict:
        """Return empty forecast when no data."""
        return {
            'summary': {
                'monthly_income': 0,
                'monthly_expenses': 0,
                'current_net': 0,
                'essential_spending': 0,
                'discretionary_spending': 0,
                'savings_capacity': 0,
                'savings_capacity_yearly': 0,
            },
            'suggested_cuts': [],
            'gray_charge_savings': 0,
            'goal_projections': [],
            'health_score': {'overall': 0, 'rating': 'unknown'},
            'insights': []
        }


def forecast_savings(db: DBSession, session_id: str) -> dict:
    """Convenience function to run forecast."""
    forecaster = GoalForecaster(db, session_id)
    return forecaster.analyze()
