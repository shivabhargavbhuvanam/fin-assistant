"""
Fortune Cookie Generator Service

Generates specific, actionable financial fortunes based on user's
spending patterns. Uses OpenAI with focused prompts for data-driven advice.

Features:
    - Data-driven fortune cookie style advice
    - References actual financial data (categories, amounts, counts)
    - Validates fortunes for specificity
    - Fallback fortunes when API unavailable
    - Color-coded sentiment (gold/orange/red)

Author: Smart Financial Coach Team
"""

import os
import random
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class FortuneSentiment(str, Enum):
    """Fortune sentiment determines cookie color."""
    POSITIVE = "positive"  # Gold cookie - good news
    NEUTRAL = "neutral"    # Orange cookie - informational
    WARNING = "warning"    # Red cookie - needs attention


@dataclass
class FinancialStats:
    """Anonymized financial statistics for fortune generation."""
    top_spending_category: str
    top_spending_amount: float
    top_spending_percent_change: Optional[float]
    subscription_count: int
    gray_charge_count: int
    gray_charge_total: float  # Total $ from gray charges
    savings_rate: float  # As percentage
    anomaly_count: int
    monthly_trend: str  # "increasing", "decreasing", "stable"
    net_amount: float
    total_spending: float
    total_income: float


@dataclass
class Fortune:
    """Generated fortune with metadata."""
    text: str
    sentiment: FortuneSentiment
    lucky_number: Optional[str] = None  # Formatted as "$XXX"


# =============================================================================
# Fallback Fortunes (when API unavailable) - Now more specific
# =============================================================================

def get_dynamic_fallback_fortunes(stats: FinancialStats) -> list[Fortune]:
    """Generate fallback fortunes that reference actual user data."""
    fortunes = []
    
    # Gray charges fallback
    if stats.gray_charge_count >= 1:
        fortunes.append(Fortune(
            f"{stats.gray_charge_count} forgotten charges drain ${stats.gray_charge_total:.0f} monthly... audit and reclaim",
            FortuneSentiment.WARNING,
            f"${stats.gray_charge_total * 12:.0f}"
        ))
    
    # Subscription fallback
    if stats.subscription_count >= 3:
        potential_savings = stats.subscription_count * 10  # Estimate $10/subscription
        fortunes.append(Fortune(
            f"{stats.subscription_count} subscriptions active... cancel one unused, save ${potential_savings:.0f} yearly",
            FortuneSentiment.NEUTRAL,
            f"${potential_savings:.0f}"
        ))
    
    # Top spending fallback
    if stats.top_spending_amount > 0:
        savings_10pct = stats.top_spending_amount * 0.1
        fortunes.append(Fortune(
            f"{stats.top_spending_category} at ${stats.top_spending_amount:.0f}/month... reduce 10% to save ${savings_10pct:.0f}",
            FortuneSentiment.NEUTRAL,
            f"${savings_10pct:.0f}"
        ))
    
    # Negative net fallback
    if stats.net_amount < 0:
        fortunes.append(Fortune(
            f"Spending exceeds income by ${abs(stats.net_amount):.0f}... time to review and rebalance",
            FortuneSentiment.WARNING
        ))
    
    # Positive savings fallback
    if stats.savings_rate > 10:
        fortunes.append(Fortune(
            f"Strong {stats.savings_rate:.0f}% savings rate... small optimizations could boost it further",
            FortuneSentiment.POSITIVE
        ))
    
    # Anomaly fallback
    if stats.anomaly_count >= 1:
        fortunes.append(Fortune(
            f"{stats.anomaly_count} unusual transactions detected... review them to avoid surprises",
            FortuneSentiment.WARNING
        ))
    
    # Generic fallbacks if nothing else applies
    if not fortunes:
        fortunes = [
            Fortune(
                f"Track your {stats.top_spending_category} spending of ${stats.top_spending_amount:.0f} for optimization",
                FortuneSentiment.NEUTRAL
            ),
            Fortune(
                "Review your recurring charges monthly... hidden savings await discovery",
                FortuneSentiment.NEUTRAL
            ),
        ]
    
    return fortunes


# =============================================================================
# OpenAI Prompt Templates - Improved for specificity
# =============================================================================

FORTUNE_SYSTEM_PROMPT = """You are a mystical financial advisor who speaks in fortune cookie style.

Your fortunes must:
1. DIRECTLY reference the user's actual data (specific categories, amounts, counts)
2. Give ONE clear, actionable insight
3. Be under 20 words
4. Sound mystical but make logical sense

STRUCTURE: [Observation] + [Actionable wisdom]

Good examples:
- "Dining consumes $450 monthly... cooking at home three times saves $180"
- "Five subscriptions detected. Cancel one unused service, save $120 yearly"
- "Gray charges drain $29 monthly in shadows... audit and reclaim"
- "Coffee spending rises 20%... your future $200 awaits in a travel mug"
- "3 anomalies lurk in your ledger... investigate before they multiply"

Bad examples (DON'T DO THIS):
- "The moon watches your spending" (no data reference)
- "Be wise with money" (too vague)
- "Dragons guard your treasure" (makes no sense)
- "Consider your choices carefully" (generic advice)

Always be specific, practical, and grounded in their actual numbers."""

FORTUNE_USER_TEMPLATE = """Generate ONE fortune cookie message (max 20 words) based on this specific data:

KEY INSIGHT TO ADDRESS:
{insight_focus}

FINANCIAL DATA:
- Top spending: {top_category} at ${top_amount:.0f}/month
- Active subscriptions: {subscription_count}
- Forgotten charges (gray): {gray_charge_count} costing ${gray_charge_total:.0f}/month
- Savings rate: {savings_rate:.0f}%
- Unusual transactions: {anomaly_count}
- Net amount: ${net:.0f} ({net_status})

INSTRUCTIONS:
Reference the KEY INSIGHT and give specific, actionable advice using the actual numbers.
Format: [What's happening] + [What to do about it]

Fortune:"""


class FortuneGenerator:
    """
    Generates data-driven financial fortunes using OpenAI.
    
    Falls back to dynamic fortunes if API unavailable.
    """
    
    def __init__(self):
        raw_key = os.getenv("OPENAI_API_KEY", "")
        self.api_key = raw_key.strip() if raw_key else None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = None
        
        if self.api_key and self.api_key.startswith("sk-"):
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"⚠️ Fortune generator: OpenAI init failed: {e}")
                self.client = None
    
    async def generate(self, stats: FinancialStats) -> Fortune:
        """
        Generate a fortune based on financial stats.
        
        Args:
            stats: Anonymized financial statistics.
            
        Returns:
            Fortune object with text, sentiment, and optional lucky number.
        """
        # Try OpenAI first
        if self.client:
            try:
                fortune = await self._generate_with_ai(stats)
                if fortune:
                    return fortune
            except Exception as e:
                print(f"⚠️ Fortune generation failed, using fallback: {e}")
        
        # Fallback to dynamic fortunes
        return self._get_fallback_fortune(stats)
    
    def _determine_primary_insight(self, stats: FinancialStats) -> str:
        """
        Determine THE most important insight to highlight in the fortune.
        This makes fortunes more focused and actionable.
        """
        # Priority 1: Gray charges (forgotten subscriptions)
        if stats.gray_charge_count >= 2:
            return f"You have {stats.gray_charge_count} forgotten subscriptions draining ${stats.gray_charge_total:.0f} monthly"
        
        # Priority 2: High anomaly count
        if stats.anomaly_count >= 2:
            return f"{stats.anomaly_count} unusual transactions detected that need review"
        
        # Priority 3: Negative net (spending more than earning)
        if stats.net_amount < 0:
            return f"Monthly spending exceeds income by ${abs(stats.net_amount):.0f}"
        
        # Priority 4: High subscription count
        if stats.subscription_count >= 5:
            return f"{stats.subscription_count} active subscriptions - some may be unnecessary"
        
        # Priority 5: Single gray charge
        if stats.gray_charge_count == 1:
            return f"1 forgotten charge of ${stats.gray_charge_total:.0f}/month hiding in shadows"
        
        # Priority 6: Top spending category is high
        if stats.total_income > 0 and stats.top_spending_amount > stats.total_income * 0.3:
            return f"{stats.top_spending_category} spending is ${stats.top_spending_amount:.0f}/month - opportunity to optimize"
        
        # Priority 7: Good savings rate (positive message)
        if stats.savings_rate > 15:
            return f"Strong {stats.savings_rate:.0f}% savings rate - small optimizations could boost it further"
        
        # Priority 8: Any anomalies
        if stats.anomaly_count == 1:
            return "1 unusual transaction detected - worth investigating"
        
        # Default: general spending observation
        return f"Top spending is {stats.top_spending_category} at ${stats.top_spending_amount:.0f}/month"

    def _validate_fortune(self, text: str, stats: FinancialStats) -> bool:
        """
        Validate that fortune makes sense and references actual data.
        Returns False if fortune seems generic or nonsensical.
        """
        text_lower = text.lower()
        
        # Check if fortune is too generic (bad signs)
        generic_phrases = [
            "be wise", "think carefully", "consider your", "remember to",
            "don't forget", "always", "never", "the path to", "journey of",
            "true wealth", "greatest treasure", "money cannot", "happiness"
        ]
        
        # If it starts with generic advice, it's probably bad
        for phrase in generic_phrases:
            if phrase in text_lower[:50]:  # Check first 50 chars
                return False
        
        # Fortune should reference at least ONE concrete element
        concrete_references = [
            stats.top_spending_category.lower() in text_lower,
            str(stats.subscription_count) in text,
            str(stats.gray_charge_count) in text,
            str(stats.anomaly_count) in text,
            "$" in text,  # References a dollar amount
            "%" in text,  # References a percentage
        ]
        
        # At least one concrete reference required
        if not any(concrete_references):
            return False
        
        # Fortune shouldn't be too short or too long
        word_count = len(text.split())
        if word_count < 6 or word_count > 25:
            return False
        
        return True

    async def _generate_with_ai(self, stats: FinancialStats) -> Optional[Fortune]:
        """Generate fortune using OpenAI with focused insights and validation."""
        
        # Determine the PRIMARY insight to focus on
        insight_focus = self._determine_primary_insight(stats)
        
        user_prompt = FORTUNE_USER_TEMPLATE.format(
            insight_focus=insight_focus,
            top_category=stats.top_spending_category,
            top_amount=stats.top_spending_amount,
            subscription_count=stats.subscription_count,
            gray_charge_count=stats.gray_charge_count,
            gray_charge_total=stats.gray_charge_total,
            savings_rate=stats.savings_rate,
            anomaly_count=stats.anomaly_count,
            net=stats.net_amount,
            net_status="saving" if stats.net_amount > 0 else "spending more than earning"
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FORTUNE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Reduced from 0.9 for more consistent results
            max_tokens=60,
            timeout=15
        )
        
        fortune_text = response.choices[0].message.content.strip()
        
        # Clean up
        fortune_text = fortune_text.strip('"\'').strip()
        # Remove "Fortune:" prefix if present
        if fortune_text.lower().startswith("fortune:"):
            fortune_text = fortune_text[8:].strip()
        
        # Validate fortune makes sense
        if not self._validate_fortune(fortune_text, stats):
            print(f"⚠️ Generated fortune failed validation: {fortune_text}")
            # Use fallback instead
            return self._get_fallback_fortune(stats)
        
        sentiment = self._determine_sentiment(stats, fortune_text)
        lucky_number = self._extract_lucky_number(fortune_text)
        
        return Fortune(
            text=fortune_text,
            sentiment=sentiment,
            lucky_number=lucky_number
        )
    
    def _determine_sentiment(self, stats: FinancialStats, text: str) -> FortuneSentiment:
        """Determine fortune sentiment based on financial health and text."""
        text_lower = text.lower()
        
        # Check for warning indicators in text
        warning_words = ["beware", "warning", "danger", "shadow", "lurk", "drain", 
                        "bleed", "exceeds", "overspend", "audit", "review", "unusual"]
        if any(word in text_lower for word in warning_words):
            return FortuneSentiment.WARNING
        
        # Check stats for warnings
        if stats.gray_charge_count >= 2:
            return FortuneSentiment.WARNING
        if stats.anomaly_count >= 2:
            return FortuneSentiment.WARNING
        if stats.net_amount < 0:
            return FortuneSentiment.WARNING
        
        # Positive indicators
        positive_words = ["save", "saving", "strong", "boost", "grow", "reclaim", "opportunity"]
        if any(word in text_lower for word in positive_words):
            return FortuneSentiment.POSITIVE
        
        if stats.savings_rate > 20:
            return FortuneSentiment.POSITIVE
        if stats.net_amount > stats.total_income * 0.1:
            return FortuneSentiment.POSITIVE
        
        return FortuneSentiment.NEUTRAL
    
    def _extract_lucky_number(self, text: str) -> Optional[str]:
        """Extract lucky number (dollar amount) from fortune text."""
        import re
        
        # Look for dollar amounts
        matches = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        if matches:
            # Return the largest amount as the "lucky number"
            amounts = []
            for m in matches:
                try:
                    amount = float(m.replace('$', '').replace(',', ''))
                    amounts.append((m, amount))
                except ValueError:
                    continue
            if amounts:
                # Return the one that looks most like a savings amount
                amounts.sort(key=lambda x: x[1], reverse=True)
                return amounts[0][0]
        
        return None
    
    def _get_fallback_fortune(self, stats: FinancialStats) -> Fortune:
        """Get an appropriate fallback fortune based on stats."""
        # Get dynamic fallbacks based on actual user data
        fallbacks = get_dynamic_fallback_fortunes(stats)
        
        # Determine appropriate sentiment based on stats
        if stats.gray_charge_count >= 2 or stats.anomaly_count >= 2 or stats.net_amount < 0:
            sentiment = FortuneSentiment.WARNING
        elif stats.savings_rate > 15 or stats.net_amount > 0:
            sentiment = FortuneSentiment.POSITIVE
        else:
            sentiment = FortuneSentiment.NEUTRAL
        
        # Try to find one matching the sentiment
        matching = [f for f in fallbacks if f.sentiment == sentiment]
        if matching:
            return random.choice(matching)
        
        # Otherwise return any fallback
        return random.choice(fallbacks) if fallbacks else Fortune(
            f"Review your {stats.top_spending_category} spending for savings opportunities",
            FortuneSentiment.NEUTRAL
        )


# =============================================================================
# Helper function for building stats from dashboard data
# =============================================================================

def build_financial_stats(
    summary: dict,
    anomaly_count: int,
    recurring_charges: list,
    insights: list
) -> FinancialStats:
    """
    Build FinancialStats from dashboard data.
    
    Args:
        summary: Spending summary with by_category dict.
        anomaly_count: Number of anomalies detected.
        recurring_charges: List of recurring charge dicts.
        insights: List of insight dicts.
        
    Returns:
        FinancialStats object.
    """
    # Find top spending category
    by_category = summary.get("by_category", {})
    
    # Filter to spending categories (negative amounts)
    spending_cats = {
        k: v for k, v in by_category.items() 
        if isinstance(v, dict) and v.get("amount", 0) < 0
    }
    
    if spending_cats:
        top_cat_name = max(spending_cats, key=lambda k: abs(spending_cats[k].get("amount", 0)))
        top_cat_data = spending_cats[top_cat_name]
        top_amount = abs(top_cat_data.get("amount", 0))
    else:
        top_cat_name = "General"
        top_amount = 0
    
    # Count subscriptions and gray charges
    subscription_count = len(recurring_charges)
    gray_charge_count = sum(1 for r in recurring_charges if r.get("is_gray_charge", False))
    
    # Calculate gray charge total
    gray_charge_total = sum(
        abs(r.get("amount", 0)) 
        for r in recurring_charges 
        if r.get("is_gray_charge", False)
    )
    
    # Calculate savings rate
    total_income = summary.get("total_income", 0)
    total_spending = abs(summary.get("total_spending", 0))
    net = summary.get("net", 0)
    
    if total_income > 0:
        savings_rate = (net / total_income) * 100
    else:
        savings_rate = 0
    
    # Determine trend (simplified - could be enhanced with actual trend data)
    if net > total_income * 0.1:
        trend = "stable"
    elif net > 0:
        trend = "slightly decreasing"
    else:
        trend = "increasing"  # spending is increasing
    
    return FinancialStats(
        top_spending_category=top_cat_name,
        top_spending_amount=top_amount,
        top_spending_percent_change=None,
        subscription_count=subscription_count,
        gray_charge_count=gray_charge_count,
        gray_charge_total=gray_charge_total,
        savings_rate=max(0, savings_rate),  # Don't show negative savings rate
        anomaly_count=anomaly_count,
        monthly_trend=trend,
        net_amount=net,
        total_spending=total_spending,
        total_income=total_income
    )
