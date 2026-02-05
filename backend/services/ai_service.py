"""
OpenAI wrapper with retry logic, rate limiting, and error handling.

Features:
    - Exponential backoff retry for transient failures
    - Rate limit handling (429 errors)
    - Token usage tracking
    - Graceful fallback when API unavailable

Author: Smart Financial Coach Team
"""

import os
import json
import asyncio
from typing import Optional
from functools import wraps
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Retry Decorator with Exponential Backoff
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_errors: tuple = None
):
    """
    Decorator for async functions that implements retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff calculation.
        retryable_errors: Tuple of exception types to retry on.
    """
    if retryable_errors is None:
        retryable_errors = (Exception,)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Check for rate limit errors (429)
                    if "rate_limit" in error_str or "429" in error_str:
                        # Use longer delay for rate limits
                        delay = min(delay * 2, max_delay)
                        print(
                            f"⏳ Rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                    elif attempt < max_retries:
                        print(
                            f"⚠️ API error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1}): {e}")

                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        raise last_exception

            raise last_exception
        return wrapper
    return decorator


class AIService:
    """
    Wrapper for OpenAI API with retry logic and rate limit handling.

    Features:
        - Automatic retry with exponential backoff
        - Rate limit (429) handling
        - Token usage tracking
        - Graceful fallback when API unavailable
    """

    # Rate limit settings
    MAX_RETRIES = 3
    INITIAL_DELAY = 1.0
    MAX_DELAY = 60.0

    def __init__(self):
        raw_key = os.getenv("OPENAI_API_KEY", "")
        self.api_key = raw_key.strip() if raw_key else None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = None

        # Token usage tracking
        self.total_tokens_used = 0
        self.request_count = 0

        # Only initialize client if API key is available and valid
        if self.api_key and self.api_key.startswith("sk-"):
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=self.api_key)
                print(f"✅ OpenAI client initialized (model: {self.model})")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            print("⚠️ OpenAI API key not configured. AI features will use fallback mode.")

    def _track_usage(self, response) -> None:
        """Track token usage from API response."""
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens_used += response.usage.total_tokens
            self.request_count += 1

    def get_usage_stats(self) -> dict:
        """Get current usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "request_count": self.request_count,
            "avg_tokens_per_request": (
                self.total_tokens_used / self.request_count
                if self.request_count > 0 else 0
            )
        }

    async def categorize_transactions(
        self, aggregated_patterns: list[dict], categories: list[str]
    ) -> list[dict]:
        """
        Categorize aggregated transaction patterns using function calling.

        Privacy-safe: Receives aggregated patterns (merchant_id, counts, hints)
        instead of raw transaction descriptions.
        """
        # If no client available, return empty (fallback to rules only)
        if not self.client:
            print("⚠️ AI categorization skipped - no API key")
            return []

        # Format aggregated patterns for AI (no raw merchant names)
        pattern_text = "\n".join(
            f"Pattern {p['merchant_id']}: {p['transaction_count']} transactions, "
            f"avg ${p['avg_amount']:.2f}, hints: {', '.join(p.get('category_hints', []))}"
            for p in aggregated_patterns
        )

        try:
            response = await self._call_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial transaction categorizer. Based on transaction patterns (count, average amount, category hints), categorize each pattern into the most appropriate category.",
                    },
                    {
                        "role": "user",
                        "content": f"Categorize these transaction patterns into one of these categories: {', '.join(categories)}\n\nPatterns:\n{pattern_text}",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "categorize_patterns",
                            "description": "Categorize aggregated transaction patterns",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "categorizations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "merchant_id": {"type": "string"},
                                                "category": {
                                                    "type": "string",
                                                    "enum": categories,
                                                },
                                                "confidence": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "maximum": 1,
                                                },
                                            },
                                            "required": ["merchant_id", "category", "confidence"],
                                        },
                                    }
                                },
                                "required": ["categorizations"],
                            },
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {
                    "name": "categorize_patterns"}},
                timeout=30,
            )

            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result.get("categorizations", [])

        except Exception as e:
            print(f"AI categorization error after retries: {e}")
            return []

    async def _call_with_retry(self, **kwargs) -> any:
        """
        Make OpenAI API call with retry logic for rate limits.

        Implements exponential backoff for:
        - 429 Rate Limit errors
        - 500/502/503 Server errors
        - Network timeouts
        """
        last_exception = None
        delay = self.INITIAL_DELAY

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    **kwargs
                )
                self._track_usage(response)
                return response

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if retryable error
                is_rate_limit = "rate_limit" in error_str or "429" in error_str
                is_server_error = any(code in error_str for code in [
                                      "500", "502", "503"])
                is_timeout = "timeout" in error_str

                if is_rate_limit or is_server_error or is_timeout:
                    if attempt < self.MAX_RETRIES:
                        wait_time = delay * (2 if is_rate_limit else 1)
                        print(
                            f"⏳ API error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.MAX_RETRIES + 1})")
                        await asyncio.sleep(wait_time)
                        delay = min(delay * 2, self.MAX_DELAY)
                        continue

                # Non-retryable error or max retries reached
                raise last_exception

        raise last_exception

    async def generate_insights(self, context: dict) -> list[dict]:
        """Generate financial insights using function calling."""
        # If no client available, use fallback insights
        if not self.client:
            print("⚠️ AI insights skipped - using fallback insights")
            return self._fallback_insights(context)

        system_prompt = """You are a friendly, knowledgeable financial coach. Your job is to
analyze spending data and provide actionable insights that help users improve their
financial health.

Guidelines:
- Be specific: use actual numbers from the data
- Be actionable: every insight should suggest a concrete step
- Be encouraging: celebrate wins, not just problems
- Be explainable: always explain WHY you're making a recommendation
- Prioritize: most impactful insights first (priority 1 = highest)

Generate 4-6 insights covering:
1. One spending pattern insight (largest category, trend)
2. One anomaly alert if any exist (unusual transactions)
3. One subscription/recurring review (especially gray charges)
4. One savings opportunity (concrete $ amount)
5. Optionally: a positive insight (good habit, improvement)"""

        try:
            response = await self._call_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Analyze this financial data and generate insights:\n{json.dumps(context, indent=2)}",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "generate_insights",
                            "description": "Generate personalized financial insights",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "insights": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": [
                                                        "spending",
                                                        "anomaly",
                                                        "subscription",
                                                        "savings",
                                                        "positive",
                                                    ],
                                                },
                                                "priority": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "maximum": 3,
                                                },
                                                "title": {
                                                    "type": "string",
                                                    "maxLength": 60,
                                                },
                                                "description": {
                                                    "type": "string",
                                                    "maxLength": 200,
                                                },
                                                "action": {
                                                    "type": "string",
                                                    "maxLength": 100,
                                                },
                                                "reasoning": {
                                                    "type": "string",
                                                    "maxLength": 150,
                                                },
                                                "confidence": {
                                                    "type": "number",
                                                    "minimum": 0,
                                                    "maximum": 1,
                                                },
                                                "data": {"type": "object"},
                                            },
                                            "required": [
                                                "type",
                                                "priority",
                                                "title",
                                                "description",
                                                "reasoning",
                                                "confidence",
                                            ],
                                        },
                                    }
                                },
                                "required": ["insights"],
                            },
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {
                    "name": "generate_insights"}},
                timeout=30,
            )

            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result.get("insights", [])

        except Exception as e:
            print(f"AI insight generation error after retries: {e}")
            return self._fallback_insights(context)

    async def generate_goal_advice(
        self, context: dict, target_amount: float, suggested_cuts: list[dict]
    ) -> str:
        """Generate personalized goal advice."""
        # If no client available, return fallback advice
        if not self.client:
            return "Focus on reducing non-essential spending first. Small changes add up over time!"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a supportive financial coach. Give brief, encouraging advice about achieving savings goals.",
                    },
                    {
                        "role": "user",
                        "content": f"""User wants to save ${target_amount:.2f}/month.
Current spending summary: {json.dumps(context, indent=2)}
Suggested cuts: {json.dumps(suggested_cuts, indent=2)}

Give 2-3 sentences of encouraging, practical advice.""",
                    },
                ],
                timeout=20,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"AI goal advice error: {e}")
            return "Focus on reducing non-essential spending first. Small changes add up over time!"

    def _fallback_insights(self, context: dict) -> list[dict]:
        """Generate basic insights without AI if API fails."""
        insights = []
        summary = context.get("spending_summary", {})

        # Basic spending insight
        if summary.get("by_category"):
            categories = summary["by_category"]
            # Filter to spending categories only (negative amounts)
            spending_cats = {k: v for k,
                             v in categories.items() if v.get("amount", 0) < 0}

            if spending_cats:
                largest = max(
                    spending_cats.items(), key=lambda x: abs(x[1].get("amount", 0))
                )
                insights.append(
                    {
                        "type": "spending",
                        "priority": 1,
                        "title": f"Largest spending: {largest[0]}",
                        "description": f"You spent ${abs(largest[1].get('amount', 0)):.2f} on {largest[0]} this period.",
                        "action": f"Review your {largest[0]} spending for potential savings.",
                        "reasoning": "This is your highest spending category.",
                        "confidence": 0.85,
                        "data": {"category": largest[0], "amount": largest[1].get("amount", 0)},
                    }
                )

        # Gray charges insight
        gray_total = context.get("gray_charges_total", 0)
        if gray_total != 0:
            insights.append(
                {
                    "type": "subscription",
                    "priority": 2,
                    "title": "Unknown recurring charges found",
                    "description": f"You have ${abs(gray_total):.2f}/month in small, unidentified recurring charges.",
                    "action": "Review these charges and cancel any you don't recognize.",
                    "reasoning": "Small recurring charges often go unnoticed but add up.",
                    "confidence": 0.80,
                    "data": {"amount": gray_total},
                }
            )

        # Anomalies insight
        anomalies = context.get("anomalies", [])
        if anomalies:
            high_anomalies = [
                a for a in anomalies if a.get("severity") == "high"]
            if high_anomalies:
                insights.append(
                    {
                        "type": "anomaly",
                        "priority": 1,
                        "title": f"{len(high_anomalies)} unusual transaction(s) detected",
                        "description": f"Found {len(high_anomalies)} transactions that are significantly higher than your typical spending.",
                        "action": "Review these transactions to ensure they were intentional.",
                        "reasoning": "These transactions deviate significantly from your normal spending patterns.",
                        "confidence": 0.90,
                        "data": {"count": len(high_anomalies)},
                    }
                )

        # Net savings insight
        total_income = summary.get("total_income", 0)
        total_spending = summary.get("total_spending", 0)
        net = summary.get("net", total_income + total_spending)

        if net > 0:
            savings_rate = (net / total_income *
                            100) if total_income > 0 else 0
            insights.append(
                {
                    "type": "positive",
                    "priority": 3,
                    "title": "Positive cash flow",
                    "description": f"You saved ${net:.2f} this period ({savings_rate:.0f}% savings rate).",
                    "action": "Keep up the good work!",
                    "reasoning": "Maintaining positive cash flow is key to building financial health.",
                    "confidence": 0.95,
                    "data": {"net": net, "savings_rate": savings_rate},
                }
            )
        elif net < 0:
            insights.append(
                {
                    "type": "savings",
                    "priority": 1,
                    "title": "Spending exceeds income",
                    "description": f"You spent ${abs(net):.2f} more than you earned this period.",
                    "action": "Review discretionary spending to get back on track.",
                    "reasoning": "Spending more than you earn can lead to debt accumulation.",
                    "confidence": 0.95,
                    "data": {"deficit": abs(net)},
                }
            )

        return insights

    async def check_connection(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.client:
            return False
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
