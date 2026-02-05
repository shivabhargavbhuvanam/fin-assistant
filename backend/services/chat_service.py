"""
Module: chat_service.py
Description: Conversational AI service with function calling for financial insights.

This service provides:
    - Natural language understanding of financial queries
    - Function calling to access transaction data, anomalies, insights
    - Streaming responses for real-time chat experience
    - Context-aware conversations with memory

Author: Smart Financial Coach Team
Created: 2025-01-31

Usage:
    chat_service = ChatService(db, ai_service)
    async for chunk in chat_service.chat(session_id, "What are my biggest expenses?"):
        print(chunk)
"""

import os
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional
from sqlalchemy.orm import Session as DBSession

from models import (
    Session, Transaction, Category, Anomaly, 
    RecurringCharge, Insight, Conversation, Message
)


class ChatService:
    """
    Conversational AI service for financial coaching.
    
    Uses function calling to access financial data and provide
    contextual, personalized responses.
    """

    # System prompt for the financial coach - Dinera persona
    SYSTEM_PROMPT = """You are Dinera, an AI financial assistant designed ONLY to help users with:
- Understanding their financial data
- Explaining spending patterns
- Identifying anomalies and recurring expenses
- Providing personalized, practical financial insights

You have access to the user's financial data through function calls. Use these tools to provide specific, data-driven advice.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENT ENFORCEMENT (CRITICAL - CHECK EVERY MESSAGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before answering, check:
1. Is the user's question related to personal finance, spending, budgeting, or money behavior?
2. If YES → Answer normally using the guidelines below.
3. If NO → Refuse politely and redirect.

You MUST NOT:
- Answer questions unrelated to personal finance or money behavior
- Provide general knowledge, life advice, or technical help
- Engage in chit-chat, jokes, or off-topic conversations
- Discuss politics, health, relationships, or other non-financial topics

REFUSAL TEMPLATE (use when question is off-topic):
"I'm Dinera, designed to help only with financial insights and spending analysis.

For example, you can ask me:
• Why was my spending higher this month?
• Do I have any unnecessary subscriptions?
• Where can I realistically cut costs?
• What are my biggest expense categories?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STYLE (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING RULES - STRICTLY ENFORCED:
- NO markdown formatting (no **, no ##, no -, no •)
- NO bullet points or lists
- NO headers or sections
- Write in plain, flowing sentences only
- Use commas to separate items, not lists

BEFORE SENDING, VERIFY:
✓ Under 100 words for simple questions
✓ Uses at least 2 contractions (you're, don't, can't, won't, I'd, that's)
✓ Has specific dollar amounts from their data
✓ Sounds natural when read aloud
✓ Has exactly 1 clear action item
✓ Doesn't ask multiple questions
✓ Doesn't end with "let me know" or "feel free"
✓ Uses round numbers ($150, not $147.23)
✓ Starts naturally (never "Based on..." or "According to...")

IF RESPONSE IS TOO LONG:
- Cut to 1-2 key points only
- Combine sentences with commas
- Remove all ending questions

IF SOUNDS TOO FORMAL:
- Add contractions (you are → you're)
- Change "one should" → "you should"
- Remove "based on", "according to", "it appears"
- Start with "So", "Well", "Looks like", "Here's the thing"

IF TOO VAGUE:
- Add specific dollar amounts
- Name the exact categories
- State one clear action

IF TOO ROBOTIC:
- Vary sentence length
- Use conversational starters ("Nice!", "Okay so", "Here's what I see")
- Add light personality
- Remove formal corporate language

GOOD EXAMPLE:
"So your biggest expense is housing at $1,400 a month, followed by dining at around $470. You're spending about $150 weekly on eating out, which adds up. Try cooking at home two more nights a week and you could save close to $200 monthly."

BAD EXAMPLE:
"**Housing:** $1,388\n**Dining:** $469\n\nBased on your spending patterns, it appears that you may want to consider reducing your dining expenses. Let me know if you have any questions!"

Available Data:
- User's transaction history (categorized)
- Detected anomalies (unusual spending)
- Recurring charges and subscriptions
- AI-generated insights
- Spending summaries by category"""

    # Function definitions for the LLM
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_spending_summary",
                "description": "Get a summary of spending by category, including totals for income, expenses, and net savings.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_anomalies",
                "description": "Get detected anomalies - transactions that are unusually high or different from normal patterns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "all"],
                            "description": "Filter by severity level. Default is 'all'."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_recurring_charges",
                "description": "Get all detected recurring charges and subscriptions, including gray charges (small forgotten subscriptions).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_gray_charges_only": {
                            "type": "boolean",
                            "description": "If true, only return gray charges (small, possibly forgotten subscriptions)."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_insights",
                "description": "Get AI-generated financial insights and recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "insight_type": {
                            "type": "string",
                            "enum": ["spending", "anomaly", "subscription", "savings", "positive", "all"],
                            "description": "Filter insights by type. Default is 'all'."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_category_details",
                "description": "Get detailed spending information for a specific category.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category_name": {
                            "type": "string",
                            "description": "The category name to get details for (e.g., 'Dining', 'Groceries', 'Subscriptions')."
                        }
                    },
                    "required": ["category_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_transactions",
                "description": "Search transactions by description or merchant name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term to find in transaction descriptions."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return. Default is 10."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_months",
                "description": "Compare spending between the current and previous month.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

    def __init__(self, db: DBSession):
        """
        Initialize the chat service.
        
        Args:
            db: Database session for accessing financial data.
        """
        self.db = db
        self.client = None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        # Initialize OpenAI client if available
        raw_key = os.getenv("OPENAI_API_KEY", "")
        api_key = raw_key.strip() if raw_key else None  # Strip whitespace!
        
        if api_key and api_key.startswith("sk-"):
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=api_key)
                print(f"✅ ChatService: OpenAI client initialized")
            except Exception as e:
                print(f"⚠️ ChatService: Failed to initialize OpenAI: {e}")

    async def chat(
        self,
        session_id: str,
        user_message: str,
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat message and stream the response.
        
        Args:
            session_id: The financial data session ID.
            user_message: The user's message.
            conversation_id: Optional existing conversation ID.
            
        Yields:
            Chunks of the assistant's response.
        """
        # Get or create conversation
        conversation = self._get_or_create_conversation(session_id, conversation_id)
        
        # Save user message
        self._save_message(conversation.id, "user", user_message)
        
        # If no OpenAI client, use fallback
        if not self.client:
            fallback_response = await self._fallback_response(session_id, user_message)
            self._save_message(conversation.id, "assistant", fallback_response)
            yield fallback_response
            return
        
        # Build messages for the API
        messages = self._build_messages(conversation, user_message)
        
        try:
            # First call - may include tool calls
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
                stream=False  # First call non-streaming to handle tools
            )
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls if any
            if assistant_message.tool_calls:
                # Execute tool calls
                tool_results = await self._execute_tool_calls(
                    session_id, 
                    assistant_message.tool_calls
                )
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Add tool results
                for tool_call, result in zip(assistant_message.tool_calls, tool_results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                # Get final response with tool results (streaming)
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True
                )
                
                full_response = ""
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content
                
                # Save assistant response
                self._save_message(conversation.id, "assistant", full_response)
            
            else:
                # No tool calls, stream the response directly
                if assistant_message.content:
                    self._save_message(conversation.id, "assistant", assistant_message.content)
                    yield assistant_message.content
                else:
                    # Re-call with streaming
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True
                    )
                    
                    full_response = ""
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield content
                    
                    self._save_message(conversation.id, "assistant", full_response)
        
        except Exception as e:
            error_msg = f"I'm having trouble processing your request. Error: {str(e)}"
            self._save_message(conversation.id, "assistant", error_msg)
            yield error_msg

    async def _execute_tool_calls(self, session_id: str, tool_calls) -> list:
        """Execute tool calls and return results with proper error handling."""
        results = []
        
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            
            try:
                # Execute the appropriate function with error handling
                if func_name == "get_spending_summary":
                    result = self._get_spending_summary(session_id)
                elif func_name == "get_anomalies":
                    result = self._get_anomalies(session_id, args.get("severity", "all"))
                elif func_name == "get_recurring_charges":
                    result = self._get_recurring_charges(session_id, args.get("include_gray_charges_only", False))
                elif func_name == "get_insights":
                    result = self._get_insights(session_id, args.get("insight_type", "all"))
                elif func_name == "get_category_details":
                    result = self._get_category_details(session_id, args.get("category_name", ""))
                elif func_name == "search_transactions":
                    result = self._search_transactions(session_id, args.get("query", ""), args.get("limit", 10))
                elif func_name == "compare_months":
                    result = self._compare_months(session_id)
                else:
                    result = {"error": f"Unknown function: {func_name}"}
            except Exception as e:
                print(f"⚠️ Tool execution error ({func_name}): {e}")
                result = {"error": f"Failed to execute {func_name}: {str(e)}"}
            
            results.append(result)
        
        return results

    # ==========================================================================
    # Tool Implementation Functions
    # ==========================================================================

    def _get_spending_summary(self, session_id: str) -> dict:
        """Get spending summary by category."""
        transactions = self.db.query(Transaction).filter(
            Transaction.session_id == session_id
        ).all()
        
        categories = {c.id: c for c in self.db.query(Category).all()}
        
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_spending = sum(t.amount for t in transactions if t.amount < 0)
        
        by_category = {}
        for t in transactions:
            if t.category_id and t.category_id in categories:
                cat = categories[t.category_id]
                if cat.name not in by_category:
                    by_category[cat.name] = {"amount": 0, "count": 0}
                by_category[cat.name]["amount"] += t.amount
                by_category[cat.name]["count"] += 1
        
        return {
            "total_income": total_income,
            "total_spending": total_spending,
            "net": total_income + total_spending,
            "transaction_count": len(transactions),
            "by_category": by_category
        }

    def _get_anomalies(self, session_id: str, severity: str = "all") -> dict:
        """Get detected anomalies."""
        query = self.db.query(Anomaly).filter(Anomaly.session_id == session_id)
        
        if severity != "all":
            query = query.filter(Anomaly.severity == severity)
        
        anomalies = query.all()
        
        result = []
        for a in anomalies:
            txn = self.db.query(Transaction).filter(Transaction.id == a.transaction_id).first()
            if txn:
                result.append({
                    "description": txn.description,
                    "amount": a.actual_value,
                    "expected": a.expected_value,
                    "severity": a.severity,
                    "explanation": a.explanation,
                    "date": txn.date.isoformat() if txn.date else None
                })
        
        return {"anomalies": result, "count": len(result)}

    def _get_recurring_charges(self, session_id: str, gray_only: bool = False) -> dict:
        """Get recurring charges."""
        query = self.db.query(RecurringCharge).filter(
            RecurringCharge.session_id == session_id
        )
        
        if gray_only:
            query = query.filter(RecurringCharge.is_gray_charge == True)
        
        charges = query.all()
        
        result = []
        total_monthly = 0
        for r in charges:
            monthly_amount = abs(r.average_amount) if r.frequency_days >= 25 else abs(r.average_amount) * 4
            total_monthly += monthly_amount
            
            result.append({
                "description": r.description_pattern,
                "amount": r.average_amount,
                "frequency": "monthly" if r.frequency_days >= 25 else "weekly",
                "is_gray_charge": r.is_gray_charge,
                "occurrences": r.occurrence_count
            })
        
        return {
            "recurring_charges": result,
            "count": len(result),
            "total_monthly": total_monthly,
            "gray_charge_count": len([r for r in result if r["is_gray_charge"]])
        }

    def _get_insights(self, session_id: str, insight_type: str = "all") -> dict:
        """Get AI-generated insights."""
        query = self.db.query(Insight).filter(Insight.session_id == session_id)
        
        if insight_type != "all":
            query = query.filter(Insight.type == insight_type)
        
        insights = query.order_by(Insight.priority).all()
        
        result = [
            {
                "type": i.type,
                "title": i.title,
                "description": i.description,
                "action": i.action,
                "reasoning": i.reasoning,
                "priority": i.priority
            }
            for i in insights
        ]
        
        return {"insights": result, "count": len(result)}

    def _get_category_details(self, session_id: str, category_name: str) -> dict:
        """Get detailed spending for a category."""
        category = self.db.query(Category).filter(
            Category.name.ilike(f"%{category_name}%")
        ).first()
        
        if not category:
            return {"error": f"Category '{category_name}' not found"}
        
        transactions = self.db.query(Transaction).filter(
            Transaction.session_id == session_id,
            Transaction.category_id == category.id
        ).order_by(Transaction.date.desc()).limit(20).all()
        
        total = sum(t.amount for t in transactions)
        
        return {
            "category": category.name,
            "total_amount": total,
            "transaction_count": len(transactions),
            "recent_transactions": [
                {
                    "description": t.description,
                    "amount": t.amount,
                    "date": t.date.isoformat() if t.date else None
                }
                for t in transactions[:10]
            ]
        }

    def _search_transactions(self, session_id: str, query: str, limit: int = 10) -> dict:
        """Search transactions by description."""
        transactions = self.db.query(Transaction).filter(
            Transaction.session_id == session_id,
            Transaction.description.ilike(f"%{query}%")
        ).limit(limit).all()
        
        categories = {c.id: c.name for c in self.db.query(Category).all()}
        
        return {
            "query": query,
            "results": [
                {
                    "description": t.description,
                    "amount": t.amount,
                    "date": t.date.isoformat() if t.date else None,
                    "category": categories.get(t.category_id, "Unknown")
                }
                for t in transactions
            ],
            "count": len(transactions)
        }

    def _compare_months(self, session_id: str) -> dict:
        """Compare spending between months."""
        from models import Delta
        
        deltas = self.db.query(Delta).filter(
            Delta.session_id == session_id
        ).all()
        
        categories = {c.id: c.name for c in self.db.query(Category).all()}
        
        significant_changes = []
        for d in deltas:
            if d.change_percent and abs(d.change_percent) > 10:
                significant_changes.append({
                    "category": categories.get(d.category_id, "Unknown"),
                    "current_amount": d.current_amount,
                    "previous_amount": d.previous_amount,
                    "change_percent": d.change_percent,
                    "direction": "increase" if d.change_percent > 0 else "decrease"
                })
        
        return {
            "current_month": deltas[0].current_month if deltas else None,
            "previous_month": deltas[0].previous_month if deltas else None,
            "significant_changes": significant_changes
        }

    # ==========================================================================
    # Helper Functions
    # ==========================================================================

    def _get_or_create_conversation(
        self, 
        session_id: str, 
        conversation_id: Optional[str]
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        if conversation_id:
            conversation = self.db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                return conversation
        
        # Create new conversation
        conversation = Conversation(
            id=str(uuid.uuid4()),
            session_id=session_id
        )
        self.db.add(conversation)
        self.db.commit()
        
        return conversation

    def _save_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        tool_calls: Optional[list] = None
    ) -> Message:
        """Save a message to the conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls
        )
        self.db.add(message)
        self.db.commit()
        return message

    def _build_messages(self, conversation: Conversation, user_message: str) -> list:
        """Build the messages list for the API call."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add conversation history (last 10 messages for context)
        for msg in conversation.messages[-10:]:
            if msg.role in ["user", "assistant"]:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages

    async def _fallback_response(self, session_id: str, user_message: str) -> str:
        """Generate intelligent response without AI using soft intent matching."""
        user_lower = user_message.lower().strip()
        
        # Detect intent with soft matching
        intent = self._detect_intent(user_lower)
        
        if intent == 'spending_summary':
            return self._respond_spending_summary(session_id)
        
        elif intent == 'anomalies':
            return self._respond_anomalies(session_id)
        
        elif intent == 'recurring':
            return self._respond_recurring(session_id)
        
        elif intent == 'insights':
            return self._respond_insights(session_id)
        
        elif intent == 'savings':
            return self._respond_savings(session_id)
        
        elif intent == 'category':
            # Try to extract category name
            category = self._extract_category(user_lower)
            if category:
                return self._respond_category(session_id, category)
            return self._respond_spending_summary(session_id)
        
        elif intent == 'compare':
            return self._respond_compare(session_id)
        
        elif intent == 'search':
            # Try to extract search term
            term = self._extract_search_term(user_lower)
            if term:
                return self._respond_search(session_id, term)
            return self._respond_spending_summary(session_id)
        
        elif intent == 'greeting':
            return self._respond_greeting(session_id)
        
        elif intent == 'thanks':
            return "You're welcome! Let me know if you have any other questions about your finances."
        
        elif intent == 'yes_continue':
            # User said yes/sure/ok - give them insights
            return self._respond_insights(session_id)
        
        elif intent == 'no_stop':
            return "No problem! I'm here whenever you need help with your finances."
        
        else:
            # SOFT FALLBACK: Try to be helpful instead of showing menu
            return self._respond_smart_fallback(session_id, user_lower)

    def _detect_intent(self, text: str) -> str:
        """Detect user intent with soft matching and synonyms."""
        
        # Intent patterns (order matters - more specific first)
        intent_patterns = {
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                'good evening', 'howdy', 'sup', "what's up", 'yo'
            ],
            'thanks': [
                'thank', 'thanks', 'thx', 'ty', 'appreciate', 'helpful', 'great'
            ],
            'yes_continue': [
                'yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'go ahead', 
                'tell me', 'show me', 'please do', 'continue'
            ],
            'no_stop': [
                'no', 'nope', 'nah', 'stop', "don't", 'never mind', 'cancel'
            ],
            'anomalies': [
                'anomal', 'unusual', 'weird', 'strange', 'odd', 'suspicious',
                'different', 'unexpected', 'out of ordinary', 'flag', 'alert',
                'problem', 'issue', 'concern', 'wrong', 'mistake'
            ],
            'recurring': [
                'subscription', 'subscriptions', 'recurring', 'monthly charge',
                'gray charge', 'forgotten', 'auto', 'automatic', 'repeat',
                'netflix', 'spotify', 'membership', 'renew', 'billing'
            ],
            'savings': [
                'save', 'saving', 'savings', 'cut', 'reduce', 'less',
                'budget', 'money left', 'afford', 'goal', 'target',
                'how can i', 'tips', 'ways to', 'help me'
            ],
            'insights': [
                'insight', 'insights', 'advice', 'recommend', 'suggestion',
                'suggest', 'tip', 'tips', 'idea', 'ideas', 'analysis',
                'analyze', 'review', 'assessment', 'feedback', 'what should'
            ],
            'compare': [
                'compare', 'comparison', 'vs', 'versus', 'difference',
                'last month', 'previous', 'change', 'trend', 'over time',
                'month over month', 'better', 'worse'
            ],
            'category': [
                'dining', 'restaurant', 'food', 'eat', 'eating',
                'grocery', 'groceries', 'shopping', 'shop', 'bought',
                'entertainment', 'fun', 'coffee', 'starbucks', 'drinks',
                'transport', 'uber', 'lyft', 'gas', 'car', 'travel',
                'utilities', 'electric', 'water', 'internet', 'phone',
                'healthcare', 'medical', 'doctor', 'pharmacy', 'health',
                'housing', 'rent', 'mortgage', 'home'
            ],
            'search': [
                'find', 'search', 'look for', 'where', 'which', 'show me',
                'list', 'all my', 'transactions for', 'payments to'
            ],
            'spending_summary': [
                'spend', 'spent', 'spending', 'expense', 'expenses', 
                'total', 'summary', 'overview', 'breakdown', 'report',
                'how much', 'money', 'cost', 'paid', 'balance', 'income',
                'budget', 'finances', 'financial', 'status', 'this month'
            ],
        }
        
        # Check each intent pattern
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return intent
        
        # Check for question patterns that imply spending summary
        question_patterns = ['how', 'what', 'show', 'tell']
        money_patterns = ['much', 'spend', 'cost', 'money', 'total', 'budget']
        
        has_question = any(q in text for q in question_patterns)
        has_money = any(m in text for m in money_patterns)
        
        if has_question and has_money:
            return 'spending_summary'
        
        # Check for number/amount queries
        if any(c.isdigit() for c in text) or '$' in text:
            return 'search'
        
        return 'unknown'

    def _extract_category(self, text: str) -> str:
        """Extract category name from user message."""
        category_mappings = {
            'dining': ['dining', 'restaurant', 'restaurants', 'eat', 'eating out', 'takeout', 'delivery'],
            'groceries': ['grocery', 'groceries', 'supermarket', 'food shopping', 'whole foods', 'trader joe'],
            'shopping': ['shopping', 'shop', 'store', 'amazon', 'online shopping', 'retail'],
            'entertainment': ['entertainment', 'fun', 'movies', 'games', 'streaming', 'netflix', 'spotify'],
            'transportation': ['transport', 'transportation', 'uber', 'lyft', 'gas', 'car', 'fuel', 'parking'],
            'utilities': ['utilities', 'utility', 'electric', 'water', 'gas bill', 'internet', 'phone'],
            'healthcare': ['healthcare', 'health', 'medical', 'doctor', 'pharmacy', 'medicine'],
            'housing': ['housing', 'rent', 'mortgage', 'home'],
            'subscriptions': ['subscription', 'subscriptions', 'monthly'],
            'coffee': ['coffee', 'starbucks', 'cafe', 'latte', 'espresso'],
        }
        
        for category, keywords in category_mappings.items():
            for keyword in keywords:
                if keyword in text:
                    return category
        
        return None

    def _extract_search_term(self, text: str) -> str:
        """Extract search term from user message."""
        # Remove common words to find the search term
        remove_words = [
            'find', 'search', 'show', 'me', 'my', 'all', 'the', 'for',
            'look', 'transactions', 'payments', 'where', 'which', 'what',
            'list', 'to', 'from', 'at', 'in', 'on', 'a', 'an'
        ]
        
        words = text.split()
        search_words = [w for w in words if w not in remove_words and len(w) > 2]
        
        if search_words:
            return ' '.join(search_words[:2])  # Take first 2 meaningful words
        return None

    # =========================================================================
    # Response Methods
    # =========================================================================

    def _respond_spending_summary(self, session_id: str) -> str:
        """Generate spending summary response."""
        summary = self._get_spending_summary(session_id)
        
        income = summary['total_income']
        spending = abs(summary['total_spending'])
        net = summary['net']
        
        response = f"""Here's your financial summary:

💰 **Income:** ${income:,.2f}
💸 **Spending:** ${spending:,.2f}
{'✅' if net >= 0 else '⚠️'} **Net:** ${net:,.2f}

**Top spending categories:**
{self._format_top_categories(summary['by_category'])}

Would you like me to look at any specific category in detail?"""
        
        return response

    def _respond_anomalies(self, session_id: str) -> str:
        """Generate anomalies response."""
        anomalies = self._get_anomalies(session_id)
        
        if anomalies['count'] == 0:
            return "✅ Good news! I didn't detect any unusual transactions. Your spending patterns look consistent."
        
        response = f"⚠️ I found **{anomalies['count']} unusual transaction(s)**:\n\n"
        
        for a in anomalies['anomalies'][:5]:
            severity_icon = '🔴' if a.get('severity') == 'high' else '🟡'
            response += f"{severity_icon} **{a['description']}**: ${abs(a['amount']):,.2f}\n"
            response += f"   Expected: ${abs(a['expected']):,.2f} | {a.get('explanation', '')}\n\n"
        
        response += "Would you like me to explain any of these in more detail?"
        return response

    def _respond_recurring(self, session_id: str) -> str:
        """Generate recurring charges response."""
        charges = self._get_recurring_charges(session_id)
        gray = [c for c in charges['recurring_charges'] if c['is_gray_charge']]
        regular = [c for c in charges['recurring_charges'] if not c['is_gray_charge']]
        
        response = f"🔄 You have **{charges['count']} recurring charges** totaling **${charges['total_monthly']:,.2f}/month**.\n\n"
        
        if gray:
            response += f"⚠️ **{len(gray)} possible forgotten subscriptions:**\n"
            for c in gray[:5]:
                response += f"• {c['description']}: ${abs(c['amount']):,.2f}/mo\n"
            response += "\n"
        
        if regular:
            response += "**Regular subscriptions:**\n"
            for c in regular[:5]:
                response += f"• {c['description']}: ${abs(c['amount']):,.2f}/mo\n"
        
        return response

    def _respond_insights(self, session_id: str) -> str:
        """Generate insights response."""
        insights = self._get_insights(session_id)
        
        if insights['count'] == 0:
            # Fall back to spending summary with tips
            return self._respond_savings(session_id)
        
        response = "💡 **Here are my top recommendations:**\n\n"
        
        for i, insight in enumerate(insights['insights'][:3], 1):
            response += f"**{i}. {insight['title']}**\n"
            response += f"{insight['description']}\n"
            if insight.get('action'):
                response += f"→ {insight['action']}\n"
            response += "\n"
        
        return response

    def _respond_savings(self, session_id: str) -> str:
        """Generate savings advice response."""
        insights = self._get_insights(session_id)
        charges = self._get_recurring_charges(session_id, gray_only=True)
        summary = self._get_spending_summary(session_id)
        
        response = "💰 **Here's how you could save money:**\n\n"
        
        # Check gray charges
        if charges['count'] > 0:
            total = sum(abs(c['amount']) for c in charges['recurring_charges'])
            response += f"1. **Cancel forgotten subscriptions** - You have {charges['count']} small recurring charges totaling ${total:,.2f}/month (${total*12:,.2f}/year)\n\n"
        
        # Get savings insights
        savings_insights = [i for i in insights.get('insights', []) if 'save' in i.get('action', '').lower() or i.get('type') == 'savings']
        
        for i, insight in enumerate(savings_insights[:2], 2 if charges['count'] > 0 else 1):
            response += f"{i}. **{insight['title']}**\n   {insight.get('action', insight['description'])}\n\n"
        
        # Top spending categories
        by_cat = summary.get('by_category', {})
        discretionary = [(k, v) for k, v in by_cat.items() if not v.get('is_essential', False) and v['amount'] < 0]
        discretionary.sort(key=lambda x: x[1]['amount'])
        
        if discretionary:
            top = discretionary[0]
            response += f"Your biggest discretionary spending is **{top[0]}** at ${abs(top[1]['amount']):,.2f}. Reducing this by 20% would save ${abs(top[1]['amount']) * 0.2:,.2f}."
        
        return response

    def _respond_category(self, session_id: str, category: str) -> str:
        """Generate category-specific response."""
        details = self._get_category_details(session_id, category)
        
        if 'error' in details:
            return self._respond_spending_summary(session_id)
        
        response = f"📊 **{details['category']} Spending:**\n\n"
        response += f"**Total:** ${abs(details['total_amount']):,.2f}\n"
        response += f"**Transactions:** {details['transaction_count']}\n\n"
        
        if details.get('recent_transactions'):
            response += "**Recent transactions:**\n"
            for t in details['recent_transactions'][:5]:
                response += f"• {t['description']}: ${abs(t['amount']):,.2f} ({t['date']})\n"
        
        return response

    def _respond_compare(self, session_id: str) -> str:
        """Generate month comparison response."""
        comparison = self._compare_months(session_id)
        
        if not comparison.get('significant_changes'):
            return "I don't have enough data to compare months yet. Need at least 2 months of transactions."
        
        response = f"📈 **Spending Changes: {comparison.get('previous_month', 'Last Month')} → {comparison.get('current_month', 'This Month')}**\n\n"
        
        for change in comparison['significant_changes'][:5]:
            icon = '📈' if change['direction'] == 'increase' else '📉'
            response += f"{icon} **{change['category']}**: {abs(change['change_percent']):.0f}% {change['direction']}\n"
            response += f"   ${abs(change['previous_amount']):,.2f} → ${abs(change['current_amount']):,.2f}\n\n"
        
        return response

    def _respond_search(self, session_id: str, term: str) -> str:
        """Generate search results response."""
        results = self._search_transactions(session_id, term, limit=10)
        
        if results['count'] == 0:
            return f"I couldn't find any transactions matching '{term}'. Try a different search term?"
        
        response = f"🔍 **Found {results['count']} transaction(s) matching '{term}':**\n\n"
        
        for t in results['results'][:7]:
            response += f"• {t['description']}: ${abs(t['amount']):,.2f} ({t['date']})\n"
        
        if results['count'] > 7:
            response += f"\n... and {results['count'] - 7} more"
        
        return response

    def _respond_greeting(self, session_id: str) -> str:
        """Generate greeting response with context."""
        summary = self._get_spending_summary(session_id)
        anomalies = self._get_anomalies(session_id)
        
        response = "👋 Hi! I'm your financial coach. "
        
        # Add personalized context
        if anomalies['count'] > 0:
            response += f"I noticed {anomalies['count']} unusual transaction(s) you might want to review. "
        
        net = summary.get('net', 0)
        if net > 0:
            response += f"Good news - you're ${net:,.2f} in the positive this period!"
        elif net < 0:
            response += f"You're currently ${abs(net):,.2f} over budget. Want me to help you find savings?"
        
        response += "\n\nWhat would you like to know about your spending?"
        return response

    def _respond_smart_fallback(self, session_id: str, user_message: str) -> str:
        """Smart fallback that tries to be helpful rather than showing menu."""
        
        # Check if it looks like a follow-up question
        follow_up_words = ['more', 'detail', 'explain', 'elaborate', 'what about', 'and', 'also', 'other']
        if any(word in user_message for word in follow_up_words):
            # Give more insights
            return self._respond_insights(session_id)
        
        # Check if asking about specific amount
        if '$' in user_message or any(c.isdigit() for c in user_message):
            summary = self._get_spending_summary(session_id)
            return f"Looking at your finances: You've spent ${abs(summary['total_spending']):,.2f} total. Would you like a breakdown by category?"
        
        # Check message length - short messages often need context
        if len(user_message.split()) <= 3:
            # Very short - probably a follow-up, give insights
            return self._respond_insights(session_id)
        
        # Default: Try to give a helpful response based on their data
        summary = self._get_spending_summary(session_id)
        anomalies = self._get_anomalies(session_id)
        
        response = f"I'm not quite sure what you're asking, but here's what I can tell you:\n\n"
        response += f"• You've spent ${abs(summary['total_spending']):,.2f} this period\n"
        
        if anomalies['count'] > 0:
            response += f"• I found {anomalies['count']} unusual transaction(s) worth reviewing\n"
        
        response += "\nYou can ask me things like:\n"
        response += "• \"What are my biggest expenses?\"\n"
        response += "• \"How can I save money?\"\n"
        response += "• \"Show me my subscriptions\""
        
        return response

    def _format_top_categories(self, by_category: dict) -> str:
        """Format top spending categories."""
        sorted_cats = sorted(
            [(k, v) for k, v in by_category.items() if v['amount'] < 0],
            key=lambda x: x[1]['amount']
        )[:5]
        
        return "\n".join([
            f"• {name}: ${abs(data['amount']):,.2f} ({data['count']} transactions)"
            for name, data in sorted_cats
        ])

    def get_suggested_prompts(self, session_id: str) -> list[str]:
        """Get contextual suggested prompts based on the data."""
        prompts = [
            "How much did I spend this month?",
            "What are my biggest expenses?",
        ]
        
        # Check for anomalies
        anomalies = self._get_anomalies(session_id)
        if anomalies['count'] > 0:
            prompts.insert(0, f"Tell me about the {anomalies['count']} unusual transactions")
        
        # Check for gray charges
        charges = self._get_recurring_charges(session_id, gray_only=True)
        if charges['count'] > 0:
            prompts.append(f"What are these {charges['count']} forgotten subscriptions?")
        
        prompts.extend([
            "How can I save more money?",
            "Compare my spending to last month",
            "Show me my subscription costs"
        ])
        
        return prompts[:6]  # Return max 6 prompts
