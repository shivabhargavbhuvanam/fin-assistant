"""Hybrid rule-based + AI transaction categorization."""

import re
from sqlalchemy.orm import Session as DBSession

from models import Transaction, Category
from .ai_service import AIService
from .privacy_layer import get_privacy_layer


class Categorizer:
    """Hybrid rule-based + AI categorization with confidence scores."""

    # Rule-based patterns (covers ~70% of transactions)
    KEYWORD_RULES = {
        r"NETFLIX|SPOTIFY|HULU|DISNEY\+|HBO|YOUTUBE|APPLE\s*(TV|MUSIC)|AMAZON\s*PRIME": "Subscriptions",
        r"UBER|LYFT|SHELL|CHEVRON|EXXON|BP|MOBIL|TEXACO|GAS|FUEL|PARKING|METRO|TRANSIT": "Transportation",
        r"AMAZON(?!\s*PRIME)|TARGET|WALMART|COSTCO|BEST\s*BUY|IKEA|HOME\s*DEPOT|LOWES": "Shopping",
        r"STARBUCKS|MCDONALD|CHIPOTLE|DOORDASH|GRUBHUB|UBER\s*EATS|RESTAURANT|PIZZA|BURGER|CAFE|COFFEE|DINER": "Dining",
        r"WHOLE\s*FOODS|TRADER\s*JOE|SAFEWAY|KROGER|GROCERY|PUBLIX|ALDI|WEGMANS|MARKET|FRESH": "Groceries",
        r"RENT|MORTGAGE|PROPERTY|LANDLORD|APARTMENT|HOUSING": "Housing",
        r"ELECTRIC|GAS\s*BILL|WATER\s*BILL|INTERNET|COMCAST|VERIZON|AT&T|T-MOBILE|UTILITY|PG&E|SEWAGE": "Utilities",
        r"CVS|WALGREENS|PHARMACY|DOCTOR|HOSPITAL|MEDICAL|HEALTH|DENTAL|VISION|CLINIC": "Healthcare",
        r"PAYROLL|DIRECT\s*DEPOSIT|SALARY|WAGE|EMPLOYER|INCOME|DIVIDEND|INTEREST\s*PAYMENT": "Income",
        r"TRANSFER|VENMO|ZELLE|CASHAPP|PAYPAL|WIRE": "Transfer",
        r"MOVIE|CINEMA|THEATER|CONCERT|TICKET|GAME|SPORT|GYM|FITNESS|CLUB": "Entertainment",
    }

    def __init__(self, db: DBSession, ai_service: AIService):
        self.db = db
        self.ai_service = ai_service
        self.privacy_layer = get_privacy_layer()
        self._category_map = None

    @property
    def category_map(self) -> dict[str, Category]:
        """Lazy-load category mapping."""
        if self._category_map is None:
            categories = self.db.query(Category).all()
            self._category_map = {c.name: c for c in categories}
        return self._category_map

    async def categorize_all(self, session_id: str) -> int:
        """Categorize all transactions for a session."""
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.category_id.is_(None))
            .all()
        )

        if not transactions:
            return 0

        categorized_count = 0
        uncategorized = []

        # First pass: rule-based categorization
        for txn in transactions:
            category_name, confidence = self._try_rules(txn.description)
            if category_name and category_name in self.category_map:
                txn.category_id = self.category_map[category_name].id
                txn.category_confidence = confidence
                txn.category_source = "rule"
                categorized_count += 1
            else:
                uncategorized.append(txn)

        # Second pass: AI categorization for unknowns
        if uncategorized:
            ai_results = await self._categorize_with_ai(uncategorized)
            for txn, category_name, confidence in ai_results:
                if category_name in self.category_map:
                    txn.category_id = self.category_map[category_name].id
                    txn.category_confidence = confidence
                    txn.category_source = "ai"
                    categorized_count += 1
                else:
                    # Fallback to "Other"
                    txn.category_id = self.category_map["Other"].id
                    txn.category_confidence = 0.5
                    txn.category_source = "fallback"
                    categorized_count += 1

        self.db.commit()
        return categorized_count

    def _try_rules(self, description: str) -> tuple[str | None, float]:
        """Try to categorize using keyword rules."""
        desc_upper = description.upper()

        for pattern, category in self.KEYWORD_RULES.items():
            if re.search(pattern, desc_upper):
                return category, 0.95

        return None, 0.0

    async def _categorize_with_ai(
        self, transactions: list[Transaction]
    ) -> list[tuple[Transaction, str, float]]:
        """Batch AI categorization for unknown transactions using privacy-safe data."""
        category_names = list(self.category_map.keys())

        # Use privacy layer to aggregate transactions (no raw merchant names sent)
        aggregated = self.privacy_layer.aggregate_for_categorization(transactions)
        
        if not aggregated:
            return [(txn, "Other", 0.5) for txn in transactions]
        
        # Call AI with aggregated patterns instead of raw descriptions
        results = await self.ai_service.categorize_transactions(aggregated, category_names)

        # Build mapping from merchant_id to category
        category_by_merchant = {}
        for r in results:
            if 'merchant_id' in r:
                category_by_merchant[r['merchant_id']] = (r.get('category', 'Other'), r.get('confidence', 0.7))
        
        # Apply categories back to original transactions
        categorized = []
        for txn in transactions:
            merchant_id = self.privacy_layer._hash_merchant(
                self.privacy_layer._normalize_merchant(txn.description)
            )
            if merchant_id in category_by_merchant:
                cat_name, confidence = category_by_merchant[merchant_id]
                categorized.append((txn, cat_name, confidence))
            else:
                # Fallback if no result
                categorized.append((txn, "Other", 0.5))

        return categorized
