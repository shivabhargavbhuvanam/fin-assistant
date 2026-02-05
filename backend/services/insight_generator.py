"""AI-powered insight generation with explainability."""

import pandas as pd
from sqlalchemy.orm import Session as DBSession

from models import Transaction, Anomaly, RecurringCharge, Delta, Insight, Category
from .ai_service import AIService
from .privacy_layer import get_privacy_layer


class InsightGenerator:
    """Generate personalized, explainable financial insights."""

    def __init__(self, db: DBSession, ai_service: AIService):
        self.db = db
        self.ai_service = ai_service
        self.privacy_layer = get_privacy_layer()

    async def generate(self, session_id: str) -> int:
        """Generate insights for a session."""
        # Build privacy-safe context (aggregated only)
        context = self._build_context(session_id)

        # Generate insights using AI
        ai_insights = await self.ai_service.generate_insights(context)

        # Store insights
        insights_created = 0
        for insight_data in ai_insights:
            insight = Insight(
                session_id=session_id,
                type=insight_data.get("type", "spending"),
                priority=insight_data.get("priority", 2),
                title=insight_data.get("title", ""),
                description=insight_data.get("description", ""),
                action=insight_data.get("action"),
                reasoning=insight_data.get("reasoning", ""),
                confidence=insight_data.get("confidence", 0.8),
                data=insight_data.get("data"),
            )
            self.db.add(insight)
            insights_created += 1

        self.db.commit()
        return insights_created

    def _build_context(self, session_id: str) -> dict:
        """Build privacy-safe context for AI - aggregated data only."""
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .all()
        )

        categories = {c.id: c for c in self.db.query(Category).all()}

        # Calculate spending summary
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_spending = sum(t.amount for t in transactions if t.amount < 0)

        # By category breakdown
        by_category = {}
        for t in transactions:
            if t.category_id and t.category_id in categories:
                cat = categories[t.category_id]
                if cat.name not in by_category:
                    by_category[cat.name] = {
                        "amount": 0,
                        "count": 0,
                        "is_essential": cat.is_essential,
                    }
                by_category[cat.name]["amount"] += t.amount
                by_category[cat.name]["count"] += 1

        # Get anomalies summary
        anomalies = (
            self.db.query(Anomaly).filter(Anomaly.session_id == session_id).all()
        )
        anomaly_summary = []
        for a in anomalies:
            txn = (
                self.db.query(Transaction)
                .filter(Transaction.id == a.transaction_id)
                .first()
            )
            if txn and txn.category_id:
                cat = categories.get(txn.category_id)
                anomaly_summary.append(
                    {
                        "category": cat.name if cat else "Unknown",
                        "amount": a.actual_value,
                        "typical": a.expected_value,
                        "severity": a.severity,
                    }
                )

        # Get recurring charges (anonymized for AI)
        recurring = (
            self.db.query(RecurringCharge)
            .filter(RecurringCharge.session_id == session_id)
            .all()
        )
        recurring_summary = []
        gray_charges_total = 0
        for r in recurring:
            cat = categories.get(r.category_id)
            freq_str = "monthly" if r.frequency_days >= 25 else "weekly"
            # Privacy: Send category and amount, NOT merchant name
            recurring_summary.append(
                {
                    "category": cat.name if cat else "Subscription",
                    "amount": r.average_amount,
                    "frequency": freq_str,
                    "is_gray_charge": r.is_gray_charge,
                    # NO merchant name sent to AI
                }
            )
            if r.is_gray_charge:
                gray_charges_total += r.average_amount

        # Get deltas
        deltas = self.db.query(Delta).filter(Delta.session_id == session_id).all()
        delta_summary = []
        for d in deltas:
            cat = categories.get(d.category_id)
            if d.change_percent and abs(d.change_percent) > 10:
                delta_summary.append(
                    {
                        "category": cat.name if cat else "Unknown",
                        "change_percent": d.change_percent,
                        "direction": "increase" if d.change_percent > 0 else "decrease",
                        "current": d.current_amount,
                        "previous": d.previous_amount,
                    }
                )

        return {
            "spending_summary": {
                "total_spending": total_spending,
                "total_income": total_income,
                "net": total_income + total_spending,
                "by_category": by_category,
            },
            "anomalies": anomaly_summary,
            "recurring_charges": recurring_summary,
            "gray_charges_total": gray_charges_total,
            "deltas": delta_summary,
            "transaction_count": len(transactions),
        }

    def calculate_deltas(self, session_id: str) -> int:
        """Calculate month-over-month spending changes."""
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.category_id.isnot(None))
            .all()
        )

        if not transactions:
            return 0

        categories = {c.id: c for c in self.db.query(Category).all()}

        # Build DataFrame
        data = [
            {
                "date": t.date,
                "amount": t.amount,
                "category_id": t.category_id,
            }
            for t in transactions
        ]
        df = pd.DataFrame(data)
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

        months = sorted(df["month"].unique())
        if len(months) < 2:
            return 0

        deltas_created = 0
        current_month = months[-1]
        previous_month = months[-2]

        for category_id in df["category_id"].unique():
            current = df[(df["month"] == current_month) & (df["category_id"] == category_id)][
                "amount"
            ].sum()
            previous = df[(df["month"] == previous_month) & (df["category_id"] == category_id)][
                "amount"
            ].sum()

            if previous != 0:
                change_percent = ((current - previous) / abs(previous)) * 100
            else:
                change_percent = 100 if current != 0 else 0

            delta = Delta(
                session_id=session_id,
                category_id=category_id,
                current_month=str(current_month),
                previous_month=str(previous_month),
                current_amount=current,
                previous_amount=previous,
                change_amount=current - previous,
                change_percent=change_percent,
            )
            self.db.add(delta)
            deltas_created += 1

        self.db.commit()
        return deltas_created
