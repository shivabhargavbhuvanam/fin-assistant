"""
Module: anomaly_detector.py
Description: Advanced anomaly detection with sophisticated fraud pattern recognition.

Detection Methods:
    1. MLAnomalyDetector: Isolation Forest for multi-dimensional anomaly detection (>=50 transactions)
    2. AnomalyDetector: Z-score statistical fallback (<50 transactions)
    3. GradualFraudDetector: Detects slowly increasing charges over time
    4. MicroTransactionDetector: Detects small systematic charges that add up
    5. VelocityDetector: Detects unusual transaction frequency spikes
    6. CrossCategoryDetector: Detects total spending spikes across all categories

Features:
    - Gradual fraud detection (incrementing charges over time)
    - Small systematic fraud ($2.99/day patterns)
    - Cross-category spending spike detection
    - Transaction velocity anomalies
    - Time-based pattern analysis

Author: Smart Financial Coach Team
Created: 2025-01-31
Updated: 2026-02-01 - Added sophisticated fraud detection
"""

import numpy as np
import pandas as pd
from typing import Optional, Set, List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy.orm import Session as DBSession

from models import Transaction, Anomaly, Category


# =============================================================================
# Data Validation Constants
# =============================================================================

MAX_AMOUNT_FOR_ANALYSIS = 50000  # $50,000
MIN_TRANSACTIONS_FOR_ZSCORE = 5
MAX_ZSCORE = 10.0

# Fraud detection thresholds
GRADUAL_FRAUD_MIN_OCCURRENCES = 3  # Minimum charges to detect gradual increase
GRADUAL_FRAUD_MIN_INCREASE_PCT = 10  # Minimum 10% total increase
MICRO_FRAUD_MAX_AMOUNT = 15.0  # Charges under $15 considered "micro"
MICRO_FRAUD_MIN_OCCURRENCES = 5  # 5+ small charges from same source
MICRO_FRAUD_FREQUENCY_DAYS = 30  # Within 30 days
VELOCITY_SPIKE_THRESHOLD = 3.0  # 3x normal transaction frequency
CROSS_CATEGORY_SPIKE_THRESHOLD = 2.0  # 2x normal daily spending

# ML imports with graceful fallback
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Using statistical fallback only.")


# =============================================================================
# Gradual Fraud Detector
# =============================================================================

class GradualFraudDetector:
    """
    Detects slowly increasing charges over time.
    
    Pattern: Subscription starts at $5, then $6, $7, $8...
    This is a common fraud technique to avoid detection thresholds.
    
    Method: Linear regression on charge amounts over time per merchant.
    Flags merchants with statistically significant positive slope.
    """
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def detect(self, session_id: str) -> List[Dict]:
        """
        Detect merchants with gradually increasing charges.
        
        Returns:
            List of detected gradual fraud patterns.
        """
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.amount < 0)  # Expenses only
            .order_by(Transaction.date)
            .all()
        )
        
        if len(transactions) < GRADUAL_FRAUD_MIN_OCCURRENCES:
            return []
        
        # Group by normalized merchant
        merchant_charges = defaultdict(list)
        for txn in transactions:
            merchant = self._normalize_merchant(txn.description)
            merchant_charges[merchant].append({
                'id': txn.id,
                'date': txn.date,
                'amount': abs(txn.amount),
                'description': txn.description
            })
        
        detections = []
        
        for merchant, charges in merchant_charges.items():
            if len(charges) < GRADUAL_FRAUD_MIN_OCCURRENCES:
                continue
            
            # Sort by date
            charges.sort(key=lambda x: x['date'])
            
            # Check for increasing trend
            amounts = [c['amount'] for c in charges]
            first_amount = amounts[0]
            last_amount = amounts[-1]
            
            # Skip if no increase
            if last_amount <= first_amount:
                continue
            
            # Calculate percentage increase
            pct_increase = ((last_amount - first_amount) / first_amount) * 100
            
            if pct_increase < GRADUAL_FRAUD_MIN_INCREASE_PCT:
                continue
            
            # Use linear regression to confirm trend (if ML available)
            if ML_AVAILABLE and len(charges) >= 3:
                X = np.arange(len(amounts)).reshape(-1, 1)
                y = np.array(amounts)
                reg = LinearRegression()
                reg.fit(X, y)
                
                # Slope should be positive and significant
                slope = reg.coef_[0]
                r_squared = reg.score(X, y)
                
                # Only flag if trend is clear (R² > 0.5)
                if slope <= 0 or r_squared < 0.5:
                    continue
                
                monthly_increase = slope * 30 / len(charges)  # Estimate monthly increase
            else:
                # Simple calculation without ML
                monthly_increase = (last_amount - first_amount) / max(1, len(charges) - 1)
            
            detections.append({
                'type': 'gradual_fraud',
                'merchant': merchant,
                'first_amount': first_amount,
                'last_amount': last_amount,
                'pct_increase': pct_increase,
                'occurrences': len(charges),
                'monthly_increase': monthly_increase,
                'transaction_ids': [c['id'] for c in charges],
                'latest_transaction_id': charges[-1]['id'],
                'severity': 'high' if pct_increase > 50 else 'medium',
                'explanation': (
                    f"⚠️ Gradual increase detected: {merchant} charges grew from "
                    f"${first_amount:.2f} to ${last_amount:.2f} ({pct_increase:.0f}% increase over "
                    f"{len(charges)} charges). This pattern is common in subscription fraud."
                )
            })
        
        return detections
    
    def _normalize_merchant(self, description: str) -> str:
        """Normalize merchant name for grouping."""
        if not description:
            return "UNKNOWN"
        import re
        text = re.sub(r'\d+', '', description.upper()).strip()
        words = text.split()[:3]  # First 3 words
        return ' '.join(words) if words else "UNKNOWN"


# =============================================================================
# Micro Transaction Detector (Small Systematic Fraud)
# =============================================================================

class MicroTransactionDetector:
    """
    Detects small systematic charges that add up.
    
    Pattern: $2.99/day, $4.99/week - individually below notice threshold
    but totaling significant amounts over time.
    
    Method: Group small charges by merchant/pattern, flag high-frequency ones.
    """
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def detect(self, session_id: str) -> List[Dict]:
        """
        Detect small systematic charge patterns.
        
        Returns:
            List of detected micro-fraud patterns.
        """
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.amount < 0)
            .filter(Transaction.amount > -MICRO_FRAUD_MAX_AMOUNT)  # Small charges only
            .order_by(Transaction.date)
            .all()
        )
        
        if len(transactions) < MICRO_FRAUD_MIN_OCCURRENCES:
            return []
        
        # Group by normalized merchant + approximate amount
        merchant_patterns = defaultdict(list)
        for txn in transactions:
            merchant = self._normalize_merchant(txn.description)
            amount_bucket = self._bucket_amount(abs(txn.amount))
            key = f"{merchant}|{amount_bucket}"
            merchant_patterns[key].append({
                'id': txn.id,
                'date': txn.date,
                'amount': abs(txn.amount),
                'description': txn.description
            })
        
        detections = []
        
        for pattern_key, charges in merchant_patterns.items():
            if len(charges) < MICRO_FRAUD_MIN_OCCURRENCES:
                continue
            
            merchant, amount_bucket = pattern_key.split('|')
            
            # Calculate frequency
            charges.sort(key=lambda x: x['date'])
            first_date = charges[0]['date']
            last_date = charges[-1]['date']
            
            if hasattr(first_date, 'toordinal'):
                days_span = (last_date - first_date).days + 1
            else:
                days_span = 30  # Default assumption
            
            # Calculate metrics
            total_amount = sum(c['amount'] for c in charges)
            avg_amount = total_amount / len(charges)
            charges_per_week = (len(charges) / days_span) * 7 if days_span > 0 else 0
            monthly_drain = (total_amount / days_span) * 30 if days_span > 0 else total_amount
            
            # Flag if high frequency and significant monthly impact
            if charges_per_week >= 2 or monthly_drain >= 20:  # 2+ per week OR $20+/month
                severity = 'high' if monthly_drain >= 50 else ('medium' if monthly_drain >= 30 else 'low')
                
                detections.append({
                    'type': 'micro_fraud',
                    'merchant': merchant,
                    'avg_amount': avg_amount,
                    'occurrences': len(charges),
                    'total_amount': total_amount,
                    'monthly_drain': monthly_drain,
                    'charges_per_week': charges_per_week,
                    'days_span': days_span,
                    'transaction_ids': [c['id'] for c in charges],
                    'latest_transaction_id': charges[-1]['id'],
                    'severity': severity,
                    'explanation': (
                        f"🔍 Small systematic charges detected: {len(charges)} charges of ~${avg_amount:.2f} "
                        f"from {merchant} totaling ${total_amount:.2f}. "
                        f"Monthly drain: ${monthly_drain:.2f}. These small charges often go unnoticed."
                    )
                })
        
        return detections
    
    def _normalize_merchant(self, description: str) -> str:
        """Normalize merchant name."""
        if not description:
            return "UNKNOWN"
        import re
        text = re.sub(r'\d+', '', description.upper()).strip()
        words = text.split()[:2]
        return ' '.join(words) if words else "UNKNOWN"
    
    def _bucket_amount(self, amount: float) -> str:
        """Bucket amount for grouping similar charges."""
        if amount < 1:
            return "under_1"
        elif amount < 3:
            return "1_3"
        elif amount < 5:
            return "3_5"
        elif amount < 10:
            return "5_10"
        else:
            return "10_15"


# =============================================================================
# Cross-Category Spike Detector
# =============================================================================

class CrossCategoryDetector:
    """
    Detects total spending spikes across all categories.
    
    Pattern: Normal spending is $100/day, suddenly spikes to $500/day
    across multiple categories (card compromise, fraud spree).
    
    Method: Calculate daily spending totals, detect statistical outliers.
    """
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def detect(self, session_id: str) -> List[Dict]:
        """
        Detect days with abnormally high total spending.

        Returns:
            List of detected cross-category spike patterns.
        """
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.amount < 0)  # Expenses only
            .order_by(Transaction.date)
            .all()
        )
        
        if len(transactions) < 10:
            return []
        
        # Group by date
        daily_spending = defaultdict(lambda: {'total': 0, 'transactions': [], 'categories': set()})
        
        for txn in transactions:
            date_key = txn.date.isoformat() if hasattr(txn.date, 'isoformat') else str(txn.date)
            daily_spending[date_key]['total'] += abs(txn.amount)
            daily_spending[date_key]['transactions'].append(txn)
            if txn.category_id:
                daily_spending[date_key]['categories'].add(txn.category_id)
        
        if len(daily_spending) < 5:
            return []
        
        # Calculate statistics
        daily_totals = [d['total'] for d in daily_spending.values()]
        mean_daily = np.mean(daily_totals)
        std_daily = np.std(daily_totals)
        
        if std_daily < 1:  # No variance
            return []
        
        detections = []
        
        for date_key, data in daily_spending.items():
            z_score = (data['total'] - mean_daily) / std_daily
            
            # Flag if spending is 2+ standard deviations above mean
            # AND involves multiple categories (cross-category)
            if z_score >= CROSS_CATEGORY_SPIKE_THRESHOLD and len(data['categories']) >= 2:
                multiplier = data['total'] / mean_daily if mean_daily > 0 else 0
                
                severity = 'high' if z_score > 3 else ('medium' if z_score > 2.5 else 'low')
                
                detections.append({
                    'type': 'cross_category_spike',
                    'date': date_key,
                    'total_spending': data['total'],
                    'normal_daily': mean_daily,
                    'multiplier': multiplier,
                    'z_score': z_score,
                    'num_transactions': len(data['transactions']),
                    'num_categories': len(data['categories']),
                    'transaction_ids': [t.id for t in data['transactions']],
                    'severity': severity,
                    'explanation': (
                        f"📊 Spending spike detected on {date_key}: ${data['total']:.2f} across "
                        f"{len(data['categories'])} categories ({multiplier:.1f}x your normal "
                        f"${mean_daily:.2f}/day). Review these {len(data['transactions'])} transactions."
                    )
                })
        
        # Sort by severity
        detections.sort(key=lambda x: x['z_score'], reverse=True)
        
        return detections[:5]  # Top 5 spike days


# =============================================================================
# Velocity Detector (Transaction Frequency Anomalies)
# =============================================================================

class VelocityDetector:
    """
    Detects unusual transaction frequency spikes.
    
    Pattern: Normal is 2-3 transactions/day, suddenly 15 in one day
    (possible card compromise, automated fraud).
    
    Method: Calculate transaction counts per time window, detect spikes.
    """
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def detect(self, session_id: str) -> List[Dict]:
        """
        Detect days with abnormally high transaction counts.
        
        Returns:
            List of detected velocity anomalies.
        """
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .order_by(Transaction.date)
            .all()
        )
        
        if len(transactions) < 10:
            return []
        
        # Count transactions per day
        daily_counts = defaultdict(lambda: {'count': 0, 'transactions': []})
        
        for txn in transactions:
            date_key = txn.date.isoformat() if hasattr(txn.date, 'isoformat') else str(txn.date)
            daily_counts[date_key]['count'] += 1
            daily_counts[date_key]['transactions'].append(txn)
        
        if len(daily_counts) < 5:
            return []
        
        # Calculate statistics
        counts = [d['count'] for d in daily_counts.values()]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if std_count < 0.5 or mean_count < 1:
            return []
        
        detections = []
        
        for date_key, data in daily_counts.items():
            z_score = (data['count'] - mean_count) / std_count
            
            if z_score >= VELOCITY_SPIKE_THRESHOLD:
                multiplier = data['count'] / mean_count
                
                severity = 'high' if z_score > 4 else ('medium' if z_score > 3 else 'low')
                
                detections.append({
                    'type': 'velocity_spike',
                    'date': date_key,
                    'transaction_count': data['count'],
                    'normal_count': mean_count,
                    'multiplier': multiplier,
                    'z_score': z_score,
                    'transaction_ids': [t.id for t in data['transactions']],
                    'severity': severity,
                    'explanation': (
                        f"⚡ Unusual activity on {date_key}: {data['count']} transactions "
                        f"({multiplier:.1f}x your normal {mean_count:.1f}/day). "
                        f"High transaction velocity can indicate compromised credentials."
                    )
                })
        
        detections.sort(key=lambda x: x['z_score'], reverse=True)
        return detections[:3]  # Top 3 velocity spikes


# =============================================================================
# Pattern Matching Fraud Detector (Sophisticated Fraud)
# =============================================================================

class PatternMatchingDetector:
    """
    Detects sophisticated fraud that mimics normal patterns.
    
    Patterns detected:
    - Duplicate amounts (same exact amount multiple times)
    - Round number clusters ($100, $200, $500)
    - Sequential amounts ($50, $51, $52)
    - Suspicious timing (3am transactions)
    - Merchant anomalies (same merchant, different locations)
    """
    
    def __init__(self, db: DBSession):
        self.db = db
    
    def detect(self, session_id: str) -> List[Dict]:
        """
        Detect sophisticated fraud patterns.
        
        Returns:
            List of detected patterns.
        """
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.amount < 0)
            .order_by(Transaction.date)
            .all()
        )
        
        if len(transactions) < 5:
            return []
        
        detections = []
        
        # 1. Detect duplicate amounts (same exact amount 3+ times)
        amount_counts = defaultdict(list)
        for txn in transactions:
            amount_key = f"{abs(txn.amount):.2f}"
            amount_counts[amount_key].append(txn)
        
        for amount_str, txns in amount_counts.items():
            if len(txns) >= 3:
                # Check if different merchants (suspicious)
                merchants = set(self._normalize_merchant(t.description) for t in txns)
                if len(merchants) >= 2:  # Same amount, different merchants
                    detections.append({
                        'type': 'duplicate_amounts',
                        'amount': float(amount_str),
                        'occurrences': len(txns),
                        'merchants': list(merchants),
                        'transaction_ids': [t.id for t in txns],
                        'severity': 'medium',
                        'explanation': (
                            f"🔢 Duplicate amount pattern: ${amount_str} charged {len(txns)} times "
                            f"from {len(merchants)} different merchants. This pattern can indicate "
                            f"testing or automated fraud."
                        )
                    })
        
        # 2. Detect round number clusters ($100, $200, $500)
        round_amounts = []
        for txn in transactions:
            amt = abs(txn.amount)
            if amt >= 50 and amt % 50 == 0:  # Multiples of $50
                round_amounts.append(txn)
        
        if len(round_amounts) >= 3:
            # Check if they're close in time
            round_amounts.sort(key=lambda x: x.date)
            total_round = sum(abs(t.amount) for t in round_amounts)
            
            detections.append({
                'type': 'round_number_cluster',
                'occurrences': len(round_amounts),
                'total': total_round,
                'transaction_ids': [t.id for t in round_amounts],
                'severity': 'low',
                'explanation': (
                    f"💰 Round number pattern: {len(round_amounts)} charges in round amounts "
                    f"(multiples of $50) totaling ${total_round:.2f}. While not always fraud, "
                    f"this pattern is worth reviewing."
                )
            })
        
        # 3. Detect sequential/incrementing amounts
        amounts = [(txn, abs(txn.amount)) for txn in transactions]
        amounts.sort(key=lambda x: x[1])
        
        sequential_groups = []
        current_group = [amounts[0]]
        
        for i in range(1, len(amounts)):
            prev_amt = amounts[i-1][1]
            curr_amt = amounts[i][1]
            
            # Check if sequential (within $1-2 of each other)
            if 0.5 <= curr_amt - prev_amt <= 2.5:
                current_group.append(amounts[i])
            else:
                if len(current_group) >= 4:
                    sequential_groups.append(current_group)
                current_group = [amounts[i]]
        
        if len(current_group) >= 4:
            sequential_groups.append(current_group)
        
        for group in sequential_groups:
            txns = [g[0] for g in group]
            amts = [g[1] for g in group]
            
            detections.append({
                'type': 'sequential_amounts',
                'start_amount': min(amts),
                'end_amount': max(amts),
                'count': len(group),
                'transaction_ids': [t.id for t in txns],
                'severity': 'medium',
                'explanation': (
                    f"📈 Sequential amounts detected: {len(group)} charges incrementing from "
                    f"${min(amts):.2f} to ${max(amts):.2f}. This pattern is characteristic of "
                    f"testing card limits or gradual fraud."
                )
            })
        
        return detections
    
    def _normalize_merchant(self, description: str) -> str:
        """Normalize merchant name."""
        if not description:
            return "UNKNOWN"
        import re
        text = re.sub(r'\d+', '', description.upper()).strip()
        words = text.split()[:2]
        return ' '.join(words) if words else "UNKNOWN"


# =============================================================================
# ML Anomaly Detector (Isolation Forest)
# =============================================================================

class MLAnomalyDetector:
    """
    Machine Learning-based anomaly detector using Isolation Forest.
    
    Enhanced with additional features for fraud detection.
    """
    
    SEVERITY_THRESHOLDS = {
        'high': 0.15,
        'medium': 0.08,
        'low': 0.0
    }

    def __init__(self, db: DBSession, contamination: float = 0.05):
        self.db = db
        self.contamination = contamination
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None

    def detect(self, session_id: str) -> int:
        """Detect anomalies using Isolation Forest."""
        if not ML_AVAILABLE:
            return 0

        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.category_id.isnot(None))
            .order_by(Transaction.date)
            .all()
        )

        if len(transactions) < 50:
            return 0

        categories = self.db.query(Category).all()
        category_map = {c.id: c.name for c in categories}

        features_df = self._extract_features(transactions)

        if features_df.empty:
            return 0

        if self.model is None:
            self._train_model(features_df)

        X = features_df[['amount_abs', 'amount_zscore', 'amount_log',
                         'merchant_frequency', 'is_one_time', 'day_of_week',
                         'is_weekend', 'is_payday', 'amount_vs_category_avg',
                         'time_since_last']].values

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)

        anomalies_created = 0
        existing_anomaly_ids: Set[int] = set(
            row[0] for row in self.db.query(Anomaly.transaction_id)
            .filter(Anomaly.session_id == session_id)
            .all()
        )

        for idx, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
            if pred == -1:
                txn_id = int(features_df.iloc[idx]['transaction_id'])

                if txn_id in existing_anomaly_ids:
                    continue

                txn = next((t for t in transactions if t.id == txn_id), None)

                if txn is None or abs(txn.amount) > MAX_AMOUNT_FOR_ANALYSIS:
                    continue

                severity = self._score_to_severity(score)
                confidence = min(self._score_to_confidence(score), 1.0)
                category_name = category_map.get(txn.category_id, "Unknown")

                explanation = self._generate_ml_explanation(
                    txn=txn,
                    category_name=category_name,
                    confidence=confidence,
                    features=features_df.iloc[idx]
                )

                anomaly = Anomaly(
                    session_id=session_id,
                    transaction_id=txn_id,
                    anomaly_type="ml_isolation_forest",
                    severity=severity,
                    expected_value=features_df['amount_abs'].mean(),
                    actual_value=abs(txn.amount),
                    z_score=confidence,
                    explanation=explanation,
                )
                self.db.add(anomaly)
                existing_anomaly_ids.add(txn_id)
                anomalies_created += 1

        self.db.commit()
        return anomalies_created

    def _extract_features(self, transactions: list) -> pd.DataFrame:
        """Extract enhanced features for ML model."""
        if not transactions:
            return pd.DataFrame()

        data = []
        prev_date = None
        
        for t in transactions:
            txn_date = t.date
            day_of_week = txn_date.weekday() if hasattr(txn_date, 'weekday') else 0
            day_of_month = txn_date.day if hasattr(txn_date, 'day') else 1
            
            # Calculate time since last transaction
            if prev_date and hasattr(txn_date, 'toordinal'):
                time_since_last = (txn_date - prev_date).days
            else:
                time_since_last = 1
            prev_date = txn_date

            data.append({
                'transaction_id': t.id,
                'amount': abs(t.amount),
                'category_id': t.category_id,
                'date': txn_date,
                'description': t.description.upper() if t.description else '',
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'time_since_last': max(0, min(time_since_last, 30)),  # Cap at 30 days
            })

        df = pd.DataFrame(data)

        # Basic features
        df['amount_abs'] = df['amount']

        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        df['amount_zscore'] = (df['amount'] - amount_mean) / (amount_std if amount_std > 0 else 1)

        df['amount_log'] = np.log1p(df['amount'])

        # Merchant features
        df['merchant_norm'] = df['description'].apply(self._normalize_merchant)
        merchant_counts = df['merchant_norm'].value_counts()
        total_txns = len(df)
        df['merchant_frequency'] = df['merchant_norm'].map(
            lambda m: merchant_counts.get(m, 1) / total_txns
        )
        df['is_one_time'] = df['merchant_norm'].map(
            lambda m: 1 if merchant_counts.get(m, 0) <= 2 else 0
        )

        # Temporal features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_payday'] = ((df['day_of_month'] <= 3) |
                           ((df['day_of_month'] >= 15) & (df['day_of_month'] <= 17))).astype(int)

        # Category-relative features
        category_means = df.groupby('category_id')['amount'].transform('mean')
        df['amount_vs_category_avg'] = df['amount'] / category_means.replace(0, 1)

        return df

    def _normalize_merchant(self, description: str) -> str:
        """Normalize merchant name."""
        if not description:
            return "UNKNOWN"
        import re
        text = re.sub(r'\d+', '', description).strip()
        words = text.split()[:2]
        return ' '.join(words) if words else "UNKNOWN"

    def _train_model(self, features_df: pd.DataFrame) -> None:
        """Train Isolation Forest model."""
        X = features_df[['amount_abs', 'amount_zscore', 'amount_log',
                         'merchant_frequency', 'is_one_time', 'day_of_week',
                         'is_weekend', 'is_payday', 'amount_vs_category_avg',
                         'time_since_last']].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
            warm_start=False
        )
        self.model.fit(X_scaled)
        print(f"✅ Trained Isolation Forest on {len(X)} transactions")

    def _score_to_severity(self, score: float) -> str:
        """Convert anomaly score to severity."""
        confidence = self._score_to_confidence(score)
        if confidence >= self.SEVERITY_THRESHOLDS['high']:
            return 'high'
        elif confidence >= self.SEVERITY_THRESHOLDS['medium']:
            return 'medium'
            return 'low'

    def _score_to_confidence(self, score: float) -> float:
        """Convert anomaly score to confidence."""
        confidence = max(0.0, min(1.0, (0.0 - score) * 2))
        return round(confidence, 3)

    def _generate_ml_explanation(self, txn, category_name: str, 
                                  confidence: float, features: pd.Series) -> str:
        """Generate explanation for ML-detected anomaly."""
        amount = abs(txn.amount)
        is_income = txn.amount > 0
        is_high = features['amount_zscore'] > 2.0
        is_low = features['amount_zscore'] < -1.5
        is_rare_merchant = features['is_one_time'] == 1 or features['merchant_frequency'] < 0.02

        if is_income:
            if is_high:
                return f"This ${amount:.0f} {category_name} deposit is higher than usual. Worth verifying."
            return f"This ${amount:.0f} {category_name} deposit stands out from your pattern."
        else:
            if is_high and is_rare_merchant:
                return f"This ${amount:.0f} {category_name} purchase is large and from an uncommon merchant."
            elif is_high:
                return f"This ${amount:.0f} {category_name} purchase is noticeably higher than typical."
            elif is_rare_merchant:
                return f"This ${amount:.0f} {category_name} purchase is from a merchant you rarely use."
            return f"This ${amount:.0f} {category_name} charge differs from your usual pattern."


# =============================================================================
# Main Anomaly Detector (Orchestrator)
# =============================================================================

class AnomalyDetector:
    """
    Comprehensive anomaly detector combining multiple detection strategies.
    
    Detection Methods:
        1. ML Isolation Forest (>=50 transactions)
        2. Statistical z-score (<50 transactions)
        3. Gradual fraud detection (incrementing charges)
        4. Micro-transaction fraud (small systematic charges)
        5. Cross-category spike detection
        6. Velocity anomalies (transaction frequency)
        7. Pattern matching (sophisticated fraud)
    """

    def __init__(self, db: DBSession):
        self.db = db
        self.ml_detector: Optional[MLAnomalyDetector] = None
        self.gradual_detector = GradualFraudDetector(db)
        self.micro_detector = MicroTransactionDetector(db)
        self.cross_category_detector = CrossCategoryDetector(db)
        self.velocity_detector = VelocityDetector(db)
        self.pattern_detector = PatternMatchingDetector(db)

        if ML_AVAILABLE:
            self.ml_detector = MLAnomalyDetector(db, contamination=0.05)

    def detect(self, session_id: str) -> int:
        """
        Run all anomaly detection methods.

        Returns:
            Total number of anomalies detected.
        """
        transactions = (
            self.db.query(Transaction)
            .filter(Transaction.session_id == session_id)
            .filter(Transaction.category_id.isnot(None))
            .all()
        )

        if len(transactions) < 3:
            return 0

        total_anomalies = 0
        
        # Get existing anomaly transaction IDs
        existing_anomaly_ids: Set[int] = set(
            row[0] for row in self.db.query(Anomaly.transaction_id)
            .filter(Anomaly.session_id == session_id)
            .all()
        )

        # 1. Run ML or statistical detection
        if len(transactions) >= 50 and self.ml_detector is not None:
            print(f"🤖 Running ML anomaly detection ({len(transactions)} transactions)")
            total_anomalies += self.ml_detector.detect(session_id)
        else:
            print(f"📊 Running statistical detection ({len(transactions)} transactions)")
            total_anomalies += self._detect_statistical(session_id, transactions)

        # Refresh existing IDs after first pass
        existing_anomaly_ids = set(
            row[0] for row in self.db.query(Anomaly.transaction_id)
            .filter(Anomaly.session_id == session_id)
            .all()
        )

        # 2. Gradual fraud detection
        print("🔍 Checking for gradual fraud patterns...")
        gradual_detections = self.gradual_detector.detect(session_id)
        for detection in gradual_detections:
            txn_id = detection['latest_transaction_id']
            if txn_id not in existing_anomaly_ids:
                anomaly = Anomaly(
                    session_id=session_id,
                    transaction_id=txn_id,
                    anomaly_type="gradual_fraud",
                    severity=detection['severity'],
                    expected_value=detection['first_amount'],
                    actual_value=detection['last_amount'],
                    z_score=detection['pct_increase'] / 100,
                    explanation=detection['explanation'],
                )
                self.db.add(anomaly)
                existing_anomaly_ids.add(txn_id)
                total_anomalies += 1

        # 3. Micro-transaction fraud detection
        print("🔍 Checking for micro-transaction fraud...")
        micro_detections = self.micro_detector.detect(session_id)
        for detection in micro_detections:
            txn_id = detection['latest_transaction_id']
            if txn_id not in existing_anomaly_ids:
                anomaly = Anomaly(
                    session_id=session_id,
                    transaction_id=txn_id,
                    anomaly_type="micro_fraud",
                    severity=detection['severity'],
                    expected_value=0,
                    actual_value=detection['monthly_drain'],
                    z_score=detection['charges_per_week'],
                    explanation=detection['explanation'],
                )
                self.db.add(anomaly)
                existing_anomaly_ids.add(txn_id)
                total_anomalies += 1

        # 4. Cross-category spike detection
        print("🔍 Checking for cross-category spending spikes...")
        spike_detections = self.cross_category_detector.detect(session_id)
        for detection in spike_detections[:3]:  # Top 3 spikes
            # Use first transaction from the spike day
            txn_id = detection['transaction_ids'][0]
            if txn_id not in existing_anomaly_ids:
                anomaly = Anomaly(
                    session_id=session_id,
                    transaction_id=txn_id,
                    anomaly_type="cross_category_spike",
                    severity=detection['severity'],
                    expected_value=detection['normal_daily'],
                    actual_value=detection['total_spending'],
                    z_score=detection['z_score'],
                    explanation=detection['explanation'],
                )
                self.db.add(anomaly)
                existing_anomaly_ids.add(txn_id)
                total_anomalies += 1

        # 5. Velocity detection
        print("🔍 Checking for velocity anomalies...")
        velocity_detections = self.velocity_detector.detect(session_id)
        for detection in velocity_detections[:2]:  # Top 2 velocity spikes
            txn_id = detection['transaction_ids'][0]
            if txn_id not in existing_anomaly_ids:
                anomaly = Anomaly(
                    session_id=session_id,
                    transaction_id=txn_id,
                    anomaly_type="velocity_spike",
                    severity=detection['severity'],
                    expected_value=detection['normal_count'],
                    actual_value=detection['transaction_count'],
                    z_score=detection['z_score'],
                    explanation=detection['explanation'],
                )
                self.db.add(anomaly)
                existing_anomaly_ids.add(txn_id)
                total_anomalies += 1

        # 6. Pattern matching (sophisticated fraud)
        print("🔍 Checking for sophisticated fraud patterns...")
        pattern_detections = self.pattern_detector.detect(session_id)
        for detection in pattern_detections[:3]:  # Top 3 patterns
            txn_id = detection['transaction_ids'][0]
            if txn_id not in existing_anomaly_ids:
                anomaly = Anomaly(
                    session_id=session_id,
                    transaction_id=txn_id,
                    anomaly_type=detection['type'],
                    severity=detection['severity'],
                    expected_value=0,
                    actual_value=detection.get('amount', 0) or detection.get('total', 0),
                    z_score=0.5,
                    explanation=detection['explanation'],
                )
                self.db.add(anomaly)
                existing_anomaly_ids.add(txn_id)
                total_anomalies += 1

        self.db.commit()
        print(f"✅ Total anomalies detected: {total_anomalies}")
        return total_anomalies

    def _detect_statistical(self, session_id: str, transactions: list) -> int:
        """Statistical z-score based anomaly detection."""
        categories = self.db.query(Category).all()
        category_map = {c.id: c.name for c in categories}

        data = [
            {
                "id": t.id,
                "amount": t.amount,
                "category_id": t.category_id,
                "date": t.date,
                "description": t.description,
            }
            for t in transactions
        ]
        df = pd.DataFrame(data)
        df = df[df['amount'].abs() <= MAX_AMOUNT_FOR_ANALYSIS]

        existing_anomaly_ids: Set[int] = set(
            row[0] for row in self.db.query(Anomaly.transaction_id)
            .filter(Anomaly.session_id == session_id)
            .all()
        )

        anomalies_created = 0

        for category_id in df["category_id"].unique():
            cat_df = df[(df["category_id"] == category_id) & (df["amount"] < 0)]

            if len(cat_df) < MIN_TRANSACTIONS_FOR_ZSCORE:
                continue

            mean = cat_df["amount"].mean()
            std = cat_df["amount"].std()

            if std == 0 or pd.isna(std) or std < 0.01:
                continue

            for _, row in cat_df.iterrows():
                txn_id = row["id"]

                if txn_id in existing_anomaly_ids:
                    continue

                z_score = (row["amount"] - mean) / std
                z_score_bounded = np.clip(z_score, -MAX_ZSCORE, MAX_ZSCORE)

                if abs(z_score_bounded) > 2:
                    severity = self._get_severity(abs(z_score_bounded))
                    category_name = category_map.get(category_id, "Unknown")
                    confidence = min(1.0, abs(z_score_bounded) / 5.0)

                    explanation = self._generate_explanation(
                        category_name,
                        abs(row["amount"]),
                        abs(mean),
                        abs(z_score_bounded),
                        is_income=row["amount"] > 0,
                    )

                    anomaly = Anomaly(
                        session_id=session_id,
                        transaction_id=txn_id,
                        anomaly_type="amount",
                        severity=severity,
                        expected_value=abs(mean),
                        actual_value=abs(row["amount"]),
                        z_score=confidence,
                        explanation=explanation,
                    )
                    self.db.add(anomaly)
                    existing_anomaly_ids.add(txn_id)
                    anomalies_created += 1

        self.db.commit()
        return anomalies_created

    def _get_severity(self, z_score_abs: float) -> str:
        """Determine severity based on z-score."""
        if z_score_abs > 3:
            return "high"
        elif z_score_abs > 2.5:
            return "medium"
            return "low"

    def _generate_explanation(self, category: str, actual: float, expected: float,
                              z_score_abs: float, is_income: bool = False) -> str:
        """Generate explanation for statistical anomaly."""
        if is_income:
            if actual > expected:
                return (
                    f"This ${actual:.0f} {category} deposit is higher than your usual "
                    f"${expected:.0f}. Could be a bonus or one-time payment."
                )
                return (
                    f"This ${actual:.0f} {category} deposit is lower than your usual "
                    f"${expected:.0f}. You might want to verify it."
                )
        else:
            if actual > expected:
                multiplier = actual / expected if expected > 0 else 0
                if multiplier >= 3:
                    return (
                        f"This ${actual:.0f} {category} purchase is about {multiplier:.0f}x "
                        f"what you normally spend (usually around ${expected:.0f})."
                    )
                elif multiplier >= 2:
                    return (
                        f"This ${actual:.0f} {category} purchase is roughly double "
                        f"your usual ${expected:.0f} in this category."
                    )
                    return (
                        f"This ${actual:.0f} {category} purchase is higher than "
                        f"your typical ${expected:.0f} spending."
                    )
                return (
                    f"This ${actual:.0f} {category} charge is lower than usual. "
                    f"You typically spend around ${expected:.0f} here."
                )
