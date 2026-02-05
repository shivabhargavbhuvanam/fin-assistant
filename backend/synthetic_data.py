"""
Module: synthetic_data.py
Description: Enhanced Synthetic Transaction Data Generator for ML training and demo purposes.

Generates realistic 6-month transaction data with:
    - Regular patterns (rent, subscriptions, utilities)
    - Multi-severity anomalies (high/medium/low)
    - Gray charges (small unknown recurring charges)
    - Temporal patterns (weekend spikes, payday spending)
    - Merchant frequency variance (habit/regular/one-time)
    - Fixed-amount subscriptions for ML pattern learning

ML Training Features:
    - 6 months of data (~480 transactions)
    - 8-10 anomalies with severity distribution
    - Weekend spending 40% higher
    - Payday spending spikes (days 1-3, 15-17)
    - Merchant frequency patterns

Author: Smart Financial Coach Team
Created: 2025-01-31
Last Modified: 2025-01-31

Usage:
    from synthetic_data import generate_synthetic_transactions
    transactions = generate_synthetic_transactions()
"""

import random
import csv
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple
from collections import defaultdict


class SyntheticDataGenerator:
    """
    Enhanced synthetic transaction data generator for ML training.
    
    Creates realistic financial data with patterns that ML models
    (Isolation Forest, Logistic Regression) can learn from.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            seed: Random seed for reproducibility.
        """
        random.seed(seed)
        # Dynamic date range: 6 months ending yesterday
        today = datetime.now()
        self.end_date = today - timedelta(days=1)
        self.start_date = self.end_date - timedelta(days=180)  # ~6 months
        
        # Merchant frequency tracking for patterns
        self.merchant_visits: Dict[str, int] = defaultdict(int)
    
    def generate(self, months: int = 6, txns_per_month: int = 80) -> List[Dict]:
        """
        Generate synthetic transactions with ML-friendly patterns.
        
        Args:
            months: Number of months of data to generate (default: 6).
            txns_per_month: Approximate transactions per month (default: 80).
        
        Returns:
            List of transaction dictionaries with date (date object), description, amount.
        """
        transactions = []
        
        # Regular patterns (recurring charges, fixed expenses)
        patterns = self._get_recurring_patterns()
        
        # Define merchant categories with frequency tiers
        habit_merchants = self._get_habit_merchants()      # 8+ visits/month
        regular_merchants = self._get_regular_merchants()  # 3-7 visits/month
        onetime_merchants = self._get_onetime_merchants()  # 1-2 visits total
        
        for month in range(months):
            month_start = self.start_date + timedelta(days=30 * month)
            month_end = month_start + timedelta(days=30)
            
            # 1. Add recurring transactions (EXACT amounts for subscriptions)
            for pattern in patterns:
                txn_date = month_start + timedelta(days=pattern['day'] - 1)
                if month_start <= txn_date < month_end:
                    # Only apply variance to non-subscription items
                    if pattern.get('is_subscription', False):
                        amount = pattern['amount']  # EXACT amount for subscriptions
                    else:
                        variance = pattern.get('variance', 0)
                        amount = pattern['amount']
                        if variance > 0:
                            amount += random.uniform(-variance, variance)
                    
                    transactions.append({
                        'date': txn_date.date(),
                        'description': pattern['name'],
                        'amount': round(amount, 2)
                    })
            
            # 2. Add habit merchant transactions (8+ visits/month)
            habit_txns = self._generate_habit_transactions(
                month_start, month_end, habit_merchants
            )
            transactions.extend(habit_txns)
            
            # 3. Add regular merchant transactions (3-7 visits/month)
            regular_txns = self._generate_regular_transactions(
                month_start, month_end, regular_merchants
            )
            transactions.extend(regular_txns)
            
            # 4. Add one-time/rare merchant transactions
            if month < 2:  # Only in first 2 months
                onetime_txns = self._generate_onetime_transactions(
                    month_start, month_end, onetime_merchants, month
                )
                transactions.extend(onetime_txns)
            
            # 5. Add random transactions with temporal patterns
            remaining_count = txns_per_month - len(habit_txns) - len(regular_txns) - 5
            random_txns = self._generate_temporal_transactions(
                month_start, month_end, max(remaining_count, 10)
            )
            transactions.extend(random_txns)
        
        # 6. Add anomalies distributed across months (skip first month)
        anomalies = self._generate_distributed_anomalies()
        transactions.extend(anomalies)
        
        # Sort by date
        transactions.sort(key=lambda x: x['date'])
        return transactions
    
    def _get_recurring_patterns(self) -> List[Dict]:
        """
        Return fixed recurring transaction patterns.
        
        Subscriptions have is_subscription=True for EXACT amounts.
        """
        return [
            # Fixed monthly expenses (Housing, Utilities)
            {'name': 'LANDLORD PAYMENT - RENT', 'amount': -1500, 'day': 1, 'category': 'Housing'},
            {'name': 'PROPERTY MANAGEMENT FEE', 'amount': -75, 'day': 1, 'category': 'Housing'},
            {'name': 'ELECTRIC COMPANY - MONTHLY', 'amount': -120, 'day': 15, 'variance': 35, 'category': 'Utilities'},
            {'name': 'GAS UTILITY BILL', 'amount': -65, 'day': 15, 'variance': 20, 'category': 'Utilities'},
            {'name': 'INTERNET SERVICE PROVIDER', 'amount': -79.99, 'day': 18, 'category': 'Utilities'},
            {'name': 'WATER & SEWER UTILITY', 'amount': -45, 'day': 20, 'variance': 10, 'category': 'Utilities'},
            
            # Subscriptions - EXACT amounts (no variance) for ML pattern learning
            {'name': 'NETFLIX SUBSCRIPTION', 'amount': -15.99, 'day': 5, 'category': 'Subscriptions', 'is_subscription': True},
            {'name': 'SPOTIFY PREMIUM', 'amount': -9.99, 'day': 5, 'category': 'Subscriptions', 'is_subscription': True},
            {'name': 'DISNEY+ STREAMING', 'amount': -7.99, 'day': 8, 'category': 'Subscriptions', 'is_subscription': True},
            {'name': 'HULU SUBSCRIPTION', 'amount': -14.99, 'day': 10, 'category': 'Subscriptions', 'is_subscription': True},
            {'name': 'GYM MEMBERSHIP', 'amount': -49.99, 'day': 7, 'category': 'Entertainment', 'is_subscription': True},
            {'name': 'AMAZON PRIME MEMBERSHIP', 'amount': -14.99, 'day': 12, 'category': 'Subscriptions', 'is_subscription': True},
            
            # Gray charges (small, unknown recurring) - EXACT for ML detection
            {'name': 'UNKNOWN APP PURCHASE', 'amount': -2.99, 'day': 12, 'category': 'Unknown', 'is_subscription': True},
            {'name': 'APP STORE CHARGE', 'amount': -4.99, 'day': 20, 'category': 'Unknown', 'is_subscription': True},
            {'name': 'MYSTERY SUBSCRIPTION', 'amount': -1.99, 'day': 25, 'category': 'Unknown', 'is_subscription': True},
            {'name': 'DIGITAL SERVICE FEE', 'amount': -3.49, 'day': 28, 'category': 'Unknown', 'is_subscription': True},
            
            # Income (twice per month - typical bi-weekly paycheck)
            {'name': 'PAYCHECK - DIRECT DEPOSIT', 'amount': 3200, 'day': 1, 'category': 'Income'},
            {'name': 'PAYCHECK - DIRECT DEPOSIT', 'amount': 3200, 'day': 15, 'category': 'Income'},
        ]
    
    def _get_habit_merchants(self) -> List[Dict]:
        """
        Return habit merchants (8+ visits/month).
        
        These are daily habits like coffee, lunch spots.
        """
        return [
            {'name': 'STARBUCKS', 'category': 'Dining', 'amount_range': (4.50, 7.50), 'visits_per_month': (18, 25)},
            {'name': 'DUNKIN DONUTS', 'category': 'Dining', 'amount_range': (3.50, 6.00), 'visits_per_month': (8, 12)},
            {'name': 'CHIPOTLE MEXICAN', 'category': 'Dining', 'amount_range': (10.00, 15.00), 'visits_per_month': (8, 12)},
            {'name': 'PANERA BREAD', 'category': 'Dining', 'amount_range': (9.00, 14.00), 'visits_per_month': (6, 10)},
        ]
    
    def _get_regular_merchants(self) -> List[Dict]:
        """
        Return regular merchants (3-7 visits/month).
        
        Weekly shopping, gas, etc.
        """
        return [
            {'name': 'WHOLE FOODS MARKET', 'category': 'Groceries', 'amount_range': (45, 120), 'visits_per_month': (4, 6)},
            {'name': 'TRADER JOES', 'category': 'Groceries', 'amount_range': (35, 85), 'visits_per_month': (3, 5)},
            {'name': 'SHELL GAS STATION', 'category': 'Transportation', 'amount_range': (35, 55), 'visits_per_month': (4, 6)},
            {'name': 'CHEVRON FUEL', 'category': 'Transportation', 'amount_range': (40, 60), 'visits_per_month': (3, 5)},
            {'name': 'TARGET STORE', 'category': 'Shopping', 'amount_range': (25, 85), 'visits_per_month': (3, 5)},
            {'name': 'CVS PHARMACY', 'category': 'Healthcare', 'amount_range': (12, 45), 'visits_per_month': (2, 4)},
        ]
    
    def _get_onetime_merchants(self) -> List[Dict]:
        """
        Return one-time/rare merchants (1-2 visits total across all months).
        
        These are unusual purchases that ML should flag as potentially anomalous.
        """
        return [
            {'name': 'IKEA FURNITURE', 'category': 'Shopping', 'amount_range': (150, 400)},
            {'name': 'BEST BUY ELECTRONICS', 'category': 'Shopping', 'amount_range': (100, 350)},
            {'name': 'HOME DEPOT HARDWARE', 'category': 'Shopping', 'amount_range': (50, 200)},
            {'name': 'CONCERT VENUE TICKETS', 'category': 'Entertainment', 'amount_range': (80, 200)},
            {'name': 'AIRLINE TICKETS', 'category': 'Travel', 'amount_range': (200, 500)},
            {'name': 'HOTEL BOOKING', 'category': 'Travel', 'amount_range': (150, 350)},
            {'name': 'CAR REPAIR SHOP', 'category': 'Transportation', 'amount_range': (100, 400)},
            {'name': 'DENTIST OFFICE', 'category': 'Healthcare', 'amount_range': (75, 250)},
            {'name': 'OPTOMETRIST CLINIC', 'category': 'Healthcare', 'amount_range': (100, 300)},
            {'name': 'SPORTING GOODS STORE', 'category': 'Shopping', 'amount_range': (50, 150)},
        ]
    
    def _generate_habit_transactions(
        self, 
        month_start: datetime, 
        month_end: datetime, 
        merchants: List[Dict]
    ) -> List[Dict]:
        """
        Generate transactions for habit merchants (8+ visits/month).
        
        Applies temporal patterns:
        - Weekend spending 40% higher probability
        - Payday spending spikes on days 1-3 and 15-17
        """
        transactions = []
        days_in_month = (month_end - month_start).days
        
        for merchant in merchants:
            visits = random.randint(*merchant['visits_per_month'])
            min_amt, max_amt = merchant['amount_range']
            
            for _ in range(visits):
                # Apply temporal weighting
                day_offset = self._get_temporal_day(days_in_month)
                txn_date = month_start + timedelta(days=day_offset)
                
                # Amount with slight variation
                base_amount = random.uniform(min_amt, max_amt)
                
                # Weekend spending 40% higher
                if txn_date.weekday() >= 5:  # Saturday=5, Sunday=6
                    base_amount *= 1.4
                
                transactions.append({
                    'date': txn_date.date(),
                    'description': f'{merchant["name"]} #{random.randint(100, 999)}',
                    'amount': round(-base_amount, 2)
                })
                
                self.merchant_visits[merchant['name']] += 1
        
        return transactions
    
    def _generate_regular_transactions(
        self, 
        month_start: datetime, 
        month_end: datetime, 
        merchants: List[Dict]
    ) -> List[Dict]:
        """
        Generate transactions for regular merchants (3-7 visits/month).
        """
        transactions = []
        days_in_month = (month_end - month_start).days
        
        for merchant in merchants:
            visits = random.randint(*merchant['visits_per_month'])
            min_amt, max_amt = merchant['amount_range']
            
            for _ in range(visits):
                day_offset = self._get_temporal_day(days_in_month)
                txn_date = month_start + timedelta(days=day_offset)
                
                amount = random.uniform(min_amt, max_amt)
                
                # Weekend boost for shopping/entertainment
                if txn_date.weekday() >= 5 and merchant['category'] in ['Shopping', 'Entertainment']:
                    amount *= 1.4
                
                transactions.append({
                    'date': txn_date.date(),
                    'description': merchant['name'],
                    'amount': round(-amount, 2)
                })
                
                self.merchant_visits[merchant['name']] += 1
        
        return transactions
    
    def _generate_onetime_transactions(
        self, 
        month_start: datetime, 
        month_end: datetime, 
        merchants: List[Dict],
        month_index: int
    ) -> List[Dict]:
        """
        Generate one-time merchant transactions.
        
        These only appear once or twice in the entire dataset.
        """
        transactions = []
        days_in_month = (month_end - month_start).days
        
        # Select 2-3 one-time merchants per month (first 2 months only)
        selected = random.sample(merchants, min(3, len(merchants)))
        
        for merchant in selected:
            # Skip if already used
            if self.merchant_visits[merchant['name']] > 0:
                continue
            
            day_offset = random.randint(3, days_in_month - 3)
            txn_date = month_start + timedelta(days=day_offset)
            
            min_amt, max_amt = merchant['amount_range']
            amount = random.uniform(min_amt, max_amt)
            
            transactions.append({
                'date': txn_date.date(),
                'description': merchant['name'],
                'amount': round(-amount, 2)
            })
            
            self.merchant_visits[merchant['name']] += 1
        
        return transactions
    
    def _generate_temporal_transactions(
        self, 
        month_start: datetime, 
        month_end: datetime, 
        count: int
    ) -> List[Dict]:
        """
        Generate random transactions with temporal patterns.
        
        Patterns applied:
        - Weekend spending 40% higher probability
        - Payday spending (days 1-3, 15-17) 60% more transactions
        """
        categories = {
            'Groceries': [
                'SAFEWAY GROCERY', 'KROGER SUPERMARKET', 'SPROUTS MARKET',
                'ALDI STORE', 'PUBLIX SUPER'
            ],
            'Dining': [
                'MCDONALD RESTAURANT', 'SUBWAY SANDWICH', 'TACO BELL',
                'BURGER KING', 'WENDYS', 'PIZZA HUT', 'DOMINOS PIZZA',
                'LOCAL DINER', 'CHINESE RESTAURANT', 'THAI KITCHEN'
            ],
            'Delivery': [
                'DOORDASH ORDER', 'UBER EATS', 'GRUBHUB DELIVERY', 'INSTACART'
            ],
            'Transportation': [
                'UBER RIDE', 'LYFT RIDE', 'PARKING METER', 'PARKING GARAGE',
                'METRO TRANSIT', 'TOLL ROAD'
            ],
            'Shopping': [
                'AMAZON MARKETPLACE', 'WALMART STORE', 'COSTCO WHOLESALE',
                'MARSHALLS', 'TJ MAXX', 'ROSS STORE'
            ],
            'Entertainment': [
                'MOVIE THEATER', 'BOWLING ALLEY', 'ARCADE CENTER',
                'MINI GOLF', 'ESCAPE ROOM'
            ],
        }
        
        amount_ranges = {
            'Groceries': (25, 90),
            'Dining': (8, 35),
            'Delivery': (15, 40),
            'Transportation': (8, 35),
            'Shopping': (20, 100),
            'Entertainment': (15, 60),
        }
        
        transactions = []
        days_in_month = (month_end - month_start).days
        
        for _ in range(count):
            category = random.choice(list(categories.keys()))
            merchant = random.choice(categories[category])
            min_amt, max_amt = amount_ranges[category]
            amount = round(random.uniform(min_amt, max_amt), 2)
            
            # Apply temporal weighting for day selection
            day_offset = self._get_temporal_day(days_in_month)
            txn_date = month_start + timedelta(days=day_offset)
            
            # Weekend spending boost (40% higher amounts)
            if txn_date.weekday() >= 5:
                amount *= 1.4
            
            # Payday spending boost (slightly higher on payday weeks)
            day_of_month = day_offset + 1
            if day_of_month <= 3 or 15 <= day_of_month <= 17:
                amount *= 1.2
            
            transactions.append({
                'date': txn_date.date(),
                'description': merchant,
                'amount': round(-amount, 2)
            })
        
        return transactions
    
    def _get_temporal_day(self, days_in_month: int) -> int:
        """
        Get a day with temporal weighting.
        
        Applies:
        - 60% more transactions on payday weeks (days 1-3, 15-17)
        - 40% higher probability on weekends
        """
        # Create weighted day distribution
        weights = []
        for day in range(days_in_month):
            day_of_month = day + 1
            
            # Base weight
            weight = 1.0
            
            # Payday boost (days 1-3 and 15-17)
            if day_of_month <= 3 or 15 <= day_of_month <= 17:
                weight *= 1.6
            
            # Weekend boost (calculate what day of week this would be)
            # This is approximate since we don't know the actual month start
            if day % 7 >= 5:  # Simplified weekend check
                weight *= 1.4
            
            weights.append(weight)
        
        # Weighted random selection
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for day, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return day
        return days_in_month - 1
    
    def _generate_distributed_anomalies(self) -> List[Dict]:
        """
        Generate 8-10 anomalies distributed across months with severity levels.
        
        Distribution:
        - 3 high-severity (>3x normal)
        - 3 medium-severity (2-3x normal)
        - 2-4 low-severity (1.5-2x normal)
        """
        anomalies = []
        
        # High severity anomalies (>3x normal) - 3 total
        high_severity = [
            {
                'description': 'LUXURY ELECTRONICS PURCHASE',
                'amount': -2487.99,
                'month_offset': 1,
                'day': random.randint(5, 25),
                'reason': 'Extremely high tech purchase (4x normal shopping)'
            },
            {
                'description': 'EMERGENCY ROOM VISIT',
                'amount': -1850.00,
                'month_offset': 3,
                'day': random.randint(5, 25),
                'reason': 'Very high medical expense (6x normal healthcare)'
            },
            {
                'description': 'JEWELRY STORE PURCHASE',
                'amount': -1299.00,
                'month_offset': 4,
                'day': random.randint(5, 25),
                'reason': 'Major jewelry purchase (10x normal shopping)'
            },
        ]
        
        # Medium severity anomalies (2-3x normal) - 3 total
        medium_severity = [
            {
                'description': 'RESTAURANT CATERING SERVICE',
                'amount': -523.45,
                'month_offset': 1,
                'day': random.randint(5, 25),
                'reason': 'Very high dining expense (3x normal)'
            },
            {
                'description': 'DESIGNER CLOTHING STORE',
                'amount': -389.00,
                'month_offset': 2,
                'day': random.randint(5, 25),
                'reason': 'High fashion purchase (2.5x normal shopping)'
            },
            {
                'description': 'AUTO PARTS AND SERVICE',
                'amount': -445.00,
                'month_offset': 4,
                'day': random.randint(5, 25),
                'reason': 'Major car repair (2.5x normal transportation)'
            },
        ]
        
        # Low severity anomalies (1.5-2x normal) - 3 total
        low_severity = [
            {
                'description': 'STEAKHOUSE DINNER',
                'amount': -148.50,
                'month_offset': 2,
                'day': random.randint(5, 25),
                'reason': 'High-end dining (1.8x normal dining)'
            },
            {
                'description': 'SPA AND WELLNESS CENTER',
                'amount': -175.00,
                'month_offset': 3,
                'day': random.randint(5, 25),
                'reason': 'Premium self-care (1.7x normal entertainment)'
            },
            {
                'description': 'WINE AND SPIRITS SHOP',
                'amount': -165.00,
                'month_offset': 5,
                'day': random.randint(5, 25),
                'reason': 'Large beverage purchase (1.6x normal groceries)'
            },
            {
                'description': 'PREMIUM GAS STATION',
                'amount': -95.00,
                'month_offset': 5,
                'day': random.randint(5, 25),
                'reason': 'Full tank premium fuel (1.5x normal gas)'
            },
        ]
        
        all_anomalies = high_severity + medium_severity + low_severity
        
        for anomaly in all_anomalies:
            month_offset = anomaly['month_offset']
            day = anomaly['day']
            
            txn_date = self.start_date + timedelta(days=30 * month_offset + day)
            
            anomalies.append({
                'date': txn_date.date(),
                'description': anomaly['description'],
                'amount': anomaly['amount']
            })
        
        return anomalies
    
    def to_csv(self, transactions: List[Dict], filename: str = 'sample_transactions.csv') -> str:
        """
        Export transactions to CSV file.
        
        Args:
            transactions: List of transaction dictionaries.
            filename: Output filename.
        
        Returns:
            The filename written.
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'description', 'amount'])
            writer.writeheader()
            for txn in transactions:
                row = {
                    'date': txn['date'].isoformat() if hasattr(txn['date'], 'isoformat') else txn['date'],
                    'description': txn['description'],
                    'amount': txn['amount']
                }
                writer.writerow(row)
        print(f"✅ Generated {len(transactions)} transactions → {filename}")
        return filename
    
    def to_dict_list(self, transactions: List[Dict]) -> List[Dict]:
        """Return transactions as list of dicts (unchanged)."""
        return transactions
    
    def get_statistics(self, transactions: List[Dict]) -> Dict:
        """
        Get statistics about generated transactions for validation.
        
        Returns:
            Dict with counts, anomaly distribution, merchant frequency, etc.
        """
        stats = {
            'total_transactions': len(transactions),
            'months': 6,
            'per_month_avg': len(transactions) / 6,
            'anomalies': {
                'high': 3,
                'medium': 3,
                'low': 4
            },
            'merchant_frequency': {
                'habit': len([m for m, c in self.merchant_visits.items() if c >= 8]),
                'regular': len([m for m, c in self.merchant_visits.items() if 3 <= c < 8]),
                'onetime': len([m for m, c in self.merchant_visits.items() if c <= 2])
            }
        }
        return stats


# =============================================================================
# Module-level function for main.py integration
# =============================================================================

def generate_synthetic_transactions(months: int = 6, txns_per_month: int = 80) -> List[Dict]:
    """
    Generate synthetic transaction data for demo and ML training.
    
    This is the main entry point used by main.py for the /sample endpoint.
    
    Args:
        months: Number of months of data to generate (default: 6).
        txns_per_month: Approximate transactions per month (default: 80).
    
    Returns:
        List of transaction dictionaries with keys: date, description, amount.
        Date is a date object, amount is float (negative for expenses).
    
    Example:
        >>> transactions = generate_synthetic_transactions()
        >>> len(transactions)
        480  # approximately
    """
    generator = SyntheticDataGenerator(seed=42)
    return generator.generate(months=months, txns_per_month=txns_per_month)


def generate_and_save():
    """Main function to generate and save synthetic data."""
    print("🚀 Generating enhanced synthetic transaction data for ML training...")
    print("=" * 70)
    
    generator = SyntheticDataGenerator(seed=42)
    transactions = generator.generate(months=6, txns_per_month=80)
    
    # Save to CSV
    generator.to_csv(transactions, 'sample_transactions.csv')
    
    # Print summary
    print("\n📊 Dataset Summary:")
    print(f"   Total transactions: {len(transactions)}")
    print(f"   Target: 6 months × 80/month = ~480 transactions")
    first_date = transactions[0]['date']
    last_date = transactions[-1]['date']
    first_str = first_date.isoformat() if hasattr(first_date, 'isoformat') else first_date
    last_str = last_date.isoformat() if hasattr(last_date, 'isoformat') else last_date
    print(f"   Date range: {first_str} to {last_str}")
    
    # Calculate totals
    total_income = sum(t['amount'] for t in transactions if t['amount'] > 0)
    total_expenses = sum(t['amount'] for t in transactions if t['amount'] < 0)
    net = total_income + total_expenses
    
    print(f"   Total income: ${total_income:,.2f}")
    print(f"   Total expenses: ${abs(total_expenses):,.2f}")
    print(f"   Net: ${net:,.2f}")
    
    # Show anomaly distribution
    print("\n⚠️  Anomalies in Data (10 total):")
    print("-" * 70)
    print("   HIGH severity (>3x normal):")
    print("      • LUXURY ELECTRONICS PURCHASE: $2,487.99")
    print("      • EMERGENCY ROOM VISIT: $1,850.00")
    print("      • JEWELRY STORE PURCHASE: $1,299.00")
    print("   MEDIUM severity (2-3x normal):")
    print("      • RESTAURANT CATERING SERVICE: $523.45")
    print("      • DESIGNER CLOTHING STORE: $389.00")
    print("      • AUTO PARTS AND SERVICE: $445.00")
    print("   LOW severity (1.5-2x normal):")
    print("      • STEAKHOUSE DINNER: $148.50")
    print("      • SPA AND WELLNESS CENTER: $175.00")
    print("      • WINE AND SPIRITS SHOP: $165.00")
    print("      • PREMIUM GAS STATION: $95.00")
    
    # Show recurring charges
    print("\n🔄 Subscriptions (EXACT amounts for ML learning):")
    print("-" * 70)
    subscriptions = [
        ('NETFLIX SUBSCRIPTION', -15.99),
        ('SPOTIFY PREMIUM', -9.99),
        ('DISNEY+ STREAMING', -7.99),
        ('HULU SUBSCRIPTION', -14.99),
        ('GYM MEMBERSHIP', -49.99),
        ('AMAZON PRIME MEMBERSHIP', -14.99),
    ]
    for name, amount in subscriptions:
        annual = amount * 12
        print(f"   {name:30} | ${amount:>6.2f}/mo | ${annual:>7.2f}/yr")
    
    # Show gray charges
    print("\n🔍 Gray Charges (small recurring unknowns):")
    print("-" * 70)
    gray = [
        ('UNKNOWN APP PURCHASE', -2.99),
        ('APP STORE CHARGE', -4.99),
        ('MYSTERY SUBSCRIPTION', -1.99),
        ('DIGITAL SERVICE FEE', -3.49),
    ]
    total_gray = sum(a for _, a in gray)
    for name, amount in gray:
        print(f"   {name:30} | ${amount:>6.2f}/mo")
    print(f"   {'TOTAL GRAY CHARGES':30} | ${total_gray:>6.2f}/mo | ${total_gray*12:>7.2f}/yr")
    
    # Show merchant frequency
    stats = generator.get_statistics(transactions)
    print("\n📈 Merchant Frequency Patterns:")
    print("-" * 70)
    print(f"   Habit merchants (8+ visits/mo): {stats['merchant_frequency']['habit']}")
    print(f"   Regular merchants (3-7/mo):     {stats['merchant_frequency']['regular']}")
    print(f"   One-time merchants (1-2 total): {stats['merchant_frequency']['onetime']}")
    
    # Show temporal patterns
    print("\n⏰ Temporal Patterns Embedded:")
    print("-" * 70)
    print("   • Weekend spending: 40% higher amounts")
    print("   • Payday weeks (1-3, 15-17): 60% more transactions")
    print("   • Habit merchants: Daily coffee/lunch patterns")
    
    print("\n" + "=" * 70)
    print("✅ Enhanced ML training data ready!")
    print("   Use 'sample_transactions.csv' or /sample endpoint")
    
    return transactions


if __name__ == '__main__':
    generate_and_save()
