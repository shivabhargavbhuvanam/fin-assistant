"""
Module: train_models.py
Description: Train Isolation Forest model for anomaly detection.

This script trains the anomaly detection model on synthetic data.
Run ONCE during setup, then model is loaded for inference.

Why ML for Anomaly Detection (but NOT for Subscriptions):
    - Anomalies are multi-dimensional (amount, timing, frequency, merchant)
    - Isolation Forest can learn complex patterns humans miss
    - Unsupervised - doesn't need labeled data

Why RULES for Subscription Detection:
    - Subscriptions follow deterministic patterns
    - Rules are transparent and fast
    - No training needed - works on first upload
    - See: services/recurring_detector.py

Usage:
    python backend/train_models.py

Output:
    - models/anomaly_model.pkl
    - models/anomaly_scaler.pkl

Author: Smart Financial Coach Team
Created: 2025-01-31
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_data import SyntheticDataGenerator


# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = Path(__file__).parent / "models"
RANDOM_STATE = 42

# Known anomaly descriptions from synthetic data (for evaluation only)
KNOWN_ANOMALIES = {
    'LUXURY ELECTRONICS PURCHASE',
    'EMERGENCY ROOM VISIT',
    'JEWELRY STORE PURCHASE',
    'RESTAURANT CATERING SERVICE',
    'DESIGNER CLOTHING STORE',
    'AUTO PARTS AND SERVICE',
    'STEAKHOUSE DINNER',
    'SPA AND WELLNESS CENTER',
    'WINE AND SPIRITS SHOP',
    'PREMIUM GAS STATION',
}


# =============================================================================
# Feature Engineering
# =============================================================================

def extract_anomaly_features(transactions: List[Dict]) -> pd.DataFrame:
    """
    Extract features for anomaly detection.
    
    Features (8 total, aligned with anomaly_detector.py):
        - amount_abs: Absolute amount
        - amount_zscore: Z-score of amount
        - amount_log: Log-transformed amount
        - merchant_frequency: count / total
        - is_one_time: 1 if merchant appears <=2 times
        - day_of_week: 0-6
        - is_weekend: 0 or 1
        - is_payday: 0 or 1
    """
    if not transactions:
        return pd.DataFrame()
    
    # Build DataFrame
    data = []
    for t in transactions:
        txn_date = t['date']
        if isinstance(txn_date, str):
            txn_date = datetime.strptime(txn_date, '%Y-%m-%d').date()
        
        data.append({
            'description': t['description'],
            'amount': abs(t['amount']),
            'date': txn_date,
            'day_of_week': txn_date.weekday() if hasattr(txn_date, 'weekday') else 0,
            'day_of_month': txn_date.day if hasattr(txn_date, 'day') else 1,
        })
    
    df = pd.DataFrame(data)
    
    # Feature 1: Absolute amount
    df['amount_abs'] = df['amount']
    
    # Feature 2: Z-score of amount
    amount_mean = df['amount'].mean()
    amount_std = df['amount'].std()
    df['amount_zscore'] = (df['amount'] - amount_mean) / (amount_std if amount_std > 0 else 1)
    
    # Feature 3: Log-transformed amount
    df['amount_log'] = np.log1p(df['amount'])
    
    # Feature 4: Merchant frequency
    merchant_norm = df['description'].str.upper().str[:20]
    merchant_counts = merchant_norm.value_counts()
    total = len(df)
    df['merchant_frequency'] = merchant_norm.map(lambda m: merchant_counts.get(m, 1) / total)
    
    # Feature 5: Is one-time merchant
    df['is_one_time'] = merchant_norm.map(lambda m: 1 if merchant_counts.get(m, 0) <= 2 else 0)
    
    # Feature 6: Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Feature 7: Is payday period
    df['is_payday'] = ((df['day_of_month'] <= 3) | 
                       ((df['day_of_month'] >= 15) & (df['day_of_month'] <= 17))).astype(int)
    
    # Label known anomalies (for evaluation only)
    df['is_known_anomaly'] = df['description'].apply(
        lambda d: 1 if any(a in d.upper() for a in KNOWN_ANOMALIES) else 0
    )
    
    return df


# =============================================================================
# Model Training
# =============================================================================

def train_anomaly_model(transactions: List[Dict]) -> Dict:
    """
    Train Isolation Forest for anomaly detection.
    
    Args:
        transactions: List of transaction dicts.
        
    Returns:
        Dict with training results.
    """
    print("\n" + "="*60)
    print("🔍 ANOMALY DETECTION MODEL (Isolation Forest)")
    print("="*60)
    
    # Extract features
    df = extract_anomaly_features(transactions)
    print(f"\n📊 Training data: {len(df)} transactions")
    
    # Feature columns
    feature_cols = [
        'amount_abs', 'amount_zscore', 'amount_log',
        'merchant_frequency', 'is_one_time', 'day_of_week',
        'is_weekend', 'is_payday'
    ]
    
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect 5% anomalies
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    
    # Predict on training data
    predictions = model.predict(X_scaled)
    df['predicted_anomaly'] = (predictions == -1).astype(int)
    
    # Calculate scores (higher = more anomalous)
    scores = -model.score_samples(X_scaled)
    df['anomaly_score'] = scores
    
    # Evaluate against known anomalies
    known_anomalies = df[df['is_known_anomaly'] == 1]
    detected = known_anomalies[known_anomalies['predicted_anomaly'] == 1]
    
    detection_rate = len(detected) / len(known_anomalies) * 100 if len(known_anomalies) > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"   Known anomalies in data: {len(known_anomalies)}")
    print(f"   Detected by model: {len(detected)}")
    print(f"   Detection rate: {detection_rate:.1f}%")
    
    # Calculate precision/recall if we have labels
    if len(known_anomalies) > 0:
        y_true = df['is_known_anomaly'].values
        y_pred = df['predicted_anomaly'].values
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"\n   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        print(f"   F1 Score: {f1:.2%}")
    
    # Show score distribution
    print(f"\n📊 Anomaly score distribution:")
    print(f"   Min: {scores.min():.3f}")
    print(f"   Max: {scores.max():.3f}")
    print(f"   Mean: {scores.mean():.3f}")
    print(f"   Std: {scores.std():.3f}")
    
    # Show top anomalies
    top_anomalies = df.nlargest(5, 'anomaly_score')[['description', 'amount', 'anomaly_score']]
    print(f"\n🚨 Top 5 detected anomalies:")
    for _, row in top_anomalies.iterrows():
        print(f"   ${row['amount']:.2f} - {row['description'][:40]} (score: {row['anomaly_score']:.3f})")
    
    return {
        'model': model,
        'scaler': scaler,
        'detection_rate': detection_rate,
        'total_detected': df['predicted_anomaly'].sum(),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Train and save ML models."""
    print("\n" + "="*60)
    print("🚀 SMART FINANCIAL COACH - MODEL TRAINING")
    print("="*60)
    
    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Model directory: {MODEL_DIR}")
    
    # Generate synthetic training data
    print("\n📊 Generating synthetic training data...")
    generator = SyntheticDataGenerator()
    transactions = generator.generate(months=6, txns_per_month=80)
    print(f"   Generated {len(transactions)} transactions")
    
    # Train anomaly model
    anomaly_results = train_anomaly_model(transactions)
    
    # Save models
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60)
    
    anomaly_model_path = MODEL_DIR / "anomaly_model.pkl"
    anomaly_scaler_path = MODEL_DIR / "anomaly_scaler.pkl"
    
    joblib.dump(anomaly_results['model'], anomaly_model_path)
    joblib.dump(anomaly_results['scaler'], anomaly_scaler_path)
    
    print(f"\n✅ Anomaly model saved to: {anomaly_model_path}")
    print(f"✅ Anomaly scaler saved to: {anomaly_scaler_path}")
    
    # Remove old subscription model files if they exist
    old_files = [
        MODEL_DIR / "subscription_model.pkl",
        MODEL_DIR / "subscription_scaler.pkl",
        MODEL_DIR / "subscription_vectorizer.pkl",
    ]
    for old_file in old_files:
        if old_file.exists():
            old_file.unlink()
            print(f"🗑️  Removed deprecated: {old_file.name}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"""
Summary:
  📊 Anomaly Detection (Isolation Forest):
     - Detection rate: {anomaly_results['detection_rate']:.1f}%
     - Total anomalies detected: {anomaly_results['total_detected']}
     
  📋 Subscription Detection:
     - Using RULE-BASED detection (no ML needed)
     - See: services/recurring_detector.py
     - Why: Subscriptions are deterministic patterns
     
  💾 Models saved to: {MODEL_DIR}/
""")


if __name__ == "__main__":
    main()
