"""
CSV parsing, validation, and normalization with data sanity checks.

Validates:
    - Required columns exist
    - Date format is parseable
    - Amounts are reasonable (not $1M+)
    - No future dates
    - Income is positive, expenses are negative

Author: Smart Financial Coach Team
"""

import uuid
import re
from datetime import datetime, date
from io import StringIO
from typing import Optional, List, Dict
import pandas as pd
from fastapi import UploadFile
from sqlalchemy.orm import Session as DBSession

from models import Session, Transaction


class DataValidationError(ValueError):
    """Exception for data validation failures."""
    
    def __init__(self, message: str, warnings: List[str] = None):
        super().__init__(message)
        self.warnings = warnings or []


class CSVProcessor:
    """Parse, validate, and normalize transaction CSVs with sanity checks."""

    REQUIRED_COLUMNS = ["date", "description", "amount"]
    DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d"]
    
    # Data sanity limits
    MAX_TRANSACTION_AMOUNT = 100000  # $100K - flag anything higher
    MAX_SINGLE_EXPENSE = 50000       # $50K - unusual for personal finance
    MIN_DATE_YEARS_AGO = 5           # Don't accept data older than 5 years
    MAX_ROWS = 10000                 # Prevent huge uploads

    def __init__(self, db: DBSession):
        self.db = db
        self.validation_warnings: List[str] = []

    async def process(
        self, 
        file: UploadFile, 
        clerk_user_id: str = None
    ) -> tuple[str, int]:
        """
        Process uploaded CSV and create session with transactions.
        
        Args:
            file: Uploaded CSV file.
            clerk_user_id: Clerk user ID for session ownership.
            
        Returns:
            Tuple of (session_id, row_count).
        """
        self.validation_warnings = []
        
        # Read file content
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        df = pd.read_csv(StringIO(text))
        
        # Check row count
        if len(df) > self.MAX_ROWS:
            raise DataValidationError(
                f"File too large: {len(df)} rows. Maximum allowed: {self.MAX_ROWS}"
            )
        
        if len(df) == 0:
            raise DataValidationError("File is empty or has no valid data rows")

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Validate required columns
        self._validate_columns(df)

        # Normalize data
        df = self._normalize_dates(df)
        df = self._normalize_amounts(df)
        df = self._clean_descriptions(df)
        
        # Validate data sanity
        df = self._validate_data_sanity(df)

        # Create session with user ownership
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            clerk_user_id=clerk_user_id or "anonymous",
            filename=file.filename,
            row_count=len(df),
            status="processing",
            is_sample=False,
            name=file.filename.replace('.csv', '') if file.filename else "Uploaded Data",
        )
        self.db.add(session)

        # Create transactions
        for _, row in df.iterrows():
            transaction = Transaction(
                session_id=session_id,
                date=row["date"],
                description=row["description"],
                amount=row["amount"],
                raw_description=row.get("raw_description", row["description"]),
            )
            self.db.add(transaction)

        self.db.commit()
        return session_id, len(df)

    def process_synthetic(
        self, 
        transactions: list[dict],
        clerk_user_id: str = None
    ) -> tuple[str, int]:
        """
        Process synthetic transaction data.
        
        Args:
            transactions: List of transaction dictionaries.
            clerk_user_id: Clerk user ID for session ownership.
            
        Returns:
            Tuple of (session_id, row_count).
        """
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            clerk_user_id=clerk_user_id or "anonymous",
            filename="sample_transactions.csv",
            row_count=len(transactions),
            status="processing",
            is_sample=True,
            name="Sample Data",
        )
        self.db.add(session)

        for txn in transactions:
            transaction = Transaction(
                session_id=session_id,
                date=txn["date"],
                description=txn["description"],
                amount=txn["amount"],
                raw_description=txn["description"],
            )
            self.db.add(transaction)

        self.db.commit()
        return session_id, len(transactions)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        missing = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                # Check for common alternatives
                alternatives = {
                    "date": ["transaction_date", "trans_date", "txn_date"],
                    "description": ["merchant", "name", "memo", "payee"],
                    "amount": ["value", "sum", "total"],
                }
                found = False
                for alt in alternatives.get(col, []):
                    if alt in df.columns:
                        df.rename(columns={alt: col}, inplace=True)
                        found = True
                        break
                if not found:
                    missing.append(col)

        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert dates to consistent format."""

        def parse_date(val):
            if pd.isna(val):
                return None
            if isinstance(val, datetime):
                return val.date()
            val_str = str(val).strip()
            for fmt in self.DATE_FORMATS:
                try:
                    return datetime.strptime(val_str, fmt).date()
                except ValueError:
                    continue
            # Try pandas parser as fallback
            try:
                return pd.to_datetime(val_str).date()
            except Exception:
                raise ValueError(f"Cannot parse date: {val_str}")

        df["date"] = df["date"].apply(parse_date)
        df = df.dropna(subset=["date"])
        return df

    def _normalize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize amount values."""

        def parse_amount(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            val_str = str(val).strip()
            # Remove currency symbols and commas
            val_str = re.sub(r"[$,]", "", val_str)
            # Handle parentheses as negative
            if val_str.startswith("(") and val_str.endswith(")"):
                val_str = "-" + val_str[1:-1]
            try:
                return float(val_str)
            except ValueError:
                return 0.0

        df["amount"] = df["amount"].apply(parse_amount)
        return df

    def _clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize descriptions."""

        def clean_desc(val):
            if pd.isna(val):
                return "Unknown"
            desc = str(val).strip()
            # Remove extra whitespace
            desc = re.sub(r"\s+", " ", desc)
            # Remove common prefixes
            desc = re.sub(r"^(POS |CHECKCARD |DEBIT |CREDIT )", "", desc, flags=re.I)
            return desc if desc else "Unknown"

        df["raw_description"] = df["description"]
        df["description"] = df["description"].apply(clean_desc)
        return df
    
    def _validate_data_sanity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data sanity and flag/remove problematic rows.
        
        Checks:
            - No future dates
            - No extremely old dates (>5 years)
            - No unreasonably large amounts ($100K+)
            - No negative income descriptions
            - No zero amounts (unless explicitly allowed)
        
        Args:
            df: DataFrame with normalized data.
            
        Returns:
            Cleaned DataFrame with problematic rows removed.
            
        Raises:
            DataValidationError: If data has critical issues.
        """
        original_count = len(df)
        today = date.today()
        min_date = date(today.year - self.MIN_DATE_YEARS_AGO, 1, 1)
        
        # Track rows to remove
        rows_to_remove = set()
        
        # Check 1: Future dates
        future_dates = df[df['date'] > today]
        if len(future_dates) > 0:
            self.validation_warnings.append(
                f"Removed {len(future_dates)} transactions with future dates"
            )
            rows_to_remove.update(future_dates.index)
        
        # Check 2: Very old dates
        old_dates = df[df['date'] < min_date]
        if len(old_dates) > 0:
            self.validation_warnings.append(
                f"Removed {len(old_dates)} transactions older than {self.MIN_DATE_YEARS_AGO} years"
            )
            rows_to_remove.update(old_dates.index)
        
        # Check 3: Unreasonably large amounts
        large_amounts = df[df['amount'].abs() > self.MAX_TRANSACTION_AMOUNT]
        if len(large_amounts) > 0:
            self.validation_warnings.append(
                f"Removed {len(large_amounts)} transactions over ${self.MAX_TRANSACTION_AMOUNT:,}"
            )
            rows_to_remove.update(large_amounts.index)
        
        # Check 4: Flag suspicious large expenses (don't remove, just warn)
        large_expenses = df[
            (df['amount'] < 0) & 
            (df['amount'].abs() > self.MAX_SINGLE_EXPENSE) &
            (~df.index.isin(rows_to_remove))
        ]
        if len(large_expenses) > 0:
            self.validation_warnings.append(
                f"Found {len(large_expenses)} unusually large expenses (>${self.MAX_SINGLE_EXPENSE:,})"
            )
        
        # Check 5: Zero amounts
        zero_amounts = df[df['amount'] == 0]
        if len(zero_amounts) > 0:
            self.validation_warnings.append(
                f"Removed {len(zero_amounts)} transactions with zero amount"
            )
            rows_to_remove.update(zero_amounts.index)
        
        # Check 6: Income marked as negative (common data error)
        income_keywords = ['paycheck', 'salary', 'deposit', 'refund', 'dividend']
        for idx, row in df.iterrows():
            if idx in rows_to_remove:
                continue
            desc_lower = str(row.get('description', '')).lower()
            if any(keyword in desc_lower for keyword in income_keywords):
                if row['amount'] < 0:
                    # Fix: income should be positive
                    df.at[idx, 'amount'] = abs(row['amount'])
                    self.validation_warnings.append(
                        f"Fixed negative income: {row['description'][:30]}"
                    )
        
        # Remove flagged rows
        if rows_to_remove:
            df = df.drop(list(rows_to_remove))
        
        # Check if we have any data left
        if len(df) == 0:
            raise DataValidationError(
                "No valid transactions after data validation. Check date and amount formats.",
                warnings=self.validation_warnings
            )
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"⚠️ Data validation: removed {removed_count}/{original_count} rows")
            for warning in self.validation_warnings:
                print(f"   - {warning}")
        
        return df.reset_index(drop=True)