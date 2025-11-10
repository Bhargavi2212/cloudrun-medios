"""
Data splitting utilities for train/test/validation splits.

Implements random stratified splitting to maintain class distribution
across all splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from pathlib import Path


class RandomStratifiedSplitter:
    """
    Random stratified data splitter for maintaining class distribution.
    
    Splits data into train (70%), validation (15%), and test (15%) sets
    with stratification by target variable to preserve class distribution.
    """
    
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize the splitter.
        
        Args:
            train_size: Proportion of data for training (default: 0.7)
            val_size: Proportion of data for validation (default: 0.15)
            test_size: Proportion of data for testing (default: 0.15)
            random_state: Random seed for reproducibility
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError(
                f"Train, validation, and test sizes must sum to 1.0. "
                f"Got: {train_size} + {val_size} + {test_size} = {train_size + val_size + test_size}"
            )
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
    
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            save_path: Optional path to save splits as CSV files
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(self.val_size + self.test_size),
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: 50-50 of 30% → 15% val, 15% test
        # Calculate test_size relative to temp split (0.5 = 50% of temp)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,  # 50% of temp = 15% of total
            stratify=y_temp,
            random_state=self.random_state
        )
        
        print(f"Data split completed:")
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Print class distribution in each split
        self._print_class_distribution(y_train, y_val, y_test)
        
        # Save splits if path provided
        if save_path is not None:
            self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test, save_path)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _print_class_distribution(
        self,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series
    ):
        """Print class distribution for each split."""
        print("\nClass distribution:")
        print("=" * 60)
        
        all_classes = sorted(set(y_train) | set(y_val) | set(y_test))
        
        print(f"{'ESI Level':<12} {'Train':<15} {'Val':<15} {'Test':<15}")
        print("-" * 60)
        
        for esi in all_classes:
            train_count = (y_train == esi).sum()
            val_count = (y_val == esi).sum()
            test_count = (y_test == esi).sum()
            
            train_pct = train_count / len(y_train) * 100
            val_pct = val_count / len(y_val) * 100
            test_pct = test_count / len(y_test) * 100
            
            print(f"ESI {int(esi):<11} {train_count:>5} ({train_pct:>5.1f}%) "
                  f"{val_count:>5} ({val_pct:>5.1f}%) {test_count:>5} ({test_pct:>5.1f}%)")
        
        print("=" * 60)
    
    def _save_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        save_path: Path
    ):
        """Save splits to CSV files."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save train set
        train_df = X_train.copy()
        train_df['esi_level'] = y_train.values
        train_df.to_csv(save_path / "train.csv", index=False)
        
        # Save validation set
        val_df = X_val.copy()
        val_df['esi_level'] = y_val.values
        val_df.to_csv(save_path / "val.csv", index=False)
        
        # Save test set
        test_df = X_test.copy()
        test_df['esi_level'] = y_test.values
        test_df.to_csv(save_path / "test.csv", index=False)
        
        print(f"\nSplits saved to: {save_path}")


class TemporalDataSplitter:
    """
    Temporal data splitter for time-series data.
    
    Splits data by year to prevent data leakage in time-series scenarios.
    Note: This is kept for reference, but RandomStratifiedSplitter is preferred
    for stable clinical data.
    """
    
    def __init__(
        self,
        train_years: list,
        val_years: list,
        test_years: list
    ):
        """
        Initialize temporal splitter.
        
        Args:
            train_years: List of years for training (e.g., list(range(2011, 2020)))
            val_years: List of years for validation (e.g., [2020, 2021])
            test_years: List of years for testing (e.g., [2022])
        """
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
    
    def split(
        self,
        df: pd.DataFrame,
        target_col: str = 'esi_level'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data by year.
        
        Args:
            df: Full DataFrame with 'year' column
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if 'year' not in df.columns:
            raise ValueError("DataFrame must contain 'year' column for temporal splitting")
        
        # Extract features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split by year
        train_mask = df['year'].isin(self.train_years)
        val_mask = df['year'].isin(self.val_years)
        test_mask = df['year'].isin(self.test_years)
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        
        print(f"Temporal split completed:")
        print(f"  Train years {self.train_years}: {len(X_train):,} samples")
        print(f"  Val years {self.val_years}: {len(X_val):,} samples")
        print(f"  Test years {self.test_years}: {len(X_test):,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

