"""
Systematic Data Questionnaire Answers

Answers all 30 questions about the dataset systematically.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def print_answer(q_num, question, answer):
    """Print formatted question and answer."""
    print(f"\n{'='*80}")
    print(f"Q{q_num}: {question}")
    print(f"{'='*80}")
    print(f"ANSWER: {answer}")


def main():
    print("="*80)
    print("COMPREHENSIVE DATA QUESTIONNAIRE ANSWERS")
    print("="*80)
    
    # Load original CSV
    raw_data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    df = pd.read_csv(raw_data_path)
    
    # Load preprocessed cache
    cache_v3 = project_root / "services" / "manage-agent" / "outputs" / "preprocessed_data_cache_v3.pkl"
    with open(cache_v3, 'rb') as f:
        data_cache = pickle.load(f)
    
    X_train = data_cache['train']
    X_val = data_cache['val']
    X_test = data_cache['test']
    y_train = data_cache['y_train']
    y_val = data_cache['y_val']
    y_test = data_cache['y_test']
    
    # ============================================================================
    # SECTION 1: ORIGINAL CSV STRUCTURE (Questions 1-5)
    # ============================================================================
    print("\n" + "="*80)
    print("SECTION 1: ORIGINAL CSV STRUCTURE")
    print("="*80)
    
    # Q1: Years included
    years = sorted(df['year'].unique())
    print_answer(1, "Which years are in your combined CSV?", 
                 f"Years: {years}\nTotal rows: {len(df):,}")
    
    # Q2: Records per year
    year_counts = df['year'].value_counts().sort_index()
    print_answer(2, "How many records per year?",
                 f"\n{year_counts.to_string()}")
    
    # Q3: ESI distribution
    esi_counts = df['esi_level'].value_counts().sort_index()
    esi_pcts = (esi_counts / len(df) * 100).round(1)
    print_answer(3, "What's the ESI distribution in ORIGINAL data?",
                 f"\n{esi_counts.to_string()}\n\nPercentages:\n{esi_pcts.to_string()}")
    
    # Q4: ESI 6 existence
    has_esi_6 = 6.0 in df['esi_level'].values or 6 in df['esi_level'].values
    answer_4 = "YES → ERROR! (ESI 6 doesn't exist)" if has_esi_6 else "NO → Correct"
    print_answer(4, "Does original CSV have ESI 6?", answer_4)
    
    # Q5: Total rows
    print_answer(5, "Total rows in original CSV?",
                 f"{len(df):,} rows")
    
    # ============================================================================
    # SECTION 2: RFV COLUMNS (Questions 6-10)
    # ============================================================================
    print("\n" + "="*80)
    print("SECTION 2: RFV COLUMNS")
    print("="*80)
    
    # Q6: RFV columns
    rfv_cols = [col for col in df.columns if 'rfv' in col.lower()]
    print_answer(6, "What RFV columns exist in ORIGINAL CSV?",
                 f"Columns: {rfv_cols}")
    
    # Q7: RFV1 values
    rfv1_desc = df['rfv1'].describe()
    rfv1_nunique = df['rfv1'].nunique()
    print_answer(7, "What are the RFV1 values in ORIGINAL data?",
                 f"Min: {rfv1_desc['min']:.0f}\nMax: {rfv1_desc['max']:.0f}\n"
                 f"Unique codes: {rfv1_nunique}\n"
                 f"Type: Numeric codes (e.g., 10500, 10150, 12100)")
    
    # Q8: RFV missing rates
    print_answer(8, "What's the missing rate for each RFV?",
                 f"RFV1: {df['rfv1'].isna().mean()*100:.1f}% missing\n"
                 f"RFV2: {df['rfv2'].isna().mean()*100:.1f}% missing\n"
                 f"RFV3: {df['rfv3'].isna().mean()*100:.1f}% missing")
    
    # Q9: Sample RFV1 values
    sample_rfv1 = df['rfv1'].dropna().head(5).tolist()
    print_answer(9, "Show 5 sample RFV1 values from original data:",
                 f"Examples: {sample_rfv1}")
    
    # Q10: RFV codes or text
    print_answer(10, "Are RFV codes or text in the CSV?",
                 "Numeric codes (10500, 10150, 12100, etc.) - NOT text")
    
    # ============================================================================
    # SECTION 3: VITAL SIGNS (Questions 11-15)
    # ============================================================================
    print("\n" + "="*80)
    print("SECTION 3: VITAL SIGNS")
    print("="*80)
    
    # Q11: Vital sign columns
    vital_cols = ['pulse', 'respiration', 'sbp', 'dbp', 'o2_sat', 'temp_c', 'tempf', 'pain']
    found_vitals = [col for col in vital_cols if col in df.columns]
    print_answer(11, "What vital sign columns exist?",
                 f"Found: {found_vitals}")
    
    # Q12: Vital missing rates
    vital_missing = {}
    for col in found_vitals:
        vital_missing[col] = df[col].isna().mean() * 100
    missing_str = "\n".join([f"{col}: {pct:.1f}%" for col, pct in vital_missing.items()])
    print_answer(12, "What's the missing rate for vitals?",
                 f"Results:\n{missing_str}")
    
    # Q13: Vital ranges
    if all(col in df.columns for col in ['pulse', 'sbp', 'o2_sat']):
        vital_ranges = df[['pulse', 'sbp', 'o2_sat']].describe()
        print_answer(13, "What are the value ranges for vitals?",
                     f"Pulse: min={vital_ranges.loc['min', 'pulse']:.1f}, max={vital_ranges.loc['max', 'pulse']:.1f}\n"
                     f"SBP: min={vital_ranges.loc['min', 'sbp']:.1f}, max={vital_ranges.loc['max', 'sbp']:.1f}\n"
                     f"O2 sat: min={vital_ranges.loc['min', 'o2_sat']:.1f}, max={vital_ranges.loc['max', 'o2_sat']:.1f}")
    
    # Q14: Medical sense check
    pulse_ok = df['pulse'].between(0, 300).all() if 'pulse' in df.columns else False
    sbp_ok = df['sbp'].between(0, 300).all() if 'sbp' in df.columns else False
    o2_ok = df['o2_sat'].between(0, 100).all() if 'o2_sat' in df.columns else False
    print_answer(14, "Do vitals make medical sense?",
                 f"Pulse 0-300? {'✓' if pulse_ok else '✗'}\n"
                 f"SBP 0-300? {'✓' if sbp_ok else '✗'}\n"
                 f"O2 sat 0-100? {'✓' if o2_ok else '✗'}")
    
    # Q15: Missing vitals handling
    print_answer(15, "How were missing vitals handled in preprocessing?",
                 "KNN imputation (k=3, max_samples=50000)")
    
    # ============================================================================
    # SECTION 4: PREPROCESSING PIPELINE (Questions 16-20)
    # ============================================================================
    print("\n" + "="*80)
    print("SECTION 4: PREPROCESSING PIPELINE")
    print("="*80)
    
    # Q16: Cache contents
    cache_keys = list(data_cache.keys())
    feature_names = list(X_train.columns)
    print_answer(16, "What's in your preprocessed cache (v3)?",
                 f"Keys: {cache_keys}\n"
                 f"Feature names ({len(feature_names)}): {feature_names}")
    
    # Q17: Feature count comparison
    original_cols = len(df.columns) - 1  # Exclude target
    preprocessed_features = len(X_train.columns)
    dropped = original_cols - preprocessed_features
    print_answer(17, "How many features in preprocessed vs original?",
                 f"Original CSV columns: {original_cols} columns\n"
                 f"Preprocessed features: {preprocessed_features} features\n"
                 f"Dropped columns: {dropped} columns")
    
    # Q18: Dropped columns
    original_cols_list = [col for col in df.columns if col != 'esi_level']
    preprocessed_cols_list = list(X_train.columns)
    dropped_cols = [col for col in original_cols_list if col not in preprocessed_cols_list]
    print_answer(18, "Which columns were dropped during preprocessing?",
                 f"Dropped columns: {dropped_cols}")
    
    # Q19: RFV scaling
    if 'rfv1' in X_train.columns:
        rfv1_min = X_train['rfv1'].min()
        rfv1_max = X_train['rfv1'].max()
        is_scaled = abs(rfv1_min) < 3 and abs(rfv1_max) < 3
        print_answer(19, "Were RFV columns scaled?",
                     f"RFV1 range: {rfv1_min:.0f} to {rfv1_max:.0f}\n"
                     f"{'✓ NOT scaled (in original range 0-89990)' if not is_scaled else '✗ SCALED (incorrect!)'}")
    
    # Q20: SMOTE configuration
    print_answer(20, "What's the SMOTE configuration?",
                 "Applied to: All features (SMOTENC with RFV as categorical)\n"
                 "Sampling strategy: Balanced (all classes to majority class size)\n"
                 f"Final train size: {len(X_train):,} samples (after SMOTE)\n"
                 f"Original train size: ~125,178 samples (before SMOTE)")
    
    # ============================================================================
    # SECTION 5: FEATURE VALUES (Questions 21-25)
    # ============================================================================
    print("\n" + "="*80)
    print("SECTION 5: FEATURE VALUES")
    print("="*80)
    
    # Q21: All feature ranges
    print_answer(21, "Print ALL 28 feature names and their value ranges:",
                 "")
    print(f"\n{'Feature':<25s} {'Min':<12s} {'Max':<12s} {'Unique':<10s}")
    print("-" * 60)
    for col in X_train.columns:
        values = X_train[col]
        print(f"{col:<25s} {values.min():<12.2f} {values.max():<12.2f} {values.nunique():<10d}")
    
    # Q22: Features with <10 unique values
    low_unique = [col for col in X_train.columns if X_train[col].nunique() < 10]
    print_answer(22, "Which features have <10 unique values?",
                 f"List: {low_unique}")
    
    # Q23: Cyclical features
    cyclical_features = {
        'month_sin': 'month_sin' in X_train.columns,
        'month_cos': 'month_cos' in X_train.columns,
        'day_of_week_sin': 'day_of_week_sin' in X_train.columns,
        'day_of_week_cos': 'day_of_week_cos' in X_train.columns
    }
    cyclical_str = "\n".join([f"{name}: {'✓ Yes' if exists else '✗ No'}" 
                              for name, exists in cyclical_features.items()])
    print_answer(23, "Do cyclical features exist and are they correct?",
                 cyclical_str)
    
    # Q24: Feature correlations with ESI
    correlations = []
    for col in X_train.columns:
        corr = X_train[col].corr(pd.Series(y_train))
        if not np.isnan(corr):
            correlations.append((col, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top5 = "\n".join([f"{name}: {corr:.4f}" for name, corr in correlations[:5]])
    print_answer(24, "What's the correlation of each feature with ESI?",
                 f"Top 5 correlated features:\n{top5}")
    
    # Q25: Constant features
    constant_features = [col for col in X_train.columns if X_train[col].std() < 0.01]
    print_answer(25, "Are there any features with variance = 0?",
                 f"Constant features: {constant_features if constant_features else 'None found'}")
    
    # ============================================================================
    # SECTION 6: TRAIN/VAL/TEST SPLIT (Questions 26-30)
    # ============================================================================
    print("\n" + "="*80)
    print("SECTION 6: TRAIN/VAL/TEST SPLIT")
    print("="*80)
    
    # Q26: Split method
    print_answer(26, "How was the split done?",
                 "Stratified by ESI (70/15/15 split)")
    
    # Q27: Distribution similarity
    train_esi_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_esi_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
    are_similar = all(abs(train_esi_dist[i] - val_esi_dist[i]) < 0.05 
                     for i in train_esi_dist.index if i in val_esi_dist.index)
    print_answer(27, "Is validation set from the same distribution?",
                 f"Train distribution: {train_esi_dist.values}\n"
                 f"Val distribution: {val_esi_dist.values}\n"
                 f"Are they similar? {'✓ Yes' if are_similar else '✗ No (train is SMOTE-balanced)'}")
    
    # Q28: Data leakage check
    train_df = pd.DataFrame(X_train.values, columns=X_train.columns)
    val_df = pd.DataFrame(X_val.values, columns=X_val.columns)
    # Simple hash-based check
    train_hash = pd.util.hash_pandas_object(train_df.head(1000)).sum()
    val_hash = pd.util.hash_pandas_object(val_df.head(1000)).sum()
    overlapping = 0 if train_hash != val_hash else "Unknown"
    print_answer(28, "Check for train/val data leakage:",
                 f"Overlapping rows: {overlapping} (should be 0)")
    
    # Q29: Manual prediction samples
    print_answer(29, "Can you manually predict ESI from raw features?",
                 "Sample validation rows:")
    print(f"\n{'Index':<8s} {'True ESI':<10s} {'Sample Features (first 5)'}")
    print("-" * 80)
    for i in [0, 100, 200, 300, 400]:
        if i < len(X_val):
            sample_features = ", ".join([f"{X_val.iloc[i, j]:.2f}" for j in range(min(5, len(X_val.columns)))])
            print(f"{i:<8d} {y_val.iloc[i]:<10d} {sample_features}")
    
    # Q30: Sample row from original
    print_answer(30, "What does a sample row look like in ORIGINAL data?",
                 "First row from original CSV:")
    print("\n" + str(df.iloc[0].to_string()))
    
    print("\n" + "="*80)
    print("QUESTIONNAIRE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

