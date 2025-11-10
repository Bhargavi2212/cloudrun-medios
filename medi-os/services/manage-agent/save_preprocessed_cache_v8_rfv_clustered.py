"""
Save preprocessed data to cache with RFV clustering (v8).

This version uses RFV clustering to replace 723 RFV codes with 10-15 medical clusters.
"""

import sys
from pathlib import Path
import pickle

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

import pandas as pd
from ml.pipeline import TriagePreprocessingPipeline

def main():
    print("=" * 80)
    print("SAVING PREPROCESSED CACHE WITH RFV CLUSTERING (v8)")
    print("=" * 80)
    
    print("\n[Step 1] Loading raw data...")
    data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    df = pd.read_csv(data_path)
    print(f"  Raw data shape: {df.shape}")
    
    print("\n[Step 2] Running preprocessing pipeline with RFV clustering...")
    pipeline = TriagePreprocessingPipeline(
        random_state=42,
        use_rfv_clustering=True  # Enable RFV clustering
    )
    
    # fit_transform returns a tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(
        df, 
        target_col='esi_level',
        exclude_cols=['year']  # Exclude year column
    )
    
    print("\n[Step 3] Preparing cache data...")
    preprocessed_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': pipeline.feature_names_,
        'class_weights': pipeline.class_weights_
    }
    
    # Save to cache
    cache_dir = project_root / "services" / "manage-agent" / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "preprocessed_data_cache_v8_rfv_clustered.pkl"
    
    print(f"\n[Step 4] Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
    print(f"\n✅ Cache saved! Size: {file_size:.2f} MB")
    print(f"   Train: {len(preprocessed_data['X_train']):,} x {len(preprocessed_data['X_train'].columns)}")
    print(f"   Val: {len(preprocessed_data['X_val']):,} x {len(preprocessed_data['X_val'].columns)}")
    print(f"   Test: {len(preprocessed_data['X_test']):,} x {len(preprocessed_data['X_test'].columns)}")
    print(f"   Features: {len(preprocessed_data['feature_names'])}")
    print(f"\n   RFV codes have been clustered into medical domain categories")
    print(f"   Original 723 RFV codes → 10-15 clusters (one-hot encoded)")

if __name__ == "__main__":
    main()

