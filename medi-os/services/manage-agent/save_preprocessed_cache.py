"""
Save preprocessed data to cache for comparison analysis.
Run this once after preprocessing to avoid re-running the pipeline.
"""

import sys
from pathlib import Path
import pickle

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

import pandas as pd
from ml.pipeline import TriagePreprocessingPipeline

def main():
    print("Loading raw data...")
    data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")
    
    print("\nRunning preprocessing pipeline...")
    pipeline = TriagePreprocessingPipeline(random_state=42)
    # fit_transform returns a tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(
        df, 
        target_col='esi_level',
        exclude_cols=['year']  # Exclude year column
    )
    
    preprocessed_data = {
        'train': X_train,
        'val': X_val,
        'test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    # Save to cache
    cache_dir = project_root / "services" / "manage-agent" / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "preprocessed_data_cache.pkl"
    
    print(f"\nSaving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Cache saved! Size: {file_size:.2f} MB")
    print(f"   Train: {len(preprocessed_data['train']):,} x {len(preprocessed_data['train'].columns)}")
    print(f"   Val: {len(preprocessed_data['val']):,} x {len(preprocessed_data['val'].columns)}")
    print(f"   Test: {len(preprocessed_data['test']):,} x {len(preprocessed_data['test'].columns)}")

if __name__ == "__main__":
    main()

