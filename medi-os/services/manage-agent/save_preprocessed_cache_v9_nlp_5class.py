"""
Save preprocessed data to cache with NLP embeddings + 5-class severity (v9).

Stage 1: RFV sentence embeddings (semantic embeddings from text labels)
Stage 2: 5-class severity mapping (7 ESI classes → 5 severity levels)
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
    print("SAVING PREPROCESSED CACHE WITH NLP EMBEDDINGS + 5-CLASS SEVERITY (v9)")
    print("=" * 80)
    
    print("\n[Step 1] Loading raw data...")
    data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    df = pd.read_csv(data_path)
    print(f"  Raw data shape: {df.shape}")
    
    print("\n[Step 2] Running preprocessing pipeline with:")
    print("  - Stage 1: RFV sentence embeddings (semantic embeddings)")
    print("  - Stage 2: 5-class severity mapping (7 ESI → 5 severity)")
    
    pipeline = TriagePreprocessingPipeline(
        random_state=42,
        use_rfv_embeddings=True,      # Stage 1: Sentence embeddings
        embedding_model="all-MiniLM-L6-v2",  # Fast, 384 dims
        embedding_pca=True,           # Reduce to 20 dims
        embedding_pca_components=20,
        use_5class_severity=True      # Stage 2: 5-class mapping
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
        'class_weights': pipeline.class_weights_,
        'version': 'v9_nlp_5class',
        'description': 'NLP embeddings + 5-class severity mapping'
    }
    
    # Save to cache
    cache_dir = project_root / "services" / "manage-agent" / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "preprocessed_data_cache_v9_nlp_5class.pkl"
    
    print(f"\n[Step 4] Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
    print(f"\n✅ Cache saved! Size: {file_size:.2f} MB")
    print(f"   Train: {len(preprocessed_data['X_train']):,} x {len(preprocessed_data['X_train'].columns)}")
    print(f"   Val: {len(preprocessed_data['X_val']):,} x {len(preprocessed_data['X_val'].columns)}")
    print(f"   Test: {len(preprocessed_data['X_test']):,} x {len(preprocessed_data['X_test'].columns)}")
    print(f"   Features: {len(preprocessed_data['feature_names'])}")
    print(f"\n   Stage 1: RFV codes converted to sentence embeddings (20 dims per RFV)")
    print(f"   Stage 2: ESI levels mapped to 5-class severity (1-5)")

if __name__ == "__main__":
    main()

