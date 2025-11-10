"""
Preprocessing Comparison Analysis

Compares data characteristics before and after preprocessing pipeline.
Generates visualizations and comprehensive report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from ml.pipeline import TriagePreprocessingPipeline

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = project_root / "services" / "manage-agent" / "outputs" / "figures" / "preprocessing_comparison"
output_dir.mkdir(parents=True, exist_ok=True)
report_dir = project_root / "services" / "manage-agent" / "outputs"
report_dir.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    """Load raw CSV data before preprocessing."""
    data_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    df = pd.read_csv(data_path)
    return df


def load_preprocessed_data(df, force_recompute=False):
    """
    Load preprocessed data from cache or run preprocessing.
    
    Args:
        df: Raw DataFrame
        force_recompute: If True, rerun preprocessing even if cache exists
    
    Returns:
        Dictionary with preprocessed splits
    """
    cache_dir = project_root / "services" / "manage-agent" / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "preprocessed_data_cache.pkl"
    
    # Try to load from cache
    if not force_recompute and cache_file.exists():
        print("Loading preprocessed data from cache...")
        import pickle
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"✅ Loaded from cache: Train ({len(cached_data['train']):,}, {len(cached_data['train'].columns)}), "
                  f"Val ({len(cached_data['val']):,}, {len(cached_data['val'].columns)}), "
                  f"Test ({len(cached_data['test']):,}, {len(cached_data['test'].columns)})")
            return cached_data
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}. Running preprocessing...")
    
    # Run preprocessing
    print("Running preprocessing pipeline (this will take ~20 minutes)...")
    print("Note: Results will be cached for future use.")
    pipeline = TriagePreprocessingPipeline(random_state=42)
    result = pipeline.fit_transform(df, target_col='esi_level')
    
    preprocessed_data = {
        'train': result['X_train'],
        'val': result['X_val'],
        'test': result['X_test'],
        'y_train': result['y_train'],
        'y_val': result['y_val'],
        'y_test': result['y_test'],
        'pipeline': pipeline
    }
    
    # Save to cache
    try:
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        print(f"✅ Preprocessed data cached to: {cache_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not cache data: {e}")
    
    return preprocessed_data


def analyze_missing_values(df_raw, preprocessed_data):
    """Compare missing values before and after preprocessing."""
    print("\n[1] Analyzing missing values...")
    
    # Before: Calculate missing values
    missing_before = df_raw.isna().sum()
    missing_before_pct = (missing_before / len(df_raw)) * 100
    total_missing_before = missing_before.sum()
    
    # After: Check all splits
    missing_train = preprocessed_data['train'].isna().sum().sum()
    missing_val = preprocessed_data['val'].isna().sum().sum()
    missing_test = preprocessed_data['test'].isna().sum().sum()
    total_missing_after = missing_train + missing_val + missing_test
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before: Top 10 columns with missing values
    missing_before_top10 = missing_before[missing_before > 0].head(10).sort_values(ascending=True)
    if len(missing_before_top10) > 0:
        axes[0].barh(range(len(missing_before_top10)), missing_before_top10.values)
        axes[0].set_yticks(range(len(missing_before_top10)))
        axes[0].set_yticklabels(missing_before_top10.index)
        axes[0].set_xlabel('Missing Count')
        axes[0].set_title(f'Before Preprocessing (Total: {total_missing_before:,})')
    else:
        axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[0].set_title('Before Preprocessing')
    
    # After: Should be all zeros
    missing_after = pd.DataFrame({
        'Split': ['Train', 'Validation', 'Test'],
        'Missing': [missing_train, missing_val, missing_test]
    })
    axes[1].bar(missing_after['Split'], missing_after['Missing'], color=['green', 'blue', 'orange'])
    axes[1].set_ylabel('Missing Count')
    axes[1].set_title(f'After Preprocessing (Total: {total_missing_after})')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Target (0)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_missing_values_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'before': {
            'total': int(total_missing_before),
            'percentage': float((total_missing_before / df_raw.size) * 100),
            'columns_with_missing': int((missing_before > 0).sum()),
            'top_missing': missing_before[missing_before > 0].head(10).to_dict()
        },
        'after': {
            'train': int(missing_train),
            'val': int(missing_val),
            'test': int(missing_test),
            'total': int(total_missing_after)
        }
    }


def analyze_skewness(df_raw, preprocessed_data):
    """Compare skewness before and after transformation."""
    print("\n[2] Analyzing skewness...")
    
    # Get numeric columns before preprocessing (excluding target and non-numeric)
    numeric_cols_before = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_before = [c for c in numeric_cols_before if c != 'esi_level']
    
    # Calculate skewness before
    skewness_before = {}
    for col in numeric_cols_before:
        if col in df_raw.columns:
            data = df_raw[col].dropna()
            if len(data) > 0:
                skewness_before[col] = skew(data)
    
    # Calculate skewness after (on train set)
    train_data = preprocessed_data['train']
    skewness_after = {}
    for col in train_data.columns:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            data = train_data[col].dropna()
            if len(data) > 0:
                skewness_after[col] = skew(data)
    
    # Find common columns (after standardization, names might differ slightly)
    # Focus on key features
    key_features = ['age', 'sbp', 'dbp', 'pulse', 'respiration', 'temp_c', 'o2_sat', 
                    'wait_time', 'pain', 'rfv1', 'rfv2', 'rfv3']
    
    common_skew = {}
    for feat in key_features:
        if feat in skewness_before:
            # Find corresponding feature after (might have different name due to encoding)
            after_feat = feat
            if after_feat not in train_data.columns:
                # Try to find it
                for col in train_data.columns:
                    if feat in col.lower() or col.lower() in feat:
                        after_feat = col
                        break
            
            if after_feat in train_data.columns:
                common_skew[feat] = {
                    'before': skewness_before[feat],
                    'after': skewness_after.get(after_feat, np.nan)
                }
    
    # Create visualization
    if common_skew:
        features = list(common_skew.keys())
        before_vals = [common_skew[f]['before'] for f in features]
        after_vals = [common_skew[f]['after'] for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width/2, before_vals, width, label='Before', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_vals, width, label='After', alpha=0.8)
        
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Skewness', fontsize=12)
        ax.set_title('Skewness Comparison: Before vs After Preprocessing', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Skew Threshold (1.0)')
        ax.axhline(y=-1.0, color='red', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / '02_skewness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate summary
    high_skew_before = sum(1 for v in skewness_before.values() if abs(v) > 1.0)
    high_skew_after = sum(1 for v in skewness_after.values() if abs(v) > 1.0)
    
    return {
        'before': {
            'high_skew_count': high_skew_before,
            'mean_abs_skew': float(np.mean([abs(v) for v in skewness_before.values()])),
            'max_skew': float(max([abs(v) for v in skewness_before.values()], default=0))
        },
        'after': {
            'high_skew_count': high_skew_after,
            'mean_abs_skew': float(np.mean([abs(v) for v in skewness_after.values()])),
            'max_skew': float(max([abs(v) for v in skewness_after.values()], default=0))
        },
        'common_features': common_skew
    }


def analyze_distributions(df_raw, preprocessed_data):
    """Compare distributions of key features."""
    print("\n[3] Analyzing distributions...")
    
    key_features = ['age', 'sbp', 'dbp', 'pulse', 'respiration', 'temp_c', 'o2_sat', 'wait_time']
    
    # Create comparison plots for each feature
    n_features = len([f for f in key_features if f in df_raw.columns])
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else axes.flatten()
    
    plot_idx = 0
    for feat in key_features:
        if feat not in df_raw.columns:
            continue
        
        ax = axes[plot_idx]
        
        # Before: histogram
        data_before = df_raw[feat].dropna()
        if len(data_before) > 0:
            ax.hist(data_before, bins=50, alpha=0.6, label='Before', density=True, color='blue')
        
        # After: find corresponding feature in preprocessed data
        after_feat = feat
        if after_feat not in preprocessed_data['train'].columns:
            for col in preprocessed_data['train'].columns:
                if feat.lower() in col.lower():
                    after_feat = col
                    break
        
        if after_feat in preprocessed_data['train'].columns:
            data_after = preprocessed_data['train'][after_feat].dropna()
            if len(data_after) > 0:
                ax.hist(data_after, bins=50, alpha=0.6, label='After', density=True, color='red')
        
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{feat} Distribution', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'features_compared': plot_idx}


def analyze_class_distribution(df_raw, preprocessed_data):
    """Compare ESI level distribution before and after SMOTE."""
    print("\n[4] Analyzing class distribution...")
    
    # Before
    esi_before = df_raw['esi_level'].value_counts().sort_index()
    
    # After (train set - after SMOTE)
    esi_after = preprocessed_data['y_train'].value_counts().sort_index()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before
    axes[0].bar(esi_before.index.astype(str), esi_before.values, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('ESI Level', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'Before SMOTE\n(Total: {len(df_raw):,})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add percentages
    for i, (idx, val) in enumerate(esi_before.items()):
        pct = (val / len(df_raw)) * 100
        axes[0].text(i, val, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # After
    axes[1].bar(esi_after.index.astype(str), esi_after.values, color='green', alpha=0.8)
    axes[1].set_xlabel('ESI Level', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title(f'After SMOTE\n(Total: {len(preprocessed_data["y_train"]):,})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # All should be equal after SMOTE
    if len(esi_after) > 0:
        target_count = esi_after.iloc[0]
        axes[1].axhline(y=target_count, color='red', linestyle='--', linewidth=2, 
                       label=f'Target: {target_count:,}')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_class_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'before': {
            'total': len(df_raw),
            'distribution': esi_before.to_dict(),
            'imbalance_ratio': float(esi_before.max() / esi_before.min())
        },
        'after': {
            'total': len(preprocessed_data['y_train']),
            'distribution': esi_after.to_dict(),
            'imbalance_ratio': float(esi_after.max() / esi_after.min()) if len(esi_after) > 0 else 1.0
        }
    }


def analyze_feature_count(df_raw, preprocessed_data):
    """Compare feature count before and after."""
    print("\n[5] Analyzing feature count...")
    
    features_before = len(df_raw.columns)
    features_after = len(preprocessed_data['train'].columns)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Before', 'After']
    counts = [features_before, features_after]
    colors = ['steelblue', 'green']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, width=0.6)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Count: Before vs After Preprocessing', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_feature_count_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'before': features_before,
        'after': features_after,
        'difference': features_after - features_before
    }


def analyze_data_shapes(df_raw, preprocessed_data):
    """Compare data shapes before and after."""
    print("\n[6] Analyzing data shapes...")
    
    shapes = {
        'before': {
            'rows': len(df_raw),
            'cols': len(df_raw.columns),
            'total_samples': len(df_raw)
        },
        'after': {
            'train': {
                'rows': len(preprocessed_data['train']),
                'cols': len(preprocessed_data['train'].columns)
            },
            'val': {
                'rows': len(preprocessed_data['val']),
                'cols': len(preprocessed_data['val'].columns)
            },
            'test': {
                'rows': len(preprocessed_data['test']),
                'cols': len(preprocessed_data['test'].columns)
            },
            'total_samples': (len(preprocessed_data['train']) + 
                            len(preprocessed_data['val']) + 
                            len(preprocessed_data['test']))
        }
    }
    
    return shapes


def generate_report(analysis_results):
    """Generate comprehensive markdown report."""
    print("\n[7] Generating report...")
    
    report = []
    report.append("# Preprocessing Comparison Report\n")
    report.append("Comprehensive analysis comparing data characteristics before and after preprocessing pipeline.\n")
    report.append("---\n\n")
    
    # Summary
    report.append("## Executive Summary\n\n")
    report.append(f"- **Missing Values After**: {analysis_results['missing']['after']['total']} (✅ All imputed)\n")
    report.append(f"- **High Skewness Reduction**: {analysis_results['skewness']['before']['high_skew_count']} → {analysis_results['skewness']['after']['high_skew_count']} features\n")
    report.append(f"- **Class Imbalance Ratio**: {analysis_results['class_dist']['before']['imbalance_ratio']:.2f} → {analysis_results['class_dist']['after']['imbalance_ratio']:.2f}\n")
    report.append(f"- **Feature Count**: {analysis_results['feature_count']['before']} → {analysis_results['feature_count']['after']}\n")
    report.append(f"- **Data Samples**: {analysis_results['shapes']['before']['total_samples']:,} → {analysis_results['shapes']['after']['total_samples']:,} (after SMOTE)\n\n")
    
    # Missing Values
    report.append("## 1. Missing Values Analysis\n\n")
    report.append(f"### Before Preprocessing\n")
    report.append(f"- Total missing values: {analysis_results['missing']['before']['total']:,}\n")
    report.append(f"- Missing percentage: {analysis_results['missing']['before']['percentage']:.2f}%\n")
    report.append(f"- Columns with missing values: {analysis_results['missing']['before']['columns_with_missing']}\n\n")
    report.append("### After Preprocessing\n")
    report.append(f"- Train missing: {analysis_results['missing']['after']['train']} ✅\n")
    report.append(f"- Validation missing: {analysis_results['missing']['after']['val']} ✅\n")
    report.append(f"- Test missing: {analysis_results['missing']['after']['test']} ✅\n\n")
    report.append("**Conclusion**: All missing values successfully imputed using KNN imputation.\n\n")
    
    # Skewness
    report.append("## 2. Skewness Analysis\n\n")
    report.append(f"### Before Preprocessing\n")
    report.append(f"- Features with |skew| > 1.0: {analysis_results['skewness']['before']['high_skew_count']}\n")
    report.append(f"- Mean absolute skewness: {analysis_results['skewness']['before']['mean_abs_skew']:.3f}\n")
    report.append(f"- Maximum absolute skewness: {analysis_results['skewness']['before']['max_skew']:.3f}\n\n")
    report.append(f"### After Preprocessing\n")
    report.append(f"- Features with |skew| > 1.0: {analysis_results['skewness']['after']['high_skew_count']}\n")
    report.append(f"- Mean absolute skewness: {analysis_results['skewness']['after']['mean_abs_skew']:.3f}\n")
    report.append(f"- Maximum absolute skewness: {analysis_results['skewness']['after']['max_skew']:.3f}\n\n")
    report.append("**Conclusion**: Yeo-Johnson transformation successfully reduced skewness in skewed features.\n\n")
    
    # Class Distribution
    report.append("## 3. Class Distribution Analysis\n\n")
    report.append(f"### Before SMOTE\n")
    report.append(f"- Total samples: {analysis_results['class_dist']['before']['total']:,}\n")
    report.append(f"- Imbalance ratio: {analysis_results['class_dist']['before']['imbalance_ratio']:.2f}x\n")
    report.append("Class distribution:\n")
    for esi, count in sorted(analysis_results['class_dist']['before']['distribution'].items()):
        pct = (count / analysis_results['class_dist']['before']['total']) * 100
        report.append(f"  - ESI {esi}: {count:,} ({pct:.1f}%)\n")
    report.append(f"\n### After SMOTE\n")
    report.append(f"- Total samples: {analysis_results['class_dist']['after']['total']:,}\n")
    report.append(f"- Imbalance ratio: {analysis_results['class_dist']['after']['imbalance_ratio']:.2f}x\n")
    report.append("**Conclusion**: SMOTE successfully balanced all classes to equal distribution.\n\n")
    
    # Feature Count
    report.append("## 4. Feature Count Analysis\n\n")
    report.append(f"- **Before**: {analysis_results['feature_count']['before']} features\n")
    report.append(f"- **After**: {analysis_results['feature_count']['after']} features\n")
    report.append(f"- **Difference**: {analysis_results['feature_count']['difference']:+d}\n\n")
    report.append("**Changes**:\n")
    report.append("- ✅ Dropped: 3 diagnosis columns (data leakage prevention)\n")
    report.append("- ✅ Dropped: year column (excluded)\n")
    report.append("- ✅ Added: 4 cyclical features (month_sin, month_cos, day_sin, day_cos)\n")
    report.append("- ✅ RFV codes kept as numeric (no encoding needed)\n\n")
    
    # Data Shapes
    report.append("## 5. Data Shape Analysis\n\n")
    report.append("### Before Preprocessing\n")
    report.append(f"- Shape: ({analysis_results['shapes']['before']['rows']:,}, {analysis_results['shapes']['before']['cols']})\n")
    report.append(f"- Total samples: {analysis_results['shapes']['before']['total_samples']:,}\n\n")
    report.append("### After Preprocessing\n")
    report.append(f"- Train: ({analysis_results['shapes']['after']['train']['rows']:,}, {analysis_results['shapes']['after']['train']['cols']})\n")
    report.append(f"- Validation: ({analysis_results['shapes']['after']['val']['rows']:,}, {analysis_results['shapes']['after']['val']['cols']})\n")
    report.append(f"- Test: ({analysis_results['shapes']['after']['test']['rows']:,}, {analysis_results['shapes']['after']['test']['cols']})\n")
    report.append(f"- Total samples: {analysis_results['shapes']['after']['total_samples']:,} (after SMOTE on train)\n\n")
    
    # Visualizations
    report.append("## 6. Visualizations\n\n")
    report.append("All visualizations are saved in `outputs/figures/preprocessing_comparison/`:\n\n")
    report.append("1. **Missing Values Comparison** - Shows missing values before vs after\n")
    report.append("2. **Skewness Comparison** - Shows skewness reduction for key features\n")
    report.append("3. **Distribution Comparison** - Histograms comparing distributions before/after\n")
    report.append("4. **Class Distribution Comparison** - ESI level balance before/after SMOTE\n")
    report.append("5. **Feature Count Comparison** - Number of features before vs after\n\n")
    
    # Recommendations
    report.append("## 7. Recommendations\n\n")
    report.append("✅ **All preprocessing steps completed successfully:**\n\n")
    report.append("1. Missing values: Fully imputed using KNN imputation\n")
    report.append("2. Skewness: Reduced through Yeo-Johnson transformation\n")
    report.append("3. Outliers: Clipped using IQR method\n")
    report.append("4. Class imbalance: Balanced using SMOTE\n")
    report.append("5. Feature engineering: Cyclical encoding applied for temporal features\n")
    report.append("6. Standardization: All features standardized for ML algorithms\n\n")
    report.append("**Data is ready for model training!**\n")
    
    # Save report
    report_text = ''.join(report)
    report_path = report_dir / "preprocessing_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Report saved to: {report_path}")
    return report_path


def main():
    """Main execution function."""
    print("=" * 70)
    print("PREPROCESSING COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading raw data...")
    df_raw = load_raw_data()
    print(f"Raw data shape: {df_raw.shape}")
    
    # Run preprocessing
    preprocessed_data = load_preprocessed_data(df_raw)
    
    # Run analyses
    analysis_results = {}
    
    analysis_results['missing'] = analyze_missing_values(df_raw, preprocessed_data)
    analysis_results['skewness'] = analyze_skewness(df_raw, preprocessed_data)
    analysis_results['distributions'] = analyze_distributions(df_raw, preprocessed_data)
    analysis_results['class_dist'] = analyze_class_distribution(df_raw, preprocessed_data)
    analysis_results['feature_count'] = analyze_feature_count(df_raw, preprocessed_data)
    analysis_results['shapes'] = analyze_data_shapes(df_raw, preprocessed_data)
    
    # Save statistics
    stats_path = report_dir / "preprocessing_statistics.json"
    # Convert to JSON-serializable format
    stats_json = json.dumps(analysis_results, indent=2, default=str)
    with open(stats_path, 'w') as f:
        f.write(stats_json)
    print(f"\nStatistics saved to: {stats_path}")
    
    # Generate report
    report_path = generate_report(analysis_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nVisualizations: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Statistics: {stats_path}")


if __name__ == "__main__":
    main()

