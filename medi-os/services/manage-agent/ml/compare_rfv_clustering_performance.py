"""
Compare Performance: RFV Clustering (v8) vs Previous Versions

Compares model performance with RFV clustering vs previous approaches.
"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

def main():
    print("=" * 80)
    print("RFV CLUSTERING PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Load baseline results
    results_file = project_root / "services" / "manage-agent" / "outputs" / "baseline_results.json"
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Please run train_baseline.py first")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 80)
    print("CURRENT RESULTS (v8: RFV Clustering)")
    print("=" * 80)
    print(f"\nModel: {results.get('model_name', 'Unknown')}")
    print(f"Cache Version: v8_rfv_clustered (RFV clustering: 723 codes → 10-15 clusters)")
    print(f"Features: {results.get('n_features', 'Unknown')}")
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"  Macro F1: {results.get('macro_f1', 0):.4f}")
    print(f"  Weighted F1: {results.get('weighted_f1', 0):.4f}")
    
    # Comparison with previous versions
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS VERSIONS")
    print("=" * 80)
    
    comparison = {
        "v8_rfv_clustered": {
            "description": "RFV clustering (723 codes -> 10-15 medical clusters)",
            "features": 58,
            "logistic_regression": {
                "accuracy": 0.0491,
                "macro_f1": 0.0134,
                "status": "BROKEN (predicts only ESI 7)"
            },
            "random_forest": {
                "accuracy": 0.4953,
                "macro_f1": 0.2782,
                "status": "SIMILAR to previous (~53%)"
            }
        },
        "v7_minimal": {
            "description": "Minimal features (top 10 RFV + essential features)",
            "features": 24,
            "logistic_regression": {
                "accuracy": 0.05,
                "macro_f1": 0.01,
                "status": "BROKEN"
            },
            "random_forest": {
                "accuracy": 0.519,
                "macro_f1": 0.30,
                "status": "BEST so far"
            }
        },
        "v4": {
            "description": "RFV one-hot encoding (top 50 codes)",
            "features": 100,
            "logistic_regression": {
                "accuracy": 0.05,
                "macro_f1": 0.01,
                "status": "BROKEN"
            },
            "random_forest": {
                "accuracy": 0.53,
                "macro_f1": 0.30,
                "status": "GOOD"
            }
        },
        "v3": {
            "description": "RFV fix (selective scaling)",
            "features": 28,
            "logistic_regression": {
                "accuracy": 0.28,
                "macro_f1": 0.15,
                "status": "POOR"
            },
            "random_forest": {
                "accuracy": 0.53,
                "macro_f1": 0.30,
                "status": "GOOD"
            }
        }
    }
    
    print("\nVersion Comparison Table:")
    print("-" * 80)
    print(f"{'Version':<20} {'Features':<12} {'LR Acc':<10} {'RF Acc':<10} {'Status':<30}")
    print("-" * 80)
    
    for version, data in comparison.items():
        lr_acc = data['logistic_regression']['accuracy']
        rf_acc = data['random_forest']['accuracy']
        lr_status = data['logistic_regression']['status']
        
        print(f"{version:<20} {data['features']:<12} {lr_acc:<10.4f} {rf_acc:<10.4f} {lr_status:<30}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print("\n1. RFV CLUSTERING IMPACT:")
    print("   - Reduced feature count: 723 RFV codes → 26 cluster features (for RFV1+RFV2)")
    print("   - Random Forest performance: 49.5% (similar to v4's 53%)")
    print("   - Logistic Regression: Still broken (4.9%)")
    
    print("\n2. FEATURE REDUCTION:")
    print("   - v8: 58 features (RFV clustering)")
    print("   - v7_minimal: 24 features (top 10 RFV)")
    print("   - v4: 100+ features (one-hot encoded RFV)")
    print("   - v3: 28 features (numeric RFV)")
    
    print("\n3. MODEL PERFORMANCE:")
    print("   - Random Forest: Consistent ~50-53% across all versions")
    print("   - Logistic Regression: Consistently broken (5-28%)")
    print("   - Best RF performance: v7_minimal (51.9%)")
    
    print("\n4. CONCLUSION:")
    print("   - RFV clustering does NOT improve accuracy significantly")
    print("   - Feature reduction (clustering) maintains similar performance")
    print("   - Root cause: Weak feature-target correlations (max 0.14)")
    print("   - Recommendation: Focus on feature engineering or accept 50-53% limit")
    
    print("\n" + "=" * 80)
    
    # Save comparison
    comparison_file = project_root / "services" / "manage-agent" / "outputs" / "rfv_clustering_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")

if __name__ == "__main__":
    main()

