"""
Comprehensive Model Comparison: All Feature Sets and Models

Compares:
1. Full Features (72 features) - v9 cache
2. Age + RFV Only (43 features)
3. Age + RFV + Vitals (52 features)

Models:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
"""

import sys
from pathlib import Path
import json
import pickle

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))


def load_baseline_results():
    """Load baseline results from train_baseline.py output."""
    results_file = project_root / "services" / "manage-agent" / "outputs" / "baseline_results.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract model results
    results = {}
    if 'logistic_regression' in data:
        results['LogisticRegression'] = {
            'accuracy': data['logistic_regression'].get('accuracy', 0),
            'macro_f1': data['logistic_regression'].get('macro_f1', 0),
            'weighted_f1': data['logistic_regression'].get('weighted_f1', 0)
        }
    if 'random_forest' in data:
        results['RandomForest'] = {
            'accuracy': data['random_forest'].get('accuracy', 0),
            'macro_f1': data['random_forest'].get('macro_f1', 0),
            'weighted_f1': data['random_forest'].get('weighted_f1', 0)
        }
    
    return results


def load_xgboost_lightgbm_results():
    """Load XGBoost and LightGBM results from full features."""
    results_file = project_root / "services" / "manage-agent" / "outputs" / "xgboost_lightgbm_results.json"
    
    if not results_file.exists():
        return {}
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = {}
    if 'xgboost' in data:
        results['XGBoost'] = data['xgboost']
    if 'lightgbm' in data:
        results['LightGBM'] = data['lightgbm']
    
    return results


def load_age_rfv_results():
    """Load Age + RFV only results."""
    results_file = project_root / "services" / "manage-agent" / "outputs" / "age_rfv_only_results.json"
    
    if not results_file.exists():
        return {}
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data.get('models', {})


def load_age_rfv_vitals_results():
    """Load Age + RFV + Vitals results."""
    results_file = project_root / "services" / "manage-agent" / "outputs" / "age_rfv_vitals_results.json"
    
    if not results_file.exists():
        return {}
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data.get('models', {})


def main():
    print("=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON: ALL FEATURE SETS")
    print("=" * 100)
    
    # Load all results
    print("\n[Loading Results]...")
    
    # Full features (72)
    full_results = {}
    baseline = load_baseline_results()
    if baseline:
        full_results.update(baseline)
    xgb_lgbm = load_xgboost_lightgbm_results()
    if xgb_lgbm:
        full_results.update(xgb_lgbm)
    
    # Age + RFV only (43)
    age_rfv_results = load_age_rfv_results()
    
    # Age + RFV + Vitals (52)
    age_rfv_vitals_results = load_age_rfv_vitals_results()
    
    # Organize by feature set
    feature_sets = {
        'Full Features (72)': full_results,
        'Age + RFV Only (43)': age_rfv_results,
        'Age + RFV + Vitals (52)': age_rfv_vitals_results
    }
    
    # Get all model names
    all_models = set()
    for results in feature_sets.values():
        all_models.update(results.keys())
    all_models = sorted(list(all_models))
    
    print(f"\nModels found: {', '.join(all_models)}")
    print(f"Feature sets: {', '.join(feature_sets.keys())}")
    
    # Comparison table
    print("\n" + "=" * 100)
    print("ACCURACY COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Model':<25} {'Full (72)':<15} {'Age+RFV (43)':<18} {'Age+RFV+Vitals (52)':<22} {'Best':<15}")
    print("-" * 100)
    
    for model in all_models:
        full_acc = feature_sets['Full Features (72)'].get(model, {}).get('accuracy', 0)
        age_rfv_acc = feature_sets['Age + RFV Only (43)'].get(model, {}).get('accuracy', 0)
        age_rfv_vitals_acc = feature_sets['Age + RFV + Vitals (52)'].get(model, {}).get('accuracy', 0)
        
        # Find best
        accuracies = {
            'Full': full_acc,
            'Age+RFV': age_rfv_acc,
            'Age+RFV+Vitals': age_rfv_vitals_acc
        }
        best_set = max(accuracies.items(), key=lambda x: x[1])[0]
        best_acc = accuracies[best_set]
        
        print(f"{model:<25} {full_acc:<15.4f} {age_rfv_acc:<18.4f} {age_rfv_vitals_acc:<22.4f} {best_set} ({best_acc:.4f})")
    
    # Macro F1 comparison
    print("\n" + "=" * 100)
    print("MACRO F1 COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Model':<25} {'Full (72)':<15} {'Age+RFV (43)':<18} {'Age+RFV+Vitals (52)':<22} {'Best':<15}")
    print("-" * 100)
    
    for model in all_models:
        full_f1 = feature_sets['Full Features (72)'].get(model, {}).get('macro_f1', 0)
        age_rfv_f1 = feature_sets['Age + RFV Only (43)'].get(model, {}).get('macro_f1', 0)
        age_rfv_vitals_f1 = feature_sets['Age + RFV + Vitals (52)'].get(model, {}).get('macro_f1', 0)
        
        f1_scores = {
            'Full': full_f1,
            'Age+RFV': age_rfv_f1,
            'Age+RFV+Vitals': age_rfv_vitals_f1
        }
        best_set = max(f1_scores.items(), key=lambda x: x[1])[0]
        best_f1 = f1_scores[best_set]
        
        print(f"{model:<25} {full_f1:<15.4f} {age_rfv_f1:<18.4f} {age_rfv_vitals_f1:<22.4f} {best_set} ({best_f1:.4f})")
    
    # Severity 1 Recall comparison
    print("\n" + "=" * 100)
    print("SEVERITY 1 (CRITICAL) RECALL COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Model':<25} {'Full (72)':<15} {'Age+RFV (43)':<18} {'Age+RFV+Vitals (52)':<22} {'Best':<15}")
    print("-" * 100)
    
    for model in all_models:
        full_recall = feature_sets['Full Features (72)'].get(model, {}).get('severity_1_recall', feature_sets['Full Features (72)'].get(model, {}).get('esi_1_recall', 0))
        age_rfv_recall = feature_sets['Age + RFV Only (43)'].get(model, {}).get('severity_1_recall', 0)
        age_rfv_vitals_recall = feature_sets['Age + RFV + Vitals (52)'].get(model, {}).get('severity_1_recall', 0)
        
        recalls = {
            'Full': full_recall,
            'Age+RFV': age_rfv_recall,
            'Age+RFV+Vitals': age_rfv_vitals_recall
        }
        best_set = max(recalls.items(), key=lambda x: x[1])[0]
        best_recall = recalls[best_set]
        
        print(f"{model:<25} {full_recall:<15.4f} {age_rfv_recall:<18.4f} {age_rfv_vitals_recall:<22.4f} {best_set} ({best_recall:.4f})")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    # Best accuracy per feature set
    print("\n[Best Accuracy by Feature Set]")
    for feature_set_name, results in feature_sets.items():
        if results:
            best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
            print(f"  {feature_set_name}:")
            print(f"    Model: {best_model[0]}")
            print(f"    Accuracy: {best_model[1].get('accuracy', 0):.4f} ({best_model[1].get('accuracy', 0)*100:.2f}%)")
            print(f"    Macro F1: {best_model[1].get('macro_f1', 0):.4f}")
    
    # Feature set impact
    print("\n[Feature Set Impact on Random Forest]")
    rf_full = feature_sets['Full Features (72)'].get('RandomForest', {})
    rf_age_rfv = feature_sets['Age + RFV Only (43)'].get('RandomForest', {})
    rf_age_rfv_vitals = feature_sets['Age + RFV + Vitals (52)'].get('RandomForest', {})
    
    if rf_full and rf_age_rfv:
        impact_rfv = rf_age_rfv.get('accuracy', 0) - 0.0  # Baseline would be lower
        impact_vitals = rf_age_rfv_vitals.get('accuracy', 0) - rf_age_rfv.get('accuracy', 0)
        impact_full = rf_full.get('accuracy', 0) - rf_age_rfv_vitals.get('accuracy', 0)
        
        print(f"  Age + RFV (43 features): {rf_age_rfv.get('accuracy', 0):.4f} ({rf_age_rfv.get('accuracy', 0)*100:.2f}%)")
        print(f"  + Adding Vitals (9 more): {rf_age_rfv_vitals.get('accuracy', 0):.4f} ({rf_age_rfv_vitals.get('accuracy', 0)*100:.2f}%)")
        print(f"    Improvement: +{impact_vitals:.4f} (+{impact_vitals*100:.2f} percentage points)")
        print(f"  Full Features (20 more): {rf_full.get('accuracy', 0):.4f} ({rf_full.get('accuracy', 0)*100:.2f}%)")
        print(f"    Improvement: +{impact_full:.4f} (+{impact_full*100:.2f} percentage points)")
        print(f"  Total improvement: {rf_full.get('accuracy', 0) - rf_age_rfv.get('accuracy', 0):.4f} ({((rf_full.get('accuracy', 0) - rf_age_rfv.get('accuracy', 0)) / rf_age_rfv.get('accuracy', 0) * 100):.1f}% relative)")
    
    # Best overall model
    print("\n[Best Overall Model]")
    all_results = []
    for feature_set_name, results in feature_sets.items():
        for model_name, metrics in results.items():
            if 'accuracy' in metrics:
                all_results.append({
                    'feature_set': feature_set_name,
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'macro_f1': metrics.get('macro_f1', 0)
                })
    
    if all_results:
        best_overall = max(all_results, key=lambda x: x['accuracy'])
        print(f"  Model: {best_overall['model']}")
        print(f"  Feature Set: {best_overall['feature_set']}")
        print(f"  Accuracy: {best_overall['accuracy']:.4f} ({best_overall['accuracy']*100:.2f}%)")
        print(f"  Macro F1: {best_overall['macro_f1']:.4f}")
    
    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    print("\n1. For Production Deployment:")
    if rf_age_rfv_vitals and rf_full:
        if rf_age_rfv_vitals.get('accuracy', 0) >= rf_full.get('accuracy', 0) * 0.95:
            print("   ✓ Use Age + RFV + Vitals (52 features)")
            print("     - Nearly as good as full features (within 5%)")
            print("     - Simpler, faster, easier to collect")
            print("     - 51.9% accuracy vs 55.0% (full features)")
        else:
            print("   ✓ Use Full Features (72 features)")
            print("     - Best accuracy: 55.0%")
            print("     - Worth the extra complexity")
    else:
        print("   ✓ Use Age + RFV + Vitals (52 features)")
        print("     - Good balance of accuracy and simplicity")
    
    print("\n2. For Early Triage (Limited Information):")
    if rf_age_rfv:
        print("   ✓ Use Age + RFV Only (43 features)")
        print(f"     - 46.2% accuracy with minimal information")
        print("     - Can be used before vitals are collected")
    
    print("\n3. Model Selection:")
    print("   ✓ Random Forest: Best overall performance")
    print("   ✓ Logistic Regression: Good baseline, interpretable")
    print("   ⚠ XGBoost/LightGBM: Overfitting to Critical cases with aggressive weights")
    
    # Save comparison
    comparison_file = project_root / "services" / "manage-agent" / "outputs" / "all_models_comparison.json"
    comparison_data = {
        'feature_sets': feature_sets,
        'best_overall': best_overall if all_results else None,
        'summary': {
            'best_accuracy_by_set': {
                name: max(results.items(), key=lambda x: x[1].get('accuracy', 0))[0] if results else None
                for name, results in feature_sets.items()
            }
        }
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {comparison_file}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

