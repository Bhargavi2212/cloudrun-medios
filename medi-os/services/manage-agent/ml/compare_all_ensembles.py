"""
Final Comparison: Compare all three ensemble approaches
Load results from Phase 1, 2, and 3, and generate comprehensive comparison.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

# Setup directories
outputs_dir = project_root / "services" / "manage-agent" / "outputs"
outputs_dir.mkdir(exist_ok=True)


def load_results():
    """Load results from all three phases."""
    print("\n[1] Loading results from all phases...")
    
    results = {}
    
    # Phase 1: Weighted Voting
    phase1_path = outputs_dir / "weighted_voting_results.json"
    if phase1_path.exists():
        with open(phase1_path, 'r') as f:
            phase1_data = json.load(f)
        results['Weighted Voting'] = phase1_data
        print(f"  Loaded Phase 1: Weighted Voting")
    else:
        print(f"  Warning: Phase 1 results not found: {phase1_path}")
    
    # Phase 2: Stacking
    phase2_path = outputs_dir / "stacking_with_tabnet_results.json"
    if phase2_path.exists():
        with open(phase2_path, 'r') as f:
            phase2_data = json.load(f)
        results['Stacking'] = phase2_data
        print(f"  Loaded Phase 2: Stacking")
    else:
        print(f"  Warning: Phase 2 results not found: {phase2_path}")
    
    # Phase 3: Selective Stacking
    phase3_path = outputs_dir / "selective_stacking_results.json"
    if phase3_path.exists():
        with open(phase3_path, 'r') as f:
            phase3_data = json.load(f)
        results['Selective Stacking'] = phase3_data
        print(f"  Loaded Phase 3: Selective Stacking")
    else:
        print(f"  Warning: Phase 3 results not found: {phase3_path}")
    
    return results


def extract_metrics(results):
    """Extract metrics from all results."""
    metrics_dict = {}
    
    # Phase 1: Weighted Voting
    if 'Weighted Voting' in results:
        phase1 = results['Weighted Voting']
        if 'optimized' in phase1:
            metrics_dict['Weighted Voting (Optimized)'] = phase1['optimized']['metrics']
        else:
            metrics_dict['Weighted Voting (Best Manual)'] = phase1['best_manual']['metrics']
    
    # Phase 2: Stacking
    if 'Stacking' in results:
        phase2 = results['Stacking']
        best_meta = phase2.get('best_meta_learner', 'LogisticRegression')
        metrics_dict[f'Stacking ({best_meta})'] = phase2['best_metrics']
        
        # Also include all meta-learners if available
        if 'results' in phase2:
            for meta_name, meta_data in phase2['results'].items():
                metrics_dict[f'Stacking ({meta_name})'] = meta_data['metrics']
    
    # Phase 3: Selective Stacking
    if 'Selective Stacking' in results:
        phase3 = results['Selective Stacking']
        metrics_dict['Selective Stacking'] = phase3['best_metrics']
    
    return metrics_dict


def print_comparison(metrics_dict):
    """Print comprehensive comparison table."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE ENSEMBLE COMPARISON")
    print("=" * 120)
    print(f"{'Method':<35} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Critical Recall':<15} {'Status':<10}")
    print("-" * 120)
    
    # Sort by critical recall (descending)
    sorted_methods = sorted(
        metrics_dict.items(),
        key=lambda x: x[1].get('critical_recall', 0) or 0,
        reverse=True
    )
    
    for method_name, metrics in sorted_methods:
        crit_recall = metrics.get('critical_recall', 0) or 0
        
        # Determine status
        if crit_recall > 0.12:
            status = "GREAT"
        elif crit_recall > 0.10:
            status = "GOOD"
        else:
            status = "NEEDS WORK"
        
        print(f"{method_name:<35} {metrics['accuracy']:<12.4f} {metrics['macro_f1']:<12.4f} "
              f"{metrics['weighted_f1']:<12.4f} {crit_recall:<15.4f} {status:<10}")
    
    print("-" * 120)
    
    # Find best method
    best_method = max(
        metrics_dict.items(),
        key=lambda x: x[1].get('critical_recall', 0) or 0
    )
    
    print(f"\n  Best Method: {best_method[0]}")
    print(f"    Accuracy: {best_method[1]['accuracy']:.4f}")
    print(f"    Macro F1: {best_method[1]['macro_f1']:.4f}")
    print(f"    Weighted F1: {best_method[1]['weighted_f1']:.4f}")
    print(f"    Critical Recall: {best_method[1].get('critical_recall', 0):.4f}")
    
    if best_method[1].get('critical_recall', 0) > 0.12:
        print("    Status: GREAT! (>12%)")
    elif best_method[1].get('critical_recall', 0) > 0.10:
        print("    Status: GOOD! (>10%)")
    else:
        print("    Status: Needs improvement (<10%)")
    
    print("=" * 120)


def generate_summary_report(results, metrics_dict):
    """Generate comprehensive summary report."""
    print("\n[2] Generating summary report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_methods': len(metrics_dict),
            'best_method': max(
                metrics_dict.items(),
                key=lambda x: x[1].get('critical_recall', 0) or 0
            )[0],
            'best_critical_recall': max(
                m.get('critical_recall', 0) or 0 for m in metrics_dict.values()
            )
        },
        'methods': {}
    }
    
    # Add metrics for each method
    for method_name, metrics in metrics_dict.items():
        crit_recall = metrics.get('critical_recall', 0) or 0
        report['methods'][method_name] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'critical_recall': crit_recall,
            'status': 'GREAT' if crit_recall > 0.12 else ('GOOD' if crit_recall > 0.10 else 'NEEDS WORK')
        }
    
    # Add phase-specific details
    if 'Weighted Voting' in results:
        phase1 = results['Weighted Voting']
        report['weighted_voting'] = {
            'best_weights': phase1.get('best_manual', {}).get('weights', []),
            'optimized': 'optimized' in phase1
        }
    
    if 'Stacking' in results:
        phase2 = results['Stacking']
        report['stacking'] = {
            'best_meta_learner': phase2.get('best_meta_learner', 'Unknown'),
            'base_estimators': phase2.get('base_estimators', [])
        }
    
    if 'Selective Stacking' in results:
        phase3 = results['Selective Stacking']
        report['selective_stacking'] = {
            'best_threshold': phase3.get('best_threshold', 0),
            'attention_method': phase3.get('attention_method', 'Unknown'),
            'tabnet_usage_pct': phase3.get('best_metrics', {}).get('tabnet_usage_pct', 0)
        }
    
    return report


def main():
    """Main function."""
    print("=" * 120)
    print("FINAL COMPARISON: ALL ENSEMBLE APPROACHES")
    print("=" * 120)
    
    # Load results
    results = load_results()
    
    if not results:
        print("\n  Error: No results found! Please run Phase 1, 2, and 3 first.")
        return
    
    # Extract metrics
    metrics_dict = extract_metrics(results)
    
    # Print comparison
    print_comparison(metrics_dict)
    
    # Generate summary report
    report = generate_summary_report(results, metrics_dict)
    
    # Save final comparison
    print("\n[3] Saving final comparison...")
    comparison_path = outputs_dir / "all_ensemble_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump({
            'comparison': metrics_dict,
            'report': report,
            'detailed_results': results
        }, f, indent=2)
    print(f"  Final comparison saved to: {comparison_path}")
    
    # Print recommendations
    print("\n" + "=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)
    
    best_method = max(
        metrics_dict.items(),
        key=lambda x: x[1].get('critical_recall', 0) or 0
    )
    
    print(f"\n1. Best Method for Critical Recall: {best_method[0]}")
    print(f"   Critical Recall: {best_method[1].get('critical_recall', 0):.4f}")
    
    # Find best overall (balanced)
    best_overall = max(
        metrics_dict.items(),
        key=lambda x: (
            x[1].get('critical_recall', 0) or 0,
            x[1]['accuracy']
        )
    )
    
    if best_overall[0] != best_method[0]:
        print(f"\n2. Best Balanced Method: {best_overall[0]}")
        print(f"   Accuracy: {best_overall[1]['accuracy']:.4f}")
        print(f"   Critical Recall: {best_overall[1].get('critical_recall', 0):.4f}")
    
    print("\n3. Key Insights:")
    if best_method[1].get('critical_recall', 0) > 0.12:
        print("   - Critical recall target (>12%) ACHIEVED!")
    elif best_method[1].get('critical_recall', 0) > 0.10:
        print("   - Critical recall target (>10%) ACHIEVED!")
        print("   - Consider further optimization to reach >12%")
    else:
        print("   - Critical recall target (<10%) NOT MET")
        print("   - Consider: feature engineering, class weights, or different models")
    
    print("=" * 120)


if __name__ == "__main__":
    main()

