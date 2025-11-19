"""
Compare Experiments Script

Loads all experiment results and creates comprehensive comparison reports
across feature sets and models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "data" / "receptionist_models_v2"


def load_experiment_summary() -> dict:
    """Load experiment summary JSON."""
    summary_file = OUTPUT_DIR / "experiments" / "experiment_summary.json"
    if not summary_file.exists():
        logger.error(f"Experiment summary not found: {summary_file}")
        return {}

    with open(summary_file) as f:
        return json.load(f)


def load_all_metrics() -> dict:
    """Load all metrics files."""
    metrics_dir = OUTPUT_DIR / "metrics"
    if not metrics_dir.exists():
        logger.error(f"Metrics directory not found: {metrics_dir}")
        return {}

    metrics = {}

    # Load validation metrics
    for file in metrics_dir.glob("val_metrics_*.csv"):
        df = pd.read_csv(file)
        if len(df) > 0:
            # Extract model name and feature set from filename
            # Format: val_metrics_model_name_feature_set.csv
            parts = file.stem.replace("val_metrics_", "").split("_")
            if len(parts) >= 2:
                feature_set = parts[-1]  # Last part is feature set
                model_name = "_".join(parts[:-1])  # Everything else is model name
                key = f"{model_name}_{feature_set}"
                metrics[key] = {
                    "type": "validation",
                    "model": model_name,
                    "feature_set": feature_set,
                    "data": df.iloc[0].to_dict(),
                }

    # Load test metrics
    for file in metrics_dir.glob("test_metrics_*.csv"):
        df = pd.read_csv(file)
        if len(df) > 0:
            # Format: test_metrics_model_name_feature_set.csv
            parts = file.stem.replace("test_metrics_", "").split("_")
            if len(parts) >= 2:
                feature_set = parts[-1]
                model_name = "_".join(parts[:-1])
                key = f"{model_name}_{feature_set}_test"
                metrics[key] = {
                    "type": "test",
                    "model": model_name,
                    "feature_set": feature_set,
                    "data": df.iloc[0].to_dict(),
                }

    return metrics


def create_comparison_matrix(
    experiment_summary: dict, all_metrics: dict
) -> pd.DataFrame:
    """Create comparison matrix: Feature Set x Model x Metrics."""
    comparison_data = []

    feature_sets = experiment_summary.get("feature_sets", ["A", "B", "C", "D"])
    models = experiment_summary.get("models", [])

    for feature_set in feature_sets:
        for model in models:
            # Get validation results
            val_key = f"{model.lower().replace(' ', '_')}_{feature_set}"
            val_metrics = all_metrics.get(val_key, {})

            # Get test results
            test_key = f"{model.lower().replace(' ', '_')}_{feature_set}_test"
            test_metrics = all_metrics.get(test_key, {})

            if val_metrics or test_metrics:
                row = {
                    "Feature_Set": feature_set,
                    "Model": model,
                    "Val_ESI_1_2_Recall": val_metrics.get("data", {}).get(
                        "esi_1_2_recall", None
                    ),
                    "Val_Accuracy": val_metrics.get("data", {}).get("accuracy", None),
                    "Val_Macro_F1": val_metrics.get("data", {}).get("macro_f1", None),
                    "Test_ESI_1_2_Recall": test_metrics.get("data", {}).get(
                        "esi_1_2_recall", None
                    ),
                    "Test_Accuracy": test_metrics.get("data", {}).get("accuracy", None),
                    "Test_Macro_F1": test_metrics.get("data", {}).get("macro_f1", None),
                }
                comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def identify_best_models(comparison_df: pd.DataFrame) -> dict:
    """Identify best models based on ESI 1-2 recall."""
    best_models = {}

    # Best overall (test set)
    if "Test_ESI_1_2_Recall" in comparison_df.columns:
        best_overall = comparison_df.nlargest(1, "Test_ESI_1_2_Recall")
        if len(best_overall) > 0:
            best_models["overall"] = best_overall.iloc[0].to_dict()

    # Best per feature set (test set)
    for feature_set in ["A", "B", "C", "D"]:
        fs_df = comparison_df[comparison_df["Feature_Set"] == feature_set]
        if len(fs_df) > 0 and "Test_ESI_1_2_Recall" in fs_df.columns:
            best_fs = fs_df.nlargest(1, "Test_ESI_1_2_Recall")
            if len(best_fs) > 0:
                best_models[f"feature_set_{feature_set}"] = best_fs.iloc[0].to_dict()

    # Best per model type (test set)
    for model in comparison_df["Model"].unique():
        model_df = comparison_df[comparison_df["Model"] == model]
        if len(model_df) > 0 and "Test_ESI_1_2_Recall" in model_df.columns:
            best_model = model_df.nlargest(1, "Test_ESI_1_2_Recall")
            if len(best_model) > 0:
                best_models[
                    f"model_{model.lower().replace(' ', '_')}"
                ] = best_model.iloc[0].to_dict()

    return best_models


def generate_summary_report(
    experiment_summary: dict, comparison_df: pd.DataFrame, best_models: dict
) -> str:
    """Generate markdown summary report."""
    report = []
    report.append("# Experiment Summary Report")
    report.append("=" * 80)
    report.append("")

    # Overview
    report.append("## Overview")
    report.append("")
    report.append(
        f"- Feature Sets Evaluated: {', '.join(experiment_summary.get('feature_sets', []))}"  # noqa: E501
    )
    report.append(
        f"- Models Evaluated: {', '.join(experiment_summary.get('models', []))}"
    )
    report.append("")

    # Best Overall Model
    report.append("## Best Overall Model (Test Set)")
    report.append("")
    if "overall" in best_models:
        best = best_models["overall"]
        report.append(f"- **Model**: {best.get('Model', 'N/A')}")
        report.append(f"- **Feature Set**: {best.get('Feature_Set', 'N/A')}")
        report.append(f"- **ESI 1-2 Recall**: {best.get('Test_ESI_1_2_Recall', 0):.4f}")
        report.append(f"- **Accuracy**: {best.get('Test_Accuracy', 0):.4f}")
        report.append(f"- **Macro F1**: {best.get('Test_Macro_F1', 0):.4f}")
    else:
        report.append("No test results available.")
    report.append("")

    # Best Per Feature Set
    report.append("## Best Model Per Feature Set (Test Set)")
    report.append("")
    for fs in ["A", "B", "C", "D"]:
        key = f"feature_set_{fs}"
        if key in best_models:
            best = best_models[key]
            report.append(f"### Feature Set {fs}")
            report.append(f"- **Model**: {best.get('Model', 'N/A')}")
            report.append(
                f"- **ESI 1-2 Recall**: {best.get('Test_ESI_1_2_Recall', 0):.4f}"
            )
            report.append(f"- **Accuracy**: {best.get('Test_Accuracy', 0):.4f}")
            report.append(f"- **Macro F1**: {best.get('Test_Macro_F1', 0):.4f}")
            report.append("")

    # Comparison Table
    report.append("## Full Comparison Matrix")
    report.append("")
    report.append("### Validation Set Results")
    report.append("")
    val_cols = [
        "Feature_Set",
        "Model",
        "Val_ESI_1_2_Recall",
        "Val_Accuracy",
        "Val_Macro_F1",
    ]
    if all(col in comparison_df.columns for col in val_cols):
        val_df = comparison_df[val_cols].sort_values(
            ["Feature_Set", "Val_ESI_1_2_Recall"], ascending=[True, False]
        )
        # Convert to markdown table manually for compatibility
        report.append("| " + " | ".join(val_df.columns) + " |")
        report.append("| " + " | ".join(["---"] * len(val_df.columns)) + " |")
        for _, row in val_df.iterrows():
            report.append(
                "| "
                + " | ".join(
                    [str(val) if val is not None else "N/A" for val in row.values]
                )
                + " |"
            )
        report.append("")

    report.append("### Test Set Results")
    report.append("")
    test_cols = [
        "Feature_Set",
        "Model",
        "Test_ESI_1_2_Recall",
        "Test_Accuracy",
        "Test_Macro_F1",
    ]
    if all(col in comparison_df.columns for col in test_cols):
        test_df = comparison_df[test_cols].sort_values(
            ["Feature_Set", "Test_ESI_1_2_Recall"], ascending=[True, False]
        )
        # Convert to markdown table manually for compatibility
        report.append("| " + " | ".join(test_df.columns) + " |")
        report.append("| " + " | ".join(["---"] * len(test_df.columns)) + " |")
        for _, row in test_df.iterrows():
            report.append(
                "| "
                + " | ".join(
                    [str(val) if val is not None else "N/A" for val in row.values]
                )
                + " |"
            )
        report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")
    if "overall" in best_models:
        best = best_models["overall"]
        esi_recall = best.get("Test_ESI_1_2_Recall", 0)
        if esi_recall >= 0.75:
            report.append(
                f"- [OK] ESI 1-2 recall ({esi_recall:.2%}) meets target threshold (>= 75%)"  # noqa: E501
            )
        elif esi_recall >= 0.50:
            report.append(
                f"- [WARN] ESI 1-2 recall ({esi_recall:.2%}) is below target but above 50%"  # noqa: E501
            )
        else:
            report.append(
                f"- [ERROR] ESI 1-2 recall ({esi_recall:.2%}) is below 50% - needs improvement"  # noqa: E501
            )

        report.append(
            f"- **Recommended Model**: {best.get('Model', 'N/A')} with Feature Set {best.get('Feature_Set', 'N/A')}"  # noqa: E501
        )
    report.append("")

    return "\n".join(report)


def main():
    """Main comparison function."""
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 80)

    # Load data
    experiment_summary = load_experiment_summary()
    if not experiment_summary:
        logger.error("Failed to load experiment summary. Exiting.")
        return

    all_metrics = load_all_metrics()
    logger.info(f"Loaded {len(all_metrics)} metric files")

    # Create comparison matrix
    comparison_df = create_comparison_matrix(experiment_summary, all_metrics)
    logger.info(f"Created comparison matrix: {comparison_df.shape}")

    # Save comparison matrix
    comparison_file = OUTPUT_DIR / "experiments" / "best_models_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Saved comparison matrix to: {comparison_file}")

    # Identify best models
    best_models = identify_best_models(comparison_df)
    logger.info(f"Identified {len(best_models)} best model categories")

    # Generate summary report
    report = generate_summary_report(experiment_summary, comparison_df, best_models)

    # Save report
    report_file = OUTPUT_DIR / "reports" / "experiment_summary.md"
    with open(report_file, "w") as f:
        f.write(report)
    logger.info(f"Saved summary report to: {report_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    if "overall" in best_models:
        best = best_models["overall"]
        logger.info(
            f"Best Overall Model: {best.get('Model', 'N/A')} (Feature Set {best.get('Feature_Set', 'N/A')})"  # noqa: E501
        )
        logger.info(f"  ESI 1-2 Recall: {best.get('Test_ESI_1_2_Recall', 0):.4f}")
        logger.info(f"  Accuracy: {best.get('Test_Accuracy', 0):.4f}")

    logger.info("\nComparison complete!")


if __name__ == "__main__":
    main()
