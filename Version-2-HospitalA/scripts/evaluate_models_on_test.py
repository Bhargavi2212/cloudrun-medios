"""
Quick script to evaluate all saved models on the test set.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# Paths
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/receptionist_models")

# Load test data
print("Loading test data...")
X_test = pd.read_csv(DATA_DIR / "X_test_final.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Extract receptionist features
receptionist_features = [
    "age",
    "ambulance_arrival",
    "seen_72h",
    "injury",
    "rfv1_cluster_Ear_Nose_Throat",
    "rfv1_cluster_Fever_Infection",
    "rfv1_cluster_Gastrointestinal",
    "rfv1_cluster_General_Symptoms",
    "rfv1_cluster_Mental_Health",
    "rfv1_cluster_Musculoskeletal",
    "rfv1_cluster_Neurological",
    "rfv1_cluster_Other",
    "rfv1_cluster_Respiratory",
    "rfv1_cluster_Skin",
    "rfv1_cluster_Trauma_Injury",
    "rfv1_cluster_Urinary_Genitourinary",
]

X_test_receptionist = X_test[receptionist_features].copy()

# Fix RFV clusters (assign default to rows with all zeros)
rfv_cols = [col for col in receptionist_features if col.startswith("rfv1_cluster_")]
zero_mask = X_test_receptionist[rfv_cols].sum(axis=1) == 0
if zero_mask.sum() > 0:
    if "rfv1_cluster_Other" in rfv_cols:
        X_test_receptionist.loc[zero_mask, "rfv1_cluster_Other"] = 1
    else:
        # Assign to most common cluster
        most_common = X_test_receptionist[rfv_cols].sum().idxmax()
        X_test_receptionist.loc[zero_mask, most_common] = 1

print(f"Test set shape: {X_test_receptionist.shape}")
print(f"Test set ESI distribution:\n{y_test.value_counts().sort_index()}")

# Models to evaluate
models_to_evaluate = {
    "Logistic Regression": "receptionist_lr.pkl",
    "Decision Tree": "receptionist_dt.pkl",
    "Random Forest": "receptionist_rf.pkl",
    "Stacking Ensemble": "receptionist_stacking_meta.pkl",
    "XGBoost": "receptionist_xgboost_calibrated.pkl",
}

# Evaluate each model
results = []

for model_name, model_file in models_to_evaluate.items():
    model_path = OUTPUT_DIR / "models" / model_file
    if not model_path.exists():
        print(f"[SKIP] {model_name}: Model file not found")
        continue

    print(f"\nEvaluating {model_name}...")
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test_receptionist)

    # Handle XGBoost (labels are 0-4, need to convert to 1-5)
    if "XGBoost" in model_name:
        y_pred = y_pred + 1

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[1, 2, 3, 4, 5], zero_division=0
    )
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ESI 1-2 combined recall
    esi_1_2_mask = y_test.isin([1, 2])
    if esi_1_2_mask.sum() > 0:
        esi_1_2_true = y_test[esi_1_2_mask]
        esi_1_2_pred = y_pred[esi_1_2_mask]
        tp_esi1 = ((esi_1_2_true == 1) & (esi_1_2_pred == 1)).sum()
        tp_esi2 = ((esi_1_2_true == 2) & (esi_1_2_pred == 2)).sum()
        total_esi1_2 = len(esi_1_2_true)
        esi_1_2_recall = (tp_esi1 + tp_esi2) / total_esi1_2 if total_esi1_2 > 0 else 0.0
    else:
        esi_1_2_recall = 0.0

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "ESI_1_2_Recall": esi_1_2_recall,
            "ESI_1_Recall": recall[0] if len(recall) > 0 else 0.0,
            "ESI_2_Recall": recall[1] if len(recall) > 1 else 0.0,
            "ESI_1_Precision": precision[0] if len(precision) > 0 else 0.0,
            "ESI_2_Precision": precision[1] if len(precision) > 1 else 0.0,
            "Weighted_F1": weighted_f1,
            "ESI_3_Recall": recall[2] if len(recall) > 2 else 0.0,
            "ESI_4_Recall": recall[3] if len(recall) > 3 else 0.0,
            "ESI_5_Recall": recall[4] if len(recall) > 4 else 0.0,
        }
    )

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ESI 1-2 Recall: {esi_1_2_recall:.4f}")
    print(f"  ESI 1 Recall: {recall[0]:.4f}")
    print(f"  ESI 2 Recall: {recall[1]:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

# Create comparison DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ESI_1_2_Recall", ascending=False)

print("\n" + "=" * 80)
print("TEST SET RESULTS - ALL MODELS")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
output_path = OUTPUT_DIR / "metrics" / "receptionist_model_comparison_test_set.csv"
results_df.to_csv(output_path, index=False)
print(f"\nSaved results to: {output_path}")
