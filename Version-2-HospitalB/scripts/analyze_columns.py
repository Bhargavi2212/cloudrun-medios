"""Comprehensive column analysis for NHAMCS triage dataset.

This script analyzes each column to understand:
- Data type
- Missing values
- Distribution
- Skewness
- Unique values
- Value ranges
- Whether it's nominal/ordinal
- Appropriate preprocessing strategy
"""

from pathlib import Path

import pandas as pd
from scipy import stats

DATA_DIR = Path(r"D:\Hackathons\Cloud Run\Version -2\data")
OUTPUT_DIR = Path(r"D:\Hackathons\Cloud Run\Version -2\data\column_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_DIR / "nhamcs_triage_dataset.csv")

print("=" * 80)
print("COMPREHENSIVE COLUMN ANALYSIS FOR NHAMCS TRIAGE DATASET")
print("=" * 80)
print(f"\nDataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Total Records: {len(df):,}")

# Check for duplicates
print("\n" + "=" * 80)
print("DUPLICATE RECORDS ANALYSIS")
print("=" * 80)
duplicate_count = df.duplicated().sum()
print(f"Total duplicate rows: {duplicate_count:,} ({duplicate_count/len(df)*100:.2f}%)")

if duplicate_count > 0:
    print("\nSample duplicate rows:")
    print(df[df.duplicated(keep=False)].head(10))
else:
    print("SUCCESS: No duplicate rows found")

# Column-by-column analysis
print("\n" + "=" * 80)
print("DETAILED COLUMN ANALYSIS")
print("=" * 80)

column_analysis = []

for col in df.columns:
    print("\n" + "-" * 80)
    print(f"COLUMN: {col.upper()}")
    print("-" * 80)

    col_data = df[col]
    analysis = {
        "column": col,
        "dtype": str(col_data.dtype),
        "total_count": len(col_data),
        "non_null_count": col_data.notna().sum(),
        "null_count": col_data.isna().sum(),
        "null_percentage": (col_data.isna().sum() / len(col_data) * 100),
    }

    # Basic info
    print(f"Data Type: {col_data.dtype}")
    print(f"Total Values: {len(col_data):,}")
    print(f"Non-Null Values: {col_data.notna().sum():,}")
    print(
        f"Null Values: {col_data.isna().sum():,} ({analysis['null_percentage']:.2f}%)"
    )

    # Check if column should be excluded (>5% missing)
    if analysis["null_percentage"] > 5:
        print("WARNING: >5% missing - Consider excluding this column")
        analysis["exclude_recommendation"] = "Yes (>5% missing)"
    else:
        analysis["exclude_recommendation"] = "No"

    # Analyze based on data type
    if col_data.dtype in ["int64", "float64"]:
        # Numerical column
        non_null_data = col_data.dropna()

        if len(non_null_data) > 0:
            analysis["is_numerical"] = True
            analysis["is_categorical"] = False

            # Check if it's actually categorical (low unique values)
            unique_count = non_null_data.nunique()
            analysis["unique_values"] = unique_count

            print(f"Unique Values: {unique_count:,}")

            if unique_count <= 20:
                print("WARNING: Low unique values - Might be categorical")
                print(f"Unique values: {sorted(non_null_data.unique())}")
                analysis["might_be_categorical"] = True
            else:
                analysis["might_be_categorical"] = False

            # Statistical measures
            analysis["mean"] = (
                float(non_null_data.mean()) if len(non_null_data) > 0 else None
            )
            analysis["median"] = (
                float(non_null_data.median()) if len(non_null_data) > 0 else None
            )
            analysis["std"] = (
                float(non_null_data.std()) if len(non_null_data) > 0 else None
            )
            analysis["min"] = (
                float(non_null_data.min()) if len(non_null_data) > 0 else None
            )
            analysis["max"] = (
                float(non_null_data.max()) if len(non_null_data) > 0 else None
            )
            analysis["q25"] = (
                float(non_null_data.quantile(0.25)) if len(non_null_data) > 0 else None
            )
            analysis["q75"] = (
                float(non_null_data.quantile(0.75)) if len(non_null_data) > 0 else None
            )

            print(f"Mean: {analysis['mean']:.2f}" if analysis["mean"] else "N/A")
            print(f"Median: {analysis['median']:.2f}" if analysis["median"] else "N/A")
            print(f"Std Dev: {analysis['std']:.2f}" if analysis["std"] else "N/A")
            print(f"Min: {analysis['min']:.2f}" if analysis["min"] else "N/A")
            print(f"Max: {analysis['max']:.2f}" if analysis["max"] else "N/A")
            print(f"Q25: {analysis['q25']:.2f}" if analysis["q25"] else "N/A")
            print(f"Q75: {analysis['q75']:.2f}" if analysis["q75"] else "N/A")

            # Skewness
            if len(non_null_data) > 0:
                skewness = stats.skew(non_null_data)
                analysis["skewness"] = float(skewness)
                print(f"Skewness: {skewness:.3f}")

                if abs(skewness) > 1:
                    print(
                        "WARNING: Highly skewed (|skewness| > 1) - Consider transformation"  # noqa: E501
                    )
                    analysis["needs_transformation"] = True
                elif abs(skewness) > 0.5:
                    print(
                        "WARNING: Moderately skewed (|skewness| > 0.5) - May need transformation"  # noqa: E501
                    )
                    analysis["needs_transformation"] = "Maybe"
                else:
                    print("OK: Normal distribution")
                    analysis["needs_transformation"] = False
            else:
                analysis["skewness"] = None
                analysis["needs_transformation"] = None

            # Outliers (IQR method)
            if len(non_null_data) > 0 and unique_count > 10:
                Q1 = non_null_data.quantile(0.25)
                Q3 = non_null_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = non_null_data[
                    (non_null_data < lower_bound) | (non_null_data > upper_bound)
                ]
                outlier_pct = len(outliers) / len(non_null_data) * 100
                analysis["outlier_count"] = len(outliers)
                analysis["outlier_percentage"] = float(outlier_pct)
                print(f"Outliers (IQR method): {len(outliers):,} ({outlier_pct:.2f}%)")

                if outlier_pct > 10:
                    print(
                        "WARNING: High outlier percentage - Consider outlier treatment"
                    )
                    analysis["needs_outlier_treatment"] = True
                elif outlier_pct > 5:
                    analysis["needs_outlier_treatment"] = "Maybe"
                else:
                    analysis["needs_outlier_treatment"] = False
            else:
                analysis["outlier_count"] = None
                analysis["outlier_percentage"] = None
                analysis["needs_outlier_treatment"] = None

            # Value distribution
            if unique_count <= 20:
                print("\nValue Distribution:")
                value_counts = non_null_data.value_counts().head(10)
                for val, count in value_counts.items():
                    pct = count / len(non_null_data) * 100
                    print(f"  {val}: {count:,} ({pct:.2f}%)")

    elif col_data.dtype == "object":
        # Categorical/string column
        non_null_data = col_data.dropna()
        unique_count = non_null_data.nunique()
        analysis["is_numerical"] = False
        analysis["is_categorical"] = True
        analysis["unique_values"] = unique_count

        print(f"Unique Values: {unique_count:,}")

        # Check if ordinal or nominal
        print("\nValue Distribution:")
        value_counts = non_null_data.value_counts().head(15)
        for val, count in value_counts.items():
            pct = count / len(non_null_data) * 100
            print(f"  {val}: {count:,} ({pct:.2f}%)")

        # Determine if ordinal
        # For now, we'll mark as nominal unless we know it's ordinal
        analysis["is_ordinal"] = False  # Will be determined manually
        analysis["is_nominal"] = True

    # Column-specific analysis and recommendations
    print("\n--- COLUMN-SPECIFIC ANALYSIS ---")

    if col == "esi_level":
        print("TARGET VARIABLE: Emergency Severity Index (1-5)")
        print("  - ESI 1: Most critical (resuscitation)")
        print("  - ESI 2: Very urgent")
        print("  - ESI 3: Urgent")
        print("  - ESI 4: Less urgent")
        print("  - ESI 5: Least urgent")
        print("  - Type: ORDINAL (ordered categories)")
        print("  - Encoding: Ordinal encoding (1,2,3,4,5) or one-hot")
        analysis[
            "column_description"
        ] = "Target variable - Emergency Severity Index (1-5)"
        analysis["is_ordinal"] = True
        analysis["is_nominal"] = False
        analysis["preprocessing"] = "Target - no preprocessing needed"

    elif col == "pulse":
        print("VITAL SIGN: Heart rate (beats per minute)")
        print("  - Normal range: 60-100 bpm")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation (vital sign)")
        analysis["column_description"] = "Heart rate in beats per minute"
        analysis["preprocessing"] = "Median imputation, check for outliers"

    elif col == "respiration":
        print("VITAL SIGN: Respiratory rate (breaths per minute)")
        print("  - Normal range: 12-20 bpm")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        analysis["column_description"] = "Respiratory rate in breaths per minute"
        analysis["preprocessing"] = "Median imputation, check for outliers"

    elif col == "sbp":
        print("VITAL SIGN: Systolic Blood Pressure (mmHg)")
        print("  - Normal range: 90-140 mmHg")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        analysis["column_description"] = "Systolic blood pressure in mmHg"
        analysis["preprocessing"] = "Median imputation, check for outliers"

    elif col == "dbp":
        print("VITAL SIGN: Diastolic Blood Pressure (mmHg)")
        print("  - Normal range: 60-90 mmHg")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        analysis["column_description"] = "Diastolic blood pressure in mmHg"
        analysis["preprocessing"] = "Median imputation, check for outliers"

    elif col == "o2_sat":
        print("VITAL SIGN: Oxygen Saturation (%)")
        print("  - Normal range: 95-100%")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        analysis["column_description"] = "Oxygen saturation percentage"
        analysis["preprocessing"] = "Median imputation, check for outliers"

    elif col == "temp_c":
        print("VITAL SIGN: Temperature (Celsius)")
        print("  - Normal range: 36.1-37.2°C")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        analysis["column_description"] = "Body temperature in Celsius"
        analysis["preprocessing"] = "Median imputation, check for outliers"

    elif col == "gcs":
        print("VITAL SIGN: Glasgow Coma Scale (3-15)")
        print("  - Range: 3 (worst) to 15 (best)")
        print("  - Type: ORDINAL NUMERICAL")
        print("  - Missing strategy: Mode imputation or exclude if >5% missing")
        analysis["column_description"] = "Glasgow Coma Scale score (3-15)"
        analysis["is_ordinal"] = True
        analysis["preprocessing"] = "Mode imputation or exclude if high missing"

    elif col == "pain":
        print("CLINICAL: Pain Scale (0-10)")
        print("  - Range: 0 (no pain) to 10 (severe pain)")
        print("  - Type: ORDINAL NUMERICAL")
        print("  - Missing strategy: Median imputation")
        analysis["column_description"] = "Pain scale score (0-10)"
        analysis["is_ordinal"] = True
        analysis["preprocessing"] = "Median imputation"

    elif col == "age":
        print("DEMOGRAPHIC: Patient Age (years)")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        print("  - May benefit from binning/grouping")
        analysis["column_description"] = "Patient age in years"
        analysis["preprocessing"] = "Median imputation, consider binning"

    elif col == "month":
        print("TEMPORAL: Month of visit (1-12)")
        print("  - Type: ORDINAL NUMERICAL (cyclic)")
        print("  - Encoding: Cyclic encoding (sin/cos) or ordinal")
        analysis["column_description"] = "Month of visit (1-12)"
        analysis["is_ordinal"] = True
        analysis["is_cyclic"] = True
        analysis["preprocessing"] = "Cyclic encoding or ordinal encoding"

    elif col == "rfv1_cluster":
        print("CLINICAL: Reason for Visit Cluster")
        print("  - Type: NOMINAL CATEGORICAL")
        print("  - Values: 13 clusters (Musculoskeletal, Respiratory, etc.)")
        print("  - Encoding: One-hot encoding or target encoding")
        analysis["column_description"] = "Reason for visit cluster (13 categories)"
        analysis["is_nominal"] = True
        analysis["is_ordinal"] = False
        analysis["preprocessing"] = "One-hot encoding or target encoding"

    elif col == "ambulance_arrival":
        print("CLINICAL: Ambulance Arrival (binary)")
        print("  - Type: BINARY CATEGORICAL")
        print("  - Values: 1=Yes, 2=No, -9/-8=Missing")
        print("  - Encoding: Binary (0/1) after handling missing")
        analysis["column_description"] = "Whether patient arrived by ambulance"
        analysis["is_nominal"] = True
        analysis["preprocessing"] = "Binary encoding (0/1), handle missing"

    elif col == "injury":
        print("CLINICAL: Injury-related visit (binary)")
        print("  - Type: BINARY CATEGORICAL")
        print("  - Values: 0=No, 1=Yes")
        print("  - Encoding: Binary (0/1)")
        analysis["column_description"] = "Whether visit is injury-related"
        analysis["is_nominal"] = True
        analysis["preprocessing"] = "Binary encoding (already 0/1)"

    elif col == "seen_72h":
        print("HISTORY: Seen in last 72 hours (binary)")
        print("  - Type: BINARY CATEGORICAL")
        print("  - Values: 1=Yes, 2=No, -9/-8=Missing")
        print("  - Encoding: Binary (0/1) after handling missing")
        analysis["column_description"] = "Whether patient seen in last 72 hours"
        analysis["is_nominal"] = True
        analysis["preprocessing"] = "Binary encoding (0/1), handle missing"

    elif col == "discharged_7d":
        print("HISTORY: Discharged in last 7 days (binary)")
        print("  - Type: BINARY CATEGORICAL")
        print("  - Values: 1=Yes, 2=No, -9/-8=Missing")
        print("  - Encoding: Binary (0/1) after handling missing")
        analysis["column_description"] = "Whether patient discharged in last 7 days"
        analysis["is_nominal"] = True
        analysis["preprocessing"] = "Binary encoding (0/1), handle missing"

    elif col in [
        "cebvd",
        "chf",
        "ed_dialysis",
        "hiv",
        "diabetes",
        "no_chronic_conditions",
    ]:
        print(f"CHRONIC CONDITION: {col.replace('_', ' ').title()}")
        print("  - Type: BINARY CATEGORICAL")
        print("  - Values: 0=No, 1=Yes")
        print("  - Encoding: Binary (already 0/1)")
        analysis[
            "column_description"
        ] = f"Chronic condition: {col.replace('_', ' ').title()}"
        analysis["is_nominal"] = True
        analysis["preprocessing"] = "Binary encoding (already 0/1)"

    elif col == "wait_time":
        print("OPERATIONAL: Wait time (minutes)")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        print("  - May have outliers")
        analysis["column_description"] = "Wait time in minutes"
        analysis["preprocessing"] = "Median imputation, handle outliers"

    elif col == "length_of_visit":
        print("OPERATIONAL: Length of visit (minutes)")
        print("  - Type: CONTINUOUS NUMERICAL")
        print("  - Missing strategy: Median imputation")
        print("  - May have outliers")
        analysis["column_description"] = "Length of visit in minutes"
        analysis["preprocessing"] = "Median imputation, handle outliers"

    elif col == "past_visits":
        print("HISTORY: Past visits count")
        print("  - Type: DISCRETE NUMERICAL")
        print("  - Missing strategy: Mode or 0 imputation")
        analysis["column_description"] = "Number of past visits"
        analysis["preprocessing"] = "Mode or 0 imputation"

    elif col in ["year", "line_num"]:
        print(f"METADATA: {col.replace('_', ' ').title()}")
        print("  - Type: IDENTIFIER/METADATA")
        print("  - Action: EXCLUDE from model (not a feature)")
        analysis["column_description"] = f"Metadata: {col.replace('_', ' ').title()}"
        analysis["exclude_recommendation"] = "Yes (metadata/identifier)"
        analysis["preprocessing"] = "Exclude from features"

    column_analysis.append(analysis)

# Save analysis to JSON
analysis_df = pd.DataFrame(column_analysis)
analysis_df.to_csv(OUTPUT_DIR / "column_analysis_summary.csv", index=False)
analysis_df.to_json(
    OUTPUT_DIR / "column_analysis_summary.json", orient="records", indent=2
)

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print(f"\nAnalysis saved to: {OUTPUT_DIR}")
print("  - column_analysis_summary.csv")
print("  - column_analysis_summary.json")

# Summary statistics
print("\nColumns by type:")
numerical = [
    c
    for c in column_analysis
    if c.get("is_numerical") and not c.get("might_be_categorical")
]
categorical = [
    c
    for c in column_analysis
    if c.get("is_categorical") or c.get("might_be_categorical")
]
print(f"  Numerical: {len(numerical)}")
print(f"  Categorical: {len(categorical)}")

print("\nColumns with >5% missing (should exclude):")
high_missing = [c for c in column_analysis if c.get("null_percentage", 0) > 5]
for col_info in high_missing:
    print(f"  {col_info['column']}: {col_info['null_percentage']:.2f}% missing")

print("\nColumns needing transformation (high skewness):")
high_skew = [c for c in column_analysis if c.get("needs_transformation") is True]
for col_info in high_skew:
    print(f"  {col_info['column']}: skewness = {col_info.get('skewness', 'N/A'):.3f}")

print("\nSUCCESS: Column analysis complete!")
