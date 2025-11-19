"""Check missing values in Reason for Visit (RFV) variables."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"D:\Hackathons\Cloud Run\Version -2\data")

# Load dataset
df = pd.read_csv(DATA_DIR / "nhamcs_triage_dataset.csv")

# Find RFV columns
rfv_cols = [c for c in df.columns if "rfv" in c.lower()]
print("=" * 70)
print("REASON FOR VISIT (RFV) MISSING VALUE ANALYSIS")
print("=" * 70)

print(f"\nRFV Columns Found: {rfv_cols}")

# Check missing values
print("\n" + "-" * 70)
print("MISSING VALUES:")
print("-" * 70)
missing_counts = df[rfv_cols].isnull().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)

missing_df = pd.DataFrame(
    {
        "Column": missing_counts.index,
        "Missing Count": missing_counts.values,
        "Missing %": missing_pct.values,
        "Non-Missing Count": (len(df) - missing_counts.values),
        "Non-Missing %": (100 - missing_pct.values),
    }
)

print(missing_df.to_string(index=False))

# Check for special missing value codes (like -9, -8, 0, etc.)
print("\n" + "-" * 70)
print("SPECIAL VALUES (Potential Missing Codes):")
print("-" * 70)

for col in rfv_cols:
    print(f"\n{col}:")
    # Check for common missing codes
    special_values = df[col].value_counts().head(10)
    print("  Top 10 values:")
    for val, count in special_values.items():
        pct = count / len(df) * 100
        print(f"    {val}: {count:,} ({pct:.2f}%)")

    # Check for negative values (often missing codes in NHAMCS)
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(
            f"  Negative values: {negative_count:,} ({(negative_count/len(df)*100):.2f}%)"  # noqa: E501
        )

    # Check for zero values
    zero_count = (df[col] == 0).sum()
    if zero_count > 0:
        print(f"  Zero values: {zero_count:,} ({(zero_count/len(df)*100):.2f}%)")

# Check if any record has all RFV values missing
print("\n" + "-" * 70)
print("RECORDS WITH ALL RFV VALUES MISSING:")
print("-" * 70)
all_rfv_missing = df[rfv_cols].isnull().all(axis=1).sum()
print(
    f"Records with all RFV missing: {all_rfv_missing:,} ({(all_rfv_missing/len(df)*100):.2f}%)"  # noqa: E501
)

# Check records with at least one RFV
at_least_one_rfv = df[rfv_cols].notnull().any(axis=1).sum()
print(
    f"Records with at least one RFV: {at_least_one_rfv:,} ({(at_least_one_rfv/len(df)*100):.2f}%)"  # noqa: E501
)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total records: {len(df):,}")
print(
    f"Records with RFV1: {df[rfv_cols[0]].notna().sum():,} ({(df[rfv_cols[0]].notna().sum()/len(df)*100):.2f}%)"  # noqa: E501
)
if len(rfv_cols) > 1:
    print(
        f"Records with RFV2: {df[rfv_cols[1]].notna().sum():,} ({(df[rfv_cols[1]].notna().sum()/len(df)*100):.2f}%)"  # noqa: E501
    )
if len(rfv_cols) > 2:
    print(
        f"Records with RFV3: {df[rfv_cols[2]].notna().sum():,} ({(df[rfv_cols[2]].notna().sum()/len(df)*100):.2f}%)"  # noqa: E501
    )
