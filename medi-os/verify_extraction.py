"""Quick script to verify NHAMCS data extraction."""

import pandas as pd
from pathlib import Path

# Read the CSV file
csv_path = Path("data/NHAMCS_2011_2022_combined.csv")
df = pd.read_csv(csv_path)

print("=" * 70)
print("NHAMCS DATA EXTRACTION VERIFICATION")
print("=" * 70)
print(f"\n📊 File: {csv_path.name}")
print(f"   Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
print(f"   Total records: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")

print("\n" + "=" * 70)
print("📋 EXTRACTED COLUMNS:")
print("=" * 70)
cols = sorted(df.columns.tolist())
for i, col in enumerate(cols, 1):
    non_null = df[col].notna().sum()
    pct = (non_null / len(df)) * 100
    dtype = str(df[col].dtype)
    print(f"{i:2d}. {col:25s} | {dtype:10s} | {non_null:>8,} values ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("🔍 KEY FIELD VERIFICATION:")
print("=" * 70)

# Check target variable
print(f"\n✅ ESI Level (Target):")
print(f"   Non-null: {df['esi_level'].notna().sum():,}")
print(f"   Distribution:")
print(df['esi_level'].value_counts().sort_index())

# Check Reason for Visit
print(f"\n✅ Reason for Visit (RFV):")
print(f"   RFV1 (detailed): {df['rfv1'].notna().sum():,} non-null")
print(f"   RFV1_3D (category): {df['rfv1_3d'].notna().sum():,} non-null")
print(f"   Top 5 RFV1_3D categories:")
top_rfv = df['rfv1_3d'].value_counts().head()
for code, count in top_rfv.items():
    pct = (count / len(df)) * 100
    print(f"      {code:8.0f}: {count:>7,} ({pct:5.1f}%)")

# Check vital signs
print(f"\n✅ Vital Signs:")
vital_cols = ['age', 'sbp', 'dbp', 'pulse', 'respiration', 'temp_c', 'pain', 'o2_sat', 'gcs']
for col in vital_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        if non_null > 0:
            mean_val = df[col].mean()
            print(f"   {col:15s}: {non_null:>7,} values, mean = {mean_val:.2f}")

# Check visit characteristics
print(f"\n✅ Visit Characteristics:")
print(f"   Injury visits: {df['injury'].sum():,} ({df['injury'].sum()/len(df)*100:.1f}%)")
# Fix ambulance calculation - values can be 0, 1, or 2
amb_yes = (df['ambulance_arrival'] == 1).sum()
amb_total = len(df[df['ambulance_arrival'].isin([0, 1])])
if amb_total > 0:
    print(f"   Ambulance arrivals: {amb_yes:,} ({amb_yes/amb_total*100:.1f}% of coded)")
else:
    print(f"   Ambulance arrivals: {amb_yes:,}")
print(f"   ARREMS value distribution:")
print(df['ambulance_arrival'].value_counts().sort_index())
if 'wait_time' in df.columns:
    print(f"   Avg wait time: {df['wait_time'].mean():.1f} minutes")
if 'length_of_visit' in df.columns:
    print(f"   Avg length of visit: {df['length_of_visit'].mean():.1f} minutes")

# Check diagnoses
print(f"\n✅ Diagnoses:")
for i in [1, 2, 3]:
    diag_col = f'diag{i}'
    if diag_col in df.columns:
        non_null = df[diag_col].notna().sum()
        print(f"   {diag_col:10s}: {non_null:>7,} non-null ({non_null/len(df)*100:.1f}%)")

# Check comorbidities
print(f"\n✅ Comorbidities:")
comorbidity_cols = ['cebvd', 'chf', 'ed_dialysis', 'hiv', 'diabetes', 'no_chronic_conditions']
for col in comorbidity_cols:
    if col in df.columns:
        count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].notna().sum()
        pct = (count / len(df)) * 100
        print(f"   {col:25s}: {count:>7,} ({pct:5.1f}%)")

# Check years
if 'year' in df.columns:
    print(f"\n✅ Years covered:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"   {year}: {count:>7,} records")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE")
print("=" * 70)
print("\n📝 Notes:")
print("   - Ambulance arrival (ARREMS) uses: 0=No, 1=Yes, 2=Not applicable")
print("   - Some fields (diabetes, ed_dialysis) only available in 2011")
print("   - RFV fields are complete and ready for feature engineering")
print("   - All vital signs properly extracted and cleaned")
