"""Check for missing values in ESI level."""

import pandas as pd

df = pd.read_csv("data/nhamcs_triage_dataset.csv")

print("=" * 70)
print("ESI LEVEL MISSING VALUES ANALYSIS")
print("=" * 70)

print(f"\nTotal records: {len(df):,}")
print(f"Null/NaN values: {df['esi_level'].isnull().sum():,}")
print(f"Null percentage: {df['esi_level'].isnull().sum() / len(df) * 100:.2f}%")

print("\nUnique ESI values:")
esi_counts = df["esi_level"].value_counts().sort_index()
for val, count in esi_counts.items():
    pct = count / len(df) * 100
    print(f"  ESI {val}: {count:,} ({pct:.2f}%)")

print("\nAny negative values (missing codes)?")
neg_values = df[df["esi_level"] < 0]
if len(neg_values) > 0:
    print(f"  YES - Found {len(neg_values):,} negative values:")
    print(neg_values["esi_level"].value_counts())
else:
    print("  NO - No negative values found")

print("\nAny zero values?")
zero_values = df[df["esi_level"] == 0]
if len(zero_values) > 0:
    print(f"  YES - Found {len(zero_values):,} zero values")
else:
    print("  NO - No zero values found")

print("\nValid ESI range (1-5):")
valid_esi = df[(df["esi_level"] >= 1) & (df["esi_level"] <= 5)]
print(f"  Valid ESI values: {len(valid_esi):,} ({len(valid_esi)/len(df)*100:.2f}%)")

invalid_esi = df[(df["esi_level"] < 1) | (df["esi_level"] > 5)]
if len(invalid_esi) > 0:
    pct = len(invalid_esi) / len(df) * 100
    print(
        f"  Invalid ESI values: {len(invalid_esi):,} ({pct:.2f}%)"
    )
    print(f"  Invalid values: {sorted(invalid_esi['esi_level'].unique())}")
else:
    print("  Invalid ESI values: 0 (0.00%)")

print("\n" + "=" * 70)
print("CONCLUSION:")
if (
    df["esi_level"].isnull().sum() == 0
    and len(df[df["esi_level"] < 1]) == 0
    and len(df[df["esi_level"] > 5]) == 0
):
    print("SUCCESS: ESI level has NO missing values and all values are valid (1-5)")
else:
    print("WARNING: ESI level has missing or invalid values that need handling")
