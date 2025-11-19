"""
Explore NHAMCS dataset structure and prepare for triage model training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"D:\Hackathons\Cloud Run\medi-os\data")
COMBINED_CSV = DATA_DIR / "NHAMCS_2011_2022_combined.csv"


def main() -> None:
    """Explore the NHAMCS dataset structure."""
    print("=" * 60)
    print("NHAMCS DATA EXPLORATION")
    print("=" * 60)

    df = pd.read_csv(COMBINED_CSV)
    print(f"\n📊 Total records: {len(df):,}")
    print(f"📅 Years: {df['year'].min():.0f} - {df['year'].max():.0f}")

    print("\n" + "=" * 60)
    print("ESI LEVEL DISTRIBUTION (Triage Acuity)")
    print("=" * 60)
    esi_counts = df["esi_level"].value_counts().sort_index()
    print(esi_counts)
    print(f"\nValid ESI levels (1-5): {len(df[df['esi_level'].isin([1,2,3,4,5])]):,}")

    print("\n" + "=" * 60)
    print("KEY TRIAGE VARIABLES")
    print("=" * 60)
    triage_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "temp_c",
        "gcs",
        "pain",
        "age",
        "sex",
        "esi_level",
    ]
    print(df[triage_vars].describe())

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing = df[triage_vars].isnull().sum()
    print(
        missing[missing > 0]
        if missing.sum() > 0
        else "✅ No missing values in key variables"
    )

    print("\n" + "=" * 60)
    print("ESI LEVEL BY VITAL SIGNS")
    print("=" * 60)
    df_valid = df[df["esi_level"].isin([1, 2, 3, 4, 5])].copy()

    # SBP analysis
    df_valid["sbp_cat"] = pd.cut(
        df_valid["sbp"],
        bins=[0, 90, 120, 140, 200, 300],
        labels=[
            "<90 (hypotension)",
            "90-120 (normal)",
            "120-140 (elevated)",
            "140-200 (high)",
            ">200 (very high)",
        ],
    )
    print("\nESI distribution by SBP category:")
    sbp_esi = (
        df_valid.groupby("sbp_cat")["esi_level"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    print(sbp_esi.round(3))

    # O2 Sat analysis
    df_valid["o2_cat"] = pd.cut(
        df_valid["o2_sat"],
        bins=[0, 90, 92, 95, 100, 101],
        labels=[
            "<90 (critical)",
            "90-92 (low)",
            "92-95 (mild)",
            "95-100 (normal)",
            ">100",
        ],
    )
    print("\nESI distribution by O2 Sat category:")
    o2_esi = (
        df_valid.groupby("o2_cat")["esi_level"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    print(o2_esi.round(3))

    # Pain analysis
    print("\nESI distribution by Pain level:")
    pain_esi = (
        df_valid.groupby("pain")["esi_level"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    print(pain_esi.round(3))

    print("\n" + "=" * 60)
    print("REASON FOR VISIT (RFV) ANALYSIS")
    print("=" * 60)
    if Path(DATA_DIR / "rfv_code_mappings.json").exists():
        with open(DATA_DIR / "rfv_code_mappings.json") as f:
            rfv_mappings = json.load(f)
        print(f"✅ RFV mappings loaded: {len(rfv_mappings.get('rfv1', {}))} codes")
        print("\nTop 10 RFV1 codes:")
        top_rfv = df["rfv1"].value_counts().head(10)
        for code, count in top_rfv.items():
            code_str = f"{code:.0f}" if pd.notna(code) else "N/A"
            desc = rfv_mappings.get("rfv1", {}).get(code_str, "Unknown")
            print(f"  {code_str}: {desc} ({count:,} visits)")

    print("\n" + "=" * 60)
    print("CHRONIC CONDITIONS")
    print("=" * 60)
    chronic_vars = [
        "cebvd",
        "chf",
        "ed_dialysis",
        "hiv",
        "diabetes",
        "no_chronic_conditions",
    ]
    print("Prevalence:")
    for var in chronic_vars:
        if var in df.columns:
            count = (df[var] == 1).sum()
            pct = (df[var] == 1).sum() / len(df) * 100
            print(f"  {var}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("DATA QUALITY FOR MODEL TRAINING")
    print("=" * 60)
    df_clean = df[
        df["esi_level"].isin([1, 2, 3, 4, 5])
        & (df["pulse"] > 0)
        & (df["pulse"] < 300)
        & (df["respiration"] > 0)
        & (df["respiration"] < 60)
        & (df["sbp"] > 0)
        & (df["sbp"] < 300)
        & (df["dbp"] > 0)
        & (df["dbp"] < 200)
        & (df["o2_sat"] >= 0)
        & (df["o2_sat"] <= 100)
        & (df["temp_c"] > 30)
        & (df["temp_c"] < 45)
        & (df["gcs"] >= 3)
        & (df["gcs"] <= 15)
        & (df["pain"] >= 0)
        & (df["pain"] <= 10)
    ].copy()
    print(
        f"✅ Clean records for training: {len(df_clean):,} "
        f"({len(df_clean)/len(df)*100:.1f}%)"
    )
    print("   ESI distribution in clean data:")
    print(df_clean["esi_level"].value_counts().sort_index())


if __name__ == "__main__":
    main()
