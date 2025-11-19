"""
Combine NHAMCS ED data files (2011-2022) into a unified dataset for triage
model training.

This script:
1. Parses .sps files to extract column definitions
2. Reads fixed-width ASCII data files
3. Combines all years with consistent schema
4. Applies data cleaning and validation
5. Exports to CSV for model training
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"D:\Hackathons\Cloud Run\medi-os\data")
OUTPUT_DIR = Path(r"D:\Hackathons\Cloud Run\Version -2\data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RFV_MAPPING_FILE = Path(
    r"D:\Hackathons\Cloud Run\medi-os\data\rfv_cluster_mapping.json"
)


def parse_sps_file(sps_path: Path) -> dict[str, tuple[int, int, bool]]:
    """
    Parse SPSS .sps file to extract column definitions.

    Returns dict mapping variable name to (start_col, end_col, is_string)
    Column numbers are 1-indexed in SPSS, we'll convert to 0-indexed for Python.
    """
    col_defs: dict[str, tuple[int, int, bool]] = {}

    with open(sps_path, encoding="utf-8", errors="ignore") as f:
        in_data_list = False
        for line in f:
            line = line.strip()

            # Start of DATA LIST section
            if "DATA LIST" in line.upper():
                in_data_list = True
                continue

            # End of DATA LIST section (usually blank line or next command)
            if in_data_list and (
                not line or line.startswith("*") or "FILE=" in line.upper()
            ):
                if line and not line.startswith("*"):
                    continue
                if not line:
                    break

            if not in_data_list:
                continue

            # Parse variable definition: VARNAME start-end or VARNAME start-end (A)
            # Examples: "PULSE 42-44" or "DIAG1 103-107 (A)"
            match = re.match(r"(\w+)\s+(\d+)-(\d+)(?:\s+\(A\))?", line)
            if match:
                var_name = match.group(1)
                start = int(match.group(2)) - 1  # Convert to 0-indexed
                end = int(match.group(3))  # End is inclusive in SPSS
                is_string = "(A)" in line
                col_defs[var_name] = (start, end, is_string)

    return col_defs


def read_fixed_width_file(
    data_path: Path, col_defs: dict[str, tuple[int, int, bool]], year: int
) -> pd.DataFrame:
    """
    Read a fixed-width ASCII file using column definitions from .sps file.
    """
    # Key variables for triage model
    triage_vars = [
        "IMMEDR",  # ESI level (target)
        "PULSE",
        "RESPR",  # Respiration
        "BPSYS",  # Systolic BP
        "BPDIAS",  # Diastolic BP
        "POPCT",  # O2 saturation
        "TEMPF",  # Temperature (Fahrenheit)
        "GCS",  # Glasgow Coma Scale
        "PAINSCALE",
        "AGE",
        "SEX",
        "VMONTH",  # Month
        "VDAYR",  # Day of week
        "WAITTIME",
        "LOV",  # Length of visit
        "ARREMS",  # Ambulance arrival
        "INJURY",
        "RFV1",  # Reason for visit
        "RFV2",
        "RFV3",
        "CEBVD",  # Chronic conditions
        "CHF",
        "EDDIAL",
        "EDHIV",
        "DIABETES",
        "NOCHRON",
        "SEEN72",  # Seen in last 72h
        "DISCH7DA",  # Discharged in last 7 days
        "PASTVIS",  # Past visits
    ]

    # Filter to variables that exist in this year's schema
    available_vars = [v for v in triage_vars if v in col_defs]

    # Read file line by line
    records = []
    max_col = max(end for _, end, _ in col_defs.values())

    try:
        with open(data_path, "rb") as f:
            for line_num, line_bytes in enumerate(f, 1):
                try:
                    line = line_bytes.decode("latin-1", errors="ignore").rstrip("\r\n")
                except Exception:
                    continue

                if len(line) < max_col:
                    line = line.ljust(max_col)

                record = {"year": year, "line_num": line_num}
                for var in available_vars:
                    start, end, is_string = col_defs[var]
                    try:
                        value = line[start:end].strip()
                        if not value or value in ["", " ", "."]:
                            record[var] = None
                        elif is_string:
                            record[var] = value
                        else:
                            # Try to convert to numeric
                            try:
                                record[var] = float(value) if value else None
                            except ValueError:
                                record[var] = None
                    except Exception:
                        record[var] = None

                records.append(record)

                # Progress indicator
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num:,} lines...", end="\r")

    except FileNotFoundError:
        print(f"  WARNING: File not found: {data_path}")
        return pd.DataFrame()

    print(f"\n  SUCCESS: Read {len(records):,} records")
    return pd.DataFrame(records)


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform the combined dataset.
    """
    df = df.copy()

    # Convert temperature from Fahrenheit to Celsius
    # IMPORTANT: Temperature is stored as integer (e.g., 986 = 98.6F, 600 = 60.0F)
    # Handle missing values BEFORE conversion to avoid invalid conversions
    if "TEMPF" in df.columns:
        # Replace missing value codes in Fahrenheit before conversion
        df["TEMPF"] = df["TEMPF"].replace([0, 9, 99, 999, 9999, -9, -8], None)
        # Check temp values before conversion
        print(
            f"  TEMPF stats (raw integer): min={df['TEMPF'].min():.1f}, max={df['TEMPF'].max():.1f}, non-null={df['TEMPF'].notna().sum():,}"  # noqa: E501
        )
        # Convert: divide by 10 to get actual Fahrenheit, then convert to Celsius
        # Only convert non-null values
        df["temp_c"] = df["TEMPF"].apply(
            lambda x: ((x / 10) - 32) * 5 / 9 if pd.notna(x) else None
        )
        print(
            f"  temp_c stats: min={df['temp_c'].min():.1f}, max={df['temp_c'].max():.1f}, non-null={df['temp_c'].notna().sum():,}"  # noqa: E501
        )
        df = df.drop(columns=["TEMPF"])

    # Rename columns to match our schema
    rename_map = {
        "IMMEDR": "esi_level",
        "PULSE": "pulse",
        "RESPR": "respiration",
        "BPSYS": "sbp",
        "BPDIAS": "dbp",
        "POPCT": "o2_sat",
        "GCS": "gcs",
        "PAINSCALE": "pain",
        "AGE": "age",
        "SEX": "sex",
        "VMONTH": "month",
        "VDAYR": "day_of_week",
        "WAITTIME": "wait_time",
        "LOV": "length_of_visit",
        "ARREMS": "ambulance_arrival",
        "INJURY": "injury",
        "RFV1": "rfv1",
        "RFV2": "rfv2",
        "RFV3": "rfv3",
        "CEBVD": "cebvd",
        "CHF": "chf",
        "EDDIAL": "ed_dialysis",
        "EDHIV": "hiv",
        "DIABETES": "diabetes",
        "NOCHRON": "no_chronic_conditions",
        "SEEN72": "seen_72h",
        "DISCH7DA": "discharged_7d",
        "PASTVIS": "past_visits",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Debug: Show data before filtering
    print(f"\n  Before filtering: {len(df):,} records")
    if "esi_level" in df.columns:
        print("  ESI level distribution (raw):")
        print(df["esi_level"].value_counts().sort_index().head(10))

    # Handle missing values - replace common missing codes with NaN
    # First handle ESI level specifically (has negative missing codes)
    if "esi_level" in df.columns:
        # Replace negative values and invalid codes with NaN
        df["esi_level"] = df["esi_level"].replace([-9, -8, 0, 7, 9, 99, 999], None)
        # Convert to int if possible (1.0 -> 1), but keep as float for comparison
  #  Actually, pandas handles float(1.0) == int(1) in isin(), so we can keep as float

    # Handle missing values for other numeric vars (temp_c already handled above)
    numeric_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "gcs",
        "pain",
        "age",
    ]
    for var in numeric_vars:
        if var in df.columns:
  #  Replace common missing value codes (0, 9, 99, 999, etc.) with NaN for numeric vars
            # But keep 0 for some vars like pain, age if it's valid
            if var in ["pain", "age"]:
                df[var] = df[var].replace([999, 9999, 99, -9, -8], None)
            else:
                df[var] = df[var].replace([0, 9, 99, 999, 9999, -9, -8], None)

    # Data quality filters - use pd.isna() to allow missing values
    # Only filter out records with invalid ESI levels or completely invalid data
    mask = pd.Series([True] * len(df), index=df.index)

    # Must have valid ESI level (1-5) - use range check to handle both int and float
    if "esi_level" in df.columns:
        esi_mask = (df["esi_level"] >= 1) & (df["esi_level"] <= 5)
        print(f"  After ESI filter: {esi_mask.sum():,} records")
        mask = mask & esi_mask

    # For numeric vitals, only filter if value exists AND is out of range
    if "pulse" in df.columns:
        pulse_mask = (df["pulse"].isna()) | ((df["pulse"] > 0) & (df["pulse"] < 300))
        print(f"  After pulse filter: {(mask & pulse_mask).sum():,} records")
        mask = mask & pulse_mask
    if "respiration" in df.columns:
        resp_mask = (df["respiration"].isna()) | (
            (df["respiration"] > 0) & (df["respiration"] < 60)
        )
        print(f"  After respiration filter: {(mask & resp_mask).sum():,} records")
        mask = mask & resp_mask
    if "sbp" in df.columns:
        sbp_mask = (df["sbp"].isna()) | ((df["sbp"] > 0) & (df["sbp"] < 300))
        print(f"  After SBP filter: {(mask & sbp_mask).sum():,} records")
        mask = mask & sbp_mask
    if "dbp" in df.columns:
        dbp_mask = (df["dbp"].isna()) | ((df["dbp"] > 0) & (df["dbp"] < 200))
        print(f"  After DBP filter: {(mask & dbp_mask).sum():,} records")
        mask = mask & dbp_mask
    if "o2_sat" in df.columns:
        o2_mask = (df["o2_sat"].isna()) | ((df["o2_sat"] >= 0) & (df["o2_sat"] <= 100))
        print(f"  After O2 sat filter: {(mask & o2_mask).sum():,} records")
        mask = mask & o2_mask
    if "temp_c" in df.columns:
        temp_mask = (df["temp_c"].isna()) | ((df["temp_c"] > 30) & (df["temp_c"] < 45))
        print(f"  After temp filter: {(mask & temp_mask).sum():,} records")
        mask = mask & temp_mask
    if "gcs" in df.columns:
        gcs_mask = (df["gcs"].isna()) | ((df["gcs"] >= 3) & (df["gcs"] <= 15))
        print(f"  After GCS filter: {(mask & gcs_mask).sum():,} records")
        mask = mask & gcs_mask
    if "pain" in df.columns:
        pain_mask = (df["pain"].isna()) | ((df["pain"] >= 0) & (df["pain"] <= 10))
        print(f"  After pain filter: {(mask & pain_mask).sum():,} records")
        mask = mask & pain_mask
    if "age" in df.columns:
        age_mask = (df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 120))
        print(f"  After age filter: {(mask & age_mask).sum():,} records")
        mask = mask & age_mask

    df_clean = df[mask].copy()

    # Convert binary flags (1/2 coding to 0/1)
    binary_vars = [
        "cebvd",
        "chf",
        "ed_dialysis",
        "hiv",
        "diabetes",
        "no_chronic_conditions",
        "injury",
    ]
    for var in binary_vars:
        if var in df_clean.columns:
            df_clean[var] = (df_clean[var] == 1).astype(int)

    # Handle missing values in categorical vars
    categorical_vars = ["sex", "ambulance_arrival", "seen_72h", "discharged_7d"]
    for var in categorical_vars:
        if var in df_clean.columns:
            df_clean[var] = df_clean[var].replace([0, 9, 99, 999], None)

    # Map RFV1 codes to clusters and drop RFV2/RFV3
    if "rfv1" in df_clean.columns:
        print("\n  Mapping RFV1 codes to clusters...")

        # Load RFV cluster mapping
        if RFV_MAPPING_FILE.exists():
            with open(RFV_MAPPING_FILE) as f:
                rfv_mapping = json.load(f)

            # Map RFV1 codes to clusters
            def map_rfv_to_cluster(rfv_code):
                """Map RFV code to cluster."""
                if pd.isna(rfv_code) or rfv_code <= 0:
                    return None
                rfv_str = str(rfv_code)
                cluster = rfv_mapping.get("code_to_cluster", {}).get(rfv_str, "Unknown")
                return cluster

            df_clean["rfv1_cluster"] = df_clean["rfv1"].apply(map_rfv_to_cluster)

            # Count clusters
            cluster_counts = df_clean["rfv1_cluster"].value_counts()
            print(f"  RFV1 clusters created: {len(cluster_counts)} categories")
            print(f"  Top 5 clusters: {dict(cluster_counts.head(5))}")

            # Drop original RFV1 code (keep only cluster)
            df_clean = df_clean.drop(columns=["rfv1"])
        else:
            print(f"  WARNING: RFV mapping file not found at {RFV_MAPPING_FILE}")
            print("  Keeping RFV1 as numeric code (clustering skipped)")

    # Drop RFV2 and RFV3 columns
    if "rfv2" in df_clean.columns:
        df_clean = df_clean.drop(columns=["rfv2"])
        print("  Dropped RFV2 column")
    if "rfv3" in df_clean.columns:
        df_clean = df_clean.drop(columns=["rfv3"])
        print("  Dropped RFV3 column")

    return df_clean


def main() -> None:
    """Combine all NHAMCS ED data files."""
    print("=" * 70)
    print("NHAMCS DATA COMBINATION PIPELINE")
    print("=" * 70)

    years = list(range(2011, 2023))  # 2011-2022
    all_dataframes: list[pd.DataFrame] = []

    for year in years:
        print(f"\nProcessing year {year}...")

        # Find .sps and data files
        sps_file = DATA_DIR / f"ed{year}.sps"
        data_file = DATA_DIR / f"ED{year}"

        # Handle 2012 which might have .sav file
        if year == 2012 and not data_file.exists():
            data_file = DATA_DIR / "ed2012-spss.sav"
            if data_file.exists():
                print("  Using SPSS .sav file for 2012")
  #  For .sav, we'd need pyreadstat, skip for now or use existing combined CSV
                continue

        if not sps_file.exists():
            print(f"  WARNING: .sps file not found: {sps_file}")
            continue

        if not data_file.exists():
            print(f"  WARNING: Data file not found: {data_file}")
            continue

        # Parse .sps to get column definitions
        print(f"  Parsing {sps_file.name}...")
        col_defs = parse_sps_file(sps_file)
        print(f"  Found {len(col_defs)} variable definitions")

        # Read fixed-width file
        print(f"  Reading {data_file.name}...")
        df_year = read_fixed_width_file(data_file, col_defs, year)

        if len(df_year) > 0:
            all_dataframes.append(df_year)
            print(f"  SUCCESS: Year {year}: {len(df_year):,} records")

    if not all_dataframes:
        print("\nERROR: No data files processed. Check file paths.")
        return

    # Combine all years
    print("\n" + "=" * 70)
    print("COMBINING ALL YEARS")
    print("=" * 70)
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"SUCCESS: Combined dataset: {len(df_combined):,} records")
    print(
        f"   Years: {df_combined['year'].min():.0f} - {df_combined['year'].max():.0f}"
    )

    # Clean and transform
    print("\n" + "=" * 70)
    print("CLEANING AND TRANSFORMING DATA")
    print("=" * 70)
    df_clean = clean_and_transform(df_combined)
    print(
        f"SUCCESS: Clean dataset: {len(df_clean):,} records ({len(df_clean)/len(df_combined)*100:.1f}% retained)"  # noqa: E501
    )

    # Summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(df_clean):,}")
    print("\nESI Level distribution:")
    print(df_clean["esi_level"].value_counts().sort_index())
    print("\nKey variables:")
    key_vars = [
        "pulse",
        "respiration",
        "sbp",
        "dbp",
        "o2_sat",
        "temp_c",
        "gcs",
        "pain",
        "age",
    ]
    print(df_clean[key_vars].describe())

    # Save to CSV
    output_file = OUTPUT_DIR / "nhamcs_triage_dataset.csv"
    print(f"\nSaving to {output_file}...")
    df_clean.to_csv(output_file, index=False)
    print(
        f"SUCCESS: Dataset saved: {len(df_clean):,} records, {len(df_clean.columns)} columns"  # noqa: E501
    )
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
