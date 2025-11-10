"""
NHAMCS Data Parser Module.

Parses fixed-width format NHAMCS ED data files from 2011-2022,
extracting vital signs and acuity scores for triage model training.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from .spss_field_extractor import extract_fields_from_sps, extract_value_labels


class NHAMCSParser:
    """
    Parser for NHAMCS Emergency Department fixed-width data files.
    
    Handles parsing of multiple years of data with varying field positions,
    data cleaning, and conversion to standardized format for ML training.
    """
    
    # Fields to extract from NHAMCS data
    REQUIRED_FIELDS = [
        # Demographics
        "VMONTH", "VDAYR", "AGE", "SEX",
        # Vital signs
        "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "GCS", "ONO2",
        # Acuity and pain
        "IMMEDR", "PAINSCALE",
        # Reason for Visit (RFV) - critical for triage
        "RFV1", "RFV2", "RFV3",  # Detailed reason for visit codes
        "RFV13D", "RFV23D", "RFV33D",  # Broader categories
        # Visit characteristics
        "WAITTIME", "LOV", "ARREMS",  # Wait time, length of visit, ambulance arrival
        # Injury and diagnosis
        "INJURY", "DIAG1", "DIAG2", "DIAG3",  # Injury flag and diagnoses
        # Past history
        "PASTVIS", "SEEN72", "DISCH7DA",  # Past visits, seen in 72h, discharged in 7d
        # Comorbidities
        "CEBVD", "CHF", "EDDIAL", "EDHIV", "DIABETES", "NOCHRON",
    ]
    
    # Missing value codes in NHAMCS
    MISSING_CODES = [-9, -8, -7]
    
    def __init__(self):
        """Initialize the NHAMCS parser."""
        self.rfv_mappings: Dict[str, Dict[float, str]] = {}
    
    def parse_year(
        self, year: int, data_dir: str
    ) -> Optional[pd.DataFrame]:
        """
        Parse a single year of NHAMCS data.
        
        Args:
            year: Year to parse (e.g., 2011).
            data_dir: Directory containing .sps and raw data files.
        
        Returns:
            Cleaned DataFrame with standardized columns, or None if files not found.
        """
        data_path = Path(data_dir)
        
        # Find .sps file (handle case variations)
        sps_patterns = [f"ed{year}.sps", f"ED{year}.sps"]
        sps_file = None
        for pattern in sps_patterns:
            potential_file = data_path / pattern
            if potential_file.exists():
                sps_file = potential_file
                break
        
        if not sps_file:
            print(f"Warning: .sps file not found for year {year}")
            return None
        
        # Find raw data file (handle case variations and no extension)
        data_patterns = [f"ed{year}", f"ED{year}"]
        data_file = None
        for pattern in data_patterns:
            potential_file = data_path / pattern
            if potential_file.exists():
                data_file = potential_file
                break
        
        if not data_file:
            print(f"Warning: Raw data file not found for year {year}")
            return None
        
        print(f"Parsing {year}: {sps_file.name} -> {data_file.name}")
        
        # Extract field positions from .sps file
        try:
            all_fields = extract_fields_from_sps(str(sps_file))
        except Exception as e:
            print(f"Error extracting fields from {sps_file}: {e}")
            return None
        
        # Check if all required fields exist
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in all_fields]
        if missing_fields:
            print(f"Warning: Missing fields in {year}: {missing_fields}")
            # Continue with available fields
        
        # Build column specs for pandas.read_fwf
        # Only extract required fields that exist
        colspecs = []
        names = []
        for field in self.REQUIRED_FIELDS:
            if field in all_fields:
                start, end = all_fields[field]
                colspecs.append((start, end))
                names.append(field)
        
        # Read fixed-width file
        try:
            df = pd.read_fwf(
                str(data_file),
                colspecs=colspecs,
                names=names,
                dtype=str,  # Read as strings first to handle missing codes
                encoding="latin-1",  # Common encoding for NHAMCS files
                on_bad_lines="skip",
            )
        except Exception as e:
            print(f"Error reading {data_file}: {e}")
            return None
        
        if df.empty:
            print(f"Warning: No data read from {data_file}")
            return None
        
        # Convert to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Handle missing values (codes: -9, -8, -7, and NaN)
        for col in df.columns:
            df[col] = df[col].replace(self.MISSING_CODES, np.nan)
        
        # Remove rows with missing IMMEDR (target variable - required)
        if "IMMEDR" in df.columns:
            initial_rows = len(df)
            df = df.dropna(subset=["IMMEDR"])
            removed = initial_rows - len(df)
            if removed > 0:
                print(f"  Removed {removed} rows with missing IMMEDR")
        
        # Convert TEMPF to Celsius
        # TEMPF is stored as tenths of Fahrenheit (e.g., 986 = 98.6°F)
        # Formula: (temp_f/10 - 32) * 5/9
        if "TEMPF" in df.columns:
            df["temp_c"] = (df["TEMPF"] / 10.0 - 32) * 5 / 9
            df = df.drop(columns=["TEMPF"])
        else:
            df["temp_c"] = np.nan
        
        # Fill missing vitals with median
        vital_cols = ["AGE", "BPSYS", "BPDIAS", "PULSE", "RESPR", "POPCT", "PAINSCALE", "GCS", "WAITTIME", "LOV"]
        for col in vital_cols:
            if col in df.columns:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
        
        # Fill missing temp_c with median
        if "temp_c" in df.columns:
            median_temp = df["temp_c"].median()
            if pd.notna(median_temp):
                df["temp_c"] = df["temp_c"].fillna(median_temp)
        
        # Fill missing categorical/binary fields with 0 or mode
        binary_cols = ["INJURY", "ARREMS", "SEEN72", "DISCH7DA", "CEBVD", "CHF", "EDDIAL", "EDHIV", "DIABETES", "ONO2"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill RFV fields with 0 (missing = no reason listed)
        rfv_cols = ["RFV1", "RFV2", "RFV3", "RFV13D", "RFV23D", "RFV33D"]
        for col in rfv_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill past visit counts with 0
        if "PASTVIS" in df.columns:
            df["PASTVIS"] = df["PASTVIS"].fillna(0)
        
        # Rename columns to standardized names BEFORE creating text columns
        rename_map = {
            "VMONTH": "month",
            "VDAYR": "day_of_week",
            "AGE": "age",
            "SEX": "sex",
            "PULSE": "pulse",
            "RESPR": "respiration",
            "BPSYS": "sbp",
            "BPDIAS": "dbp",
            "POPCT": "o2_sat",
            "GCS": "gcs",
            "ONO2": "on_oxygen",
            "IMMEDR": "esi_level",
            "PAINSCALE": "pain",
            "RFV1": "rfv1",
            "RFV2": "rfv2",
            "RFV3": "rfv3",
            "RFV13D": "rfv1_3d",
            "RFV23D": "rfv2_3d",
            "RFV33D": "rfv3_3d",
            "WAITTIME": "wait_time",
            "LOV": "length_of_visit",
            "ARREMS": "ambulance_arrival",
            "INJURY": "injury",
            "DIAG1": "diag1",
            "DIAG2": "diag2",
            "DIAG3": "diag3",
            "PASTVIS": "past_visits",
            "SEEN72": "seen_72h",
            "DISCH7DA": "discharged_7d",
            "CEBVD": "cebvd",
            "CHF": "chf",
            "EDDIAL": "ed_dialysis",
            "EDHIV": "hiv",
            "DIABETES": "diabetes",
            "NOCHRON": "no_chronic_conditions",
        }
        
        df = df.rename(columns=rename_map)
        
        # Extract and store RFV code-to-text mappings for inference
        # Keep RFV columns as numeric codes (efficient for ML training)
        rfv_fields_to_label = {
            "rfv1": "RFV1",
            "rfv2": "RFV2", 
            "rfv3": "RFV3",
            "rfv1_3d": "RFV13D",
            "rfv2_3d": "RFV23D",
            "rfv3_3d": "RFV33D"
        }
        
        rfv_found = []
        for field_name, label_field in rfv_fields_to_label.items():
            if field_name in df.columns:
                try:
                    # Extract value labels from .sps file (code → text mapping)
                    value_labels = extract_value_labels(str(sps_file), label_field)
                    
                    if value_labels:
                        # Store mappings for this field (merge across years)
                        if field_name not in self.rfv_mappings:
                            self.rfv_mappings[field_name] = {}
                        # Update with new mappings (later years may have more codes)
                        self.rfv_mappings[field_name].update(value_labels)
                        rfv_found.append(field_name)
                except Exception as e:
                    print(f"  Warning: Error extracting labels for {field_name}: {e}")
        
        if rfv_found:
            print(f"  Keeping RFV columns as numeric codes: {rfv_found}")
            total_mappings = sum(len(m) for m in [self.rfv_mappings.get(f, {}) for f in rfv_found])
            print(f"  Stored {total_mappings} RFV code mappings for inference")
            # Ensure they're numeric (float, handle missing values)
            for field in rfv_found:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        # Keep all extracted columns (don't limit to just vitals)
        # This preserves all the rich features for ML training including numeric RFV codes
        
        print(f"  Parsed {len(df)} records from {year}")
        
        return df
    
    def parse_all_years(
        self, data_dir: str, years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Parse multiple years of NHAMCS data and combine.
        
        Args:
            data_dir: Directory containing .sps and raw data files.
            years: List of years to parse. Defaults to 2011-2022.
        
        Returns:
            Combined DataFrame with all years.
        """
        if years is None:
            years = list(range(2011, 2023))
        
        dataframes = []
        parsed_years = []
        
        for year in years:
            df = self.parse_year(year, data_dir)
            if df is not None and not df.empty:
                df["year"] = year
                dataframes.append(df)
                parsed_years.append(year)
        
        if not dataframes:
            print("No data successfully parsed!")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("PARSING SUMMARY")
        print("=" * 60)
        print(f"Total records: {len(combined_df):,}")
        print(f"Years parsed: {parsed_years}")
        print(f"\nESI Level distribution:")
        if "esi_level" in combined_df.columns:
            print(combined_df["esi_level"].value_counts().sort_index())
        print(f"\nVital signs statistics:")
        vital_cols = ["age", "sbp", "dbp", "pulse", "respiration", "temp_c", "pain", "o2_sat"]
        available_vital_cols = [col for col in vital_cols if col in combined_df.columns]
        if available_vital_cols:
            vital_stats = combined_df[available_vital_cols].describe()
            print(vital_stats)
        
        print(f"\nReason for Visit (RFV) distribution:")
        rfv_cols = ["rfv1_3d", "rfv2_3d", "rfv3_3d"]
        available_rfv_cols = [col for col in rfv_cols if col in combined_df.columns]
        if available_rfv_cols:
            for col in available_rfv_cols[:1]:  # Show first RFV category
                print(f"\n{col}:")
                print(combined_df[col].value_counts().head(10))
        
        print(f"\nInjury visit percentage:")
        if "injury" in combined_df.columns:
            injury_pct = (combined_df["injury"] == 1).sum() / len(combined_df) * 100
            print(f"  {injury_pct:.1f}% of visits are injury-related")
        
        print(f"\nAmbulance arrival percentage:")
        if "ambulance_arrival" in combined_df.columns:
            amb_pct = (combined_df["ambulance_arrival"] == 1).sum() / len(combined_df) * 100
            print(f"  {amb_pct:.1f}% arrived by ambulance")
        print("=" * 60)
        
        # Save RFV mappings after parsing all years
        if self.rfv_mappings:
            data_dir_path = Path(data_dir)
            mappings_path = data_dir_path / "rfv_code_mappings.json"
            self.save_rfv_mappings(str(mappings_path))
        
        return combined_df
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save parsed DataFrame to CSV with metadata.
        
        Args:
            df: DataFrame to save.
            output_path: Path to output CSV file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        
        print(f"\nSaved {len(df):,} records to {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    def save_rfv_mappings(self, output_path: str):
        """
        Save RFV code-to-text mappings to JSON file for inference.
        
        Args:
            output_path: Path to save JSON file (e.g., 'data/rfv_code_mappings.json')
        """
        # Convert float keys to strings for JSON serialization
        json_mappings = {}
        for field_name, code_to_text in self.rfv_mappings.items():
            json_mappings[field_name] = {
                str(code): text for code, text in code_to_text.items()
            }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_mappings, f, indent=2, ensure_ascii=False)
        
        total_mappings = sum(len(m) for m in json_mappings.values())
        print(f"\nSaved {total_mappings} RFV code mappings to {output_path}")


if __name__ == "__main__":
    """Main execution block for parsing NHAMCS data."""
    
    # Data directory path
    data_dir = r"D:\Hackathons\Cloud Run\medi-os\data"
    
    # Initialize parser
    parser = NHAMCSParser()
    
    # Parse all years (2011-2022)
    print("Starting NHAMCS data parsing...")
    print(f"Data directory: {data_dir}\n")
    
    combined_df = parser.parse_all_years(
        data_dir=data_dir,
        years=list(range(2011, 2023))
    )
    
    if not combined_df.empty:
        # Save to CSV
        output_path = Path(data_dir) / "NHAMCS_2011_2022_combined.csv"
        parser.save_to_csv(combined_df, str(output_path))
        print("\nParsing complete!")
    else:
        print("\nParsing failed - no data to save.")

