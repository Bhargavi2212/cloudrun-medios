"""
Regenerate NHAMCS combined CSV with numeric RFV codes (not text).

This script regenerates the combined CSV file after modifying the parser
to keep RFV columns as numeric codes instead of converting to text.
"""

import sys
from pathlib import Path
import pandas as pd

# Add the services/manage-agent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core.nhamcs_parser import NHAMCSParser

if __name__ == "__main__":
    data_dir = str(Path(__file__).parent.parent.parent / 'data')
    parser = NHAMCSParser()
    
    print("Regenerating NHAMCS CSV with numeric RFV codes...")
    print(f"Data directory: {data_dir}\n")
    
    combined_df = parser.parse_all_years(
        data_dir=data_dir,
        years=list(range(2011, 2023))
    )
    
    if not combined_df.empty:
        output_path = Path(data_dir) / "NHAMCS_2011_2022_combined.csv"
        parser.save_to_csv(combined_df, str(output_path))
        
        # Verify RFV columns are numeric
        print("\nVerifying RFV columns are numeric...")
        rfv_cols = [c for c in combined_df.columns if 'rfv' in c.lower()]
        print(f"RFV columns: {rfv_cols}")
        print("\nData types:")
        for col in rfv_cols:
            print(f"  {col}: {combined_df[col].dtype}")
        
        print("\nSample RFV values (should be numeric codes):")
        print(combined_df[rfv_cols].head())
        
        # Verify mappings were saved
        mappings_path = Path(data_dir) / "rfv_code_mappings.json"
        if mappings_path.exists():
            print(f"\n✅ RFV mappings saved to: {mappings_path}")
            import json
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                total_codes = sum(len(m) for m in mappings.values())
                print(f"   Total code mappings: {total_codes}")
        else:
            print(f"\n⚠️ Warning: RFV mappings file not created")
        
        print(f"\n✅ Regeneration complete! Saved to: {output_path}")
    else:
        print("\n❌ Parsing failed - no data to save.")

