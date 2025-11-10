"""
Analyze RFV Code Distribution and Frequency

Diagnostic script to understand:
1. Which RFV codes are most common
2. How many unique codes exist
3. Distribution across ESI levels
4. Semantic grouping potential
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'manage-agent'))

from ml.rfv_mapper import RFVTextToCodeMapper


def print_section(title, char="="):
    """Print a formatted section header."""
    print("\n" + char * 80)
    print(f"{title}")
    print(char * 80)


def load_rfv_mappings():
    """Load RFV code-to-text mappings."""
    mappings_path = project_root / "data" / "rfv_code_mappings.json"
    if not mappings_path.exists():
        print(f"ERROR: Mappings file not found: {mappings_path}")
        return None
    
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    return mappings


def analyze_rfv_distribution():
    """Analyze RFV code distribution in the dataset."""
    
    print("=" * 80)
    print("RFV CODE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Load original CSV (before one-hot encoding)
    print_section("STEP 1: LOADING DATA")
    csv_path = project_root / "data" / "NHAMCS_2011_2022_combined.csv"
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Load RFV mappings
    print_section("STEP 2: LOADING RFV MAPPINGS")
    mappings = load_rfv_mappings()
    if mappings is None:
        return
    
    rfv1_mappings = mappings.get('rfv1', {})
    print(f"  RFV1 codes with labels: {len(rfv1_mappings)}")
    
    # Analyze RFV1 distribution
    print_section("STEP 3: RFV1 CODE DISTRIBUTION")
    
    if 'rfv1' not in df.columns:
        print("ERROR: 'rfv1' column not found in CSV")
        return
    
    # Count RFV1 codes
    rfv1_counts = df['rfv1'].value_counts()
    print(f"\nTotal unique RFV1 codes in data: {len(rfv1_counts)}")
    print(f"Non-zero RFV1 codes: {(df['rfv1'] != 0).sum():,} ({100 * (df['rfv1'] != 0).sum() / len(df):.1f}%)")
    print(f"Zero/missing RFV1 codes: {(df['rfv1'] == 0).sum():,} ({100 * (df['rfv1'] == 0).sum() / len(df):.1f}%)")
    
    # Top 50 most common RFV1 codes
    print("\nTop 50 Most Common RFV1 Codes:")
    print("-" * 80)
    print(f"{'Code':<12} {'Count':<12} {'%':<8} {'Text Label':<50}")
    print("-" * 80)
    
    top_codes = []
    for code, count in rfv1_counts.head(50).items():
        code_str = str(code)
        if code_str in rfv1_mappings:
            label = rfv1_mappings[code_str]
            if len(label) > 47:
                label = label[:44] + "..."
        else:
            label = "N/A"
        
        pct = 100 * count / len(df)
        print(f"{code:<12} {count:<12} {pct:<8.2f} {label:<50}")
        top_codes.append((code, count, label))
    
    # Analyze RFV1 by ESI level
    print_section("STEP 4: RFV1 DISTRIBUTION BY ESI LEVEL")
    
    if 'esi_level' not in df.columns:
        print("WARNING: 'esi_level' column not found")
    else:
        esi_levels = sorted(df['esi_level'].dropna().unique())
        print(f"\nESI levels: {esi_levels}")
        
        # For each ESI level, show top 10 RFV1 codes
        for esi in esi_levels:
            esi_df = df[df['esi_level'] == esi]
            if len(esi_df) == 0:
                continue
            
            esi_rfv1_counts = esi_df['rfv1'].value_counts()
            print(f"\nESI {esi} (n={len(esi_df):,}):")
            print(f"  Top 10 RFV1 codes:")
            for code, count in esi_rfv1_counts.head(10).items():
                code_str = str(code)
                label = rfv1_mappings.get(code_str, "N/A")
                if len(label) > 60:
                    label = label[:57] + "..."
                pct = 100 * count / len(esi_df)
                print(f"    {code}: {count:,} ({pct:.1f}%) - {label}")
    
    # Analyze code ranges
    print_section("STEP 5: RFV1 CODE RANGE ANALYSIS")
    
    non_zero_rfv1 = df[df['rfv1'] != 0]['rfv1']
    print(f"\nRFV1 code statistics (non-zero only):")
    print(f"  Min: {non_zero_rfv1.min():.0f}")
    print(f"  Max: {non_zero_rfv1.max():.0f}")
    print(f"  Mean: {non_zero_rfv1.mean():.2f}")
    print(f"  Median: {non_zero_rfv1.median():.0f}")
    
    # Code range groups (based on NHAMCS structure)
    print("\nCode range groups (NHAMCS structure):")
    ranges = {
        "10000-10999": "General symptoms",
        "11000-11999": "Psychological symptoms",
        "12000-12999": "Neurological symptoms",
        "13000-13999": "Cardiovascular symptoms",
        "14000-14999": "Respiratory symptoms",
        "15000-15999": "Gastrointestinal symptoms",
        "16000-16999": "Urinary symptoms",
        "17000-17999": "Reproductive symptoms",
        "18000-18999": "Skin symptoms",
        "19000-19999": "Musculoskeletal symptoms",
        "20000-29999": "Diseases and conditions",
        "30000-39999": "Examinations and tests",
        "40000-49999": "Treatments and procedures",
        "50000-59999": "Injuries",
        "60000-69999": "Follow-up and results",
        "70000-79999": "Administrative and other",
    }
    
    for code_range, description in ranges.items():
        start, end = map(int, code_range.split('-'))
        count = ((non_zero_rfv1 >= start) & (non_zero_rfv1 <= end)).sum()
        if count > 0:
            pct = 100 * count / len(non_zero_rfv1)
            print(f"  {code_range}: {count:,} ({pct:.1f}%) - {description}")
    
    # Semantic grouping analysis
    print_section("STEP 6: SEMANTIC GROUPING ANALYSIS")
    
    print("\nAnalyzing semantic groups for clustering...")
    
    # Group codes by keywords in their labels
    semantic_groups = {
        "Chest/Cardiac": [],
        "Respiratory/Breathing": [],
        "Gastrointestinal/Abdominal": [],
        "Neurological/Head": [],
        "Musculoskeletal/Pain": [],
        "Skin/Rash": [],
        "Urinary/Genitourinary": [],
        "Fever/Infection": [],
        "Trauma/Injury": [],
        "Mental Health/Psychological": [],
    }
    
    keywords = {
        "Chest/Cardiac": ["chest", "heart", "cardiac", "palpitation"],
        "Respiratory/Breathing": ["breath", "cough", "wheez", "respir", "sputum"],
        "Gastrointestinal/Abdominal": ["stomach", "abdominal", "nausea", "vomit", "diarrhea", "constipation", "bowel"],
        "Neurological/Head": ["headache", "head", "dizzy", "vertigo", "seizure", "convulsion"],
        "Musculoskeletal/Pain": ["back", "neck", "shoulder", "knee", "joint", "muscle", "fracture", "sprain"],
        "Skin/Rash": ["rash", "skin", "lesion", "wound", "burn"],
        "Urinary/Genitourinary": ["urine", "urinary", "bladder", "kidney", "menstrual", "vaginal"],
        "Fever/Infection": ["fever", "infection", "chills", "sepsis"],
        "Trauma/Injury": ["injury", "trauma", "accident", "fracture", "laceration"],
        "Mental Health/Psychological": ["depression", "anxiety", "psych", "suicide", "alcohol", "drug"],
    }
    
    # Classify codes into semantic groups
    for code_str, label in rfv1_mappings.items():
        label_lower = label.lower()
        classified = False
        
        for group, group_keywords in keywords.items():
            if any(keyword in label_lower for keyword in group_keywords):
                semantic_groups[group].append((code_str, label))
                classified = True
                break
        
        if not classified:
            # Add to "Other" if not classified
            if "Other" not in semantic_groups:
                semantic_groups["Other"] = []
            semantic_groups["Other"].append((code_str, label))
    
    print("\nSemantic groups (based on label keywords):")
    for group, codes in semantic_groups.items():
        if codes:
            print(f"  {group}: {len(codes)} codes")
            # Show top 5 codes in this group by frequency
            group_codes_numeric = [float(code_str) for code_str, _ in codes]
            group_counts = df[df['rfv1'].isin(group_codes_numeric)]['rfv1'].value_counts()
            if len(group_counts) > 0:
                print(f"    Top 5 by frequency:")
                for code, count in group_counts.head(5).items():
                    code_str = str(code)
                    label = rfv1_mappings.get(code_str, "N/A")
                    if len(label) > 50:
                        label = label[:47] + "..."
                    print(f"      {code}: {count:,} - {label}")
    
    # Summary
    print_section("STEP 7: SUMMARY AND RECOMMENDATIONS")
    
    print("\nKey Findings:")
    print(f"  1. Total unique RFV1 codes: {len(rfv1_counts)}")
    print(f"  2. Top 50 codes cover: {100 * rfv1_counts.head(50).sum() / len(df):.1f}% of records")
    print(f"  3. Top 10 codes cover: {100 * rfv1_counts.head(10).sum() / len(df):.1f}% of records")
    print(f"  4. Semantic groups identified: {len([g for g in semantic_groups.values() if g])}")
    
    print("\nClustering Recommendations:")
    print("  1. Medical domain clustering: Group codes by semantic categories (10-15 clusters)")
    print("  2. Frequency-based clustering: Top 10-15 codes as individual clusters, rest as 'Other'")
    print("  3. Hybrid approach: Semantic groups for common codes, frequency for rare codes")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_rfv_distribution()

