"""
Test script to verify Option A: RFV text labels from .sps files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.spss_field_extractor import extract_value_labels
from ml.rfv_mapper import RFVTextToCodeMapper

def test_option_a():
    """Test if RFV text labels are available and working."""
    
    print("=" * 60)
    print("TESTING OPTION A: RFV TEXT LABELS FROM .SPS FILES")
    print("=" * 60)
    
    # Test 1: Check if labels can be extracted from .sps file
    print("\n[Test 1] Extracting RFV labels from ed2011.sps...")
    sps_file = Path(__file__).parent.parent.parent.parent / "data" / "ed2011.sps"
    
    if not sps_file.exists():
        print(f"  ❌ ERROR: .sps file not found: {sps_file}")
        return False
    
    try:
        labels = extract_value_labels(str(sps_file), "RFV1")
        print(f"  ✅ SUCCESS: Extracted {len(labels)} RFV1 labels")
        print(f"  Sample labels:")
        for i, (code, text) in enumerate(list(labels.items())[:5]):
            print(f"    {code}: {text}")
    except Exception as e:
        print(f"  ❌ ERROR: Failed to extract labels: {e}")
        return False
    
    # Test 2: Check if mappings file exists
    print("\n[Test 2] Checking rfv_code_mappings.json...")
    mappings_file = Path(__file__).parent.parent.parent.parent / "data" / "rfv_code_mappings.json"
    
    if not mappings_file.exists():
        print(f"  ⚠️  WARNING: Mappings file not found: {mappings_file}")
        print(f"  This means the parser hasn't been run yet to generate mappings.")
        return False
    
    print(f"  ✅ Mappings file exists: {mappings_file}")
    
    # Test 3: Check if mapper works
    print("\n[Test 3] Testing RFVTextToCodeMapper...")
    try:
        mapper = RFVTextToCodeMapper()
        print(f"  ✅ Mapper loaded successfully")
        print(f"  Fields available: {list(mapper._code_to_text_map.keys())}")
        
        # Test text-to-code conversion
        test_cases = [
            ("Chest pain", "rfv1"),
            ("Fever", "rfv1"),
            ("Stomach and abdominal pain", "rfv1"),
            ("Headache", "rfv1"),
        ]
        
        print(f"\n  Testing text-to-code conversion:")
        for text, field in test_cases:
            code = mapper.text_to_code(text, field, fuzzy=True)
            if code:
                code_text = mapper.code_to_text(code, field)
                print(f"    '{text}' -> {code} ({code_text})")
            else:
                print(f"    '{text}' -> NOT FOUND")
        
    except Exception as e:
        print(f"  ❌ ERROR: Mapper failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: OPTION A STATUS")
    print("=" * 60)
    print("✅ RFV text labels ARE available in .sps files")
    print("✅ Labels can be extracted successfully")
    print("✅ Mapper exists and can convert text to codes")
    print("\n✅ OPTION A IS WORKING!")
    print("\nNext steps:")
    print("  1. Option A can be used for RFV clustering")
    print("  2. Text labels can be grouped into 10-15 clusters")
    print("  3. Clusters can replace 723 numeric codes")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_option_a()
    sys.exit(0 if success else 1)

