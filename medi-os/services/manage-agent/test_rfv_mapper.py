"""
Test RFV text-to-code mapping for inference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml.rfv_mapper import RFVTextToCodeMapper

if __name__ == "__main__":
    print("Testing RFV Text-to-Code Mapper\n")
    
    try:
        mapper = RFVTextToCodeMapper.load()
        print("[OK] Mapper loaded successfully\n")
        
        # Test cases
        test_cases = [
            ("Chest pain", "rfv1"),
            ("Abdominal pain", "rfv1"),
            ("Shortness of breath", "rfv2"),
            ("Headache", "rfv1"),
            ("Fever", "rfv2"),
        ]
        
        print("Testing text → code conversion:")
        print("-" * 60)
        
        for text, field in test_cases:
            code = mapper.text_to_code(text, field)
            if code:
                # Reverse lookup for verification
                text_back = mapper.code_to_text(code, field)
                print(f"  '{text}' → {code:.0f} → '{text_back}'")
            else:
                print(f"  '{text}' → NOT FOUND")
        
        print("\n[OK] Mapper test complete!")
        
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        print("   Please run the parser first to generate mappings.")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

