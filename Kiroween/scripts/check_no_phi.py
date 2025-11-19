#!/usr/bin/env python3
"""
No PHI Data Check Script

This script ensures no Protected Health Information (PHI) is committed to the repository.
It checks for patterns that might indicate real patient data.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_file_for_phi(file_path: Path) -> List[Tuple[int, str]]:
    """Check a file for potential PHI patterns."""
    phi_violations = []
    
    # PHI patterns to detect
    patterns = {
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'Phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'MRN': r'\b[A-Z]{2,3}\d{6,10}\b',
        'Date of Birth': r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b',
        'Credit Card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'Real Names': r'\b(John|Jane|Michael|Sarah|David|Lisa|Robert|Mary)\s+(Smith|Johnson|Williams|Brown|Jones|Garcia|Miller|Davis)\b',
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        for line_no, line in enumerate(content.split('\n'), 1):
            for pattern_name, pattern in patterns.items():
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Skip if it's in a comment explaining the pattern
                    if '# Example:' in line or '# Pattern:' in line or 'TODO' in line:
                        continue
                        
                    phi_violations.append((
                        line_no,
                        f"Potential {pattern_name} detected: {match.group()}"
                    ))
                    
    except Exception as e:
        phi_violations.append((0, f"Error reading file: {str(e)}"))
        
    return phi_violations


def main() -> int:
    """Main PHI check function."""
    print("🛡️  Checking for PHI data...")
    
    # Get files to check from command line arguments
    files_to_check = []
    if len(sys.argv) > 1:
        files_to_check = [Path(arg) for arg in sys.argv[1:] if Path(arg).exists()]
    else:
        # Check all relevant files if no arguments provided
        for directory in ['services', 'shared', 'data', 'apps']:
            if Path(directory).exists():
                for file_path in Path(directory).rglob('*'):
                    if file_path.is_file() and file_path.suffix in ['.py', '.json', '.csv', '.txt', '.md']:
                        files_to_check.append(file_path)
    
    total_violations = 0
    
    for file_path in files_to_check:
        violations = check_file_for_phi(file_path)
        
        if violations:
            print(f"\n📁 {file_path}")
            for line_no, violation in violations:
                print(f"  ⚠️  Line {line_no}: {violation}")
                total_violations += 1
    
    print(f"\n📊 PHI Check Results:")
    print(f"   Files checked: {len(files_to_check)}")
    print(f"   Potential PHI violations: {total_violations}")
    
    if total_violations > 0:
        print("\n❌ PHI check failed! Potential real patient data detected.")
        print("   Please ensure all data is synthetic and remove any real PHI.")
        return 1
    else:
        print("\n✅ PHI check passed! No real patient data detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())