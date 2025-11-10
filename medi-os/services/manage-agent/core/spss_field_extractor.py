"""
SPSS Field Extractor Module.

Extracts field positions from SPSS definition files (.sps) for parsing
fixed-width format NHAMCS data files.
"""

import re
from pathlib import Path
from typing import Dict, Tuple


def extract_fields_from_sps(sps_file_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Extract field positions from SPSS .sps file.

    Parses the DATA LIST section and extracts field names with their
    start and end positions, converting from 1-indexed SPSS format
    to 0-indexed Python slicing format.

    Args:
        sps_file_path: Path to the .sps definition file.

    Returns:
        Dictionary mapping field names to (start, end) tuples for Python slicing.
        Format: {'VMONTH': (0, 2), 'AGE': (3, 6), ...}

    Raises:
        FileNotFoundError: If the .sps file doesn't exist.
        ValueError: If the DATA LIST section cannot be found or parsed.
    """
    sps_path = Path(sps_file_path)
    
    if not sps_path.exists():
        raise FileNotFoundError(f"SPSS definition file not found: {sps_file_path}")
    
    with open(sps_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Find DATA LIST section
    # Pattern: DATA LIST FILE=TEMP / followed by field definitions
    data_list_match = re.search(
        r"DATA\s+LIST\s+(?:FILE=\w+\s*)?/.*?\n((?:\s+\w+\s+\d+(?:-\d+)?.*?\n)+)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    
    if not data_list_match:
        raise ValueError(f"Could not find DATA LIST section in {sps_file_path}")
    
    field_section = data_list_match.group(1)
    
    # Pattern to match field definitions:
    # - Single position: FIELDNAME 21
    # - Range: FIELDNAME 1-2 or FIELDNAME 38-41
    # - With type: FIELDNAME 7-10 (A)
    field_pattern = re.compile(r"(\w+)\s+(\d+)(?:-(\d+))?(?:\s*\([A-Z]\))?")
    
    fields: Dict[str, Tuple[int, int]] = {}
    
    for line in field_section.split("\n"):
        match = field_pattern.search(line.strip())
        if match:
            field_name = match.group(1)
            start_pos = int(match.group(2))  # 1-indexed SPSS position
            
            # If no end position specified, it's a single character field
            if match.group(3):
                end_pos = int(match.group(3))  # 1-indexed SPSS position
            else:
                end_pos = start_pos  # Single character field
            
            # Convert to 0-indexed Python slicing: (start-1, end)
            # SPSS 1-2 means characters at positions 1 and 2 (1-indexed)
            # Python [0:2] gives characters at indices 0 and 1 (0-indexed)
            python_start = start_pos - 1
            python_end = end_pos
            
            fields[field_name] = (python_start, python_end)
    
    if not fields:
        raise ValueError(f"No valid field definitions found in {sps_file_path}")
    
    return fields


def extract_value_labels(sps_file_path: str, field_name: str) -> Dict[float, str]:
    """
    Extract value labels for a specific field from SPSS .sps file.
    
    Parses the VALUE LABELS section and extracts numeric codes with their
    text descriptions. Handles fields that share labels (like RFV1/RFV2/RFV3).
    
    Args:
        sps_file_path: Path to the .sps definition file.
        field_name: Name of the field (e.g., 'RFV1', 'RFV13D').
    
    Returns:
        Dictionary mapping numeric codes (as float) to text descriptions.
        Format: {10050.0: 'Chills', 10100.0: 'Fever', ...}
    """
    sps_path = Path(sps_file_path)
    
    if not sps_path.exists():
        raise FileNotFoundError(f"SPSS definition file not found: {sps_file_path}")
    
    with open(sps_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    labels: Dict[float, str] = {}
    field_upper = field_name.upper()
    lines = content.split('\n')
    
    # Find VALUE LABELS section that contains our field
    # Sections can start with "/ RFV3" but contain RFV1, RFV2, RFV3
    # Format:
    #   / RFV3
    #     RFV2
    #     RFV1
    #     10050 "Chills"
    #     10100 "Fever"
    #     ...
    
    in_section = False
    found_field = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if this line starts a section with "/"
        if stripped.startswith("/"):
            # Check if this section is for our field or a related field
            # RFV1, RFV2, RFV3 share labels; RFV13D, RFV23D, RFV33D share labels
            if field_upper in stripped or any(
                (field_upper.startswith("RFV") and f.startswith("RFV") and 
                 (field_upper.endswith("3D") == f.endswith("3D"))) 
                for f in stripped.split()
            ):
                in_section = True
                found_field = field_upper in stripped
                continue
        
        # If we're in a relevant section, check continuation lines
        if in_section:
            # Check if this line is a field name continuation (like "  RFV1")
            if stripped == field_upper:
                found_field = True
                continue
            
            # If this is another "/" section, stop
            if stripped.startswith("/"):
                break
            
            # If we hit ".", end of VALUE LABELS section
            if stripped == ".":
                break
            
            # Skip other field names in the section header
            if stripped and stripped.isupper() and len(stripped) < 15 and not any(c in stripped for c in '()'):
                continue
            
            # Try to parse label line: code "Description"
            # Pattern 1: unquoted numeric code
            label_match = re.match(r'^\s+(-?\d+(?:\.\d+)?)\s+"([^"]+)"', line)
            
            # Pattern 2: quoted code (like '10050')
            if not label_match:
                label_match = re.match(r"^\s+['\"](-?\d+(?:\.\d+)?)['\"]\s+\"([^\"]+)\"", line)
            
            if label_match:
                code_str = label_match.group(1)
                description = label_match.group(2)
                try:
                    code = float(code_str)
                    labels[code] = description
                except (ValueError, TypeError):
                    pass
    
    return labels

