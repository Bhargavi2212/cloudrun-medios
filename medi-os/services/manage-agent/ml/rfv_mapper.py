"""
RFV Text-to-Code Mapper for Inference.

Converts text RFV descriptions (from user input) to numeric codes
for model prediction at inference time.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
import re


class RFVTextToCodeMapper:
    """
    Maps RFV text descriptions to numeric codes for inference.
    
    Handles text input from users (e.g., "Chest pain") and converts
    to numeric codes (e.g., 10501.0) that the model expects.
    """
    
    def __init__(self, mappings_path: Optional[str] = None):
        """
        Initialize mapper.
        
        Args:
            mappings_path: Path to rfv_code_mappings.json file.
                          If None, auto-detect in data directory.
        """
        if mappings_path is None:
            # Auto-detect: look for mappings in data directory
            project_root = Path(__file__).parent.parent.parent.parent
            mappings_path = project_root / "data" / "rfv_code_mappings.json"
        
        self.mappings_path = Path(mappings_path)
        self._code_to_text_map: Dict[str, Dict[str, str]] = {}  # Internal dict (avoid conflict with method)
        self._text_to_code_map: Dict[str, Dict[str, float]] = {}  # Internal dict (avoid conflict with method)
        self._load_mappings()
    
    def _load_mappings(self):
        """Load RFV code-to-text mappings from JSON file."""
        if not self.mappings_path.exists():
            raise FileNotFoundError(
                f"RFV mappings file not found: {self.mappings_path}\n"
                "Please run the parser to generate rfv_code_mappings.json"
            )
        
        with open(self.mappings_path, 'r', encoding='utf-8') as f:
            self._code_to_text_map = json.load(f)
        
        # Create reverse mapping: text (lowercase) → code
        self._text_to_code_map = {}
        for field_name, code_text_map in self._code_to_text_map.items():
            self._text_to_code_map[field_name] = {}
            for code_str, text in code_text_map.items():
                # Normalize text: lowercase, strip whitespace
                text_normalized = text.lower().strip()
                code = float(code_str)
                
                # Store normalized text → code
                if text_normalized not in self._text_to_code_map[field_name]:
                    self._text_to_code_map[field_name][text_normalized] = code
                else:
                    # If duplicate text, keep the code with smaller value (most common)
                    existing_code = self._text_to_code_map[field_name][text_normalized]
                    if code < existing_code:
                        self._text_to_code_map[field_name][text_normalized] = code
        
        print(f"Loaded RFV mappings: {len(self._code_to_text_map)} fields, "
              f"{sum(len(m) for m in self._code_to_text_map.values())} total codes")
    
    def text_to_code(
        self,
        rfv_text: str,
        field: str = "rfv1",
        fuzzy: bool = True
    ) -> Optional[float]:
        """
        Convert RFV text description to numeric code.
        
        Args:
            rfv_text: Text description (e.g., "Chest pain", "stomach ache")
            field: RFV field name (rfv1, rfv2, rfv3, rfv1_3d, rfv2_3d, rfv3_3d)
            fuzzy: If True, try fuzzy matching if exact match fails
            
        Returns:
            Numeric code (float) or None if not found
        """
        if field not in self._text_to_code_map:
            raise ValueError(f"Unknown RFV field: {field}. Available: {list(self._text_to_code_map.keys())}")
        
        field_mappings = self._text_to_code_map[field]
        
        # Normalize input text
        text_normalized = rfv_text.lower().strip()
        
        # Try exact match first
        if text_normalized in field_mappings:
            return field_mappings[text_normalized]
        
        if not fuzzy:
            return None
        
        # Try fuzzy matching
        # 1. Partial match (text contains the description)
        for mapped_text, code in field_mappings.items():
            if text_normalized in mapped_text or mapped_text in text_normalized:
                return code
        
        # 2. Word-based matching (check if key words match)
        text_words = set(text_normalized.split())
        best_match = None
        best_score = 0
        
        for mapped_text, code in field_mappings.items():
            mapped_words = set(mapped_text.split())
            # Count overlapping words
            overlap = len(text_words & mapped_words)
            if overlap > 0:
                score = overlap / max(len(text_words), len(mapped_words))
                if score > best_score:
                    best_score = score
                    best_match = code
        
        if best_match and best_score > 0.3:  # At least 30% word overlap
            return best_match
        
        # 3. Return None if no match found
        return None
    
    def get_all_codes_for_text(
        self,
        text: str,
        fuzzy: bool = True
    ) -> Dict[str, Optional[float]]:
        """
        Find matching codes for text across all RFV fields.
        
        Args:
            text: RFV text description
            fuzzy: Enable fuzzy matching
            
        Returns:
            Dictionary mapping field names to codes (or None if not found)
        """
        results = {}
        for field in self._text_to_code_map.keys():
            results[field] = self.text_to_code(text, field, fuzzy)
        return results
    
    def code_to_text(
        self,
        code: float,
        field: str = "rfv1"
    ) -> Optional[str]:
        """
        Convert numeric code back to text (for interpretability).
        
        Args:
            code: Numeric RFV code
            field: RFV field name
            
        Returns:
            Text description or None if not found
        """
        if field not in self._code_to_text_map:
            return None
        
        code_str = str(code)
        return self._code_to_text_map[field].get(code_str)
    
    @classmethod
    def load(cls, mappings_path: Optional[str] = None) -> 'RFVTextToCodeMapper':
        """
        Load mapper from mappings file.
        
        Args:
            mappings_path: Path to mappings file (auto-detect if None)
            
        Returns:
            RFVTextToCodeMapper instance
        """
        return cls(mappings_path)

