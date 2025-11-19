#!/usr/bin/env python3
"""
Patient Privacy Validation Script

This script validates that code follows patient privacy requirements:
- No hospital/provider metadata in portable profiles
- Proper data anonymization
- No PHI in logs or error messages
- Compliance with privacy-first architecture
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


class PrivacyValidator(ast.NodeVisitor):
    """AST visitor to validate patient privacy requirements."""
    
    def __init__(self):
        self.errors: List[Tuple[int, str]] = []
        self.warnings: List[Tuple[int, str]] = []
        self.phi_patterns = {
            'hospital_name', 'provider_name', 'doctor_name', 'location',
            'address', 'phone', 'ssn', 'medical_record_number', 'mrn'
        }
        
    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for assignments that might expose PHI."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id.lower()
                
                # Check for PHI variable names
                if any(phi in var_name for phi in self.phi_patterns):
                    self.warnings.append((
                        node.lineno,
                        f"Variable '{target.id}' may contain PHI - ensure proper handling"
                    ))
                    
                # Check for hospital metadata in portable profiles
                if 'portable' in var_name or 'export' in var_name:
                    if isinstance(node.value, ast.Dict):
                        for key in node.value.keys:
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                key_name = key.value.lower()
                                if any(phi in key_name for phi in 
                                       ['hospital', 'provider', 'doctor', 'location']):
                                    self.errors.append((
                                        node.lineno,
                                        f"Portable profile contains hospital metadata: '{key.value}'"
                                    ))
                                    
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for privacy violations."""
        # Check logging calls for PHI
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['info', 'debug', 'warning', 'error', 'critical']:
                for arg in node.args:
                    if isinstance(arg, ast.JoinedStr):  # f-string
                        self._check_fstring_for_phi(arg, node.lineno)
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        self._check_string_for_phi(arg.value, node.lineno)
                        
        # Check for database queries that might expose PHI
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['execute', 'query', 'select']:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        query = arg.value.lower()
                        if any(phi in query for phi in 
                               ['hospital_name', 'provider_name', 'doctor_name']):
                            self.errors.append((
                                node.lineno,
                                f"Database query may expose PHI: {phi}"
                            ))
                            
        self.generic_visit(node)
        
    def visit_Return(self, node: ast.Return) -> None:
        """Check return statements for PHI exposure."""
        if isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    key_name = key.value.lower()
                    if any(phi in key_name for phi in 
                           ['hospital', 'provider', 'doctor', 'location']):
                        self.warnings.append((
                            node.lineno,
                            f"Return value may contain PHI: '{key.value}'"
                        ))
                        
        self.generic_visit(node)
        
    def _check_fstring_for_phi(self, node: ast.JoinedStr, line_no: int) -> None:
        """Check f-string for PHI exposure."""
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                if isinstance(value.value, ast.Name):
                    var_name = value.value.id.lower()
                    if any(phi in var_name for phi in self.phi_patterns):
                        self.warnings.append((
                            line_no,
                            f"F-string may log PHI variable: '{value.value.id}'"
                        ))
                        
    def _check_string_for_phi(self, text: str, line_no: int) -> None:
        """Check string literal for PHI patterns."""
        text_lower = text.lower()
        phi_found = []
        
        for phi in self.phi_patterns:
            if phi in text_lower:
                phi_found.append(phi)
                
        if phi_found:
            self.warnings.append((
                line_no,
                f"String may contain PHI references: {', '.join(phi_found)}"
            ))


def validate_file(file_path: Path) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Validate a single Python file for privacy compliance."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for hardcoded PHI patterns in comments and strings
        phi_regex_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{10}\b',  # Phone number pattern
            r'\b[A-Z]{2}\d{8}\b',  # Medical record number pattern
        ]
        
        errors = []
        warnings = []
        
        for i, line in enumerate(content.split('\n'), 1):
            for pattern in phi_regex_patterns:
                if re.search(pattern, line):
                    warnings.append((i, f"Line may contain PHI pattern: {pattern}"))
                    
        # AST validation
        tree = ast.parse(content)
        validator = PrivacyValidator()
        validator.visit(tree)
        
        errors.extend(validator.errors)
        warnings.extend(validator.warnings)
        
        return errors, warnings
        
    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")], []
    except Exception as e:
        return [(0, f"Error parsing file: {str(e)}")], []


def main() -> int:
    """Main validation function."""
    print("🔒 Running Patient Privacy Validation...")
    
    # Find all Python files in services and shared directories
    python_files = []
    for directory in ['services', 'shared']:
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(Path(root) / file)
    
    total_errors = 0
    total_warnings = 0
    
    for file_path in python_files:
        errors, warnings = validate_file(file_path)
        
        if errors or warnings:
            print(f"\n📁 {file_path}")
            
            for line_no, error in errors:
                print(f"  ❌ Line {line_no}: {error}")
                total_errors += 1
                
            for line_no, warning in warnings:
                print(f"  ⚠️  Line {line_no}: {warning}")
                total_warnings += 1
    
    print(f"\n📊 Privacy Validation Results:")
    print(f"   Files checked: {len(python_files)}")
    print(f"   Errors: {total_errors}")
    print(f"   Warnings: {total_warnings}")
    
    if total_errors > 0:
        print("\n❌ Privacy validation failed! Please fix errors before committing.")
        return 1
    elif total_warnings > 0:
        print("\n⚠️  Privacy validation passed with warnings.")
        return 0
    else:
        print("\n✅ Privacy validation passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())