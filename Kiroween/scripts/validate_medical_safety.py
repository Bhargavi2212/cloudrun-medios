#!/usr/bin/env python3
"""
Medical Safety Validation Script

This script validates that code follows medical safety best practices:
- No hardcoded medical values that could cause harm
- Proper error handling for critical medical functions
- Validation of medical data ranges
- Proper logging for medical decisions
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


class MedicalSafetyValidator(ast.NodeVisitor):
    """AST visitor to validate medical safety requirements."""
    
    def __init__(self):
        self.errors: List[Tuple[int, str]] = []
        self.warnings: List[Tuple[int, str]] = []
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Validate function definitions for medical safety."""
        # Check for medical functions without proper error handling
        if any(keyword in node.name.lower() for keyword in 
               ['triage', 'classify', 'diagnose', 'prescribe', 'dose', 'medication']):
            if not self._has_error_handling(node):
                self.errors.append((
                    node.lineno,
                    f"Medical function '{node.name}' must have proper error handling"
                ))
                
        # Check for missing docstrings on medical functions
        if any(keyword in node.name.lower() for keyword in 
               ['triage', 'classify', 'diagnose', 'prescribe']):
            if not ast.get_docstring(node):
                self.errors.append((
                    node.lineno,
                    f"Medical function '{node.name}' must have comprehensive docstring"
                ))
                
        self.generic_visit(node)
        
    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for hardcoded medical values."""
        if isinstance(node.value, ast.Constant):
            # Check for suspicious hardcoded medical values
            if isinstance(node.value.value, (int, float)):
                value = node.value.value
                # Common dangerous hardcoded values
                if value in [120, 80, 98.6, 37.0, 100, 60]:  # Common vital signs
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id.lower()
                            if any(keyword in var_name for keyword in 
                                   ['bp', 'pressure', 'temp', 'heart', 'pulse', 'dose']):
                                self.warnings.append((
                                    node.lineno,
                                    f"Hardcoded medical value {value} in '{var_name}' - consider using constants"
                                ))
                                
        self.generic_visit(node)
        
    def visit_Compare(self, node: ast.Compare) -> None:
        """Check for medical range validations."""
        # Look for medical comparisons without proper bounds checking
        if isinstance(node.left, ast.Name):
            var_name = node.left.id.lower()
            if any(keyword in var_name for keyword in 
                   ['age', 'weight', 'height', 'bp', 'hr', 'temp']):
                # Should have both upper and lower bounds for medical values
                if len(node.ops) == 1:
                    self.warnings.append((
                        node.lineno,
                        f"Medical variable '{var_name}' should have both upper and lower bound validation"
                    ))
                    
        self.generic_visit(node)
        
    def _has_error_handling(self, node: ast.FunctionDef) -> bool:
        """Check if function has proper error handling."""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
            if isinstance(child, ast.Raise):
                return True
        return False


def validate_file(file_path: Path) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Validate a single Python file for medical safety."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        validator = MedicalSafetyValidator()
        validator.visit(tree)
        
        return validator.errors, validator.warnings
        
    except SyntaxError as e:
        return [(e.lineno or 0, f"Syntax error: {e.msg}")], []
    except Exception as e:
        return [(0, f"Error parsing file: {str(e)}")], []


def main() -> int:
    """Main validation function."""
    print("Running Medical Safety Validation...")
    
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
            print(f"\nFile: {file_path}")
            
            for line_no, error in errors:
                print(f"  ERROR Line {line_no}: {error}")
                total_errors += 1
                
            for line_no, warning in warnings:
                print(f"  WARNING Line {line_no}: {warning}")
                total_warnings += 1
    
    print(f"\nMedical Safety Validation Results:")
    print(f"   Files checked: {len(python_files)}")
    print(f"   Errors: {total_errors}")
    print(f"   Warnings: {total_warnings}")
    
    if total_errors > 0:
        print("\nMedical safety validation failed! Please fix errors before committing.")
        return 1
    elif total_warnings > 0:
        print("\nMedical safety validation passed with warnings.")
        return 0
    else:
        print("\nMedical safety validation passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())