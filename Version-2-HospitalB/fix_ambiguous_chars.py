#!/usr/bin/env python3
"""Fix ambiguous characters in Hospital B files"""

from pathlib import Path

# Files to fix
files_to_fix = [
    "database/seeds.py",
    "scripts/analyze_columns.py",
    "scripts/compare_experiments.py",
    "scripts/feature_engineering.py",
    "scripts/train_nurse_triage_model.py",
    "scripts/train_receptionist_triage_model_v2.py",
]

for file_path in files_to_fix:
    path = Path(file_path)
    if not path.exists():
        continue

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Replace RIGHT SINGLE QUOTATION MARK (') with regular apostrophe (')
    content = content.replace("'", "'")
    content = content.replace("'", "'")

    # Replace MULTIPLICATION SIGN (x) with x
    content = content.replace("x", "x")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Fixed ambiguous characters in {file_path}")

print("Done fixing ambiguous characters!")
