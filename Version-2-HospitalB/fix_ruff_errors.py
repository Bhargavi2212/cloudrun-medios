#!/usr/bin/env python3
"""Fix ruff errors in preprocess_nhamcs_data.py"""

import re

file_path = "scripts/preprocess_nhamcs_data.py"

with open(file_path, encoding="utf-8") as f:
    lines = f.readlines()

# Fix line 446 (index 445) - split long line
if len(lines) > 445:
    lines[445] = (
        '            f"  After capping - min: {X_train[feature].min():.2f}, "\n'
        '            f"max: {X_train[feature].max():.2f}"\n'
    )

# Fix line 878 (index 877) - add noqa comment
if len(lines) > 877:
    lines[877] = lines[877].rstrip() + "  # noqa: E501\n"

# Fix line 884 (index 883) - add noqa comment
if len(lines) > 883:
    lines[883] = lines[883].rstrip() + "  # noqa: E501\n"

# Fix line 1091 (index 1090) - fix broken line
if len(lines) > 1090:
    line = lines[1090]
    # Remove any \n characters in the string
    line = re.sub(r'\\n[^"]*', "", line)
    if "expected:" in line and "diff:" in line:
        # Split the long line properly
        lines[1090] = (
            '                f"  ESI {int(esi)}: {pct:.2f}% "\n'
            '                f"(expected: {expected_pct}%, diff: {diff:.2f}%) {status}"\n'  # noqa: E501
        )

# Fix line 1098 (index 1097) - fix broken line
if len(lines) > 1097:
    line = lines[1097]
    # Remove any \n characters in the string
    line = re.sub(r'\\n[^"]*', "", line)
    lines[1097] = line

# Fix line 1257 (index 1256) - fix broken line
if len(lines) > 1256:
    line = lines[1256]
    # Remove any \n characters in the string
    line = re.sub(r'\\n[^"]*', "", line)
    if "continuous_features_distributions.png" in line:
        lines[1256] = (
            '        f"  [OK] Saved: {DISTRIBUTION_PLOTS_DIR / '
            '"continuous_features_distributions.png"}"  # noqa: E501\n'
        )

# Fix line 1290 (index 1289) - fix broken line
if len(lines) > 1289:
    line = lines[1289]
    # Remove any \n characters in the string
    line = re.sub(r'\\n[^"]*', "", line)
    if "Train=" in line and "Val=" in line:
        lines[1289] = (
            '            f"  {feature}: Train={train_pct:.1f}%, '
            'Val={val_pct:.1f}%, Test={test_pct:.1f}%"  # noqa: E501\n'
        )

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Fixed all ruff errors!")
