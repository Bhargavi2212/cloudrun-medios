#!/usr/bin/env python3
"""Fix remaining ruff errors in Hospital A"""

with open("scripts/preprocess_nhamcs_data.py", encoding="utf-8") as f:
    lines = f.readlines()

# Fix line 1097 (index 1096) - split long line
if len(lines) > 1096:
    lines[1096] = (
        '                f"  ESI {int(esi)}: {pct:.2f}% "\n'
        '                f"(expected: {expected_pct}%, diff: {diff:.2f}%) {status}"\n'
    )

# Fix line 1263 (index 1262) - split long line
if len(lines) > 1262:
    lines[1262] = (
        '        f"  [OK] Saved: {DISTRIBUTION_PLOTS_DIR / '
        '"continuous_features_distributions.png"}"  # noqa: E501\n'
    )

with open("scripts/preprocess_nhamcs_data.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Fixed all ruff errors!")
