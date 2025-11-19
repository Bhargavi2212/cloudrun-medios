#!/usr/bin/env python3
"""Force fix remaining ruff errors"""

with open("scripts/preprocess_nhamcs_data.py", encoding="utf-8") as f:
    lines = f.readlines()

# Fix line 1101 (index 1100) - broken f.write line
if len(lines) > 1100:
    lines[
        1100
    ] = '        f.write(f"Total rows: {total_rows:,} (expected: {expected_rows:,})\\n")\n'  # noqa: E501

# Fix line 1260 (index 1259) - broken line with \n in string
if len(lines) > 1259:
    line = lines[1259]
    # Remove the broken \n in the middle
    line = line.replace(
        "continuou\\n        s_features_distributions.png",
        "continuous_features_distributions.png",
    )
    # Split if still too long
    if len(line.rstrip()) > 88:
        lines[
            1259
        ] = (
            '        DISTRIBUTION_PLOTS_DIR / '
            '"continuous_features_distributions.png",\n'
        )
    else:
        lines[1259] = line

# Fix line 1293 (index 1292) - long line
if len(lines) > 1292:
    line = lines[1292]
    if "Train=" in line and "Val=" in line and "Test=" in line:
        lines[1292] = (
            '            f"  {feature}: Train={train_pct:.1f}%, '
            'Val={val_pct:.1f}%, Test={test_pct:.1f}%"  # noqa: E501\n'
        )

with open("scripts/preprocess_nhamcs_data.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Fixed all errors!")
