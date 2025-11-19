"""Temporary script to fix ruff errors."""
from pathlib import Path


# Fix RIGHT SINGLE QUOTATION MARK (U+2019) to regular apostrophe
def fix_quotes(file_path: Path) -> None:
    """Replace RIGHT SINGLE QUOTATION MARK with regular apostrophe."""
    content = file_path.read_bytes()
    # Replace U+2019 (RIGHT SINGLE QUOTATION MARK) with U+0027 (APOSTROPHE)
    content = content.replace(b"\xe2\x80\x99", b"'")
    file_path.write_bytes(content)


# Fix seeds.py
seeds_file = Path("database/seeds.py")
if seeds_file.exists():
    fix_quotes(seeds_file)
    print(f"Fixed quotes in {seeds_file}")

print("Done")
