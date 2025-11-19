"""Quick script to check summarizer agent status."""
import os
import sys

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

print("=" * 60)
print("SUMMARIZER AGENT STATUS CHECK")
print("=" * 60)

# Check environment variables
gemini_key = os.getenv("GEMINI_API_KEY")
print(f"\n1. GEMINI_API_KEY: {'SET' if gemini_key else 'NOT SET'}")
if gemini_key:
    print(f"   Length: {len(gemini_key)} characters")

database_url = os.getenv("DATABASE_URL")
print(f"\n2. DATABASE_URL: {'SET' if database_url else 'NOT SET'}")
if database_url:
    scheme = (
        database_url.split("://")[0]
        if "://" in database_url
        else "unknown"
    )
    print(f"   Scheme: {scheme}")

# Check if google-generativeai is installed
try:
    import google.generativeai as genai  # noqa: F401

    print("\n3. google-generativeai package: INSTALLED")
except ImportError:
    print("\n3. google-generativeai package: NOT INSTALLED")
    print("   Install with: poetry add google-generativeai")

# Check debug log
log_path = "debug_summarizer.log"
if os.path.exists(log_path):
    print(f"\n4. Debug log: ✅ EXISTS at {log_path}")
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
        print(f"   Total lines: {len(lines)}")
        # Check for recent errors
        error_lines = [line for line in lines if "ERROR" in line.upper()]
        if error_lines:
            print(f"   ⚠️  Found {len(error_lines)} error lines")
            print(f"   Last error: {error_lines[-1][:100]}...")
else:
    print("\n4. Debug log: ❌ NOT FOUND (will be created on first run)")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Make sure Summarizer Agent is RESTARTED (port 8003)")
print("2. Generate a summary in the UI")
print("3. Check terminal for [SUMMARIZE] log messages")
print("4. Check debug_summarizer.log for detailed errors")
print("=" * 60)
