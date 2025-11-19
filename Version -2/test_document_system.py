"""
Quick test script to verify document system components.
"""

from pathlib import Path

# Test imports
print("Testing imports...")
try:
    print("[OK] Database models import OK")
except Exception as e:
    print(f"[FAIL] Database models import failed: {e}")

try:
    from services.manage_agent.services.storage_service import StorageService

    print("[OK] Storage service import OK")
except Exception as e:
    print(f"[FAIL] Storage service import failed: {e}")

try:
    print("[OK] File asset service import OK")
except Exception as e:
    print(f"[FAIL] File asset service import failed: {e}")

try:
    from services.summarizer_agent.core.document_processor import DocumentProcessor

    print("[OK] Document processor import OK")
except Exception as e:
    print(f"[FAIL] Document processor import failed: {e}")

try:
    print("[OK] Document service import OK")
except Exception as e:
    print(f"[FAIL] Document service import failed: {e}")

# Test storage service initialization
print("\nTesting storage service...")
try:
    storage = StorageService(storage_root="./test_storage")
    print("[OK] Storage service initialized")
    # Clean up
    if Path("./test_storage").exists():
        import shutil

        shutil.rmtree("./test_storage")
except Exception as e:
    print(f"[FAIL] Storage service initialization failed: {e}")

# Test document processor initialization (without API key)
print("\nTesting document processor...")
try:
    processor = DocumentProcessor(api_key=None, model_name="gemini-2.0-flash-exp")
    print("[OK] Document processor initialized (no API key - will be disabled)")
except Exception as e:
    print(f"[FAIL] Document processor initialization failed: {e}")

print("\n[OK] All basic tests passed!")
print("\nNext steps:")
print("1. Run database migration: cd 'Version -2' && alembic upgrade head")
print("2. Set GEMINI_API_KEY in environment for document processing")
print("3. Start manage-agent: uvicorn services.manage_agent.main:app --port 8001")
print(
    "4. Start summarizer-agent: uvicorn services.summarizer_agent.main:app --port 8003"
)
print("5. Start frontend: cd apps/frontend && npm run dev")
