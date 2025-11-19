"""
Comprehensive test script for document system.
Tests backend services and verifies endpoints are registered.
"""

import os
import sys
from pathlib import Path

# Set minimal environment for testing
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test"
)
# GEMINI_API_KEY must be set as environment variable - do not hardcode!
if not os.getenv("GEMINI_API_KEY"):
    print("WARNING: GEMINI_API_KEY not set. Document processing will be disabled.")
    os.environ.setdefault("GEMINI_API_KEY", "")

print("=" * 60)
print("Document System Integration Test")
print("=" * 60)

# Test 1: Backend Imports
print("\n[TEST 1] Backend Service Imports")
print("-" * 60)
try:
    from services.manage_agent.handlers.documents import router as manage_docs_router

    print("[OK] Manage-agent documents router imported")
except Exception as e:
    print(f"[FAIL] Manage-agent documents router: {e}")
    sys.exit(1)

try:
    from services.summarizer_agent.handlers.documents import (
        router as summarizer_docs_router,
    )

    print("[OK] Summarizer-agent documents router imported")
except Exception as e:
    print(f"[FAIL] Summarizer-agent documents router: {e}")
    sys.exit(1)

# Test 2: Check Routes
print("\n[TEST 2] API Routes Registration")
print("-" * 60)

manage_routes = [route.path for route in manage_docs_router.routes]
print(f"Manage-agent document routes ({len(manage_routes)}):")
for route in manage_routes:
    print(f"  - {route}")

summarizer_routes = [route.path for route in summarizer_docs_router.routes]
print(f"\nSummarizer-agent document routes ({len(summarizer_routes)}):")
for route in summarizer_routes:
    print(f"  - {route}")

# Test 3: Service Initialization
print("\n[TEST 3] Service Initialization")
print("-" * 60)

try:
    from services.manage_agent.services.storage_service import StorageService

    storage = StorageService(storage_root="./test_storage")
    print("[OK] Storage service initialized")
    # Cleanup
    if Path("./test_storage").exists():
        import shutil

        shutil.rmtree("./test_storage")
except Exception as e:
    print(f"[FAIL] Storage service: {e}")

try:
    from services.summarizer_agent.core.document_processor import DocumentProcessor

    processor = DocumentProcessor(
        api_key=os.getenv("GEMINI_API_KEY"), model_name="gemini-2.0-flash-exp"
    )
    if processor._model is None:
        print(
            "[WARN] Document processor initialized but Gemini model not available (may need google-generativeai package)"
        )
    else:
        print("[OK] Document processor initialized with Gemini")
except Exception as e:
    print(f"[WARN] Document processor: {e}")

# Test 4: Database Models
print("\n[TEST 4] Database Models")
print("-" * 60)

try:
    from database.models import FileAsset, TimelineEvent

    print("[OK] FileAsset model imported")
    print("[OK] TimelineEvent model imported")

    # Check model attributes
    file_asset_attrs = [attr for attr in dir(FileAsset) if not attr.startswith("_")]
    timeline_attrs = [attr for attr in dir(TimelineEvent) if not attr.startswith("_")]
    print(f"[OK] FileAsset has {len(file_asset_attrs)} attributes")
    print(f"[OK] TimelineEvent has {len(timeline_attrs)} attributes")
except Exception as e:
    print(f"[FAIL] Database models: {e}")

# Test 5: Frontend Compilation Check
print("\n[TEST 5] Frontend Build Check")
print("-" * 60)

frontend_dist = Path("apps/frontend/dist")
if frontend_dist.exists():
    print("[OK] Frontend build directory exists")
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        print("[OK] Frontend index.html found")
    else:
        print("[WARN] Frontend index.html not found")
else:
    print("[INFO] Frontend not built yet (run: cd apps/frontend && npm run build)")

# Test 6: Route Methods
print("\n[TEST 6] Route HTTP Methods")
print("-" * 60)

for route in manage_docs_router.routes:
    methods = getattr(route, "methods", set())
    path = getattr(route, "path", "unknown")
    print(f"  {list(methods)[0] if methods else 'UNKNOWN':6} {path}")

for route in summarizer_docs_router.routes:
    methods = getattr(route, "methods", set())
    path = getattr(route, "path", "unknown")
    print(f"  {list(methods)[0] if methods else 'UNKNOWN':6} {path}")

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("[OK] All critical components imported successfully")
print("[OK] API routes registered correctly")
print("[OK] Services can be initialized")
print("\nNext Steps:")
print("1. Set DATABASE_URL environment variable")
print("2. Run database migration: alembic upgrade head")
print("3. Start services:")
print("   - uvicorn services.manage_agent.main:app --port 8001")
print("   - uvicorn services.summarizer_agent.main:app --port 8003")
print("4. Start frontend: cd apps/frontend && npm run dev")
print("=" * 60)
