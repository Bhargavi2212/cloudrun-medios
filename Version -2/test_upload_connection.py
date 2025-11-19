"""Test script to verify upload endpoint is accessible."""
import sys

import requests

print("=" * 60)
print("TESTING UPLOAD ENDPOINT CONNECTION")
print("=" * 60)

# Test 1: Health check
print("\n1. Testing health endpoint...")
try:
    r = requests.get("http://localhost:8001/health", timeout=5)
    print(f"   [OK] Health check: {r.status_code} - {r.json()}")
except Exception as e:
    print(f"   [FAIL] Health check failed: {e}")
    sys.exit(1)

# Test 2: Test documents endpoint
print("\n2. Testing documents test endpoint...")
try:
    r = requests.get(
        "http://localhost:8001/manage/documents/test-connection", timeout=5
    )
    print(f"   [OK] Test endpoint: {r.status_code} - {r.json()}")
except Exception as e:
    print(f"   [FAIL] Test endpoint failed: {e}")

# Test 3: Try uploading a dummy file
print("\n3. Testing upload endpoint (without file - should fail with 422)...")
try:
    r = requests.post("http://localhost:8001/manage/documents/upload", timeout=5)
    print(f"   Response: {r.status_code}")
    if r.status_code == 422:
        print("   [OK] Endpoint is accessible (expected validation error)")
    else:
        print(f"   [WARN] Unexpected status: {r.status_code}")
        print(f"   Response: {r.text[:200]}")
except Exception as e:
    print(f"   ❌ Upload endpoint failed: {e}")

# Test 4: Check CORS
print("\n4. Testing CORS headers...")
try:
    r = requests.options(
        "http://localhost:8001/manage/documents/upload",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
        timeout=5,
    )
    print(f"   CORS preflight: {r.status_code}")
    cors_headers = {
        k: v for k, v in r.headers.items() if k.lower().startswith("access-control")
    }
    if cors_headers:
        print(f"   [OK] CORS headers present: {list(cors_headers.keys())}")
    else:
        print("   [WARN] No CORS headers found")
except Exception as e:
    print(f"   [WARN] CORS test failed: {e}")

print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
print("If all tests pass, the backend is accessible.")
print("If upload fails, check:")
print("  1. Is manage-agent running on port 8001?")
print("  2. Is the service restarted after code changes?")
print("  3. Check browser console for JavaScript errors")
print("  4. Check browser Network tab for the actual request")
print("=" * 60)
