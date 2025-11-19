"""Test script to verify the UUID fix in summarizer agent."""
import asyncio

import httpx


async def test_summarizer():
    """Test the summarizer agent with a real patient ID."""
    # Get a patient ID from the database or use a known one
    # For now, let's test the test endpoint first
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test 1: Check if service is running
            print("=" * 60)
            print("TEST 1: Checking if summarizer agent is running...")
            resp = await client.get("http://localhost:8003/summarizer/test")
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    print(f"Response: {resp.json()}")
                except Exception:
                    print(f"Response text: {resp.text[:200]}")
            else:
                print(f"Error: {resp.text[:200]}")
            print()

            # Test 2: Try to generate a summary with a known patient
            # You'll need to replace these with actual IDs from your database
            print("=" * 60)
            print("TEST 2: Testing summary generation...")
            print(
                "Note: Replace patient_id and encounter_ids with real values from your DB"  # noqa: E501
            )

            # Example - you may need to adjust these
            test_patient_id = (
                "22967efe-ba34-46a1-99af-5d49b0c57d27"  # From your earlier logs
            )
            test_encounter_id = (
                "bc491043-445b-43da-8951-7a231bf430c5"  # From your earlier logs
            )

            summary_resp = await client.post(
                "http://localhost:8003/summarizer/generate-summary",
                json={
                    "patient_id": test_patient_id,
                    "encounter_ids": [test_encounter_id],
                    "highlights": ["Test summary generation"],
                },
            )
            print(f"Status: {summary_resp.status_code}")
            if summary_resp.status_code == 201:
                data = summary_resp.json()
                print(f"Summary ID: {data.get('id')}")
                print(f"Model Version: {data.get('model_version')}")
                print(f"Confidence: {data.get('confidence_score')}")
                print(f"Summary Text Length: {len(data.get('summary_text', ''))}")
                print(f"Has Structured Data: {bool(data.get('structured_data'))}")
                print(f"Summary Preview: {data.get('summary_text', '')[:200]}...")
                print("\n✅ SUCCESS: Summary generated without UUID error!")
            else:
                print(f"Error: {summary_resp.text}")
                print("\n❌ FAILED: Summary generation returned error")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_summarizer())
