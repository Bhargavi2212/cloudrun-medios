"""Check the latest summary for a patient."""
import asyncio
from uuid import UUID

from sqlalchemy import select

from database.models import Summary
from database.session import get_session, init_engine


async def check_summary(patient_id: str):
    """Check the latest summary for a patient."""
    # Initialize database
    init_engine()

    async for session in get_session():
        stmt = (
            select(Summary)
            .where(Summary.patient_id == UUID(patient_id))
            .order_by(Summary.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        summary = result.scalar_one_or_none()

        if summary:
            print("=" * 60)
            print("LATEST SUMMARY FOUND")
            print("=" * 60)
            print(f"ID: {summary.id}")
            print(f"Model Version: {summary.model_version}")
            print(f"Confidence: {summary.confidence_score}")
            text_len = len(summary.summary_text) if summary.summary_text else 0
            print(f"Text Length: {text_len}")
            print(f"Has Structured Data: {bool(summary.structured_data)}")
            print("\nText Preview (first 500 chars):")
            print("-" * 60)
            if summary.summary_text:
                try:
                    print(summary.summary_text[:500])
                    if len(summary.summary_text) > 500:
                        print("...")
                except UnicodeEncodeError:
                    print(
                        summary.summary_text[:500]
                        .encode("utf-8", errors="replace")
                        .decode("utf-8")
                    )
                    if len(summary.summary_text) > 500:
                        print("...")
            else:
                print("(empty)")
            print("-" * 60)

            # Check if it's stub text
            if (
                "recent encounters" in summary.summary_text
                and "No highlights provided" in summary.summary_text
            ):
                print(
                    "\n[WARNING] This appears to be STUB text, not a "
                    "Gemini-generated summary!"
                )
            elif "gemini" in summary.model_version.lower():
                print("\n[OK] This appears to be a Gemini-generated summary!")
            else:
                print("\n[INFO] Model version does not indicate Gemini usage.")
        else:
            print(f"No summary found for patient {patient_id}")
        break


if __name__ == "__main__":
    import sys

    patient_id = (
        sys.argv[1] if len(sys.argv) > 1 else "22967efe-ba34-46a1-99af-5d49b0c57d27"
    )
    asyncio.run(check_summary(patient_id))
