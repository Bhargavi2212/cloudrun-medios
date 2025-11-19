"""Verify database tables were created successfully."""

import asyncio
import os

import asyncpg


async def verify_tables():
    """Check if file_assets and timeline_events tables exist."""
    try:
        # Get database credentials from environment - do not hardcode!
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "medi_os_v2_b")

        if not db_password:
            raise ValueError(
                "DB_PASSWORD environment variable not set. "
                "Please set it before running this script."
            )

        conn = await asyncpg.connect(
            f"postgresql://{db_user}:{db_password}@localhost:5432/{db_name}"
        )

        # Check for file_assets table
        file_assets = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'file_assets')"  # noqa: E501
        )

        # Check for timeline_events table
        timeline_events = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'timeline_events')"  # noqa: E501
        )

        print("Database Verification:")
        print("-" * 60)
        print(f"file_assets table: {'[OK]' if file_assets else '[FAIL]'}")
        print(f"timeline_events table: {'[OK]' if timeline_events else '[FAIL]'}")

        if file_assets and timeline_events:
            # Get column counts
            file_assets_cols = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'file_assets'"  # noqa: E501
            )
            timeline_cols = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'timeline_events'"  # noqa: E501
            )
            print(f"\nfile_assets columns: {file_assets_cols}")
            print(f"timeline_events columns: {timeline_cols}")
            print("\n[OK] All tables created successfully!")
        else:
            print("\n[FAIL] Some tables are missing!")

        await conn.close()

    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(verify_tables())
