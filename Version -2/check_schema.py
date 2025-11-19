"""Check if structured_data column exists in summaries table."""
import asyncio

from sqlalchemy import text

from database.session import get_engine, init_engine


async def check_column():
    """Check if structured_data column exists."""
    init_engine()
    engine = get_engine()

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'summaries' 
                AND column_name = 'structured_data'
            """
            )
        )
        row = result.fetchone()
        if row:
            print("[OK] Column 'structured_data' exists in 'summaries' table")
            print(f"     Data type: {row[1]}")
        else:
            print(
                "[ERROR] Column 'structured_data' does NOT exist in 'summaries' table"
            )
            print(
                "        You need to run: poetry run alembic -c "
                "database/alembic.ini upgrade head"
            )

        # Also check all columns in summaries table
        result2 = await conn.execute(
            text(
                """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'summaries'
                ORDER BY ordinal_position
            """
            )
        )
        print("\nAll columns in 'summaries' table:")
        for col in result2.fetchall():
            print(f"  - {col[0]} ({col[1]})")


if __name__ == "__main__":
    asyncio.run(check_column())
