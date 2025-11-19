"""Test database connection with different credentials."""

import asyncio
import os

import asyncpg


async def test_connection(username: str, password: str, database: str = "medi_os_v2_b"):
    """Test database connection."""
    try:
        conn = await asyncpg.connect(
            f"postgresql://{username}:{password}@localhost:5432/{database}"
        )
        print(f"[OK] Connection successful with {username}:{password}")
        await conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] Connection failed with {username}:{password} - {e}")
        return False


async def main():
    """Test different credential combinations."""
    print("Testing database connections...")
    print("-" * 60)

    # Try different combinations
    # (use environment variables - do not hardcode passwords!)
    # Get from environment or prompt user
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")

    if not db_password:
        print("[WARNING] DB_PASSWORD not set. Please set it as environment variable.")
        print("Example: $env:DB_PASSWORD='your_password' (PowerShell)")
        print("Example: export DB_PASSWORD='your_password' (Bash)")
        return

    combinations = [
        (db_user, db_password),
        ("postgres", db_password),
    ]

    for username, password in combinations:
        success = await test_connection(username, password)
        if success:
            print(f"\nWorking credentials: {username}:{password}")
            db_name = "medi_os_v2_b"
            print(
                f"DATABASE_URL=postgresql+asyncpg://{username}:{password}@localhost:5432/{db_name}"
            )
            return

    print("\n[ERROR] None of the credential combinations worked.")
    print("Please verify your PostgreSQL username and password.")


if __name__ == "__main__":
    asyncio.run(main())
