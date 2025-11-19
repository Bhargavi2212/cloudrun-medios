"""Test database connection with different credentials."""

import asyncio

import asyncpg


async def test_connection(username: str, password: str, database: str = "medi_os_v2"):
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
    combinations = [
        ("Anuradha", "postgres"),
        ("postgres", "postgres"),
        ("Anuradha", "Anuradha"),
        ("postgres", "Anuradha"),
    ]
    default_db = "medi_os_v2"

    for username, password in combinations:
        success = await test_connection(username, password, default_db)
        if success:
            print(f"\nWorking credentials: {username}:{password}")
            print(
                f"DATABASE_URL=postgresql+asyncpg://{username}:{password}"
                f"@localhost:5432/{default_db}"
            )
            return

    print("\n[ERROR] None of the credential combinations worked.")
    print("Please verify your PostgreSQL username and password.")


if __name__ == "__main__":
    asyncio.run(main())
