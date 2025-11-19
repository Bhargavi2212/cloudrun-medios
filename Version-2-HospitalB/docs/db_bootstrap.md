## Database Bootstrap – Local and CI

### Local development (Windows – your machine)

- **Assumptions**
  - PostgreSQL is running locally.
  - User: `postgres`
  - Password: `Anuradha`
  - Databases:
    - `medi_os_v2_b` (main app DB)
    - `medi_os_v2_b_test` (test DB)

```powershell
cd "D:\Hackathons\Cloud Run\Version -2"

# 1) Point SQLAlchemy at your local databases
$env:DATABASE_URL = "postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_b"
$env:TEST_DATABASE_URL = "postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_b_test"

# 2) Apply schema migrations
poetry run alembic -c database/alembic.ini upgrade head

# 3) Seed demo data (patients, encounters, summaries, audit logs)
poetry run python scripts/seed_database.py

# 4) Run all backend + DB-backed tests
poetry run pytest
```

> If you ever need a clean slate, you can truncate all core tables with:
>
> ```bash
> ./scripts/reset_demo_data.sh
> ```

### CI (GitHub Actions)

- **Service container**
  - Image: `postgres:15`
  - User: `postgres`
  - Password: `postgres`
  - Databases:
    - `medi_os_v2_b` (created by the container)
    - `medi_os_v2_b_test` (created in the workflow)

The CI workflow (`.github/workflows/ci.yml`) sets:

```text
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/medi_os_v2_b
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/medi_os_v2_b_test
```

Then it runs:

```bash
poetry install
poetry run pytest
```

Because the same Alembic migrations and seed script are used in both local and CI environments, any schema or data issues will be caught before deployment.***

