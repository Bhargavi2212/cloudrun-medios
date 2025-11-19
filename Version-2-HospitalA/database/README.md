# Database Module

This package centralizes the persistence layer for Medi OS Version -2.

## Structure
- `base.py` — Declarative base and timestamp mixins.
- `models.py` — SQLAlchemy entity definitions.
- `session.py` — Async engine/session helpers and configuration.
- `crud.py` — Generic repository abstraction.
- `seeds.py` — Synthetic data seeding utilities.
- `migrations/` — Alembic environment and migration history.

## Running Migrations
```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2_a"
poetry run alembic -c database/alembic.ini upgrade head
```

## Seeding Demo Data
```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2_a"
poetry run python scripts/seed_database.py
```

The seed routine provisions:

- Three patients spanning urgent care, ED, and telehealth encounters.
- Multi-role staff participation (reception, triage nursing, physicians, federation proxy) captured in `audit_logs`.
- Cross-hospital context demonstrating how portable profiles aggregate partner updates.

Re-run `scripts/reset_demo_data.sh` if you need to wipe the tables before reseeding.

## Tests
Set `TEST_DATABASE_URL` to a disposable PostgreSQL database, then run:
```bash
TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2_a_test" \
poetry run pytest database/tests
```

