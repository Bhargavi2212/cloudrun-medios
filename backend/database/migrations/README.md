# Alembic Migrations

This directory contains Alembic migration scripts for the Medi OS database.

Commands (from repository root):

```bash
alembic -c backend/database/migrations/alembic.ini revision --autogenerate -m "message"
alembic -c backend/database/migrations/alembic.ini upgrade head
```

Migrations depend on the metadata defined in `backend/database/models.py`.

