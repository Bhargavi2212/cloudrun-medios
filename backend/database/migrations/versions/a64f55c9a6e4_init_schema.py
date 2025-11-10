"""init schema

Revision ID: a64f55c9a6e4
Revises:
Create Date: 2025-11-08 21:34:03.986982

"""
from typing import Sequence, Union

from alembic import op

from backend.database.base import Base

# revision identifiers, used by Alembic.
revision: str = "a64f55c9a6e4"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables defined in SQLAlchemy metadata."""
    bind = op.get_bind()
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """Drop all tables defined in SQLAlchemy metadata."""
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
    op.execute("DROP EXTENSION IF EXISTS citext")
