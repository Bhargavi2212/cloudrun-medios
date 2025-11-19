"""add structured_data to summaries

Revision ID: 20250117_000005
Revises: 20250117_000004
Create Date: 2025-01-17 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20250117_000005"
down_revision: str | None = "20250117_000004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add structured_data column to summaries table
    op.add_column(
        "summaries",
        sa.Column(
            "structured_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Structured timeline data for the summary",
        ),
    )


def downgrade() -> None:
    # Remove structured_data column
    op.drop_column("summaries", "structured_data")
