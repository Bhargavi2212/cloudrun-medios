"""add triage_level to consultations

Revision ID: aa5f6d6a70a4
Revises: f1d9d954e744
Create Date: 2025-11-09 15:15:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "aa5f6d6a70a4"
down_revision = "f1d9d954e744"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("consultations", sa.Column("triage_level", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("consultations", "triage_level")


