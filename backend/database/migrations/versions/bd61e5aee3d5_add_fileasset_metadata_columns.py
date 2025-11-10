"""Add metadata columns to file assets.

Revision ID: bd61e5aee3d5
Revises: aa5f6d6a70a4
Create Date: 2025-11-09 18:15:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "bd61e5aee3d5"
down_revision: Union[str, None] = "aa5f6d6a70a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("files", sa.Column("original_filename", sa.String(length=255), nullable=True))
    op.add_column("files", sa.Column("description", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("files", "description")
    op.drop_column("files", "original_filename")

