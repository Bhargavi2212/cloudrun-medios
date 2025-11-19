"""
Add metadata column to federated_model_rounds.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20250117_000004"
down_revision = "20250117_000003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = inspector.get_table_names()

    if "federated_model_rounds" not in tables:
        op.create_table(
            "federated_model_rounds",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("model_name", sa.String(length=128), nullable=False),
            sa.Column("round_id", sa.Integer(), nullable=False),
            sa.Column(
                "weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False
            ),
            sa.Column("contributor_count", sa.Integer(), nullable=False),
            sa.Column(
                "round_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.UniqueConstraint(
                "model_name", "round_id", name="uq_federated_model_round"
            ),
        )
        return

    columns = {
        column["name"] for column in inspector.get_columns("federated_model_rounds")
    }
    if "round_metadata" not in columns:
        op.add_column(
            "federated_model_rounds",
            sa.Column(
                "round_metadata",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            ),
        )


def downgrade() -> None:
    op.drop_column("federated_model_rounds", "round_metadata")
