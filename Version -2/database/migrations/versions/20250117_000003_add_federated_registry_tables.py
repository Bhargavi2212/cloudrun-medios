"""Add hospital registry and federated patient cache tables.

Revision ID: 20250117_000003
Revises: 20250117_000002
Create Date: 2025-11-17 15:20:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20250117_000003"
down_revision = "20250117_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hospital_registry",
        sa.Column("id", sa.String(length=64), nullable=False),
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
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("manage_url", sa.String(length=255), nullable=False),
        sa.Column("scribe_url", sa.String(length=255), nullable=True),
        sa.Column("summarizer_url", sa.String(length=255), nullable=True),
        sa.Column("dol_url", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "federated_patient_profiles",
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
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("mrn", sa.String(length=64), nullable=True),
        sa.Column("primary_hospital_id", sa.String(length=64), nullable=True),
        sa.Column(
            "demographics", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("summaries", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "last_snapshot_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("patient_id"),
    )

    op.create_table(
        "federated_timeline_events",
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
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_hospital_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("encounter_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("summary_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["federated_patient_profiles.patient_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "patient_id", "external_id", name="uq_federated_timeline_events_external"
        ),
    )
    op.create_index(
        "ix_federated_timeline_events_patient_ts",
        "federated_timeline_events",
        ["patient_id", "event_timestamp"],
    )

    op.create_table(
        "federated_model_rounds",
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
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("contributor_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_name", "round_id", name="uq_federated_model_round"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_federated_timeline_events_patient_ts",
        table_name="federated_timeline_events",
    )
    op.drop_table("federated_model_rounds")
    op.drop_table("federated_timeline_events")
    op.drop_table("federated_patient_profiles")
    op.drop_table("hospital_registry")
