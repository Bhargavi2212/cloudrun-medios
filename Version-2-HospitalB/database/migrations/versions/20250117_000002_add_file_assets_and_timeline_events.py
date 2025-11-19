"""
Add file_assets and timeline_events tables for document upload and processing.

Revision ID: 20250117_000002
Revises: 20231115_000001
Create Date: 2025-01-17 12:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20250117_000002"
down_revision = "20231115_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create file_assets table
    op.create_table(
        "file_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("encounter_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("document_type", sa.String(length=50), nullable=True),
        sa.Column("upload_method", sa.String(length=20), nullable=True),
        sa.Column(
            "status", sa.String(length=30), nullable=False, server_default="uploaded"
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("extraction_status", sa.String(length=30), nullable=True),
        sa.Column("extraction_confidence", sa.Float(), nullable=True),
        sa.Column("confidence_tier", sa.String(length=20), nullable=True),
        sa.Column("review_status", sa.String(length=20), nullable=True),
        sa.Column(
            "needs_manual_review",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "extraction_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "processing_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("processing_notes", sa.Text(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_file_assets_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name=op.f("fk_file_assets_encounter_id_encounters"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_file_assets")),
    )
    op.create_index(
        op.f("ix_file_assets_patient_id"), "file_assets", ["patient_id"], unique=False
    )
    op.create_index(
        op.f("ix_file_assets_encounter_id"),
        "file_assets",
        ["encounter_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_file_assets_status"), "file_assets", ["status"], unique=False
    )
    op.create_index(
        op.f("ix_file_assets_extraction_status"),
        "file_assets",
        ["extraction_status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_file_assets_review_status"),
        "file_assets",
        ["review_status"],
        unique=False,
    )

    # Create timeline_events table
    op.create_table(
        "timeline_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("encounter_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_type", sa.String(length=30), nullable=True),
        sa.Column("source_file_asset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "status", sa.String(length=30), nullable=False, server_default="pending"
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("extraction_confidence", sa.Float(), nullable=True),
        sa.Column(
            "extraction_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "doctor_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_timeline_events_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name=op.f("fk_timeline_events_encounter_id_encounters"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_file_asset_id"],
            ["file_assets.id"],
            name=op.f("fk_timeline_events_source_file_asset_id_file_assets"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_timeline_events")),
    )
    op.create_index(
        op.f("ix_timeline_events_patient_id"),
        "timeline_events",
        ["patient_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_timeline_events_encounter_id"),
        "timeline_events",
        ["encounter_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_timeline_events_event_date"),
        "timeline_events",
        ["event_date"],
        unique=False,
    )
    op.create_index(
        op.f("ix_timeline_events_source_file_asset_id"),
        "timeline_events",
        ["source_file_asset_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_timeline_events_status"), "timeline_events", ["status"], unique=False
    )
    op.create_index(
        op.f("ix_timeline_events_source_type"),
        "timeline_events",
        ["source_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_timeline_events_source_type"), table_name="timeline_events")
    op.drop_index(op.f("ix_timeline_events_status"), table_name="timeline_events")
    op.drop_index(
        op.f("ix_timeline_events_source_file_asset_id"), table_name="timeline_events"
    )
    op.drop_index(op.f("ix_timeline_events_event_date"), table_name="timeline_events")
    op.drop_index(op.f("ix_timeline_events_encounter_id"), table_name="timeline_events")
    op.drop_index(op.f("ix_timeline_events_patient_id"), table_name="timeline_events")
    op.drop_table("timeline_events")
    op.drop_index(op.f("ix_file_assets_review_status"), table_name="file_assets")
    op.drop_index(op.f("ix_file_assets_extraction_status"), table_name="file_assets")
    op.drop_index(op.f("ix_file_assets_status"), table_name="file_assets")
    op.drop_index(op.f("ix_file_assets_encounter_id"), table_name="file_assets")
    op.drop_index(op.f("ix_file_assets_patient_id"), table_name="file_assets")
    op.drop_table("file_assets")
