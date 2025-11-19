"""Enhance file assets and timeline metadata for AI summarizer pipeline.

Revision ID: e2fb1a4b4a6f
Revises: c1a1d39a8f2a
Create Date: 2025-11-17 12:00:00.000000
"""

from typing import Union, Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e2fb1a4b4a6f"
down_revision: Union[str, None] = "c1a1d39a8f2a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("files", sa.Column("upload_method", sa.String(length=20), nullable=True))
    op.add_column("files", sa.Column("raw_text", sa.Text(), nullable=True))
    op.add_column("files", sa.Column("extraction_status", sa.String(length=30), nullable=True))
    op.add_column("files", sa.Column("extraction_confidence", sa.Numeric(4, 3), nullable=True))
    op.add_column("files", sa.Column("confidence_tier", sa.String(length=20), nullable=True))
    op.add_column("files", sa.Column("review_status", sa.String(length=20), nullable=True))
    op.add_column(
        "files",
        sa.Column("needs_manual_review", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("files", sa.Column("extraction_data", sa.JSON(), nullable=True))
    op.add_column("files", sa.Column("nurse_corrections", sa.JSON(), nullable=True))
    op.add_column("files", sa.Column("linked_timeline_event_id", sa.String(length=36), nullable=True))
    op.create_foreign_key(
        "fk_files_linked_timeline_event",
        "files",
        "timeline_events",
        ["linked_timeline_event_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_files_extraction_status", "files", ["extraction_status"])
    op.create_index("ix_files_review_status", "files", ["review_status"])
    op.create_index("ix_files_confidence_tier", "files", ["confidence_tier"])
    op.alter_column("files", "needs_manual_review", server_default=None)

    op.drop_constraint("fk_timeline_events_file", "timeline_events", type_="foreignkey")
    op.drop_index("ix_timeline_events_source_file", table_name="timeline_events")
    op.alter_column("timeline_events", "source_file_id", new_column_name="source_file_asset_id")
    op.add_column("timeline_events", sa.Column("source_type", sa.String(length=30), nullable=True))
    op.add_column("timeline_events", sa.Column("extraction_confidence", sa.Numeric(4, 3), nullable=True))
    op.add_column("timeline_events", sa.Column("extraction_metadata", sa.JSON(), nullable=True))
    op.add_column(
        "timeline_events",
        sa.Column("doctor_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("timeline_events", sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("timeline_events", sa.Column("verified_by", sa.String(length=36), nullable=True))
    op.create_foreign_key(
        "fk_timeline_events_verified_by",
        "timeline_events",
        "users",
        ["verified_by"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_timeline_events_file_asset",
        "timeline_events",
        "files",
        ["source_file_asset_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_timeline_events_source_file_asset",
        "timeline_events",
        ["source_file_asset_id"],
    )
    op.create_index("ix_timeline_events_source_type", "timeline_events", ["source_type"])
    op.alter_column("timeline_events", "doctor_verified", server_default=None)


def downgrade() -> None:
    op.alter_column("timeline_events", "doctor_verified", server_default=sa.text("false"))
    op.drop_index("ix_timeline_events_source_type", table_name="timeline_events")
    op.drop_index("ix_timeline_events_source_file_asset", table_name="timeline_events")
    op.drop_constraint("fk_timeline_events_file_asset", "timeline_events", type_="foreignkey")
    op.drop_constraint("fk_timeline_events_verified_by", "timeline_events", type_="foreignkey")
    op.drop_column("timeline_events", "verified_by")
    op.drop_column("timeline_events", "verified_at")
    op.drop_column("timeline_events", "doctor_verified")
    op.drop_column("timeline_events", "extraction_metadata")
    op.drop_column("timeline_events", "extraction_confidence")
    op.drop_column("timeline_events", "source_type")
    op.alter_column("timeline_events", "source_file_asset_id", new_column_name="source_file_id")
    op.create_index("ix_timeline_events_source_file", "timeline_events", ["source_file_id"])
    op.create_foreign_key(
        "fk_timeline_events_file",
        "timeline_events",
        "files",
        ["source_file_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.drop_index("ix_files_confidence_tier", table_name="files")
    op.drop_index("ix_files_review_status", table_name="files")
    op.drop_index("ix_files_extraction_status", table_name="files")
    op.drop_constraint("fk_files_linked_timeline_event", "files", type_="foreignkey")
    op.drop_column("files", "linked_timeline_event_id")
    op.drop_column("files", "nurse_corrections")
    op.drop_column("files", "extraction_data")
    op.drop_column("files", "needs_manual_review")
    op.drop_column("files", "review_status")
    op.drop_column("files", "confidence_tier")
    op.drop_column("files", "extraction_confidence")
    op.drop_column("files", "extraction_status")
    op.drop_column("files", "raw_text")
    op.drop_column("files", "upload_method")
