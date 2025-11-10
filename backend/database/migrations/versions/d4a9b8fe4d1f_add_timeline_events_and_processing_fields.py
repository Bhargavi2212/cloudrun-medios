"""Add timeline events table and file processing metadata.

Revision ID: d4a9b8fe4d1f
Revises: bd61e5aee3d5
Create Date: 2025-11-09 19:05:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d4a9b8fe4d1f"
down_revision: Union[str, None] = "bd61e5aee3d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_enum_if_missing(enum_name: str, values: Sequence[str]) -> None:
    formatted_values = ", ".join(f"''{value}''" for value in values)
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_type WHERE typname = '{enum_name}'
            ) THEN
                EXECUTE 'CREATE TYPE {enum_name} AS ENUM ({formatted_values})';
            END IF;
        END
        $$;
        """
    )


def upgrade() -> None:
    document_status_values = ["uploaded", "processing", "completed", "needs_review", "failed"]
    timeline_status_values = ["pending", "completed", "needs_review", "failed"]
    timeline_event_values = ["document", "lab_result", "medication", "vitals", "note"]

    _create_enum_if_missing("documentprocessingstatus", document_status_values)
    _create_enum_if_missing("timelineeventstatus", timeline_status_values)
    _create_enum_if_missing("timelineeventtype", timeline_event_values)

    document_status_enum = postgresql.ENUM(
        *document_status_values,
        name="documentprocessingstatus",
        create_type=False,
    )
    timeline_status_enum = postgresql.ENUM(
        *timeline_status_values,
        name="timelineeventstatus",
        create_type=False,
    )
    timeline_type_enum = postgresql.ENUM(
        *timeline_event_values,
        name="timelineeventtype",
        create_type=False,
    )

    op.add_column("files", sa.Column("document_type", sa.String(length=50), nullable=True))
    op.add_column(
        "files",
        sa.Column(
            "status",
            document_status_enum,
            nullable=False,
            server_default="uploaded",
        ),
    )
    op.add_column("files", sa.Column("confidence", sa.Numeric(4, 3), nullable=True))
    op.add_column("files", sa.Column("processing_metadata", sa.JSON(), nullable=True))
    op.add_column("files", sa.Column("processing_notes", sa.Text(), nullable=True))
    op.add_column(
        "files",
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("files", sa.Column("last_error", sa.Text(), nullable=True))

    op.create_index("ix_files_status", "files", ["status"])

    op.create_table(
        "timeline_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("timezone('utc', now())"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("timezone('utc', now())"),
        ),
        sa.Column(
            "is_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("patient_id", sa.String(length=36), nullable=False),
        sa.Column("consultation_id", sa.String(length=36), nullable=True),
        sa.Column("source_file_id", sa.String(length=36), nullable=True),
        sa.Column("event_type", timeline_type_enum, nullable=False),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column(
            "status",
            timeline_status_enum,
            nullable=False,
            server_default="pending",
        ),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["consultation_id"],
            ["consultations.id"],
            name="fk_timeline_events_consultation",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name="fk_timeline_events_patient",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_file_id"],
            ["files.id"],
            name="fk_timeline_events_file",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_timeline_events_confidence_range",
        ),
    )
    op.create_index(
        "ix_timeline_events_patient_date",
        "timeline_events",
        ["patient_id", "event_date"],
    )
    op.create_index("ix_timeline_events_status", "timeline_events", ["status"])
    op.create_index(
        "ix_timeline_events_source_file",
        "timeline_events",
        ["source_file_id"],
    )

    op.alter_column("files", "status", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_timeline_events_source_file", table_name="timeline_events")
    op.drop_index("ix_timeline_events_status", table_name="timeline_events")
    op.drop_index("ix_timeline_events_patient_date", table_name="timeline_events")
    op.drop_table("timeline_events")

    op.drop_index("ix_files_status", table_name="files")
    op.drop_column("files", "last_error")
    op.drop_column("files", "processed_at")
    op.drop_column("files", "processing_notes")
    op.drop_column("files", "processing_metadata")
    op.drop_column("files", "confidence")
    op.drop_column("files", "status")
    op.drop_column("files", "document_type")
    op.execute("DROP TYPE IF EXISTS timelineeventtype")
    op.execute("DROP TYPE IF EXISTS timelineeventstatus")
    op.execute("DROP TYPE IF EXISTS documentprocessingstatus")

