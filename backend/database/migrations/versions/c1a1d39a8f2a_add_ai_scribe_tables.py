"""Add AI scribe session, vitals, soap note tables and extend triage predictions.

Revision ID: c1a1d39a8f2a
Revises: bd61e5aee3d5
Create Date: 2025-11-17 09:25:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1a1d39a8f2a"
down_revision: Union[str, None] = "bd61e5aee3d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scribe_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("consultation_id", sa.String(length=36), nullable=True),
        sa.Column("patient_id", sa.String(length=36), nullable=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "created",
                "streaming",
                "summarizing",
                "completed",
                "failed",
                name="scribesessionstatus",
            ),
            nullable=False,
        ),
        sa.Column("language", sa.String(length=10), nullable=True),
        sa.Column("active_audio_file_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("session_metadata", sa.JSON(), nullable=True),
        sa.Column("transcript_snapshot", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.ForeignKeyConstraint(["active_audio_file_id"], ["audio_files.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["consultation_id"], ["consultations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scribe_sessions_consultation", "scribe_sessions", ["consultation_id"], unique=False)
    op.create_index("ix_scribe_sessions_status", "scribe_sessions", ["status"], unique=False)

    op.create_table(
        "scribe_transcript_segments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("speaker_label", sa.String(length=50), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=True),
        sa.Column("end_ms", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("is_final", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("segment_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.ForeignKeyConstraint(["session_id"], ["scribe_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_scribe_segments_session_time",
        "scribe_transcript_segments",
        ["session_id", "start_ms"],
        unique=False,
    )

    op.create_table(
        "scribe_vitals",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("recorded_by", sa.String(length=36), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("heart_rate", sa.Integer(), nullable=True),
        sa.Column("respiratory_rate", sa.Integer(), nullable=True),
        sa.Column("systolic_bp", sa.Integer(), nullable=True),
        sa.Column("diastolic_bp", sa.Integer(), nullable=True),
        sa.Column("temperature_c", sa.Numeric(4, 1), nullable=True),
        sa.Column("oxygen_saturation", sa.Integer(), nullable=True),
        sa.Column("pain_score", sa.Integer(), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.ForeignKeyConstraint(["recorded_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["scribe_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scribe_vitals_session_time", "scribe_vitals", ["session_id", "recorded_at"], unique=False)

    op.create_table(
        "soap_notes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("consultation_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("specialty", sa.String(length=50), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.Column("raw_markdown", sa.Text(), nullable=True),
        sa.Column("confidence", sa.JSON(), nullable=True),
        sa.Column("tokens_prompt", sa.Integer(), nullable=True),
        sa.Column("tokens_completion", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.ForeignKeyConstraint(["consultation_id"], ["consultations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["scribe_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_soap_notes_session", "soap_notes", ["session_id", "created_at"], unique=False)
    op.create_index("ix_soap_notes_consultation", "soap_notes", ["consultation_id"], unique=False)

    op.add_column("triage_predictions", sa.Column("session_id", sa.String(length=36), nullable=True))
    op.add_column("triage_predictions", sa.Column("probabilities", sa.JSON(), nullable=True))
    op.add_column("triage_predictions", sa.Column("flagged", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("triage_predictions", sa.Column("latency_ms", sa.Integer(), nullable=True))
    op.add_column("triage_predictions", sa.Column("source", sa.String(length=50), nullable=True))
    op.create_index("ix_triage_predictions_session", "triage_predictions", ["session_id"], unique=False)
    op.create_foreign_key(
        "fk_triage_predictions_session",
        "triage_predictions",
        "scribe_sessions",
        ["session_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_triage_predictions_session", "triage_predictions", type_="foreignkey")
    op.drop_index("ix_triage_predictions_session", table_name="triage_predictions")
    op.drop_column("triage_predictions", "source")
    op.drop_column("triage_predictions", "latency_ms")
    op.drop_column("triage_predictions", "flagged")
    op.drop_column("triage_predictions", "probabilities")
    op.drop_column("triage_predictions", "session_id")

    op.drop_index("ix_soap_notes_consultation", table_name="soap_notes")
    op.drop_index("ix_soap_notes_session", table_name="soap_notes")
    op.drop_table("soap_notes")

    op.drop_index("ix_scribe_vitals_session_time", table_name="scribe_vitals")
    op.drop_table("scribe_vitals")

    op.drop_index("ix_scribe_segments_session_time", table_name="scribe_transcript_segments")
    op.drop_table("scribe_transcript_segments")

    op.drop_index("ix_scribe_sessions_status", table_name="scribe_sessions")
    op.drop_index("ix_scribe_sessions_consultation", table_name="scribe_sessions")
    op.drop_table("scribe_sessions")
    sa.Enum(
        "created",
        "streaming",
        "summarizing",
        "completed",
        "failed",
        name="scribesessionstatus",
    ).drop(op.get_bind(), checkfirst=False)

