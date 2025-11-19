"""
Initial schema for Medi OS Version -2.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20231115_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "patients",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("mrn", sa.String(length=64), nullable=False),
        sa.Column("first_name", sa.String(length=128), nullable=False),
        sa.Column("last_name", sa.String(length=128), nullable=False),
        sa.Column("dob", sa.Date(), nullable=True),
        sa.Column("sex", sa.String(length=16), nullable=True),
        sa.Column(
            "contact_info", postgresql.JSONB(astext_type=sa.Text()), nullable=True
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
            server_onupdate=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patients")),
    )
    op.create_index(op.f("ix_patients_mrn"), "patients", ["mrn"], unique=True)

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("performed_by", sa.String(length=128), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_logs")),
    )
    op.create_index(
        op.f("ix_audit_logs_entity_type"), "audit_logs", ["entity_type"], unique=False
    )

    op.create_table(
        "encounters",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "arrival_ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("disposition", sa.String(length=64), nullable=True),
        sa.Column("location", sa.String(length=128), nullable=True),
        sa.Column("acuity_level", sa.Integer(), nullable=True),
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
            server_onupdate=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_encounters_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_encounters")),
    )
    op.create_index(
        op.f("ix_encounters_patient_id"), "encounters", ["patient_id"], unique=False
    )
    op.create_index(
        op.f("ix_encounters_arrival_ts"), "encounters", ["arrival_ts"], unique=False
    )

    op.create_table(
        "summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "encounter_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
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
            server_onupdate=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_summaries_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_summaries")),
    )
    op.create_index(
        op.f("ix_summaries_patient_id"), "summaries", ["patient_id"], unique=False
    )

    op.create_table(
        "dialogue_transcripts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("encounter_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transcript", sa.Text(), nullable=False),
        sa.Column(
            "speaker_segments", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("source", sa.String(length=64), nullable=True),
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
            server_onupdate=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name=op.f("fk_dialogue_transcripts_encounter_id_encounters"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dialogue_transcripts")),
    )
    op.create_index(
        op.f("ix_dialogue_transcripts_encounter_id"),
        "dialogue_transcripts",
        ["encounter_id"],
        unique=False,
    )

    op.create_table(
        "soap_notes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("encounter_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subjective", sa.Text(), nullable=True),
        sa.Column("objective", sa.Text(), nullable=True),
        sa.Column("assessment", sa.Text(), nullable=True),
        sa.Column("plan", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
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
            server_onupdate=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["encounter_id"],
            ["encounters.id"],
            name=op.f("fk_soap_notes_encounter_id_encounters"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_soap_notes")),
    )
    op.create_index(
        op.f("ix_soap_notes_encounter_id"), "soap_notes", ["encounter_id"], unique=False
    )

    op.create_table(
        "triage_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("encounter_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("vitals", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("chief_complaint", sa.String(length=512), nullable=True),
        sa.Column("notes", sa.String(length=1024), nullable=True),
        sa.Column("triage_score", sa.Integer(), nullable=True),
        sa.Column("triage_model_version", sa.String(length=64), nullable=True),
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
            ["encounter_id"],
            ["encounters.id"],
            name=op.f("fk_triage_observations_encounter_id_encounters"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_triage_observations")),
        sa.UniqueConstraint(
            "encounter_id", name=op.f("uq_triage_observations_encounter_id")
        ),
    )
    op.create_index(
        op.f("ix_triage_observations_encounter_id"),
        "triage_observations",
        ["encounter_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_triage_observations_encounter_id"), table_name="triage_observations"
    )
    op.drop_table("triage_observations")
    op.drop_index(op.f("ix_soap_notes_encounter_id"), table_name="soap_notes")
    op.drop_table("soap_notes")
    op.drop_index(
        op.f("ix_dialogue_transcripts_encounter_id"), table_name="dialogue_transcripts"
    )
    op.drop_table("dialogue_transcripts")
    op.drop_index(op.f("ix_summaries_patient_id"), table_name="summaries")
    op.drop_table("summaries")
    op.drop_index(op.f("ix_encounters_arrival_ts"), table_name="encounters")
    op.drop_index(op.f("ix_encounters_patient_id"), table_name="encounters")
    op.drop_table("encounters")
    op.drop_index(op.f("ix_audit_logs_entity_type"), table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index(op.f("ix_patients_mrn"), table_name="patients")
    op.drop_table("patients")
