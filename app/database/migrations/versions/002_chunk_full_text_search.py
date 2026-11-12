"""Add a GIN index for lexical search over document chunks.

Revision ID: 002_chunk_full_text_search
Revises: 001_initial
"""

from alembic import op

revision = "002_chunk_full_text_search"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_chunks_content_fts ON document_chunks USING gin "
        "(to_tsvector('simple'::regconfig, content || ' ' || "
        "coalesce(section_heading, '')))"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_content_fts", table_name="document_chunks")
