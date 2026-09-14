"""rename username to user_id

Revision ID: d5f124f36ddf
Revises: 759f554ca962
Create Date: 2026-03-22 22:31:00.446852

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5f124f36ddf"
down_revision: Union[str, Sequence[str], None] = "759f554ca962"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename analysis_history.username to user_id."""
    op.alter_column("analysis_history", "username", new_column_name="user_id")


def downgrade() -> None:
    """Revert analysis_history.user_id back to username."""
    op.alter_column("analysis_history", "user_id", new_column_name="username")
