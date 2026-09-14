"""merge_heads_before_api_keys

Revision ID: dd5bd40489a3
Revises: 6fb0f926b7a1, 726d427447d7
Create Date: 2026-03-26 12:10:38.613359

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dd5bd40489a3"
down_revision: Union[str, Sequence[str], None] = ("6fb0f926b7a1", "726d427447d7")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
