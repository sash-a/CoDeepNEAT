from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src2.Evolution.Generation import Generation

instance: Optional[Generation] = None
