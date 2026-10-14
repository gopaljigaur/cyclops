"""Tools with postponed annotations, defined outside the toolkit module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Hashable


class Point(BaseModel):
    x: int
    y: int


def total(points: list[Point]) -> Hashable:
    """Sum the coordinates of the points."""
    return sum(p.x + p.y for p in points)
