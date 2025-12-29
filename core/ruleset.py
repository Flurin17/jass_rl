from __future__ import annotations

from dataclasses import dataclass, field

from .legal_moves import RuleSet


@dataclass(frozen=True)
class RulesetConfig:
    allow_stock: bool = True
    allow_weis: bool = True
    legal_moves: RuleSet = field(default_factory=RuleSet)
