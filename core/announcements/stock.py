from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..cards import Card

STOCK_POINTS = 20


def is_stock_card(card: Card, trump_suit: str) -> bool:
    return card.suit == trump_suit and card.rank in ("K", "Q")


def is_stock_announcement(played_cards: List[Card], card_played: Card, trump_suit: str) -> bool:
    if not is_stock_card(card_played, trump_suit):
        return False
    other_rank = "Q" if card_played.rank == "K" else "K"
    return any(card.suit == trump_suit and card.rank == other_rank for card in played_cards)


@dataclass
class StockTracker:
    announced_by: set[int] = field(default_factory=set)
    played_by: Dict[int, List[Card]] = field(
        default_factory=lambda: {0: [], 1: [], 2: [], 3: []}
    )

    def record_play(self, player: int, card: Card, trump_suit: str) -> int:
        if player not in self.played_by:
            self.played_by[player] = []
        points = 0
        if (
            player not in self.announced_by
            and is_stock_announcement(self.played_by[player], card, trump_suit)
        ):
            self.announced_by.add(player)
            points = STOCK_POINTS
        self.played_by[player].append(card)
        return points
