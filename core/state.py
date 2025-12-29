from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from .cards import Card, MODE_TRUMP
from .legal_moves import RuleSet, legal_cards
from .rankings import winning_card


@dataclass
class Trick:
    plays: List[Tuple[int, Card]] = field(default_factory=list)

    @property
    def cards(self) -> List[Card]:
        return [card for _, card in self.plays]

    @property
    def led_suit(self) -> Optional[str]:
        if not self.plays:
            return None
        return self.plays[0][1].suit


@dataclass(frozen=True)
class TrickResult:
    plays: List[Tuple[int, Card]]
    winner: int
    points: int
    last_trick: bool

    @property
    def cards(self) -> List[Card]:
        return [card for _, card in self.plays]


@dataclass
class GameState:
    hands: List[List[Card]]
    mode: str
    trump_suit: Optional[str]
    leader: int = 0
    trick_index: int = 0
    trick: Trick = field(default_factory=Trick)
    team_points: List[int] = field(default_factory=lambda: [0, 0])
    completed_tricks: List[TrickResult] = field(default_factory=list)

    @property
    def current_player(self) -> int:
        return (self.leader + len(self.trick.plays)) % 4

    @staticmethod
    def partner(player: int) -> int:
        return (player + 2) % 4

    @staticmethod
    def team_index(player: int) -> int:
        return 0 if player % 2 == 0 else 1

    def current_winning_player(self) -> Optional[int]:
        if not self.trick.plays:
            return None
        led_suit = self.trick.led_suit
        cards = self.trick.cards
        winning = winning_card(cards, led_suit, self.mode, self.trump_suit)
        for player, card in self.trick.plays:
            if card == winning:
                return player
        return None

    def legal_cards_for(self, player: int, ruleset: Optional[RuleSet] = None) -> List[Card]:
        if self.mode == MODE_TRUMP and self.trump_suit is None:
            raise ValueError("trump_suit is required for trump mode")
        partner_is_winning = self.current_winning_player() == self.partner(player)
        return legal_cards(
            self.hands[player],
            self.trick.cards,
            self.mode,
            trump_suit=self.trump_suit,
            partner_is_winning=partner_is_winning,
            ruleset=ruleset,
        )

    def play_card(self, player: int, card: Card, ruleset: Optional[RuleSet] = None) -> None:
        if player != self.current_player:
            raise ValueError("not this player's turn")
        if card not in self.hands[player]:
            raise ValueError("card not in hand")
        legal = self.legal_cards_for(player, ruleset)
        if card not in legal:
            raise ValueError("illegal card")
        self.hands[player].remove(card)
        self.trick.plays.append((player, card))
