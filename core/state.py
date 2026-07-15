from __future__ import annotations

from dataclasses import dataclass, field

from .cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from .legal_moves import RuleSet, legal_cards
from .rankings import winning_card


@dataclass
class Trick:
    plays: list[tuple[int, Card]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.plays) > 4:
            raise ValueError("a trick cannot contain more than four plays")
        players = [player for player, _ in self.plays]
        if len(players) != len(set(players)):
            raise ValueError("a player cannot play twice in one trick")
        for player in players:
            _validate_player(player)

    @property
    def cards(self) -> list[Card]:
        return [card for _, card in self.plays]

    @property
    def led_suit(self) -> str | None:
        if not self.plays:
            return None
        return self.plays[0][1].suit


@dataclass(frozen=True)
class TrickResult:
    plays: list[tuple[int, Card]]
    winner: int
    points: int
    last_trick: bool

    def __post_init__(self) -> None:
        if len(self.plays) != 4:
            raise ValueError("a completed trick must contain four plays")
        players = [player for player, _ in self.plays]
        if len(set(players)) != 4:
            raise ValueError("each player must play exactly once per trick")
        for player in players:
            _validate_player(player)
        if self.winner not in players:
            raise ValueError("trick winner must be one of the players")
        if (
            not isinstance(self.points, int)
            or isinstance(self.points, bool)
            or self.points < 0
        ):
            raise ValueError("trick points must be a non-negative integer")
        if not isinstance(self.last_trick, bool):
            raise ValueError("last_trick must be a boolean")

    @property
    def cards(self) -> list[Card]:
        return [card for _, card in self.plays]


@dataclass
class GameState:
    """Mutable round state with factor-applied totals in ``team_points``.

    Raw trick points remain available on ``completed_tricks``.  Use the shared
    transition helpers to keep raw events and scored totals synchronized.
    """

    hands: list[list[Card]]
    mode: str
    trump_suit: str | None
    leader: int = 0
    trick_index: int = 0
    trick: Trick = field(default_factory=Trick)
    team_points: list[int] = field(default_factory=lambda: [0, 0])
    completed_tricks: list[TrickResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.hands) != 4:
            raise ValueError("exactly four hands are required")
        _validate_contract(self.mode, self.trump_suit)
        _validate_player(self.leader, name="leader")
        if (
            not isinstance(self.trick_index, int)
            or isinstance(self.trick_index, bool)
            or not 0 <= self.trick_index <= 9
        ):
            raise ValueError("trick_index must be between 0 and 9")
        if len(self.completed_tricks) > 9:
            raise ValueError("a round cannot contain more than nine tricks")
        if len(self.team_points) != 2 or any(
            not isinstance(points, int)
            or isinstance(points, bool)
            or points < 0
            for points in self.team_points
        ):
            raise ValueError("team_points must contain two non-negative integers")
        if len(self.trick.plays) > 4:
            raise ValueError("a trick cannot contain more than four plays")
        if len(self.completed_tricks) == 9:
            self.validate_terminal()

    @property
    def is_terminal(self) -> bool:
        return len(self.completed_tricks) == 9 and all(
            len(hand) == 0 for hand in self.hands
        )

    @property
    def current_player(self) -> int:
        return (self.leader + len(self.trick.plays)) % 4

    @staticmethod
    def partner(player: int) -> int:
        _validate_player(player)
        return (player + 2) % 4

    @staticmethod
    def team_index(player: int) -> int:
        _validate_player(player)
        return 0 if player % 2 == 0 else 1

    def current_winning_player(self) -> int | None:
        if not self.trick.plays:
            return None
        led_suit = self.trick.led_suit
        assert led_suit is not None
        cards = self.trick.cards
        winning = winning_card(cards, led_suit, self.mode, self.trump_suit)
        for player, card in self.trick.plays:
            if card == winning:
                return player
        return None

    def legal_cards_for(
        self, player: int, ruleset: RuleSet | None = None
    ) -> list[Card]:
        _validate_player(player)
        if self.is_terminal or len(self.trick.plays) == 4:
            return []
        partner_is_winning = self.current_winning_player() == self.partner(player)
        return legal_cards(
            self.hands[player],
            self.trick.cards,
            self.mode,
            trump_suit=self.trump_suit,
            partner_is_winning=partner_is_winning,
            ruleset=ruleset,
        )

    def play_card(
        self, player: int, card: Card, ruleset: RuleSet | None = None
    ) -> None:
        _validate_player(player)
        if self.is_terminal:
            raise ValueError("round is already complete")
        if len(self.trick.plays) == 4:
            raise ValueError("current trick is already complete")
        if player != self.current_player:
            raise ValueError("not this player's turn")
        if card not in self.hands[player]:
            raise ValueError("card not in hand")
        legal = self.legal_cards_for(player, ruleset)
        if card not in legal:
            raise ValueError("illegal card")
        self.hands[player].remove(card)
        self.trick.plays.append((player, card))

    def validate_terminal(self) -> None:
        """Raise when a purported finished round violates core invariants."""

        if not self.is_terminal:
            raise ValueError("round is not complete")
        if len(self.completed_tricks) != 9:
            raise ValueError("a completed round must contain nine tricks")
        if any(len(result.plays) != 4 for result in self.completed_tricks):
            raise ValueError("every completed trick must contain four plays")
        if sum(len(result.plays) for result in self.completed_tricks) != 36:
            raise ValueError("a completed round must contain exactly 36 plays")
        played_cards = [
            card for result in self.completed_tricks for card in result.cards
        ]
        if len(set(played_cards)) != 36:
            raise ValueError("each card must be played exactly once")
        last_flags = [result.last_trick for result in self.completed_tricks]
        if last_flags != [False] * 8 + [True]:
            raise ValueError("only the ninth trick may be marked as the last trick")
        if self.trick_index != 9:
            raise ValueError("a completed round must have trick_index 9")
        if self.trick.plays:
            raise ValueError("a completed round cannot have an active trick")
        if self.leader != self.completed_tricks[-1].winner:
            raise ValueError("terminal leader must be the final trick winner")


def _validate_player(player: int, *, name: str = "player") -> None:
    if (
        not isinstance(player, int)
        or isinstance(player, bool)
        or not 0 <= player < 4
    ):
        raise ValueError(f"{name} must be an integer from 0 to 3")


def _validate_contract(mode: str, trump_suit: str | None) -> None:
    if mode not in (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE):
        raise ValueError(f"unknown mode: {mode}")
    if mode == MODE_TRUMP:
        if trump_suit not in SUITS:
            raise ValueError("trump_suit is required and must be a valid suit")
    elif trump_suit is not None:
        raise ValueError("trump_suit must be None for non-trump modes")
