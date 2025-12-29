from core.cards import Card, MODE_TRUMP, RANKS, SUITS
from core.game import play_round
from core.scoring import trick_points


def _card_sort_key(card: Card) -> tuple[int, int]:
    return (SUITS.index(card.suit), RANKS.index(card.rank))


def _policy_lowest(state, player):
    legal = state.legal_cards_for(player)
    return sorted(legal, key=_card_sort_key)[0]


def test_round_is_deterministic_with_seed() -> None:
    policies = [_policy_lowest] * 4
    result_a = play_round(policies, MODE_TRUMP, trump_suit="rosen", seed=123)
    result_b = play_round(policies, MODE_TRUMP, trump_suit="rosen", seed=123)
    assert result_a.play_log == result_b.play_log
    assert result_a.state.team_points == result_b.state.team_points


def test_round_completes_and_points_sum() -> None:
    policies = [_policy_lowest] * 4
    result = play_round(policies, MODE_TRUMP, trump_suit="schilten", seed=7)
    state = result.state

    assert all(len(hand) == 0 for hand in state.hands)
    assert len(state.completed_tricks) == 9
    assert sum(state.team_points) == 157

    base_points = sum(
        trick_points(trick.cards, state.mode, state.trump_suit, last_trick=False)
        for trick in state.completed_tricks
    )
    assert base_points + 5 == sum(state.team_points)
