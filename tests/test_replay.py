from cli.replay import ReplayData, load_replay, replay_game, save_replay
from core.cards import Card, MODE_TRUMP, RANKS, SUITS
from core.game import play_round


def _card_sort_key(card: Card) -> tuple[int, int]:
    return (SUITS.index(card.suit), RANKS.index(card.rank))


def _policy_lowest(state, player: int) -> Card:
    legal = state.legal_cards_for(player)
    return sorted(legal, key=_card_sort_key)[0]


def test_replay_roundtrip(tmp_path) -> None:
    seed = 99
    result = play_round([_policy_lowest] * 4, MODE_TRUMP, trump_suit="schilten", seed=seed)
    replay = ReplayData(
        mode=MODE_TRUMP,
        trump_suit="schilten",
        leader=0,
        seed=seed,
        play_log=list(result.play_log),
    )

    path = tmp_path / "replay.json"
    save_replay(path, replay)
    loaded = load_replay(path)

    state = replay_game(loaded)
    assert state.team_points == result.state.team_points
    assert all(len(hand) == 0 for hand in state.hands)
