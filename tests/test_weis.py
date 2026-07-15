from core.announcements.weis import find_weis, resolve_weis_by_player
from core.cards import MODE_TRUMP, MODE_UNEUFE, Card


def test_sequence_detection() -> None:
    cards = [Card("schilten", "6"), Card("schilten", "7"), Card("schilten", "8")]
    combos = find_weis(cards)
    assert any(c.kind == "sequence" and c.length == 3 and c.points == 20 for c in combos)


def test_sequence_points_longer_runs() -> None:
    cards = [
        Card("schilten", "6"),
        Card("schilten", "7"),
        Card("schilten", "8"),
        Card("schilten", "9"),
        Card("schilten", "10"),
        Card("schilten", "J"),
    ]
    combos = find_weis(cards)
    assert any(c.kind == "sequence" and c.length == 6 and c.points == 150 for c in combos)


def test_four_of_a_kind_detection_any_rank() -> None:
    cards = [
        Card("schilten", "6"),
        Card("rosen", "6"),
        Card("schellen", "6"),
        Card("eicheln", "6"),
    ]
    combos = find_weis(cards)
    assert any(c.kind == "four_of_a_kind" and c.points == 100 for c in combos)


def test_four_of_a_kind_detection_accepts_generator() -> None:
    cards = (Card(suit, "A") for suit in ("schellen", "rosen", "schilten", "eicheln"))
    combos = find_weis(cards)
    assert any(c.kind == "four_of_a_kind" and c.rank == "A" for c in combos)


def test_resolve_weis_by_player_awards_only_best_team_sum_of_individuals() -> None:
    announced = {
        # Team A: p0 has 4 J (200), p2 has 3-sequence (20) => total 220 if team A wins.
        0: [Card("schilten", "J"), Card("rosen", "J"), Card("schellen", "J"), Card("eicheln", "J")],
        2: [Card("schilten", "6"), Card("schilten", "7"), Card("schilten", "8")],
        # Team B: p1 has 5-sequence (100), p3 none.
        1: [
            Card("rosen", "10"),
            Card("rosen", "J"),
            Card("rosen", "Q"),
            Card("rosen", "K"),
            Card("rosen", "A"),
        ],
        3: [],
    }
    points_a, points_b, winner, best = resolve_weis_by_player(
        announced, [0, 1, 2, 3], mode=MODE_TRUMP, trump_suit="eicheln"
    )
    assert winner == 0
    assert points_a == 220
    assert points_b == 0
    assert best is not None


def test_tie_breaker_is_earlier_announce() -> None:
    announced = {
        0: [Card("schilten", "6"), Card("schilten", "7"), Card("schilten", "8")],
        1: [Card("rosen", "6"), Card("rosen", "7"), Card("rosen", "8")],
        2: [],
        3: [],
    }
    # Same value; p0 announces earlier -> team A gets its 20 points.
    points_a, points_b, winner, best = resolve_weis_by_player(
        announced, [0, 1, 2, 3], mode=MODE_TRUMP, trump_suit="eicheln"
    )
    assert winner == 0
    assert points_a == 20
    assert points_b == 0
    assert best is not None


def test_trump_sequence_wins_tie() -> None:
    announced = {
        0: [Card("schilten", "6"), Card("schilten", "7"), Card("schilten", "8")],
        1: [Card("rosen", "6"), Card("rosen", "7"), Card("rosen", "8")],
        2: [],
        3: [],
    }
    points_a, points_b, winner, _ = resolve_weis_by_player(
        announced, [0, 1, 2, 3], mode=MODE_TRUMP, trump_suit="rosen"
    )
    assert winner == 1
    assert points_a == 0
    assert points_b == 20


def test_undenufe_sequence_compare_prefers_lower_top_card() -> None:
    announced = {
        # p0: 7-8-9 (top 9), p1: 6-7-8 (top 8). In Undenufe, lower wins.
        0: [Card("schilten", "7"), Card("schilten", "8"), Card("schilten", "9")],
        1: [Card("rosen", "6"), Card("rosen", "7"), Card("rosen", "8")],
        2: [],
        3: [],
    }
    points_a, points_b, winner, _ = resolve_weis_by_player(
        announced, [0, 1, 2, 3], mode=MODE_UNEUFE, trump_suit=None
    )
    assert winner == 1
    assert points_a == 0
    assert points_b == 20


def test_undenufe_four_of_a_kind_prefers_lower_rank() -> None:
    announced = {
        0: [Card(suit, "A") for suit in ("schellen", "rosen", "schilten", "eicheln")],
        1: [Card(suit, "6") for suit in ("schellen", "rosen", "schilten", "eicheln")],
        2: [],
        3: [],
    }
    points_a, points_b, winner, best = resolve_weis_by_player(
        announced, [0, 1, 2, 3], mode=MODE_UNEUFE, trump_suit=None
    )
    assert winner == 1
    assert points_a == 0
    assert points_b == 100
    assert best is not None and best.rank == "6"
