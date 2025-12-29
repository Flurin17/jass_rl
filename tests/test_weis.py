from core.announcements.weis import WeisCombo, best_weis, find_weis, resolve_weis
from core.cards import Card


def test_sequence_detection() -> None:
    cards = [Card("schilten", "6"), Card("schilten", "7"), Card("schilten", "8")]
    combos = find_weis(cards)
    assert any(c.kind == "sequence" and c.length == 3 and c.points == 20 for c in combos)


def test_four_of_a_kind_detection() -> None:
    cards = [
        Card("schilten", "J"),
        Card("rosen", "J"),
        Card("schellen", "J"),
        Card("eicheln", "J"),
    ]
    combos = find_weis(cards)
    assert any(c.kind == "four_of_a_kind" and c.points == 200 for c in combos)


def test_resolve_weis_awards_only_best_team() -> None:
    team_a = [
        Card("schilten", "6"),
        Card("schilten", "7"),
        Card("schilten", "8"),
        Card("schilten", "J"),
        Card("rosen", "J"),
        Card("schellen", "J"),
        Card("eicheln", "J"),
    ]
    team_b = [
        Card("rosen", "10"),
        Card("rosen", "J"),
        Card("rosen", "Q"),
        Card("rosen", "K"),
        Card("rosen", "A"),
    ]
    points_a, points_b, winner, best = resolve_weis(team_a, team_b)
    assert winner == 0
    assert points_a == 220
    assert points_b == 0
    assert best is not None


def test_tie_results_in_no_points() -> None:
    team_a = [Card("schilten", "6"), Card("schilten", "7"), Card("schilten", "8")]
    team_b = [Card("rosen", "6"), Card("rosen", "7"), Card("rosen", "8")]
    points_a, points_b, winner, best = resolve_weis(team_a, team_b)
    assert points_a == 0
    assert points_b == 0
    assert winner is None
    assert best is None
