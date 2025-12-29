from core.announcements.stock import StockTracker, is_stock_announcement
from core.cards import Card


def test_stock_announcement_on_second_card() -> None:
    trump = "schilten"
    played = [Card(trump, "K")]
    assert is_stock_announcement(played, Card(trump, "Q"), trump) is True
    assert is_stock_announcement([], Card(trump, "K"), trump) is False


def test_stock_tracker_awards_points_once() -> None:
    trump = "rosen"
    tracker = StockTracker()
    # First card: no points
    assert tracker.record_play(0, Card(trump, "K"), trump) == 0
    # Second card: award
    assert tracker.record_play(0, Card(trump, "Q"), trump) == 20
    # Further plays: no additional points
    assert tracker.record_play(0, Card(trump, "Q"), trump) == 0
