"""Unit tests for scripts/market.py -- every downstream edge number
depends on consensus_fair_probability being correct, so this is tested
explicitly rather than only exercised indirectly.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import math
import pytest

from market import (
    american_to_decimal, consensus_fair_probability, best_available_price,
    true_ev, probability_edge, points_edge, moneyline_to_prob,
)


class TestAmericanToDecimal:
    def test_negative_odds(self):
        # -110 -> risk 110 to win 100 -> decimal 1.909...
        assert american_to_decimal(-110) == pytest.approx(1 + 100 / 110)

    def test_positive_odds(self):
        # +150 -> risk 100 to win 150 -> decimal 2.5
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_even_money_symmetry(self):
        # -100 and +100 should both give decimal 2.0 (even odds)
        assert american_to_decimal(-100) == pytest.approx(2.0)
        assert american_to_decimal(100) == pytest.approx(2.0)


class TestConsensusFairProbability:
    def test_single_book_matches_manual_devig(self):
        # -150/+130 at one book -- consensus of one book should equal
        # that book's own de-vigged probability exactly.
        result = consensus_fair_probability({"bookA": (-150, 130)})
        # manual: raw_a = 150/250=0.6, raw_b=100/230=0.4348; sum=1.0348
        raw_a, raw_b = 150 / 250, 100 / 230
        expected_a = raw_a / (raw_a + raw_b)
        expected_b = raw_b / (raw_a + raw_b)
        assert result["side_a_prob"] == pytest.approx(expected_a, abs=1e-9)
        assert result["side_b_prob"] == pytest.approx(expected_b, abs=1e-9)
        assert result["n_books"] == 1

    def test_probabilities_sum_to_one(self):
        result = consensus_fair_probability({
            "bookA": (-150, 130), "bookB": (-140, 120), "bookC": (-160, 140),
        })
        assert result["side_a_prob"] + result["side_b_prob"] == pytest.approx(1.0, abs=1e-9)

    def test_averages_across_books_not_just_first(self):
        # two books with meaningfully different implied probabilities --
        # consensus must land strictly between them, not equal either one.
        result = consensus_fair_probability({
            "bookA": (-200, 170),  # heavily favors side_a
            "bookB": (110, -130),  # favors side_b
        })
        prob_a_bookA = consensus_fair_probability({"bookA": (-200, 170)})["side_a_prob"]
        prob_a_bookB = consensus_fair_probability({"bookB": (110, -130)})["side_a_prob"]
        lo, hi = sorted([prob_a_bookA, prob_a_bookB])
        assert lo < result["side_a_prob"] < hi

    def test_vig_is_actually_removed(self):
        # a real two-sided vigged line should NOT sum to 1.0 before
        # de-vigging (that's the whole point of vig) -- confirm our
        # consensus output does, proving de-vig actually happened rather
        # than just averaging raw prices.
        raw_sum = moneyline_to_prob(-110) + moneyline_to_prob(-110)
        assert raw_sum > 1.0  # standard -110/-110 has ~4.76% vig
        result = consensus_fair_probability({"bookA": (-110, -110)})
        assert result["side_a_prob"] + result["side_b_prob"] == pytest.approx(1.0, abs=1e-9)
        assert result["side_a_prob"] == pytest.approx(0.5, abs=1e-9)  # symmetric price -> 50/50 fair

    def test_missing_side_drops_book(self):
        result = consensus_fair_probability({
            "bookA": (-150, 130),
            "bookB": (None, 120),  # incomplete -- can't de-vig with one side
        })
        assert result["n_books"] == 1
        assert "bookB" not in result["per_book"]

    def test_no_valid_books_returns_nan(self):
        result = consensus_fair_probability({})
        assert result["n_books"] == 0
        assert math.isnan(result["side_a_prob"])

    def test_per_book_breakdown_present_for_audit(self):
        result = consensus_fair_probability({"bookA": (-150, 130), "bookB": (-140, 120)})
        assert set(result["per_book"].keys()) == {"bookA", "bookB"}
        for book_result in result["per_book"].values():
            assert book_result["side_a_prob"] + book_result["side_b_prob"] == pytest.approx(1.0, abs=1e-9)


class TestBestAvailablePrice:
    def test_picks_highest_payout_positive_odds(self):
        book, price = best_available_price({"bookA": (120, -140), "bookB": (135, -150)}, "side_a")
        assert book == "bookB" and price == 135

    def test_picks_highest_payout_negative_odds(self):
        # -105 is a better price than -110 for the bettor (less vig)
        book, price = best_available_price({"bookA": (-110, -110), "bookB": (-105, -115)}, "side_a")
        assert book == "bookB" and price == -105

    def test_ignores_missing_side(self):
        book, price = best_available_price({"bookA": (None, -110), "bookB": (130, -145)}, "side_a")
        assert book == "bookB"

    def test_returns_none_when_no_books_have_side(self):
        assert best_available_price({"bookA": (None, -110)}, "side_a") is None


class TestTrueEV:
    def test_positive_ev_when_fair_prob_exceeds_implied(self):
        # fair prob 55%, but market price implies only 50% (-100/+100 even) -- positive EV
        ev = true_ev(fair_prob=0.55, american_odds=100)
        assert ev > 0

    def test_negative_ev_when_fair_prob_below_implied(self):
        ev = true_ev(fair_prob=0.45, american_odds=100)
        assert ev < 0

    def test_zero_ev_at_fair_price(self):
        # if fair_prob exactly matches the break-even probability implied
        # by the price offered, EV should be ~0
        assert true_ev(fair_prob=0.5, american_odds=100) == pytest.approx(0.0, abs=1e-9)
        # -150 requires implied prob 150/250=0.6 to break even
        assert true_ev(fair_prob=0.6, american_odds=-150) == pytest.approx(0.0, abs=1e-9)


class TestEdgeHelpers:
    def test_probability_edge_sign(self):
        assert probability_edge(0.55, 0.50) == pytest.approx(0.05)
        assert probability_edge(0.45, 0.50) == pytest.approx(-0.05)

    def test_points_edge_matches_backtest_convention(self):
        # fair_line - best_line, same convention as backtest.py's
        # `predicted_margin - spread_line`
        assert points_edge(fair_line=3.5, best_line=2.5) == pytest.approx(1.0)
        assert points_edge(fair_line=2.5, best_line=3.5) == pytest.approx(-1.0)
