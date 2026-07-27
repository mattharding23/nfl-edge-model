"""Layer 4: market layer.

- Opening vs. current line vs. our independent fair-value number
- Line movement velocity / steam detection
- Vig-removed implied probability for true edge/EV calculation

This module consumes odds -- it does not generate the fair-value line
itself (that's Layers 1-3, see power_ratings.py / matchup_adjustments.py
/ situational.py / backtest.py's predict_game). Opening/current line
data comes from odds_snapshots (the append-only time series
pull_odds_snapshot.py has been writing to since Step 1).

Odds API region param (resolved, re-verified live in this session): no
single `region` covers all 6 books -- `region=us2` returns espnbet but
not Bovada; `region=us` returns Bovada/Caesars but not espnbet. Always
use the explicit `bookmakers` param (odds_api.py's BOOKS list), which
bypasses region filtering and returns all 6 in one call.

Steam detection is NOT implemented pending sign-off -- see
propose_steam_definition() and CLAUDE.md for the proposed definition,
reasoning, and open questions. Don't hardcode thresholds before that
sign-off; STEAM_CONFIG below is a placeholder structure only.
"""
import numpy as np
import pandas as pd

from backtest import moneyline_to_prob, devig_two_way  # American-odds math is format-agnostic: works for h2h, spreads, or totals prices alike

BOOKS = ["bovada", "draftkings", "fanduel", "williamhill_us", "betmgm", "espnbet"]

# Placeholder only -- see propose_steam_definition(). Not used by any
# function below until the definition is signed off.
STEAM_CONFIG_PLACEHOLDER = {
    "min_move_points": None,
    "min_move_prob": None,
    "window_minutes": None,
    "min_books_corroborating": None,
}


def american_to_decimal(odds: float) -> float:
    """American odds -> decimal odds (for EV calculation)."""
    if odds < 0:
        return 1 + 100 / (-odds)
    return 1 + odds / 100


def consensus_fair_probability(book_prices: dict[str, tuple[float, float]]) -> dict:
    """No-vig CONSENSUS across books: de-vigs each book's own two-sided
    price independently, then averages the de-vigged probabilities
    across books -- not a single-book de-vig, and not an average of raw
    (still-vigged) prices, which would leave vig baked into the "fair"
    number.

    book_prices: {book_name: (side_a_american_odds, side_b_american_odds)}
    Returns: side_a_prob / side_b_prob (consensus, vig-free, sums to 1.0),
    n_books used, and the per-book de-vigged breakdown for auditability.
    """
    per_book = {}
    for book, (odds_a, odds_b) in book_prices.items():
        if odds_a is None or odds_b is None:
            continue
        raw_a, raw_b = moneyline_to_prob(odds_a), moneyline_to_prob(odds_b)
        fair_a, fair_b = devig_two_way(raw_a, raw_b)
        per_book[book] = {"side_a_prob": fair_a, "side_b_prob": fair_b}

    if not per_book:
        return {"side_a_prob": np.nan, "side_b_prob": np.nan, "n_books": 0, "per_book": {}}

    side_a_prob = float(np.mean([v["side_a_prob"] for v in per_book.values()]))
    side_b_prob = float(np.mean([v["side_b_prob"] for v in per_book.values()]))
    # renormalize -- averaging two already-normalized-per-book pairs can
    # drift a hair from summing to exactly 1.0 due to floating point;
    # cheap and exact to fix here rather than let it compound downstream.
    total = side_a_prob + side_b_prob
    return {
        "side_a_prob": side_a_prob / total,
        "side_b_prob": side_b_prob / total,
        "n_books": len(per_book),
        "per_book": per_book,
    }


def best_available_price(book_prices: dict[str, tuple[float, float]], side: str) -> tuple[str, float] | None:
    """The best (highest payout) American price for the given side
    ("side_a" or "side_b") across books -- CLAUDE.md's edge definition
    is measured against the best available line, not the consensus.
    """
    idx = 0 if side == "side_a" else 1
    candidates = [(book, prices[idx]) for book, prices in book_prices.items() if prices[idx] is not None]
    if not candidates:
        return None
    # higher American odds = better price for the bettor in both signed
    # regimes (e.g. +150 beats +120; -105 beats -110)
    return max(candidates, key=lambda x: x[1])


def true_ev(fair_prob: float, american_odds: float) -> float:
    """Expected value per $1 staked, using our vig-removed fair
    probability against a specific book's price (typically the best
    available price for that side, via best_available_price)."""
    decimal_odds = american_to_decimal(american_odds)
    return fair_prob * (decimal_odds - 1) - (1 - fair_prob)


def probability_edge(fair_prob: float, market_prob: float) -> float:
    """Simple probability-space edge: our fair probability minus the
    market's no-vig consensus probability for the same side."""
    return fair_prob - market_prob


def points_edge(fair_line: float, best_line: float) -> float:
    """Points-space edge for spread/total markets -- identical
    convention to backtest.py's `predicted_margin - spread_line`."""
    return fair_line - best_line


def load_opening_and_current(odds_snapshots: pd.DataFrame, game_id_col: str = "odds_api_event_id") -> pd.DataFrame:
    """From odds_snapshots (long format: one row per book/market/side/
    pulled_at), extracts the OPENING (earliest pulled_at) and CURRENT
    (latest pulled_at) snapshot per (event, book, market_type, side).
    """
    df = odds_snapshots.sort_values("pulled_at")
    keys = [game_id_col, "book", "market_type", "side"]
    opening = df.groupby(keys, as_index=False).first()
    current = df.groupby(keys, as_index=False).last()
    merged = opening.merge(current, on=keys, suffixes=("_opening", "_current"))
    return merged


def line_movement(merged_opening_current: pd.DataFrame, fair_line_col: str | None = None) -> pd.DataFrame:
    """Adds movement columns: current - opening (points, for spread/
    total) and, if a fair_line_col is supplied, opening-vs-fair and
    current-vs-fair deltas -- the three-way comparison CLAUDE.md's
    market layer calls for.
    """
    out = merged_opening_current.copy()
    out["line_movement"] = out["line_current"] - out["line_opening"]
    if fair_line_col and fair_line_col in out.columns:
        out["fair_vs_opening"] = out[fair_line_col] - out["line_opening"]
        out["fair_vs_current"] = out[fair_line_col] - out["line_current"]
    return out


def propose_steam_definition() -> str:
    """Returns the proposed (NOT yet signed off) steam-detection
    definition as text, for review -- see CLAUDE.md for the same
    content plus open questions. Kept in code too so the proposal and
    the (currently unimplemented) detector can't silently drift apart
    once thresholds are approved.
    """
    return (
        "PROPOSED (pending sign-off): steam = a line move of >= 1.5 points "
        "(spread/total) or >= 4 percentage points of no-vig implied probability "
        "(moneyline), occurring within a 30-minute window, corroborated by the "
        "SAME-DIRECTION move appearing at >= 3 of the 6 tracked books -- as opposed "
        "to a single book adjusting alone (idiosyncratic, not steam) or the same "
        "total move spread gradually across many hours (drift, not steam). "
        "See CLAUDE.md's Layer 4 section for reasoning and open questions."
    )


if __name__ == "__main__":
    print(propose_steam_definition())
