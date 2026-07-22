"""Odds API client for NFL full-game and halves/quarters lines.

Book list is the Penn Entertainment lineage (Barstool Sportsbook -> ESPN
BET -> theScore Bet, current live brand as of Dec 2025 after the ESPN
partnership ended and Penn relaunched under theScore Bet) plus the five
mainstream/offshore books named in CLAUDE.md. The Odds API bookmaker key
is still "espnbet" -- their catalog hasn't caught up to the rebrand yet.
If it ever gets renamed (e.g. to "thescorebet"), update BOOKS below; same
underlying operator/product either way. Keyed directly via the
`bookmakers` param, which bypasses region filtering entirely -- simpler
and more reliable than guessing which region a given book lives under.

Full-game markets (h2h/spreads/totals) are available on the list endpoint
(one call covers every upcoming game). Halves/quarters markets are only
available on the per-event endpoint, and only populate close to kickoff --
confirmed empirically, not assumed from docs.

Historical endpoints cost roughly 10x a live call per market and only go
back to ~August 2020 for NFL, confirmed empirically. Callers should budget
credits deliberately rather than looping over full historical ranges by
default.
"""
import os
from datetime import datetime, timezone

import requests

SPORT_KEY = "americanfootball_nfl"
BASE_URL = "https://api.the-odds-api.com/v4"

BOOKS = ["bovada", "draftkings", "fanduel", "williamhill_us", "betmgm", "espnbet"]

FULL_GAME_MARKETS = ["h2h", "spreads", "totals"]
PERIOD_MARKETS = [
    f"{market}_{period}"
    for market in ("h2h", "spreads", "totals")
    for period in ("h1", "h2", "q1", "q2", "q3", "q4")
]


def _get(path: str, params: dict) -> requests.Response:
    params = {"apiKey": os.environ["ODDS_API_KEY"], "oddsFormat": "american", **params}
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)
    resp.raise_for_status()
    return resp


def fetch_live_full_game_odds() -> list[dict]:
    """One call, all upcoming NFL games, full-game markets, all 6 books."""
    resp = _get(
        f"/sports/{SPORT_KEY}/odds",
        {"bookmakers": ",".join(BOOKS), "markets": ",".join(FULL_GAME_MARKETS)},
    )
    return resp.json()


def fetch_live_period_odds(event_id: str) -> dict:
    """Halves/quarters for one event. Only populates close to kickoff."""
    resp = _get(
        f"/sports/{SPORT_KEY}/events/{event_id}/odds",
        {"bookmakers": ",".join(BOOKS), "markets": ",".join(FULL_GAME_MARKETS + PERIOD_MARKETS)},
    )
    return resp.json()


def fetch_historical_events(as_of: datetime) -> list[dict]:
    """Cheap (1 credit) — list of events as they existed at a point in time."""
    resp = _get(
        f"/historical/sports/{SPORT_KEY}/events",
        {"date": as_of.isoformat().replace("+00:00", "Z")},
    )
    return resp.json()["data"]


def fetch_historical_odds(as_of: datetime, markets: list[str] | None = None) -> dict:
    """Expensive (~10 credits/market) — full-game odds snapshot at a point in time."""
    resp = _get(
        f"/historical/sports/{SPORT_KEY}/odds",
        {
            "bookmakers": ",".join(BOOKS),
            "markets": ",".join(markets or FULL_GAME_MARKETS),
            "date": as_of.isoformat().replace("+00:00", "Z"),
        },
    )
    return resp.json()


def fetch_historical_period_odds(event_id: str, as_of: datetime, markets: list[str]) -> dict:
    """Expensive (~10 credits/market) — halves/quarters snapshot at a point in time."""
    resp = _get(
        f"/historical/sports/{SPORT_KEY}/events/{event_id}/odds",
        {
            "bookmakers": ",".join(BOOKS),
            "markets": ",".join(markets),
            "date": as_of.isoformat().replace("+00:00", "Z"),
        },
    )
    return resp.json()


def normalize_event_odds(event: dict, pulled_at: datetime | None = None) -> list[dict]:
    """Flatten one event's nested bookmakers/markets/outcomes into snapshot rows."""
    pulled_at = pulled_at or datetime.now(timezone.utc)
    home_team = event["home_team"]
    away_team = event["away_team"]
    rows = []

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                name = outcome["name"]
                if name == home_team:
                    side = "home"
                elif name == away_team:
                    side = "away"
                else:
                    side = name.lower()  # "Over" / "Under"

                rows.append({
                    "odds_api_event_id": event["id"],
                    "game_id": None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": event.get("commence_time"),
                    "book": bookmaker["key"],
                    "market_type": market["key"],
                    "side": side,
                    "line": outcome.get("point"),
                    "price": outcome.get("price"),
                    "pulled_at": pulled_at.isoformat(),
                })
    return rows


if __name__ == "__main__":
    events = fetch_live_full_game_odds()
    print(f"Live events: {len(events)}")
    if events:
        sample_rows = normalize_event_odds(events[0])
        print(f"Sample event: {events[0]['home_team']} vs {events[0]['away_team']}")
        print(f"Normalized rows for this event: {len(sample_rows)}")
        for row in sample_rows[:5]:
            print(" ", row)
