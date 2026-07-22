"""Pull current live full-game NFL odds for all 6 books and write to
odds_snapshots. Each run appends a new pulled_at snapshot -- this table is
an append-only time series (see schema_new_tables.sql), not upserted.
"""
import argparse

from db import get_supabase_client
from odds_api import fetch_live_full_game_odds, normalize_event_odds


def pull_and_store(limit_events: int | None = None) -> int:
    events = fetch_live_full_game_odds()
    if limit_events is not None:
        events = events[:limit_events]

    rows = [row for event in events for row in normalize_event_odds(event)]
    if not rows:
        return 0

    client = get_supabase_client()
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        client.table("odds_snapshots").insert(rows[i:i + batch_size]).execute()
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull live NFL odds into odds_snapshots.")
    parser.add_argument(
        "--limit-events", type=int, default=None,
        help="Cap number of events pulled (useful for a small smoke test).",
    )
    args = parser.parse_args()

    n = pull_and_store(limit_events=args.limit_events)
    print(f"Wrote {n} rows to odds_snapshots.")
