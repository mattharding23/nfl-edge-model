"""Unified Step 1 data pipeline entrypoint.

Combines the three sources confirmed independently:
  1. PBP (pbp.py) -- never persisted; load_pbp() is available for the
     backtest to call directly. Not run by default since there's nothing
     to pull for a season still in progress/upcoming; pass --pbp-season /
     --pbp-weeks to smoke-test it against a completed season.
  2. historical_games (load_historical_games.py) -- upserts the target
     season(s), idempotent on game_id, so late score/line updates land
     without re-running the full 1999-2025 backfill.
  3. odds_snapshots (pull_odds_snapshot.py) -- appends one live multi-book
     snapshot for every currently listed NFL game. No historical odds
     backfill here -- that's Step 5 (halves/quarters), deliberately
     deferred; see CLAUDE.md build sequence and the credit-cost sizing
     discussion in this repo's history.

This is the script GitHub Actions cron will eventually call on the
Tue/Wed -> Thu -> ~90min-pre-kickoff schedule from CLAUDE.md.
"""
import argparse

from db import get_supabase_client
from load_historical_games import build_rows
from pull_odds_snapshot import pull_and_store


def refresh_historical_games(season: int) -> int:
    rows = build_rows(season, season)
    client = get_supabase_client()
    for i in range(0, len(rows), 500):
        client.table("historical_games").upsert(rows[i:i + 500], on_conflict="game_id").execute()
    return len(rows)


def smoke_test_pbp(season: int, weeks: list[int]) -> None:
    from pbp import load_pbp
    df = load_pbp([season], weeks=weeks)
    print(f"  Loaded {len(df)} plays across {df['game_id'].nunique()} games "
          f"for {season} weeks {weeks}. Not persisted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Step 1 data pipeline.")
    parser.add_argument("--season", type=int, default=2026, help="Season for historical_games refresh.")
    parser.add_argument("--pbp-season", type=int, default=None, help="Optional: smoke-test the PBP loader.")
    parser.add_argument("--pbp-week-start", type=int, default=1)
    parser.add_argument("--pbp-week-end", type=int, default=1)
    args = parser.parse_args()

    print("=== Source 1: PBP ===")
    if args.pbp_season is not None:
        smoke_test_pbp(args.pbp_season, list(range(args.pbp_week_start, args.pbp_week_end + 1)))
    else:
        print("  Skipped -- load_pbp() is available for the backtest to call directly (pass --pbp-season to smoke-test).")

    print("=== Source 2: historical_games ===")
    n_games = refresh_historical_games(args.season)
    print(f"  Upserted {n_games} rows for season {args.season}.")

    print("=== Source 3: odds_snapshots ===")
    n_odds = pull_and_store()
    print(f"  Wrote {n_odds} rows to odds_snapshots.")

    print("Pipeline run complete.")
