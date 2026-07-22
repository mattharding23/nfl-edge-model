"""Populate historical_games from nfl_data_py.import_schedules().

Sets spread_total_backtest_safe / moneyline_backtest_safe per row from
season-level coverage (see schedule_coverage.py) so backtest code can
filter on a column rather than re-deriving the coverage cutoff.
"""
import argparse
import math

import nfl_data_py as nfl
import pandas as pd

from db import get_supabase_client
from schedule_coverage import LINE_COLUMNS, coverage_by_season

INT_COLUMNS = [
    "home_score", "away_score", "away_moneyline", "home_moneyline",
    "away_spread_odds", "home_spread_odds", "under_odds", "over_odds",
    "away_rest", "home_rest",
]

SPREAD_TOTAL_COLS = ["spread_line", "total_line"]
MONEYLINE_COLS = [c for c in LINE_COLUMNS if c not in SPREAD_TOTAL_COLS]

COLUMNS_TO_KEEP = [
    "game_id", "season", "week", "game_type", "gameday",
    "home_team", "away_team", "home_score", "away_score",
    "spread_line", "total_line", "away_moneyline", "home_moneyline",
    "away_spread_odds", "home_spread_odds", "under_odds", "over_odds",
    "div_game", "roof", "surface", "temp", "wind", "away_rest", "home_rest",
]


def build_rows(start: int, end: int) -> list[dict]:
    seasons = list(range(start, end + 1))
    df = nfl.import_schedules(seasons)
    df = df[df["game_type"] != "PRE"].copy()

    coverage = coverage_by_season(start, end).set_index("season")
    spread_total_safe = {
        s: bool(all(coverage.loc[s, c] == 100.0 for c in SPREAD_TOTAL_COLS))
        for s in coverage.index
    }
    moneyline_safe = {
        s: bool(all(coverage.loc[s, c] == 100.0 for c in MONEYLINE_COLS))
        for s in coverage.index
    }

    df = df[COLUMNS_TO_KEEP]
    df["gameday"] = df["gameday"].astype(str)
    df["div_game"] = df["div_game"].astype(bool)

    rows = []
    for record in df.to_dict(orient="records"):
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
        for key in INT_COLUMNS:
            if record[key] is not None:
                record[key] = int(record[key])

        season = record["season"]
        record["spread_total_backtest_safe"] = spread_total_safe.get(season, False)
        record["moneyline_backtest_safe"] = moneyline_safe.get(season, False)
        rows.append(record)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load historical_games.")
    parser.add_argument("--start", type=int, default=1999)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--dry-run", action="store_true", help="Build rows but don't write.")
    args = parser.parse_args()

    rows = build_rows(args.start, args.end)
    print(f"Built {len(rows)} rows for seasons {args.start}-{args.end}.")
    print(f"Sample row: {rows[0]}")

    if args.dry_run:
        print("Dry run — not writing to Supabase.")
    else:
        client = get_supabase_client()
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            client.table("historical_games").upsert(batch, on_conflict="game_id").execute()
            print(f"Upserted rows {i}-{i + len(batch)}")
        print("Done.")
