"""Check season-by-season odds coverage in nfl_data_py.import_schedules().

Historical full-game lines come bundled into the schedules table (spread,
total, moneyline columns). Coverage is spotty in early seasons, so this
script reports per-season fill rates before any of it is treated as
backtest-ready. Read-only — writes nothing to Supabase or disk.
"""
import argparse

import nfl_data_py as nfl
import pandas as pd

LINE_COLUMNS = [
    "spread_line",
    "total_line",
    "away_moneyline",
    "home_moneyline",
    "away_spread_odds",
    "home_spread_odds",
    "under_odds",
    "over_odds",
]


def coverage_by_season(start: int, end: int) -> pd.DataFrame:
    seasons = list(range(start, end + 1))
    df = nfl.import_schedules(seasons)

    # Regular + postseason only — coverage of odds for preseason is irrelevant.
    df = df[df["game_type"] != "PRE"]

    rows = []
    for season, group in df.groupby("season"):
        n_games = len(group)
        row = {"season": season, "n_games": n_games}
        for col in LINE_COLUMNS:
            row[col] = round(group[col].notna().mean() * 100, 1)
        row["full_coverage"] = all(row[c] == 100.0 for c in LINE_COLUMNS)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("season")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report odds coverage by season.")
    parser.add_argument("--start", type=int, default=1999)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    summary = coverage_by_season(args.start, args.end)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))

    full = summary[summary["full_coverage"]]
    if len(full):
        print(f"\nFirst season with 100% coverage on all line columns: {full['season'].min()}")
    partial = summary[~summary["full_coverage"]]
    print(f"Seasons with any gap: {partial['season'].tolist()}")
