"""Play-by-play loader.

Per CLAUDE.md storage policy, raw PBP is never persisted to Supabase or to
disk — it's re-pulled fresh from nfl_data_py/nflverse every time the
backtest runs. This module exposes a single function the backtest can call
directly; the CLI entrypoint below is only for smoke-testing that pull.
"""
import argparse

import nfl_data_py as nfl
import pandas as pd


def load_pbp(seasons: list[int], weeks: list[int] | None = None) -> pd.DataFrame:
    """Pull play-by-play for the given seasons, optionally filtered to weeks."""
    df = nfl.import_pbp_data(seasons, downcast=True, cache=False)
    if weeks is not None:
        df = df[df["week"].isin(weeks)]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the PBP pull.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week-start", type=int, default=1)
    parser.add_argument("--week-end", type=int, default=1)
    args = parser.parse_args()

    weeks = list(range(args.week_start, args.week_end + 1))
    pbp = load_pbp([args.season], weeks=weeks)

    print(f"Rows: {len(pbp)}")
    print(f"Columns: {len(pbp.columns)}")
    print(f"Games: {pbp['game_id'].nunique()}")
    print(f"Weeks present: {sorted(pbp['week'].unique().tolist())}")
    print(pbp[["game_id", "week", "posteam", "defteam", "play_type", "epa"]].head(10))
