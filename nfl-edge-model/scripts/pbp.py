"""Play-by-play loader.

Per CLAUDE.md storage policy, raw PBP is never persisted to Supabase or to
disk — it's re-pulled fresh from nfl_data_py/nflverse every time the
backtest runs. This module exposes a single function the backtest can call
directly; the CLI entrypoint below is only for smoke-testing that pull.
"""
import argparse
import time

import nfl_data_py as nfl
import pandas as pd


def retry_network_call(fn, *args, retries: int = 3, delay_seconds: float = 5.0, **kwargs):
    """nfl_data_py pulls parquet files straight from GitHub over plain
    urllib with no retry of its own -- a transient connection reset (seen
    twice in one session) kills the whole call. Wrap network-dependent
    nfl_data_py calls in this rather than re-running scripts by hand after
    every blip.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < retries:
                print(f"  {fn.__name__} failed ({e!r}), retrying ({attempt}/{retries})...")
                time.sleep(delay_seconds)
    raise last_error

# nfl_data_py's PBP retroactively uses each franchise's *current* team code
# for every season (confirmed empirically: Raiders are "LV" in PBP back to
# 2018, Chargers are "LAC" back to 2015, Rams are "LA" back to 2015) --
# but import_schedules() keeps the period-accurate code for the seasons
# before the move (OAK/SD/STL). Anything that looks up PBP-derived state
# (ratings, weekly stats) using a schedule-sourced team code must go
# through normalize_team_code() first, or old-era games for these three
# franchises will silently KeyError or miss their rating entirely.
TEAM_CODE_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def normalize_team_code(code: str) -> str:
    return TEAM_CODE_ALIASES.get(code, code)


def load_pbp(seasons: list[int], weeks: list[int] | None = None) -> pd.DataFrame:
    """Pull play-by-play for the given seasons, optionally filtered to weeks."""
    df = retry_network_call(nfl.import_pbp_data, seasons, downcast=True, cache=False)
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
