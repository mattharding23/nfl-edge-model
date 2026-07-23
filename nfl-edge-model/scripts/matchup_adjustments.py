"""Layer 2: matchup-specific adjustments.

O-line vs D-line deltas and pace interaction effects for totals, fit via
recency-weighted regression over the historical window -- per CLAUDE.md,
this layer does NOT self-solve the way Layer 1's sequential Bayesian
filter does, so recency weighting has to be built explicitly here.

Recency weighting scheme
--------------------------
Exponential decay by season: weight = SEASON_DECAY_RATE ** (seasons_ago).
SEASON_DECAY_RATE=0.75 is a reasonable starting default (each season back
carries 75% of the next season's weight), not a value derived from any
backtest yet -- CLAUDE.md explicitly allows starting with a reasonable
default and tuning after real backtest results are in. Flagging this
explicitly since the exact decay rate is genuinely undetermined right
now, not something to treat as settled.

O-line/D-line proxies
------------------------
No PFF-style grades are available from our data sources, so these are
PBP-derived proxies, not literal O-line/D-line grades:
  - Run blocking: team rushing EPA/play (offense) vs. rushing EPA/play
    allowed (defense).
  - Pass protection: sack rate taken per dropback (offense, lower is
    better) vs. sack rate generated per opponent dropback (defense,
    higher is better).

Each matchup model is: actual_outcome ~ team_metric + opp_metric +
team_metric:opp_metric, fit with recency-weighted least squares. The
interaction term is what's reported as the "matchup delta" -- it's the
incremental effect beyond what you'd get from the two additive terms
alone (which Layer 1's opponent-adjustment already captures at the
whole-game level).

Player-prop spillover from funnel matchups (e.g. "bad run defense forces
more passing volume, boosting the opposing WR1's targets") is explicitly
out of scope -- not built toward here.

Sign convention: standardized to match power_ratings.py -- every
"_rating" column (offense or defense) is higher-is-better for the team/
unit it describes. Raw single-week observations (off_rush_epa,
sack_rate_allowed, etc.) stay in their natural "as observed" units since
they're also used as regression targets (literal outcomes, not ratings);
only the entering-week aggregate features consumed as model inputs
(the "_rating_to_date" columns) are sign-standardized. def_rush_rating
and def_pass_rating are the negation of raw EPA allowed (so higher =
better defense, same direction as power_ratings.py's def_rating).
pass_block_rating is the negation of sack_rate_allowed (higher = better
protection). pass_rush_rating is sack_rate_generated unchanged (already
higher = better defense).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import nfl_data_py as nfl

from pbp import load_pbp, normalize_team_code, retry_network_call

SEASON_DECAY_RATE = 0.75  # tunable; see module docstring


def compute_weekly_split_stats(seasons: list[int], pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rush/pass-split offense & defense stats, plus sack rates, per
    team-week. Layer 1's weekly stats are whole-game; Layer 2 needs the
    rush/pass split to isolate O-line/D-line-specific effects.
    """
    pbp = load_pbp(seasons) if pbp is None else pbp

    rush = pbp[(pbp["rush_attempt"] == 1) & pbp["epa"].notna() & pbp["posteam"].notna()]
    off_rush = rush.groupby(["season", "week", "posteam"]).agg(
        off_rush_epa=("epa", "mean"), off_rush_plays=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team"})
    def_rush = rush.groupby(["season", "week", "defteam"]).agg(
        def_rush_epa_allowed=("epa", "mean"), def_rush_plays=("epa", "size"),
    ).reset_index().rename(columns={"defteam": "team"})

    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["posteam"].notna()]
    pass_plays = dropbacks[dropbacks["epa"].notna()]
    off_pass = pass_plays.groupby(["season", "week", "posteam"]).agg(
        off_pass_epa=("epa", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})
    def_pass = pass_plays.groupby(["season", "week", "defteam"]).agg(
        def_pass_epa_allowed=("epa", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    off_pass_protect = dropbacks.groupby(["season", "week", "posteam"]).agg(
        dropbacks=("sack", "size"), sacks_taken=("sack", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    off_pass_protect["sack_rate_allowed"] = off_pass_protect["sacks_taken"] / off_pass_protect["dropbacks"]

    def_pass_rush = dropbacks.groupby(["season", "week", "defteam"]).agg(
        opp_dropbacks=("sack", "size"), sacks_generated=("sack", "sum"),
    ).reset_index().rename(columns={"defteam": "team"})
    def_pass_rush["sack_rate_generated"] = def_pass_rush["sacks_generated"] / def_pass_rush["opp_dropbacks"]

    pace = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()].groupby(
        ["season", "week", "posteam"]
    ).size().reset_index(name="pace").rename(columns={"posteam": "team"})

    stats = off_rush.merge(def_rush, on=["season", "week", "team"], how="outer")
    stats = stats.merge(off_pass, on=["season", "week", "team"], how="outer")
    stats = stats.merge(def_pass, on=["season", "week", "team"], how="outer")
    stats = stats.merge(off_pass_protect[["season", "week", "team", "sack_rate_allowed"]], on=["season", "week", "team"], how="outer")
    stats = stats.merge(def_pass_rush[["season", "week", "team", "sack_rate_generated"]], on=["season", "week", "team"], how="outer")
    stats = stats.merge(pace, on=["season", "week", "team"], how="outer")
    return stats.sort_values(["season", "week", "team"]).reset_index(drop=True)


def add_entering_week_features(stats: pd.DataFrame) -> pd.DataFrame:
    """Expanding (season-to-date, prior weeks only) mean of each metric,
    shifted so week W's feature only uses weeks 1..W-1 -- walk-forward
    safe by construction, same principle as Layer 1's use of pre-week
    ratings only.

    Sign-standardizes the resulting "_rating_to_date" features so higher
    is always better (see module docstring) -- the raw per-week columns
    themselves are left untouched since they double as regression targets.
    """
    stats = stats.sort_values(["team", "season", "week"]).copy()

    def _to_date(col: str) -> pd.Series:
        return (
            stats.groupby(["team", "season"])[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )

    # (source column, output name, sign) -- sign=-1 flips raw "allowed"/
    # "taken" units into the higher-is-better rating convention.
    rating_specs = [
        ("off_rush_epa", "off_rush_rating_to_date", 1),
        ("def_rush_epa_allowed", "def_rush_rating_to_date", -1),
        ("off_pass_epa", "off_pass_rating_to_date", 1),
        ("def_pass_epa_allowed", "def_pass_rating_to_date", -1),
        ("sack_rate_allowed", "pass_block_rating_to_date", -1),
        ("sack_rate_generated", "pass_rush_rating_to_date", 1),
    ]
    for source_col, out_col, sign in rating_specs:
        stats[out_col] = sign * _to_date(source_col)

    stats["pace_to_date"] = _to_date("pace")  # neutral -- no better/worse direction
    return stats


def build_game_level_dataset(seasons: list[int]) -> pd.DataFrame:
    """One row per team-game with entering-week features (this team's and
    the opponent's) plus the actual realized outcome that game.
    """
    stats = compute_weekly_split_stats(seasons)
    stats = add_entering_week_features(stats)

    games = retry_network_call(nfl.import_schedules, seasons)
    games = games[games["game_type"] != "PRE"].copy()
    # import_schedules() uses the period-accurate team code (OAK/SD/STL);
    # compute_weekly_split_stats (PBP-derived) always uses the franchise's
    # current code -- normalize so old-era relocated-franchise games match.
    games["home_team"] = games["home_team"].map(normalize_team_code)
    games["away_team"] = games["away_team"].map(normalize_team_code)

    rows = []
    for _, g in games.iterrows():
        for team, opp, team_score, opp_score in (
            (g["home_team"], g["away_team"], g["home_score"], g["away_score"]),
            (g["away_team"], g["home_team"], g["away_score"], g["home_score"]),
        ):
            team_row = stats[(stats["season"] == g["season"]) & (stats["week"] == g["week"]) & (stats["team"] == team)]
            opp_row = stats[(stats["season"] == g["season"]) & (stats["week"] == g["week"]) & (stats["team"] == opp)]
            if team_row.empty or opp_row.empty:
                continue
            team_row, opp_row = team_row.iloc[0], opp_row.iloc[0]
            rows.append({
                "season": g["season"], "week": g["week"], "team": team, "opp": opp,
                "actual_rush_epa": team_row["off_rush_epa"],
                "actual_sack_rate": (team_row["sack_rate_allowed"] if pd.notna(team_row["sack_rate_allowed"]) else np.nan),
                "actual_total_points": (g["home_score"] + g["away_score"]) if pd.notna(g["home_score"]) else np.nan,
                "team_off_rush_rating_to_date": team_row["off_rush_rating_to_date"],
                "opp_def_rush_rating_to_date": opp_row["def_rush_rating_to_date"],
                "team_pass_block_rating_to_date": team_row["pass_block_rating_to_date"],
                "opp_pass_rush_rating_to_date": opp_row["pass_rush_rating_to_date"],
                "team_pace_to_date": team_row["pace_to_date"],
                "opp_pace_to_date": opp_row["pace_to_date"],
                "team_off_pass_rating_to_date": team_row["off_pass_rating_to_date"],
                "opp_def_pass_rating_to_date": opp_row["def_pass_rating_to_date"],
            })
    return pd.DataFrame(rows)


def _season_weights(seasons: pd.Series, decay_rate: float = SEASON_DECAY_RATE) -> pd.Series:
    seasons_ago = seasons.max() - seasons
    return decay_rate ** seasons_ago


def fit_rushing_matchup_model(dataset: pd.DataFrame, decay_rate: float = SEASON_DECAY_RATE):
    d = dataset.dropna(subset=["actual_rush_epa", "team_off_rush_rating_to_date", "opp_def_rush_rating_to_date"])
    X = pd.DataFrame({
        "team_off_rush_rating": d["team_off_rush_rating_to_date"],
        "opp_def_rush_rating": d["opp_def_rush_rating_to_date"],
        "interaction": d["team_off_rush_rating_to_date"] * d["opp_def_rush_rating_to_date"],
    })
    X = sm.add_constant(X)
    weights = _season_weights(d["season"], decay_rate)
    return sm.WLS(d["actual_rush_epa"], X, weights=weights).fit()


def fit_pass_protection_model(dataset: pd.DataFrame, decay_rate: float = SEASON_DECAY_RATE):
    d = dataset.dropna(subset=["actual_sack_rate", "team_pass_block_rating_to_date", "opp_pass_rush_rating_to_date"])
    X = pd.DataFrame({
        "team_pass_block_rating": d["team_pass_block_rating_to_date"],
        "opp_pass_rush_rating": d["opp_pass_rush_rating_to_date"],
        "interaction": d["team_pass_block_rating_to_date"] * d["opp_pass_rush_rating_to_date"],
    })
    X = sm.add_constant(X)
    weights = _season_weights(d["season"], decay_rate)
    return sm.WLS(d["actual_sack_rate"], X, weights=weights).fit()


def fit_pace_total_model(dataset: pd.DataFrame, decay_rate: float = SEASON_DECAY_RATE):
    d = dataset.dropna(subset=[
        "actual_total_points", "team_pace_to_date", "opp_pace_to_date",
        "team_off_pass_rating_to_date", "opp_def_pass_rating_to_date",
    ])
    X = pd.DataFrame({
        "team_pace": d["team_pace_to_date"],
        "opp_pace": d["opp_pace_to_date"],
        "pace_interaction": d["team_pace_to_date"] * d["opp_pace_to_date"],
        "team_off_pass_rating": d["team_off_pass_rating_to_date"],
        "opp_def_pass_rating": d["opp_def_pass_rating_to_date"],
    })
    X = sm.add_constant(X)
    weights = _season_weights(d["season"], decay_rate)
    return sm.WLS(d["actual_total_points"], X, weights=weights).fit()


def predict_matchup_deltas(team_row: pd.Series, opp_row: pd.Series, rush_model, protect_model) -> dict:
    """Isolate the interaction term's contribution -- the matchup-specific
    delta beyond the additive baseline -- for one real team-vs-opp pairing.
    """
    rush_interaction_coef = rush_model.params["interaction"]
    rush_delta = rush_interaction_coef * team_row["off_rush_rating_to_date"] * opp_row["def_rush_rating_to_date"]

    protect_interaction_coef = protect_model.params["interaction"]
    protect_delta = protect_interaction_coef * team_row["pass_block_rating_to_date"] * opp_row["pass_rush_rating_to_date"]

    return {
        "rushing_matchup_delta": rush_delta,
        "pass_protection_matchup_delta": protect_delta,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check Layer 2 matchup adjustments.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    args = parser.parse_args()

    print("=== Building game-level dataset ===")
    dataset = build_game_level_dataset(args.seasons)
    print(f"Rows: {len(dataset)}")

    print(f"\n=== Fitting models (season decay rate={SEASON_DECAY_RATE}) ===")
    rush_model = fit_rushing_matchup_model(dataset)
    protect_model = fit_pass_protection_model(dataset)
    total_model = fit_pace_total_model(dataset)

    print("\n-- Rushing matchup model --")
    print(rush_model.params)
    print(f"interaction p-value: {rush_model.pvalues['interaction']:.4f}")

    print("\n-- Pass protection matchup model --")
    print(protect_model.params)
    print(f"interaction p-value: {protect_model.pvalues['interaction']:.4f}")

    print("\n-- Pace/total model --")
    print(total_model.params)
    print(f"pace_interaction p-value: {total_model.pvalues['pace_interaction']:.4f}")

    print("\n=== Sample matchup deltas on real games ===")
    stats = compute_weekly_split_stats(args.seasons)
    stats = add_entering_week_features(stats)

    latest = args.seasons[-1]
    games = retry_network_call(nfl.import_schedules, [latest])
    games = games[(games["game_type"] != "PRE") & (games["week"] == games["week"].max() - 4)]

    for _, g in games.head(5).iterrows():
        home_row = stats[(stats["season"] == g["season"]) & (stats["week"] == g["week"]) & (stats["team"] == g["home_team"])]
        away_row = stats[(stats["season"] == g["season"]) & (stats["week"] == g["week"]) & (stats["team"] == g["away_team"])]
        if home_row.empty or away_row.empty:
            continue
        deltas = predict_matchup_deltas(home_row.iloc[0], away_row.iloc[0], rush_model, protect_model)
        print(f"{g['away_team']} @ {g['home_team']} (season {g['season']} week {g['week']}): {deltas}")
