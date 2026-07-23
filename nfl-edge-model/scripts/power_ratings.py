"""Layer 1: base power ratings (Bayesian, weekly updating).

Design note on "Bayesian, weekly updating" + recency decay
------------------------------------------------------------
This is implemented as a Kalman filter, not a static-parameter Bayesian
update. That distinction matters: a standard Bayesian update on a *fixed*
unknown parameter (e.g. "team's true off rating never changes") makes the
posterior variance shrink monotonically and gives week 1 and week 15 equal
evidentiary weight forever -- it does NOT decay old evidence on its own.
Recency decay only shows up if you explicitly model the parameter as
drifting over time (a random walk) and inject process noise between
updates. That process noise (Q, below) is the mechanism that makes recent
games matter more than old ones -- see `describe_recency_decay()` for a
concrete, data-driven demonstration of the resulting effective memory.

Design note on opponent adjustment
-----------------------------------
CLAUDE.md calls for "SRS/Sagarin-style iterative adjustment." Classic
SRS/Sagarin solves a linear system over a fixed set of games simultaneously
-- which means a team's week 3 rating is influenced by their week 15
result. That's non-causal and would leak future information into a
walk-forward backtest. Instead, opponent adjustment here is folded
directly into the Kalman observation: each week's observation is adjusted
using the opponent's rating *as it stood entering that week* (before that
week's game), never using future games. This is the walk-forward-safe
version of the same idea, consistent with "Bayesian, weekly updating"
elsewhere in the same paragraph. Flagging this as a deliberate
interpretation, not an assumption made silently.

Design note on the QB layer
------------------------------
Team off_rating reflects "whatever this team's offense has looked like
recently," which implicitly bakes in whoever has been playing QB. QB
adjustment here is a *delta*: each individual QB has their own
Kalman-updated rating (opponent-pass-defense-adjusted EPA/dropback,
portable across teams), and each team has a rolling baseline of "the QB
rating we've typically gotten." The adjustment for a given week is
(this week's starter's rating - team's own recent baseline) -- zero when
the normal starter plays, negative when a backup steps in, positive when a
previously-injured starter returns. This avoids double-counting QB value
that's already inside team off_rating while still surfacing "the starter
changed" as its own signal.

Preseason prior: last year's end-of-season rating + roster turnover only
--------------------------------------------------------------------------
Vegas season win totals are NOT included yet. Confirmed empirically (and
via Odds API's own docs) that Odds API has no season-long win totals
market for NFL -- team_totals/alternate_team_totals are single-game
markets, and there's no separate outrights/futures sport entry for it the
way there is for Super Bowl winner. A Bradley-Terry-style approximation
(Super Bowl odds -> implied team strength -> expected win total via the
actual schedule) is documented in CLAUDE.md as a follow-up enhancement,
sequenced after Layer 1 is working, not a prerequisite for it.
"""
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from pbp import load_pbp, normalize_team_code, retry_network_call

LEAGUE_AVG = 0.0  # ratings are relative to league average each season

# Kalman filter parameters. OBS_VAR is set empirically from real data (see
# calibrate_noise_params()); PROCESS_VAR is the tunable "how fast do we
# forget" knob, expressed as a fraction of OBS_VAR.
DEFAULT_PROCESS_VAR_FRACTION = 0.08
PRESEASON_EXTRA_VAR = 0.04       # uncertainty injected at every season boundary
FULL_TURNOVER_EXTRA_VAR = 0.03   # additional preseason uncertainty for a fully-turned-over roster


def kalman_update(prior_mean: float, prior_var: float, obs: float, obs_var: float):
    """One Kalman update step. Returns (posterior_mean, posterior_var, gain)."""
    k = prior_var / (prior_var + obs_var)
    post_mean = prior_mean + k * (obs - prior_mean)
    post_var = (1 - k) * prior_var
    return post_mean, post_var, k


def describe_recency_decay(process_var: float, obs_var: float, weeks: int = 20) -> pd.DataFrame:
    """Explicit, checkable demonstration that old evidence decays.

    Simulates a team with a stable true rating, all observations equal to
    that rating except a single one-time shock in week 1, then tracks how
    much of that week-1 shock survives in the posterior mean each
    subsequent week. If the model didn't decay old evidence, week 1's
    influence would never shrink relative to later weeks.
    """
    true_rating = 0.0
    shock = 1.0  # arbitrary unit shock, purely for measuring persistence
    mean, var = true_rating + shock, 1.0  # start "as if" week 1 fully moved the mean
    baseline_mean, baseline_var = true_rating, 1.0  # same process, no shock, for comparison

    rows = []
    for week in range(1, weeks + 1):
        mean, var, k = kalman_update(mean, var + process_var, true_rating, obs_var)
        baseline_mean, baseline_var, _ = kalman_update(baseline_mean, baseline_var + process_var, true_rating, obs_var)
        residual_from_shock = mean - baseline_mean
        rows.append({
            "week": week,
            "gain_k": round(k, 4),
            "residual_from_week1_shock": round(residual_from_shock, 5),
            "pct_of_shock_remaining": round(100 * residual_from_shock / shock, 2),
        })
    return pd.DataFrame(rows)


def calibrate_noise_params(weekly_stats: pd.DataFrame, process_var_fraction: float = DEFAULT_PROCESS_VAR_FRACTION):
    """Derive OBS_VAR from real within-team-season variance, not a guess."""
    residuals = weekly_stats.groupby(["season", "team"])["off_epa"].transform(lambda s: s - s.mean())
    obs_var = float(residuals.var())
    process_var = obs_var * process_var_fraction
    return obs_var, process_var


def compute_weekly_team_stats(seasons: list[int], pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Offense/defense EPA per play, pace, from PBP. One row per team-week."""
    pbp = load_pbp(seasons) if pbp is None else pbp
    plays = pbp.dropna(subset=["epa", "posteam", "defteam"])
    plays = plays[plays["play_type"].isin(["pass", "run"])]

    off = plays.groupby(["season", "week", "posteam"]).agg(
        off_epa=("epa", "mean"),
        off_plays=("epa", "size"),
        off_success_rate=("success", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})

    deff = plays.groupby(["season", "week", "defteam"]).agg(
        def_epa_allowed=("epa", "mean"),
        def_plays=("epa", "size"),
        def_success_rate_allowed=("success", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})

    stats = off.merge(deff, on=["season", "week", "team"], how="outer")
    stats["pace"] = stats["off_plays"]  # offensive plays run that week; a direct pace proxy
    return stats.sort_values(["season", "week", "team"]).reset_index(drop=True)


def compute_success_rate_by_down(seasons: list[int], pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Descriptive/diagnostic only -- not fed into the Kalman state in this
    build. Reserved for Layer 3 situational use (e.g. short-yardage,
    down-specific context) or a future refinement of the core rating.
    """
    pbp = load_pbp(seasons) if pbp is None else pbp
    plays = pbp.dropna(subset=["epa", "posteam", "down", "success"])
    return plays.groupby(["season", "week", "posteam", "down"]).agg(
        success_rate=("success", "mean"),
        plays=("success", "size"),
    ).reset_index().rename(columns={"posteam": "team"})


def compute_qb_starters(seasons: list[int], pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per team-week starting QB, determined empirically from PBP dropback
    volume (whoever had the plurality of that team's dropbacks that week)
    rather than trusting the pre-game schedule listing -- this reflects
    who actually played, including in-game changes.
    """
    pbp = load_pbp(seasons) if pbp is None else pbp
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna()]
    counts = dropbacks.groupby(["season", "week", "posteam", "passer_player_id", "passer_player_name"]).size()
    counts = counts.reset_index(name="dropbacks")
    idx = counts.groupby(["season", "week", "posteam"])["dropbacks"].idxmax()
    starters = counts.loc[idx].reset_index(drop=True)
    return starters.rename(columns={"posteam": "team", "passer_player_id": "qb_id", "passer_player_name": "qb_name"})


def compute_qb_weekly_epa(seasons: list[int], pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each starting QB's own EPA/dropback that week (for the QB rating,
    independent of team-level offensive rating).
    """
    pbp = load_pbp(seasons) if pbp is None else pbp
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna() & pbp["epa"].notna()]
    qb_epa = dropbacks.groupby(["season", "week", "posteam", "passer_player_id"]).agg(
        qb_epa=("epa", "mean"),
        qb_dropbacks=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team", "passer_player_id": "qb_id"})
    return qb_epa


def compute_roster_turnover(season: int) -> pd.DataFrame:
    """Fraction of last season's offensive/defensive snaps returning on
    this year's roster, by team. Used to shrink the preseason prior
    toward league average for high-turnover teams and to widen preseason
    uncertainty accordingly.
    """
    prior_season = season - 1
    if prior_season < 2012:
        # nfl_data_py.import_snap_counts has no data before 2012 -- a real
        # coverage limit, not a transient failure. build_preseason_prior
        # already falls back to returning_pct=0.5 (average turnover
        # assumed) when no row is found, which is the right degraded
        # behavior for these early transitions rather than a crash.
        return pd.DataFrame(columns=["season", "team", "returning_off_pct", "returning_def_pct", "returning_pct"])
    snaps = retry_network_call(nfl.import_snap_counts, [prior_season])
    snaps = snaps.copy()
    snaps["team"] = snaps["team"].map(normalize_team_code)  # snap_counts uses the period-accurate code (e.g. OAK); rosters/PBP use the current one (LV)
    roster = retry_network_call(nfl.import_seasonal_rosters, [season])

    current_ids = set(roster["pfr_id"].dropna())

    team_snaps = snaps.groupby("team").agg(
        total_off_snaps=("offense_snaps", "sum"),
        total_def_snaps=("defense_snaps", "sum"),
    )

    snaps = snaps.copy()
    snaps["returning"] = snaps["pfr_player_id"].isin(current_ids)
    returning_snaps = snaps[snaps["returning"]].groupby("team").agg(
        returning_off_snaps=("offense_snaps", "sum"),
        returning_def_snaps=("defense_snaps", "sum"),
    )

    turnover = team_snaps.join(returning_snaps, how="left").fillna(0)
    turnover["returning_off_pct"] = (turnover["returning_off_snaps"] / turnover["total_off_snaps"]).clip(0, 1)
    turnover["returning_def_pct"] = (turnover["returning_def_snaps"] / turnover["total_def_snaps"]).clip(0, 1)
    turnover["returning_pct"] = (turnover["returning_off_pct"] + turnover["returning_def_pct"]) / 2
    turnover["season"] = season
    return turnover.reset_index()[["season", "team", "returning_off_pct", "returning_def_pct", "returning_pct"]]


def build_preseason_prior(team: str, season: int, final_ratings: dict, turnover: pd.DataFrame) -> tuple[float, float, float, float]:
    """Returns (off_prior_mean, off_prior_var, def_prior_mean, def_prior_var)."""
    row = turnover[turnover["team"] == team]
    returning_pct = float(row["returning_pct"].iloc[0]) if len(row) else 0.5  # unknown -> assume average turnover

    prior_off = final_ratings.get(team, {}).get("off_mean", LEAGUE_AVG)
    prior_def = final_ratings.get(team, {}).get("def_mean", LEAGUE_AVG)
    prior_off_var = final_ratings.get(team, {}).get("off_var")
    prior_def_var = final_ratings.get(team, {}).get("def_var")

    off_mean = returning_pct * prior_off + (1 - returning_pct) * LEAGUE_AVG
    def_mean = returning_pct * prior_def + (1 - returning_pct) * LEAGUE_AVG

    turnover_extra_var = (1 - returning_pct) * FULL_TURNOVER_EXTRA_VAR
    base_var = (prior_off_var if prior_off_var is not None else PRESEASON_EXTRA_VAR)
    off_var = base_var + PRESEASON_EXTRA_VAR + turnover_extra_var
    def_var = (prior_def_var if prior_def_var is not None else PRESEASON_EXTRA_VAR) + PRESEASON_EXTRA_VAR + turnover_extra_var

    return off_mean, off_var, def_mean, def_var


def run_power_ratings(seasons: list[int], process_var_fraction: float = DEFAULT_PROCESS_VAR_FRACTION,
                       pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Main driver. Returns one row per team-week with ratings *entering*
    that week (i.e. walk-forward safe -- these are the values you'd use
    to predict that week's game, not values informed by it).
    """
    pbp = load_pbp(seasons) if pbp is None else pbp  # loaded once and reused -- these three all used to redownload independently
    weekly_stats = compute_weekly_team_stats(seasons, pbp=pbp)
    obs_var, process_var = calibrate_noise_params(weekly_stats, process_var_fraction)

    starters = compute_qb_starters(seasons, pbp=pbp)
    qb_epa = compute_qb_weekly_epa(seasons, pbp=pbp)

    teams = sorted(weekly_stats["team"].dropna().unique())
    off_state = {t: {"mean": LEAGUE_AVG, "var": PRESEASON_EXTRA_VAR * 3} for t in teams}
    def_state = {t: {"mean": LEAGUE_AVG, "var": PRESEASON_EXTRA_VAR * 3} for t in teams}
    qb_state: dict[str, dict] = {}          # per QB id, portable across teams
    team_qb_baseline = {t: {"mean": LEAGUE_AVG, "var": PRESEASON_EXTRA_VAR * 3} for t in teams}
    final_ratings_by_season: dict[int, dict] = {}

    rows = []
    for season in seasons:
        if season - 1 in final_ratings_by_season:
            turnover = compute_roster_turnover(season)
            prev_final = final_ratings_by_season[season - 1]
            for t in teams:
                off_mean, off_var, def_mean, def_var = build_preseason_prior(t, season, prev_final, turnover)
                off_state[t] = {"mean": off_mean, "var": off_var}
                def_state[t] = {"mean": def_mean, "var": def_var}
                team_qb_baseline[t]["var"] += PRESEASON_EXTRA_VAR

        season_weeks = sorted(weekly_stats.loc[weekly_stats["season"] == season, "week"].unique())
        season_pbp_games = retry_network_call(nfl.import_schedules, [season])
        season_pbp_games = season_pbp_games[season_pbp_games["game_type"] != "PRE"].copy()
        season_pbp_games["home_team"] = season_pbp_games["home_team"].map(normalize_team_code)
        season_pbp_games["away_team"] = season_pbp_games["away_team"].map(normalize_team_code)

        for week in season_weeks:
            week_games = season_pbp_games[season_pbp_games["week"] == week]
            pre_week_off = {t: dict(off_state[t]) for t in teams}
            pre_week_def = {t: dict(def_state[t]) for t in teams}

            week_stats = weekly_stats[(weekly_stats["season"] == season) & (weekly_stats["week"] == week)]

            for _, game_row in week_games.iterrows():
                home, away = game_row["home_team"], game_row["away_team"]
                for team, opp in ((home, away), (away, home)):
                    if team not in off_state or opp not in off_state:
                        continue
                    team_row = week_stats[week_stats["team"] == team]
                    if team_row.empty:
                        continue
                    team_row = team_row.iloc[0]

                    opp_def_mean = pre_week_def[opp]["mean"]
                    opp_off_mean = pre_week_off[opp]["mean"]

                    if pd.notna(team_row["off_epa"]):
                        adj_off_obs = team_row["off_epa"] + opp_def_mean
                        off_state[team]["mean"], off_state[team]["var"], _ = kalman_update(
                            off_state[team]["mean"], off_state[team]["var"] + process_var, adj_off_obs, obs_var,
                        )
                    if pd.notna(team_row["def_epa_allowed"]):
                        adj_def_obs = opp_off_mean - team_row["def_epa_allowed"]
                        def_state[team]["mean"], def_state[team]["var"], _ = kalman_update(
                            def_state[team]["mean"], def_state[team]["var"] + process_var, adj_def_obs, obs_var,
                        )

            for t in teams:
                if t not in [row for pair in week_games[["home_team", "away_team"]].values for row in pair]:
                    off_state[t]["var"] += process_var
                    def_state[t]["var"] += process_var

            for _, game_row in week_games.iterrows():
                home, away = game_row["home_team"], game_row["away_team"]
                for team in (home, away):
                    if team not in off_state:
                        continue
                    pace_row = week_stats[week_stats["team"] == team]
                    pace = float(pace_row["pace"].iloc[0]) if not pace_row.empty else np.nan

                    starter_row = starters[(starters["season"] == season) & (starters["week"] == week) & (starters["team"] == team)]
                    qb_id = starter_row["qb_id"].iloc[0] if not starter_row.empty else None
                    qb_name = starter_row["qb_name"].iloc[0] if not starter_row.empty else None

                    qb_adjustment = 0.0
                    if qb_id is not None:
                        if qb_id not in qb_state:
                            qb_state[qb_id] = {"mean": LEAGUE_AVG, "var": PRESEASON_EXTRA_VAR * 3}
                        qb_adjustment = qb_state[qb_id]["mean"] - team_qb_baseline[team]["mean"]

                    rows.append({
                        "season": season, "week": week, "team": team,
                        "off_rating": pre_week_off[team]["mean"], "off_var": pre_week_off[team]["var"],
                        "def_rating": pre_week_def[team]["mean"], "def_var": pre_week_def[team]["var"],
                        "pace": pace, "qb_id": qb_id, "qb_name": qb_name, "qb_adjustment": qb_adjustment,
                    })

                    qb_epa_row = qb_epa[(qb_epa["season"] == season) & (qb_epa["week"] == week) & (qb_epa["team"] == team) & (qb_epa["qb_id"] == qb_id)]
                    if qb_id is not None and not qb_epa_row.empty:
                        opp = away if team == home else home
                        opp_def_mean = pre_week_def[opp]["mean"]
                        adj_qb_obs = float(qb_epa_row["qb_epa"].iloc[0]) + opp_def_mean
                        qb_state[qb_id]["mean"], qb_state[qb_id]["var"], _ = kalman_update(
                            qb_state[qb_id]["mean"], qb_state[qb_id]["var"] + process_var, adj_qb_obs, obs_var,
                        )
                        team_qb_baseline[team]["mean"], team_qb_baseline[team]["var"], _ = kalman_update(
                            team_qb_baseline[team]["mean"], team_qb_baseline[team]["var"] + process_var,
                            qb_state[qb_id]["mean"], obs_var,
                        )

        final_ratings_by_season[season] = {
            t: {"off_mean": off_state[t]["mean"], "off_var": off_state[t]["var"],
                "def_mean": def_state[t]["mean"], "def_var": def_state[t]["var"]}
            for t in teams
        }

    return pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check Layer 1 power ratings.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--teams", type=str, nargs="+", default=["KC", "SF", "NE", "CAR"])
    args = parser.parse_args()

    print("=== Recency decay demonstration (synthetic, before touching real data) ===")
    demo_obs_var = 0.05
    demo_process_var = demo_obs_var * DEFAULT_PROCESS_VAR_FRACTION
    decay_df = describe_recency_decay(demo_process_var, demo_obs_var, weeks=12)
    print(decay_df.to_string(index=False))

    print("\n=== Running Layer 1 on real data ===")
    ratings = run_power_ratings(args.seasons)
    print(f"Rows: {len(ratings)}")

    obs_var, process_var = calibrate_noise_params(compute_weekly_team_stats(args.seasons))
    print(f"\nCalibrated from real data: obs_var={obs_var:.4f}, process_var={process_var:.4f} "
          f"({DEFAULT_PROCESS_VAR_FRACTION:.0%} of obs_var)")

    print(f"\n=== Sample: {args.teams} across {args.seasons} ===")
    sample = ratings[ratings["team"].isin(args.teams)]
    for team in args.teams:
        print(f"\n-- {team} --")
        t = sample[sample["team"] == team][["season", "week", "off_rating", "def_rating", "pace", "qb_name", "qb_adjustment"]]
        print(t.to_string(index=False))
