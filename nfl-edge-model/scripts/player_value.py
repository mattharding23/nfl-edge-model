"""Injury-adjusted player-value sub-layer (feeds Layer 3).

Four distinct mechanisms, deliberately tagged separately so each can be
independently evaluated/killed in the backtest without touching the
others:

  - QB-injury-specific: precise -- reuses Layer 1's per-QB Kalman rating
    (power_ratings.py) and depth-chart data to identify the actual
    likely replacement.
  - skill-injury-specific: precise -- new RB/WR/TE Kalman ratings built
    here, weighted by usage share.
  - OL-injury-coarse: blunt "starters missing" headcount. No PBP-level
    attribution is possible for individual offensive linemen with our
    data sources (blocking isn't charted play-by-play).
  - DEF-injury-coarse: blunt "starters missing" headcount for DL/LB/DB.
    This is a genuine data limitation, not laziness: sacks/INTs only
    credit a fraction of defensive snaps, and a partial-attribution
    rating would systematically undervalue shutdown coverage players
    who rarely get targeted *because* they're good -- worse than no
    rating, not just noisier.

All lookups here use only the pre-game injury report (report_status is
a single final per-player-week designation, confirmed to reflect the
official pre-kickoff status, not something that could leak in-game
information) and trailing-week data (never the target week's own
outcome) -- walk-forward safe throughout.
"""
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from pbp import retry_network_call
from power_ratings import kalman_update, LEAGUE_AVG, PRESEASON_EXTRA_VAR

SKILL_PROCESS_VAR_FRACTION = 0.08
TRAILING_WEEKS = 3
OL_POSITIONS = {"T", "G", "C", "OL", "OT", "OG"}
DEF_POSITIONS = {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "CB", "S", "SS", "FS", "DB"}
QUESTIONABLE_DISCOUNT = 0.5  # ~ empirical play-through rate for "Questionable" -- a simplifying assumption, not fit from data


def position_lookup(seasons: list[int]) -> dict:
    roster = retry_network_call(nfl.import_seasonal_rosters, seasons)
    return dict(zip(roster["player_id"], roster["position"]))


def compute_skill_weekly_stats(pbp: pd.DataFrame, positions: dict) -> pd.DataFrame:
    """RB rushing EPA/carry and WR/TE receiving EPA/target, per
    player-week, with usage share (this player's touches / team's
    position-group touches that week).
    """
    rush = pbp[(pbp["rush_attempt"] == 1) & pbp["epa"].notna() & pbp["rusher_player_id"].notna()].copy()
    rush["position"] = rush["rusher_player_id"].map(positions)
    rush = rush[rush["position"].isin(["RB", "FB"])]
    rb_stats = rush.groupby(["season", "week", "posteam", "rusher_player_id", "rusher_player_name"]).agg(
        epa_per_touch=("epa", "mean"), touches=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team", "rusher_player_id": "player_id", "rusher_player_name": "player_name"})
    rb_stats["position"] = "RB"

    targets = pbp[(pbp["pass_attempt"] == 1) & pbp["epa"].notna() & pbp["receiver_player_id"].notna()].copy()
    targets["position"] = targets["receiver_player_id"].map(positions)
    targets = targets[targets["position"].isin(["WR", "TE"])]
    rec_stats = targets.groupby(["season", "week", "posteam", "receiver_player_id", "receiver_player_name", "position"]).agg(
        epa_per_touch=("epa", "mean"), touches=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team", "receiver_player_id": "player_id", "receiver_player_name": "player_name"})

    stats = pd.concat([rb_stats, rec_stats], ignore_index=True)
    team_group_touches = stats.groupby(["season", "week", "team", "position"])["touches"].transform("sum")
    stats["usage_share"] = stats["touches"] / team_group_touches
    return stats


def run_skill_player_ratings(pbp: pd.DataFrame, positions: dict) -> pd.DataFrame:
    """Kalman-filtered EPA/touch rating per player, portable across teams
    -- same recency-decay mechanism as Layer 1's QB rating, for RB/WR/TE.
    Returns ratings *entering* each week (walk-forward safe).
    """
    stats = compute_skill_weekly_stats(pbp, positions)
    residuals = stats.groupby(["season", "player_id"])["epa_per_touch"].transform(lambda s: s - s.mean())
    obs_var = float(residuals.var())
    process_var = obs_var * SKILL_PROCESS_VAR_FRACTION

    player_state: dict[str, dict] = {}
    rows = []
    for (season, week), week_df in stats.sort_values(["season", "week"]).groupby(["season", "week"]):
        for _, row in week_df.iterrows():
            pid = row["player_id"]
            if pid not in player_state:
                player_state[pid] = {"mean": LEAGUE_AVG, "var": PRESEASON_EXTRA_VAR * 3}
            rows.append({
                "season": season, "week": week, "player_id": pid, "player_name": row["player_name"],
                "team": row["team"], "position": row["position"], "usage_share": row["usage_share"],
                "rating_entering": player_state[pid]["mean"],
            })
            player_state[pid]["mean"], player_state[pid]["var"], _ = kalman_update(
                player_state[pid]["mean"], player_state[pid]["var"] + process_var, row["epa_per_touch"], obs_var,
            )
    return pd.DataFrame(rows)


def _normalize_name(name: str) -> str:
    for suffix in (" Jr.", " Sr.", " III", " II", " IV"):
        name = name.replace(suffix, "")
    return name.strip().upper()


def build_injury_lookup(seasons: list[int]) -> dict:
    inj = retry_network_call(nfl.import_injuries, seasons)
    return {
        (row.season, row.week, row.team, row.gsis_id): row.report_status
        for row in inj.itertuples()
    }


def build_injury_lookup_by_name(seasons: list[int]) -> dict:
    """(season, week, team, normalized_name) -> report_status. Used only
    for the OL/DEF coarse mechanisms, whose upstream snap-count data has
    no usable gsis_id crosswalk (see build_snap_starter_lookup) -- name
    matching is consistent with those mechanisms already being the
    blunter, coarser proxies.
    """
    inj = retry_network_call(nfl.import_injuries, seasons)
    return {
        (row.season, row.week, row.team, _normalize_name(row.full_name)): row.report_status
        for row in inj.itertuples()
    }


def build_depth_chart_lookup(seasons: list[int]) -> dict:
    """(season, week, team) -> {depth_rank: gsis_id} for QBs only."""
    dc = retry_network_call(nfl.import_depth_charts, seasons)
    qb = dc[(dc["position"] == "QB")].dropna(subset=["week", "depth_team", "gsis_id"])
    lookup: dict = {}
    for row in qb.itertuples():
        key = (row.season, int(row.week), row.club_code)
        lookup.setdefault(key, {})[int(row.depth_team)] = row.gsis_id
    return lookup


def build_snap_starter_lookup(seasons: list[int], positions: dict, top_n_ol: int = 5, top_n_def: int = 7) -> dict:
    """(season, week, team, 'OL'/'DEF') -> set of presumed-starter
    normalized names, determined from trailing-week average snap share
    (walk-forward safe).

    Keyed by normalized name, not gsis_id: snap_counts is keyed by
    pfr_player_id, and cross-referencing that to gsis_id (via either
    import_ids() or import_seasonal_rosters()'s pfr_id column) turned out
    to have near-zero coverage specifically for O-line players -- an
    upstream data gap, confirmed empirically (checked both crosswalks),
    not a join bug. Name matching against injuries' full_name sidesteps
    it and is consistent with these already being the coarser mechanisms.
    """
    snap_seasons = [s for s in seasons if s >= 2012]  # import_snap_counts has no data before 2012 (confirmed empirically)
    if not snap_seasons:
        return {}
    snaps = retry_network_call(nfl.import_snap_counts, snap_seasons)
    snaps = snaps.copy()
    snaps["position_group"] = snaps["position"].apply(
        lambda p: "OL" if p in OL_POSITIONS else ("DEF" if p in DEF_POSITIONS else None)
    )
    snaps = snaps.dropna(subset=["position_group"])
    snaps["snaps"] = snaps["offense_snaps"].fillna(0) + snaps["defense_snaps"].fillna(0)
    snaps["norm_name"] = snaps["player"].apply(_normalize_name)

    lookup: dict = {}
    for (season, team, group), grp in snaps.groupby(["season", "team", "position_group"]):
        grp = grp.sort_values("week")
        top_n = top_n_ol if group == "OL" else top_n_def
        for week in sorted(grp["week"].unique()):
            trailing = grp[(grp["week"] < week) & (grp["week"] >= week - TRAILING_WEEKS)]
            if trailing.empty:
                continue
            starters = set(trailing.groupby("norm_name")["snaps"].mean().sort_values(ascending=False).head(top_n).index)
            lookup[(season, int(week), team, group)] = starters
    return lookup


def build_presumed_qb_lookup(starters: pd.DataFrame) -> dict:
    """(season, week, team) -> presumed starting qb_id, from trailing-week
    dropback plurality (never the target week's own dropbacks).
    """
    lookup: dict = {}
    for (season, team), grp in starters.sort_values("week").groupby(["season", "team"]):
        for week in sorted(grp["week"].unique()):
            trailing = grp[(grp["week"] < week) & (grp["week"] >= week - TRAILING_WEEKS)]
            if trailing.empty:
                continue
            counts = trailing.groupby("qb_id").size()
            lookup[(season, int(week), team)] = counts.idxmax()
    return lookup


def compute_all_injury_adjustments(
    seasons: list[int], starters: pd.DataFrame, qb_rating_history: pd.DataFrame,
    skill_ratings: pd.DataFrame, positions: dict,
) -> pd.DataFrame:
    """One row per (season, week, team) with all four adjustment values.
    Built once for the whole graded range; backtest.py looks this up by
    dict, not by re-filtering DataFrames per game.
    """
    from power_ratings import qb_rating_entering, team_qb_baseline_entering

    injury_lookup = build_injury_lookup(seasons)
    injury_lookup_by_name = build_injury_lookup_by_name(seasons)
    depth_lookup = build_depth_chart_lookup(seasons)
    presumed_qb_lookup = build_presumed_qb_lookup(starters)
    snap_starter_lookup = build_snap_starter_lookup(seasons, positions)

    skill_lookup: dict = {}
    for (season, week, team, position), grp in skill_ratings.groupby(["season", "week", "team", "position"]):
        skill_lookup.setdefault((season, week, team), []).append((position, grp))
    skill_trailing: dict = {}
    for (season, team, position), grp in skill_ratings.sort_values("week").groupby(["season", "team", "position"]):
        for week in sorted(grp["week"].unique()):
            trailing = grp[(grp["week"] < week) & (grp["week"] >= week - TRAILING_WEEKS)]
            if trailing.empty:
                continue
            latest = trailing.sort_values("week").groupby("player_id").last()
            skill_trailing[(season, int(week), team, position)] = latest

    team_weeks = sorted({(s, int(w), t) for (s, w, t) in presumed_qb_lookup.keys()}
                         | {(s, int(w), t) for (s, w, t, _) in snap_starter_lookup.keys()})

    rows = []
    for season, week, team in team_weeks:
        qb_adj = 0.0
        presumed_qb = presumed_qb_lookup.get((season, week, team))
        if presumed_qb is not None:
            status = injury_lookup.get((season, week, team, presumed_qb))
            if status in ("Out", "Doubtful"):
                depth = depth_lookup.get((season, week, team), {})
                backup_id = depth.get(2)
                baseline = team_qb_baseline_entering(team, season, week, qb_rating_history)
                backup_rating = qb_rating_entering(backup_id, season, week, qb_rating_history) if backup_id else LEAGUE_AVG
                qb_adj = backup_rating - baseline

        skill_adj = 0.0
        for position in ("RB", "WR", "TE"):
            latest = skill_trailing.get((season, week, team, position))
            if latest is None:
                continue
            for player_id, prow in latest.iterrows():
                status = injury_lookup.get((season, week, team, player_id))
                if status in ("Out", "Doubtful"):
                    skill_adj -= prow["usage_share"] * prow["rating_entering"]
                elif status == "Questionable":
                    skill_adj -= QUESTIONABLE_DISCOUNT * prow["usage_share"] * prow["rating_entering"]

        ol_missing = 0
        for norm_name in snap_starter_lookup.get((season, week, team, "OL"), set()):
            status = injury_lookup_by_name.get((season, week, team, norm_name))
            if status in ("Out", "Doubtful"):
                ol_missing += 1

        def_missing = 0
        for norm_name in snap_starter_lookup.get((season, week, team, "DEF"), set()):
            status = injury_lookup_by_name.get((season, week, team, norm_name))
            if status in ("Out", "Doubtful"):
                def_missing += 1

        rows.append({
            "season": season, "week": week, "team": team,
            "qb_injury_specific": qb_adj, "skill_injury_specific": skill_adj,
            "ol_injury_coarse": ol_missing, "def_injury_coarse": def_missing,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    from pbp import load_pbp
    from power_ratings import run_power_ratings

    parser = argparse.ArgumentParser(description="Sanity-check the injury-adjusted player-value sub-layer.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    args = parser.parse_args()

    pbp = load_pbp(args.seasons)
    positions = position_lookup(args.seasons)

    print("=== Skill player ratings sample ===")
    skill_ratings = run_skill_player_ratings(pbp, positions)
    print(f"Rows: {len(skill_ratings)}")
    print(skill_ratings.sort_values("rating_entering", ascending=False).head(10).to_string(index=False))

    print("\n=== Layer 1 ratings + QB history (for injury lookups) ===")
    ratings, qb_rating_history = run_power_ratings(args.seasons, pbp=pbp)
    from power_ratings import compute_qb_starters
    starters = compute_qb_starters(args.seasons, pbp=pbp)

    print("\n=== All injury adjustments sample ===")
    adjustments = compute_all_injury_adjustments(args.seasons, starters, qb_rating_history, skill_ratings, positions)
    print(f"Rows: {len(adjustments)}")
    nonzero = adjustments[(adjustments["qb_injury_specific"] != 0) | (adjustments["skill_injury_specific"] != 0)
                           | (adjustments["ol_injury_coarse"] > 0) | (adjustments["def_injury_coarse"] > 0)]
    print(f"Non-zero rows: {len(nonzero)}")
    print(nonzero.head(15).to_string(index=False))
