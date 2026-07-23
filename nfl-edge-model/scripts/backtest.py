"""Step 3: walk-forward backtest of the core full-game model (Layers 1+2
only -- no Layer 3/4, no notification system, no shadow-mode infra).

Season boundary handling
---------------------------
Layer 1's Kalman state carries forward across seasons via the
preseason-prior blend already built into power_ratings.py (last season's
final rating + roster turnover) -- it does not reset to a flat prior each
year. Running run_power_ratings() across a continuous multi-season range
gets this automatically; nothing extra is needed here.

Calibration constants (walk-forward, strict)
-----------------------------------------------
Combining Layer 1 ratings + Layer 2 matchup deltas into an actual
predicted margin/total/win-probability requires a few constants: a
home-field-advantage/EPA-to-points scale for the margin, and a residual
std for converting margin into win probability. Layer 2's own matchup
models (rush/pass-protect/pace-total) are also fitted parameters.

None of these can be fit once on the whole backtest window without
leaking future seasons into early predictions. They're refit **once per
graded season, using only strictly prior seasons** (same season-decay
weighting as Layer 2's SEASON_DECAY_RATE) -- not per-week, since HFA/scale
are stable within a season and per-week refitting would be expensive for
no real walk-forward benefit, but still fully safe at the season level:
season S's calibration never sees season S or later.

Spread/ML sign convention
----------------------------
Confirmed empirically (not assumed): historical_games.spread_line is
denominated as the market's expected HOME margin (positive = home
favored) -- regression of actual_margin on spread_line gives slope=1.025,
and both means are ~1.9 (home field advantage). predicted_margin uses the
identical convention, so they're directly comparable.

What "CLV" means in this backtest
-------------------------------------
True CLV needs a bet-time price compared to a later closing price. We
deliberately did not backfill historical odds (Step 5, not yet done -- see
CLAUDE.md), so there's no bet-time price here, only the single closing
line already in historical_games. What's computed instead is "model fair
line vs. closing line": when our number disagrees with the closing line,
does the actual outcome land closer to OUR number or the market's? That's
the best available proxy for CLV validation right now, not literal CLV --
flagged as such throughout, not silently relabeled.
"""
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from scipy.stats import norm

from db import get_pg_connection
from pbp import load_pbp, normalize_team_code, retry_network_call
from power_ratings import run_power_ratings, compute_qb_starters
from matchup_adjustments import (
    compute_weekly_split_stats, add_entering_week_features,
    fit_rushing_matchup_model, fit_pass_protection_model, fit_pace_total_model,
    predict_matchup_deltas, SEASON_DECAY_RATE,
)
import situational
import player_value

STATE_START = 2010          # Layer 1/2 computed continuously from here (warm-up + calibration)
GRADED_START = 2013         # backtest results only reported from here (3 seasons of prior data to calibrate from)
GRADED_END = 2024

# Layer 3 feature columns fed into the incremental margin/total regressions.
# Kept as module-level lists so fit and predict always agree on order/names.
MARGIN_L3_FEATURES = [
    "rest_differential", "bye_diff", "away_travel_km", "away_tz_shift_hours",
    "div_game", "primetime", "letdown_diff", "lookahead_diff",
    "qb_injury_diff", "skill_injury_diff", "ol_injury_diff", "def_injury_diff",
]
TOTAL_L3_FEATURES = ["wind_mph", "temp_f", "precip_mm"]


def load_historical_games(start: int, end: int) -> pd.DataFrame:
    conn = get_pg_connection()
    df = pd.read_sql(
        """
        select season, week, game_id, home_team, away_team, home_score, away_score,
               spread_line, total_line, home_moneyline, away_moneyline,
               spread_total_backtest_safe, moneyline_backtest_safe,
               home_rest, away_rest, div_game, roof, gameday
        from historical_games
        where season between %s and %s and home_score is not null
        """,
        conn, params=(start, end),
    )
    conn.close()
    df["gameday"] = pd.to_datetime(df["gameday"])  # raw date objects from psycopg2 break fastparquet's type inference
    for col in ["spread_line", "total_line", "home_moneyline", "away_moneyline"]:
        df[col] = df[col].astype(float)
    # historical_games was populated from import_schedules(), which uses the
    # period-accurate team code (OAK/SD/STL); ratings/layer2 lookups are
    # keyed by PBP's code, which is always the franchise's current one.
    df["home_team"] = df["home_team"].map(normalize_team_code)
    df["away_team"] = df["away_team"].map(normalize_team_code)
    return df


def compute_layer3_features(games: pd.DataFrame, state_seasons: list[int], ratings_lookup: dict,
                             starters: pd.DataFrame, qb_rating_history: pd.DataFrame,
                             skill_ratings: pd.DataFrame, positions: dict) -> pd.DataFrame:
    """Adds all Layer 3 situational + injury feature columns to games, in
    the differenced ('positive = favors home') form the margin/total
    regressions expect. Computed once for the whole state range.
    """
    print("  Layer 3: rest/travel/timezone...")
    schedules = retry_network_call(nfl.import_schedules, state_seasons)
    schedules = schedules[schedules["game_type"] != "PRE"].copy()
    schedules["home_team"] = schedules["home_team"].map(normalize_team_code)
    schedules["away_team"] = schedules["away_team"].map(normalize_team_code)

    games = games.merge(
        schedules[["season", "week", "home_team", "away_team", "stadium_id"]],
        on=["season", "week", "home_team", "away_team"], how="left",
    )
    games = situational.compute_rest_travel_features(games, schedules)
    games["bye_diff"] = games["home_off_bye"] - games["away_off_bye"]

    print("  Layer 3: primetime flag...")
    games = situational.compute_primetime_flag(games, schedules)

    print("  Layer 3: weather (meteostat, batched per stadium)...")
    games = situational.compute_weather_features(games)

    print("  Layer 3: lookahead/letdown...")
    games = situational.compute_lookahead_letdown(games, ratings_lookup)
    games["letdown_diff"] = games["home_letdown"] - games["away_letdown"]
    games["lookahead_diff"] = games["home_lookahead"] - games["away_lookahead"]

    print("  Layer 3: injury adjustments (QB/skill/OL-coarse/DEF-coarse)...")
    injury_adj = player_value.compute_all_injury_adjustments(
        state_seasons, starters, qb_rating_history, skill_ratings, positions
    )
    home_inj = injury_adj.rename(columns={c: f"home_{c}" for c in injury_adj.columns if c not in ("season", "week", "team")})
    home_inj = home_inj.rename(columns={"team": "home_team"})
    away_inj = injury_adj.rename(columns={c: f"away_{c}" for c in injury_adj.columns if c not in ("season", "week", "team")})
    away_inj = away_inj.rename(columns={"team": "away_team"})

    games = games.merge(home_inj, on=["season", "week", "home_team"], how="left")
    games = games.merge(away_inj, on=["season", "week", "away_team"], how="left")
    for col in ("qb_injury_specific", "skill_injury_specific"):
        games[f"{col.split('_injury_')[0]}_injury_diff"] = games[f"home_{col}"].fillna(0) - games[f"away_{col}"].fillna(0)
    for col in ("ol_injury_coarse", "def_injury_coarse"):
        prefix = col.split("_injury_")[0]
        games[f"{prefix}_injury_diff"] = games[f"away_{col}"].fillna(0) - games[f"home_{col}"].fillna(0)  # more missing on AWAY favors home

    return games


def moneyline_to_prob(ml: float) -> float:
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def build_ratings_lookup(ratings: pd.DataFrame) -> dict:
    return {(r["season"], r["week"], r["team"]): r for _, r in ratings.iterrows()}


def raw_efficiency_signal(home: str, away: str, season: int, week: int,
                           ratings_lookup: dict, layer2_lookup: dict,
                           rush_model, protect_model) -> float | None:
    """(home expected epa/play - away expected epa/play) * expected plays.
    Returns None if either team's entering-week data isn't available
    (e.g. a team's first tracked week, before any prior-week rating exists).
    """
    home_rating = ratings_lookup.get((season, week, home))
    away_rating = ratings_lookup.get((season, week, away))
    if home_rating is None or away_rating is None:
        return None

    home_l2 = layer2_lookup.get((season, week, home))
    away_l2 = layer2_lookup.get((season, week, away))
    if home_l2 is None or away_l2 is None:
        return None
    if pd.isna(home_l2.get("off_rush_rating_to_date")) or pd.isna(away_l2.get("off_rush_rating_to_date")):
        return None  # first week of a team's tracked window -- no prior-week features yet

    home_deltas = predict_matchup_deltas(home_l2, away_l2, rush_model, protect_model)
    away_deltas = predict_matchup_deltas(away_l2, home_l2, rush_model, protect_model)

    home_epa = (home_rating["off_rating"] - away_rating["def_rating"]
                + home_deltas["rushing_matchup_delta"] + home_deltas["pass_protection_matchup_delta"])
    away_epa = (away_rating["off_rating"] - home_rating["def_rating"]
                + away_deltas["rushing_matchup_delta"] + away_deltas["pass_protection_matchup_delta"])

    expected_plays = np.nanmean([home_l2["pace_to_date"], away_l2["pace_to_date"]])
    if pd.isna(expected_plays):
        return None

    return (home_epa - away_epa) * expected_plays


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """SVD-based (via lstsq), not a direct normal-equations solve: the
    Layer 3 regression has 12+ features fit on modest early-season
    samples, where sparse/collinear features (bye_diff, primetime, small
    injury counts) can make X^T W X singular. lstsq degrades gracefully
    (least-norm solution) instead of raising.
    """
    sqrt_w = np.sqrt(weights)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def fit_season_calibration(train_games: pd.DataFrame, ratings_lookup: dict, layer2_lookup: dict,
                            decay_rate: float = SEASON_DECAY_RATE, use_layer3: bool = False):
    """Fit Layer 2's three matchup models, the margin-scale/HFA, residual
    std, and -- if use_layer3 -- an incremental regression of
    (actual - Layer1+2 prediction) on the Layer 3 situational/injury
    features. Layer 3 is deliberately incremental on top of the untouched
    Layer 1+2 prediction, not remixed into one combined fit -- this keeps
    each layer's contribution separately inspectable (coefficients,
    significance) rather than blending them into one opaque model.
    """
    l2_dataset = _build_l2_dataset(train_games, layer2_lookup)
    rush_model = fit_rushing_matchup_model(l2_dataset, decay_rate)
    protect_model = fit_pass_protection_model(l2_dataset, decay_rate)
    total_model = fit_pace_total_model(l2_dataset, decay_rate)

    rows = []
    for _, g in train_games.iterrows():
        signal = raw_efficiency_signal(g["home_team"], g["away_team"], g["season"], g["week"],
                                        ratings_lookup, layer2_lookup, rush_model, protect_model)
        if signal is None:
            continue
        row = {"season": g["season"], "signal": signal, "actual_margin": g["home_score"] - g["away_score"]}
        if use_layer3:
            row["week"] = g["week"]
            row["home_team"] = g["home_team"]
            row["away_team"] = g["away_team"]
            row["actual_total"] = g["home_score"] + g["away_score"]
            for col in MARGIN_L3_FEATURES + TOTAL_L3_FEATURES:
                row[col] = g.get(col)
        rows.append(row)
    margin_df = pd.DataFrame(rows)

    if len(margin_df) < 30:
        return None  # not enough training data yet

    weights = decay_rate ** (margin_df["season"].max() - margin_df["season"])
    X = np.column_stack([np.ones(len(margin_df)), margin_df["signal"]])
    beta = _weighted_lstsq(X, margin_df["actual_margin"].values, weights.values)
    hfa, scale = beta[0], beta[1]

    layer12_predicted_margin = hfa + scale * margin_df["signal"]
    residuals = margin_df["actual_margin"] - layer12_predicted_margin
    residual_std = float(np.sqrt(np.average(residuals ** 2, weights=weights)))

    calibration = {
        "rush_model": rush_model, "protect_model": protect_model, "total_model": total_model,
        "hfa": hfa, "scale": scale, "residual_std": residual_std,
        "n_train_games": len(margin_df),
        "layer3_margin_model": None, "layer3_total_model": None,
    }

    if use_layer3:
        margin_l3 = margin_df.dropna(subset=MARGIN_L3_FEATURES)
        if len(margin_l3) >= 30:
            l3_weights = decay_rate ** (margin_l3["season"].max() - margin_l3["season"])
            resid_target = margin_l3["actual_margin"] - (hfa + scale * margin_l3["signal"])
            Xl3 = np.column_stack([np.ones(len(margin_l3))] + [margin_l3[c].values for c in MARGIN_L3_FEATURES])
            beta_l3 = _weighted_lstsq(Xl3, resid_target.values, l3_weights.values)
            calibration["layer3_margin_model"] = {"coefs": beta_l3, "features": MARGIN_L3_FEATURES}
            calibration["layer3_margin_std"] = float(np.sqrt(np.average(
                (resid_target - Xl3 @ beta_l3) ** 2, weights=l3_weights,
            )))
            calibration["n_train_games_layer3"] = len(margin_l3)

        total_l3 = margin_df.dropna(subset=TOTAL_L3_FEATURES + ["actual_total"])
        if len(total_l3) >= 30:
            base_total_pred = _predict_total_base(total_l3, layer2_lookup, total_model)
            valid = pd.notna(base_total_pred)
            total_l3v = total_l3[valid]
            base_total_pred = base_total_pred[valid]
            if len(total_l3v) >= 30:
                l3_weights_t = decay_rate ** (total_l3v["season"].max() - total_l3v["season"])
                resid_target_t = total_l3v["actual_total"].values - base_total_pred
                Xl3t = np.column_stack([np.ones(len(total_l3v))] + [total_l3v[c].values for c in TOTAL_L3_FEATURES])
                beta_l3t = _weighted_lstsq(Xl3t, resid_target_t, l3_weights_t.values)
                calibration["layer3_total_model"] = {"coefs": beta_l3t, "features": TOTAL_L3_FEATURES}
                calibration["n_train_games_layer3_total"] = len(total_l3v)

    return calibration


def _predict_total_base(games: pd.DataFrame, layer2_lookup: dict, total_model) -> np.ndarray:
    preds = []
    for _, g in games.iterrows():
        home_l2 = layer2_lookup.get((g["season"], g["week"], g["home_team"]))
        away_l2 = layer2_lookup.get((g["season"], g["week"], g["away_team"]))
        if home_l2 is None or away_l2 is None or pd.isna(home_l2.get("pace_to_date")) or pd.isna(away_l2.get("pace_to_date")):
            preds.append(np.nan)
            continue
        row = pd.DataFrame([{
            "const": 1.0,
            "team_pace": home_l2["pace_to_date"], "opp_pace": away_l2["pace_to_date"],
            "pace_interaction": home_l2["pace_to_date"] * away_l2["pace_to_date"],
            "team_off_pass_rating": home_l2["off_pass_rating_to_date"], "opp_def_pass_rating": away_l2["def_pass_rating_to_date"],
        }])
        row = row[total_model.model.exog_names]
        preds.append(float(total_model.predict(row).iloc[0]))
    return np.array(preds)


def _build_l2_dataset(games: pd.DataFrame, layer2_lookup: dict) -> pd.DataFrame:
    rows = []
    for _, g in games.iterrows():
        for team, opp in ((g["home_team"], g["away_team"]), (g["away_team"], g["home_team"])):
            team_row = layer2_lookup.get((g["season"], g["week"], team))
            opp_row = layer2_lookup.get((g["season"], g["week"], opp))
            if team_row is None or opp_row is None:
                continue
            rows.append({
                "season": g["season"], "week": g["week"], "team": team, "opp": opp,
                "actual_rush_epa": team_row["off_rush_epa"],
                "actual_sack_rate": team_row["sack_rate_allowed"] if pd.notna(team_row["sack_rate_allowed"]) else np.nan,
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


def predict_game(g: pd.Series, ratings_lookup: dict, layer2_lookup: dict, calibration: dict) -> dict | None:
    signal = raw_efficiency_signal(g["home_team"], g["away_team"], g["season"], g["week"],
                                    ratings_lookup, layer2_lookup, calibration["rush_model"], calibration["protect_model"])
    if signal is None:
        return None
    predicted_margin_l12 = calibration["hfa"] + calibration["scale"] * signal

    home_l2 = layer2_lookup.get((g["season"], g["week"], g["home_team"]))
    away_l2 = layer2_lookup.get((g["season"], g["week"], g["away_team"]))
    total_row = pd.DataFrame([{
        "const": 1.0,
        "team_pace": home_l2["pace_to_date"], "opp_pace": away_l2["pace_to_date"],
        "pace_interaction": home_l2["pace_to_date"] * away_l2["pace_to_date"],
        "team_off_pass_rating": home_l2["off_pass_rating_to_date"], "opp_def_pass_rating": away_l2["def_pass_rating_to_date"],
    }])
    total_row = total_row[calibration["total_model"].model.exog_names]  # guard against add_constant column-order differences across statsmodels versions
    predicted_total_l12 = float(calibration["total_model"].predict(total_row).iloc[0])

    predicted_margin = predicted_margin_l12
    predicted_total = predicted_total_l12
    residual_std = calibration["residual_std"]

    if calibration.get("layer3_margin_model") is not None:
        model = calibration["layer3_margin_model"]
        vals = [g.get(f) for f in model["features"]]
        if all(pd.notna(v) for v in vals):
            x = np.array([1.0] + [float(v) for v in vals])
            predicted_margin = predicted_margin_l12 + float(x @ model["coefs"])
            residual_std = calibration.get("layer3_margin_std", residual_std)

    if calibration.get("layer3_total_model") is not None:
        model = calibration["layer3_total_model"]
        vals = [g.get(f) for f in model["features"]]
        if all(pd.notna(v) for v in vals):
            x = np.array([1.0] + [float(v) for v in vals])
            predicted_total = predicted_total_l12 + float(x @ model["coefs"])

    predicted_home_win_prob = float(norm.cdf(predicted_margin / residual_std))

    return {
        "predicted_margin": predicted_margin,
        "predicted_total": predicted_total,
        "predicted_home_win_prob": predicted_home_win_prob,
        "predicted_margin_l12_only": predicted_margin_l12,
    }


def run_backtest(state_start: int = STATE_START, graded_start: int = GRADED_START, graded_end: int = GRADED_END,
                  use_layer3: bool = False) -> pd.DataFrame:
    state_seasons = list(range(state_start, graded_end + 1))
    print(f"Loading PBP for {state_seasons} (once, shared across all layers)...")
    pbp = load_pbp(state_seasons)

    print("Computing Layer 1 ratings...")
    ratings, qb_rating_history = run_power_ratings(state_seasons, pbp=pbp)

    print("Computing Layer 2 split stats...")
    layer2_stats = compute_weekly_split_stats(state_seasons, pbp=pbp)
    layer2_stats = add_entering_week_features(layer2_stats)
    layer2_lookup = {(r["season"], r["week"], r["team"]): r for _, r in layer2_stats.iterrows()}
    ratings_lookup = build_ratings_lookup(ratings)

    games = load_historical_games(state_start, graded_end)

    if use_layer3:
        print("Computing player-value ratings (RB/WR/TE)...")
        positions = player_value.position_lookup(state_seasons)
        skill_ratings = player_value.run_skill_player_ratings(pbp, positions)
        starters = compute_qb_starters(state_seasons, pbp=pbp)
        games = compute_layer3_features(games, state_seasons, ratings_lookup, starters, qb_rating_history, skill_ratings, positions)

    results = []
    for season in range(graded_start, graded_end + 1):
        train_games = games[games["season"] < season]
        calibration = fit_season_calibration(train_games, ratings_lookup, layer2_lookup, use_layer3=use_layer3)
        if calibration is None:
            print(f"  Skipping {season}: insufficient training data.")
            continue
        l3_note = ""
        if use_layer3:
            n_m = calibration.get("n_train_games_layer3", 0)
            n_t = calibration.get("n_train_games_layer3_total", 0)
            l3_note = f", layer3 margin n={n_m}, layer3 total n={n_t}"
        print(f"  Season {season}: calibrated on {calibration['n_train_games']} prior games "
              f"(hfa={calibration['hfa']:.3f}, scale={calibration['scale']:.4f}, residual_std={calibration['residual_std']:.2f}{l3_note})")

        season_games = games[games["season"] == season]
        for _, g in season_games.iterrows():
            pred = predict_game(g, ratings_lookup, layer2_lookup, calibration)
            if pred is None:
                continue
            results.append({**g.to_dict(), **pred})

    return pd.DataFrame(results)


def _win_rate_test(win_rate: float, n: int, breakeven: float = 0.5238) -> dict | None:
    """Two-sided z-test of a win rate against 50% and against the -110
    breakeven price. Standard error uses 0.5 (not the observed rate) since
    we're testing the null of no edge, which is the conventional choice.
    """
    if n == 0 or pd.isna(win_rate):
        return None
    se = np.sqrt(0.5 * 0.5 / n)
    z_half = (win_rate - 0.5) / se
    z_breakeven = (win_rate - breakeven) / se
    return {
        "se": se,
        "z_vs_50pct": z_half, "p_vs_50pct": 2 * (1 - norm.cdf(abs(z_half))),
        "z_vs_breakeven": z_breakeven, "p_vs_breakeven": 2 * (1 - norm.cdf(abs(z_breakeven))),
    }


def _print_significance(win_rate: float, n: int, indent: str = "") -> None:
    test = _win_rate_test(win_rate, n)
    if test is None:
        return
    print(f"{indent}  significance: z={test['z_vs_50pct']:.2f} vs 50% (p={test['p_vs_50pct']:.3f}), "
          f"z={test['z_vs_breakeven']:.2f} vs 52.4% breakeven (p={test['p_vs_breakeven']:.3f})")


def ats_slice_analysis(results: pd.DataFrame, threshold: float = 1.0, verbose: bool = True,
                        margin_col: str = "predicted_margin") -> dict:
    """Break the pooled ATS result down by pick side, favorite/underdog,
    and disagreement magnitude -- distinguishes a specific fixable bias
    from genuinely uniform noise. Returns {label: (n, win_rate)} so
    Layer 1+2 vs Layer 1+2+3 can be compared directly (see
    compare_bias_narrowing) using margin_col to select which prediction.
    """
    d = results.dropna(subset=[margin_col, "spread_line", "home_score", "away_score"]).copy()
    actual_margin = d["home_score"] - d["away_score"]
    our_edge = d[margin_col] - d["spread_line"]
    d = d[our_edge.abs() > threshold].copy()
    edge = our_edge.loc[d.index]

    d["bet_home"] = edge > 0
    d["covered_home"] = actual_margin.loc[d.index] > d["spread_line"]
    d["win"] = np.where(d["bet_home"], d["covered_home"], ~d["covered_home"])
    d["edge_abs"] = edge.abs()
    d["picked_favorite"] = np.where(d["bet_home"], d["spread_line"] > 0, d["spread_line"] < 0)

    if verbose:
        print(f"\n=== ATS slice analysis (pooled, |disagreement|>{threshold}pt, n={len(d)}) ===")

    out = {}

    def report(mask: pd.Series, label: str) -> None:
        sub = d[mask]
        n = len(sub)
        wr = sub["win"].mean() if n else float("nan")
        out[label] = (n, wr)
        if verbose:
            print(f"  {label}: n={n}, win rate={wr:.1%}")
            _print_significance(wr, n)

    report(d["bet_home"], "Picked home")
    report(~d["bet_home"], "Picked away")
    report(d["picked_favorite"], "Picked favorite")
    report(~d["picked_favorite"], "Picked underdog")
    for lo, hi, label in [(threshold, 2, f"{threshold}-2pt"), (2, 4, "2-4pt"), (4, np.inf, "4pt+")]:
        mask = (d["edge_abs"] >= lo) & (d["edge_abs"] < hi)
        report(mask, f"Edge magnitude {label}")

    return out


def compare_bias_narrowing(results: pd.DataFrame, threshold: float = 1.0) -> None:
    """Directly answers the Step 4 question: does the favorite-side /
    large-disagreement bias found in Step 3 narrow once Layer 3 is added?
    Same slice methodology, same threshold, side by side -- using a
    single Layer 1+2+3 run's predicted_margin_l12_only (the untouched
    Layer 1+2 baseline computed inside that same run) vs predicted_margin
    (with Layer 3 applied), rather than re-running the backtest twice.
    """
    print("\n=== Bias-narrowing check: Layer 1+2 vs Layer 1+2+3 (same slices as Step 3) ===")
    before = ats_slice_analysis(results, threshold, verbose=False, margin_col="predicted_margin_l12_only")
    after = ats_slice_analysis(results, threshold, verbose=False, margin_col="predicted_margin")
    for label in before:
        n0, wr0 = before[label]
        n1, wr1 = after.get(label, (0, float("nan")))
        z0 = _win_rate_test(wr0, n0)
        z1 = _win_rate_test(wr1, n1)
        z0s = f"{z0['z_vs_50pct']:.2f}" if z0 else "n/a"
        z1s = f"{z1['z_vs_50pct']:.2f}" if z1 else "n/a"
        print(f"  {label}: L1+2 n={n0} wr={wr0:.1%} (z={z0s})  ->  L1+2+3 n={n1} wr={wr1:.1%} (z={z1s})")


def moneyline_calibration(results: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve: bins predicted home win probability and compares
    actual win rate (and the market's own implied probability) per bin --
    separates 'missing information' from 'our probability conversion is
    itself miscalibrated.'
    """
    d = results[results["moneyline_backtest_safe"]].dropna(
        subset=["predicted_home_win_prob", "home_moneyline", "away_moneyline", "home_score", "away_score"]
    ).copy()
    d["home_won"] = (d["home_score"] > d["away_score"]).astype(int)
    market_home_raw = d["home_moneyline"].apply(moneyline_to_prob)
    market_away_raw = d["away_moneyline"].apply(moneyline_to_prob)
    d["market_home_prob"] = [devig_two_way(h, a)[0] for h, a in zip(market_home_raw, market_away_raw)]

    d["bin"] = pd.qcut(d["predicted_home_win_prob"], n_bins, duplicates="drop")
    rows = []
    for b, group in d.groupby("bin", observed=True):
        rows.append({
            "bin": str(b), "n": len(group),
            "mean_predicted_prob": group["predicted_home_win_prob"].mean(),
            "actual_win_rate": group["home_won"].mean(),
            "mean_market_prob": group["market_home_prob"].mean(),
        })
    table = pd.DataFrame(rows)
    print(f"\n=== Moneyline calibration ({n_bins}-bin reliability curve, n={len(d)}) ===")
    print(table.to_string(index=False))
    return table


def grade(results: pd.DataFrame) -> None:
    print(f"\n=== Overall: {len(results)} graded games, seasons {results['season'].min()}-{results['season'].max()} ===\n")

    print("--- SPREAD ---")
    _grade_spread(results)
    print("\n--- TOTAL ---")
    _grade_total(results)
    print("\n--- MONEYLINE ---")
    _grade_moneyline(results[results["moneyline_backtest_safe"]])

    print("\n\n=== By season ===")
    for season, season_df in results.groupby("season"):
        print(f"\n-- {season} ({len(season_df)} games) --")
        _grade_spread(season_df, indent="  ")
        _grade_total(season_df, indent="  ")
        ml_df = season_df[season_df["moneyline_backtest_safe"]]
        if len(ml_df):
            _grade_moneyline(ml_df, indent="  ")
        else:
            print("  moneyline: not backtest-safe this season")


def _grade_spread(df: pd.DataFrame, indent: str = "") -> None:
    d = df.dropna(subset=["predicted_margin", "spread_line", "home_score", "away_score"])
    actual_margin = d["home_score"] - d["away_score"]
    mae_vs_actual = (d["predicted_margin"] - actual_margin).abs().mean()
    mae_vs_close = (d["predicted_margin"] - d["spread_line"]).abs().mean()

    our_edge = d["predicted_margin"] - d["spread_line"]
    market_miss = actual_margin - d["spread_line"]
    edge_corr = our_edge.corr(market_miss)

    threshold = 1.0
    bets = d[our_edge.abs() > threshold].copy()
    bets["bet_home"] = our_edge[our_edge.abs() > threshold] > 0
    bets["covered_home"] = actual_margin[bets.index] > d.loc[bets.index, "spread_line"]
    bets["win"] = np.where(bets["bet_home"], bets["covered_home"], ~bets["covered_home"])
    ats_win_rate = bets["win"].mean() if len(bets) else float("nan")

    print(f"{indent}MAE vs actual margin: {mae_vs_actual:.2f} pts | MAE vs closing line: {mae_vs_close:.2f} pts")
    print(f"{indent}corr(our disagreement w/ close, market's actual miss): {edge_corr:.3f}  [signal test -- see docstring]")
    print(f"{indent}ATS when |disagreement|>{threshold}pt: {len(bets)} bets, win rate {ats_win_rate:.1%} (breakeven ~52.4% at -110)")
    _print_significance(ats_win_rate, len(bets), indent)


def _grade_total(df: pd.DataFrame, indent: str = "") -> None:
    d = df.dropna(subset=["predicted_total", "total_line", "home_score", "away_score"])
    actual_total = d["home_score"] + d["away_score"]
    mae_vs_actual = (d["predicted_total"] - actual_total).abs().mean()
    mae_vs_close = (d["predicted_total"] - d["total_line"]).abs().mean()

    our_edge = d["predicted_total"] - d["total_line"]
    market_miss = actual_total - d["total_line"]
    edge_corr = our_edge.corr(market_miss)

    threshold = 1.0
    bets = d[our_edge.abs() > threshold].copy()
    bets["bet_over"] = our_edge[our_edge.abs() > threshold] > 0
    bets["went_over"] = actual_total[bets.index] > d.loc[bets.index, "total_line"]
    bets["win"] = np.where(bets["bet_over"], bets["went_over"], ~bets["went_over"])
    ou_win_rate = bets["win"].mean() if len(bets) else float("nan")

    print(f"{indent}MAE vs actual total: {mae_vs_actual:.2f} pts | MAE vs closing line: {mae_vs_close:.2f} pts")
    print(f"{indent}corr(our disagreement w/ close, market's actual miss): {edge_corr:.3f}  [signal test]")
    print(f"{indent}O/U when |disagreement|>{threshold}pt: {len(bets)} bets, win rate {ou_win_rate:.1%} (breakeven ~52.4% at -110)")
    _print_significance(ou_win_rate, len(bets), indent)


def _grade_moneyline(df: pd.DataFrame, indent: str = "") -> None:
    d = df.dropna(subset=["predicted_home_win_prob", "home_moneyline", "away_moneyline", "home_score", "away_score"])
    if len(d) == 0:
        print(f"{indent}no moneyline-safe games in this slice")
        return
    home_won = (d["home_score"] > d["away_score"]).astype(int)

    market_home_raw = d["home_moneyline"].apply(moneyline_to_prob)
    market_away_raw = d["away_moneyline"].apply(moneyline_to_prob)
    market_home_prob = [devig_two_way(h, a)[0] for h, a in zip(market_home_raw, market_away_raw)]
    market_home_prob = pd.Series(market_home_prob, index=d.index)

    our_brier = ((d["predicted_home_win_prob"] - home_won) ** 2).mean()
    market_brier = ((market_home_prob - home_won) ** 2).mean()

    eps = 1e-6
    p = d["predicted_home_win_prob"].clip(eps, 1 - eps)
    our_logloss = -(home_won * np.log(p) + (1 - home_won) * np.log(1 - p)).mean()
    pm = market_home_prob.clip(eps, 1 - eps)
    market_logloss = -(home_won * np.log(pm) + (1 - home_won) * np.log(1 - pm)).mean()

    print(f"{indent}n={len(d)} | Brier: ours={our_brier:.4f} vs market={market_brier:.4f} (lower is better)")
    print(f"{indent}Log-loss: ours={our_logloss:.4f} vs market={market_logloss:.4f} (lower is better)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the walk-forward backtest.")
    parser.add_argument("--layer3", action="store_true", help="Include Layer 3 situational/injury features.")
    parser.add_argument("--cache-path", type=str, default=None,
                        help="Save results to this parquet path after running (for reuse without recomputing).")
    parser.add_argument("--from-cache", type=str, default=None,
                        help="Skip run_backtest() and grade a previously-cached results file instead.")
    args = parser.parse_args()

    if args.from_cache:
        results = pd.read_parquet(args.from_cache)
    else:
        results = run_backtest(use_layer3=args.layer3)
        if args.cache_path:
            try:
                results.to_parquet(args.cache_path)
                print(f"Cached results to {args.cache_path}")
            except Exception as e:
                # Don't let a caching hiccup discard a full backtest run's results --
                # fall back to CSV, and if even that fails, just proceed to grading.
                print(f"Parquet cache failed ({e!r}), trying CSV fallback...")
                try:
                    results.to_csv(args.cache_path + ".csv", index=False)
                    print(f"Cached results to {args.cache_path}.csv instead")
                except Exception as e2:
                    print(f"CSV fallback also failed ({e2!r}) -- proceeding without caching.")

    grade(results)
    ats_slice_analysis(results)
    moneyline_calibration(results)

    if args.layer3 and "predicted_margin_l12_only" in results.columns:
        compare_bias_narrowing(results)
