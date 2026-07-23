"""Layer 3: situational/contextual features.

Rest, travel/timezone, divisional flag, primetime flag, weather, and a
lookahead/letdown feature -- all computed walk-forward safe (every
feature for week W uses only information knowable entering week W).

Stadium locations
-------------------
Built from the actual `stadium_id`s used in nfl_data_py.import_schedules()
2010-2024 (47 distinct venues), not a team->city table, because
stadium_id already correctly encodes neutral-site/international games
and Super Bowl venues -- e.g. a Chiefs "home" Super Bowl game at a
neutral site shows the real venue, not KC. This matters for travel
distance and timezone shift, which need the actual game location.

Weather
---------
meteostat, batched **once per stadium** across the full date range
(not once per game) -- meteostat's newer (2.x) API requires a station
ID, not a raw lat/lon Point, so stations are resolved once via
`meteostat.stations.nearby()` and cached. Dome games (roof column
already in historical_games) skip real weather entirely and get neutral
values, since wind/precip are irrelevant indoors.

Lookahead/letdown
-------------------
Proposed definition (validate via regression coefficient/p-value before
treating as real, per instructions -- see backtest.py): using Layer 1
ratings entering week W for every team involved (never week W+1's actual
rating, which would require knowing week W's outcome):

  letdown_score(T, W)   = opponent_rating(T, W-1) - opponent_rating(T, W)
  lookahead_score(T, W) = opponent_rating(T, W+1, but rated as of
                          entering week W) - opponent_rating(T, W)

The W+1 opponent is known in advance from the schedule (not derived from
week W's outcome), but their *rating* is evaluated at the same
entering-week-W snapshot as everything else -- using their actual
entering-week-(W+1) rating would leak week W's games.
"""
import math

import numpy as np
import pandas as pd
import meteostat

meteostat.config.block_large_requests = False  # our historical pulls deliberately span 15 seasons per stadium
import nfl_data_py as nfl

from pbp import normalize_team_code, retry_network_call

# (lat, lon, IANA timezone) for every stadium_id appearing in
# nfl_data_py.import_schedules() 2010-2024. Public, static data --
# retired stadiums included since historical games were played there.
STADIUM_LOCATIONS = {
    "ATL00": (33.7554, -84.4008, "America/New_York"),
    "ATL97": (33.7554, -84.4008, "America/New_York"),
    "BAL00": (39.2780, -76.6227, "America/New_York"),
    "BOS00": (42.0909, -71.2643, "America/New_York"),
    "BUF00": (42.7738, -78.7870, "America/New_York"),
    "BUF01": (43.6414, -79.3894, "America/Toronto"),
    "CAR00": (35.2258, -80.8528, "America/New_York"),
    "CHI98": (41.8623, -87.6167, "America/Chicago"),
    "CIN00": (39.0955, -84.5161, "America/New_York"),
    "CLE00": (41.5061, -81.6995, "America/New_York"),
    "DAL00": (32.7473, -97.0945, "America/Chicago"),
    "DEN00": (39.7439, -105.0201, "America/Denver"),
    "DET00": (42.3400, -83.0456, "America/Detroit"),
    "FRA00": (50.0686, 8.6455, "Europe/Berlin"),
    "GER00": (48.2188, 11.6247, "Europe/Berlin"),
    "GNB00": (44.5013, -88.0622, "America/Chicago"),
    "HOU00": (29.6847, -95.4107, "America/Chicago"),
    "IND00": (39.7601, -86.1639, "America/Indiana/Indianapolis"),
    "JAX00": (30.3239, -81.6373, "America/New_York"),
    "KAN00": (39.0489, -94.4839, "America/Chicago"),
    "LAX01": (33.9535, -118.3392, "America/Los_Angeles"),
    "LAX97": (33.8644, -118.2611, "America/Los_Angeles"),
    "LAX99": (34.0141, -118.2879, "America/Los_Angeles"),
    "LON00": (51.5560, -0.2795, "Europe/London"),
    "LON01": (51.4560, -0.3410, "Europe/London"),
    "LON02": (51.6043, -0.0662, "Europe/London"),
    "MEX00": (19.3029, -99.1505, "America/Mexico_City"),
    "MIA00": (25.9580, -80.2389, "America/New_York"),
    "MIN00": (44.9738, -93.2581, "America/Chicago"),
    "MIN01": (44.9737, -93.2581, "America/Chicago"),
    "MIN98": (44.9762, -93.2244, "America/Chicago"),
    "NAS00": (36.1665, -86.7713, "America/Chicago"),
    "NOR00": (29.9511, -90.0812, "America/Chicago"),
    "NYC01": (40.8135, -74.0745, "America/New_York"),
    "OAK00": (37.7516, -122.2005, "America/Los_Angeles"),
    "PHI00": (39.9008, -75.1675, "America/New_York"),
    "PHO00": (33.5276, -112.2626, "America/Phoenix"),
    "PIT00": (40.4468, -80.0158, "America/New_York"),
    "SAO00": (-23.5453, -46.4742, "America/Sao_Paulo"),
    "SDG00": (32.7831, -117.1196, "America/Los_Angeles"),
    "SEA00": (47.5952, -122.3316, "America/Los_Angeles"),
    "SFO00": (37.7135, -122.3860, "America/Los_Angeles"),
    "SFO01": (37.4032, -121.9698, "America/Los_Angeles"),
    "STL00": (38.6329, -90.1886, "America/Chicago"),
    "TAM00": (27.9759, -82.5033, "America/New_York"),
    "VEG00": (36.0909, -115.1833, "America/Los_Angeles"),
    "WAS00": (38.9076, -76.8645, "America/New_York"),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def get_team_home_stadium(team: str, season: int, schedules: pd.DataFrame) -> str | None:
    """Most common home stadium_id for this team-season (mode), excluding
    neutral-site games -- handles in-season stadium changes and
    relocations without needing a separate team->stadium table.
    """
    rows = schedules[
        (schedules["season"] == season) & (schedules["home_team"] == team) & (schedules["location"] != "Neutral")
    ]
    if rows.empty:
        return None
    return rows["stadium_id"].mode().iloc[0]


def compute_rest_travel_features(games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """rest_differential, away_travel_km, away_tz_shift_hours per game."""
    games = games.copy()
    games["rest_differential"] = games["home_rest"] - games["away_rest"]
    games["home_off_bye"] = (games["home_rest"] >= 10).astype(int)
    games["away_off_bye"] = (games["away_rest"] >= 10).astype(int)

    travel_km, tz_shift = [], []
    home_stadium_cache: dict[tuple, str | None] = {}

    for _, g in games.iterrows():
        game_stadium = STADIUM_LOCATIONS.get(g["stadium_id"])
        key = (g["away_team"], g["season"])
        if key not in home_stadium_cache:
            home_stadium_cache[key] = get_team_home_stadium(g["away_team"], g["season"], schedules)
        away_home_stadium_id = home_stadium_cache[key]
        away_home = STADIUM_LOCATIONS.get(away_home_stadium_id) if away_home_stadium_id else None

        if game_stadium is None or away_home is None:
            travel_km.append(np.nan)
            tz_shift.append(np.nan)
            continue

        travel_km.append(haversine_km(game_stadium[0], game_stadium[1], away_home[0], away_home[1]))

        game_offset = _utc_offset_hours(game_stadium[2], g["gameday"])
        home_offset = _utc_offset_hours(away_home[2], g["gameday"])
        tz_shift.append(abs(game_offset - home_offset) if game_offset is not None and home_offset is not None else np.nan)

    games["away_travel_km"] = travel_km
    games["away_tz_shift_hours"] = tz_shift
    return games


def _utc_offset_hours(tz_name: str, date) -> float | None:
    import zoneinfo
    from datetime import datetime
    try:
        d = pd.Timestamp(date).to_pydatetime()
        tz = zoneinfo.ZoneInfo(tz_name)
        offset = tz.utcoffset(datetime(d.year, d.month, d.day, 13))
        return offset.total_seconds() / 3600 if offset is not None else None
    except Exception:
        return None


def compute_primetime_flag(games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Primetime = SNF/MNF/TNF slot, from weekday+gametime (not in
    historical_games' curated columns, so pulled fresh from schedules).
    """
    sched_slim = schedules[["season", "week", "home_team", "away_team", "weekday", "gametime"]].copy()
    games = games.drop(columns=[c for c in ("weekday", "gametime") if c in games.columns])
    games = games.merge(sched_slim, on=["season", "week", "home_team", "away_team"], how="left")

    def is_primetime(row) -> int:
        if pd.isna(row["gametime"]):
            return 0
        hour = int(str(row["gametime"]).split(":")[0])
        if row["weekday"] in ("Monday", "Thursday"):
            return 1
        if row["weekday"] == "Sunday" and hour >= 19:
            return 1
        return 0

    games["primetime"] = games.apply(is_primetime, axis=1)
    return games.drop(columns=["weekday", "gametime"])


_STATION_CACHE: dict[str, str | None] = {}
_WEATHER_CACHE: dict[str, pd.DataFrame] = {}


def _nearest_station(lat: float, lon: float) -> str | None:
    key = f"{lat:.3f},{lon:.3f}"
    if key not in _STATION_CACHE:
        pt = meteostat.Point(lat, lon)
        nearby = retry_network_call(lambda: meteostat.stations.nearby(pt))
        _STATION_CACHE[key] = nearby.index[0] if len(nearby) else None
    return _STATION_CACHE[key]


def _stadium_weather_series(stadium_id: str, start, end) -> pd.DataFrame | None:
    if stadium_id in _WEATHER_CACHE:
        return _WEATHER_CACHE[stadium_id]
    lat, lon, _ = STADIUM_LOCATIONS[stadium_id]
    station = _nearest_station(lat, lon)
    if station is None:
        _WEATHER_CACHE[stadium_id] = None
        return None
    df = retry_network_call(lambda: meteostat.hourly(station, start, end).fetch())
    _WEATHER_CACHE[stadium_id] = df
    return df


def compute_weather_features(games: pd.DataFrame) -> pd.DataFrame:
    """wind_mph, temp_f, precip_mm per game. Dome games (roof column)
    get neutral values -- indoor conditions are irrelevant.
    """
    games = games.copy()
    outdoor_mask = ~games["roof"].isin(["dome", "closed"])
    outdoor_stadiums = games.loc[outdoor_mask, "stadium_id"].dropna().unique()

    if len(outdoor_stadiums):
        start = pd.Timestamp(games["gameday"].min()) - pd.Timedelta(days=1)
        end = pd.Timestamp(games["gameday"].max()) + pd.Timedelta(days=1)
        for stadium_id in outdoor_stadiums:
            if stadium_id in STADIUM_LOCATIONS:
                _stadium_weather_series(stadium_id, start, end)
                print(f"  weather loaded for {stadium_id}")

    wind_mph, temp_f, precip_mm = [], [], []
    for _, g in games.iterrows():
        if not outdoor_mask.loc[g.name] or g["stadium_id"] not in STADIUM_LOCATIONS:
            wind_mph.append(0.0)
            temp_f.append(70.0)
            precip_mm.append(0.0)
            continue
        df = _WEATHER_CACHE.get(g["stadium_id"])
        kickoff = pd.Timestamp(g["gameday"])
        if df is None or df.empty:
            wind_mph.append(np.nan)
            temp_f.append(np.nan)
            precip_mm.append(np.nan)
            continue
        nearest_idx = df.index.get_indexer([kickoff], method="nearest")[0]
        row = df.iloc[nearest_idx]
        wind_mph.append(row["wspd"] * 0.621371 if pd.notna(row["wspd"]) else np.nan)
        temp_f.append(row["temp"] * 9 / 5 + 32 if pd.notna(row["temp"]) else np.nan)
        precip_mm.append(row["prcp"] if pd.notna(row["prcp"]) else 0.0)

    games["wind_mph"] = wind_mph
    games["temp_f"] = temp_f
    games["precip_mm"] = precip_mm
    return games


def compute_lookahead_letdown(games: pd.DataFrame, ratings_lookup: dict) -> pd.DataFrame:
    """letdown_score / lookahead_score per team, merged onto each game as
    home_letdown/away_letdown/home_lookahead/away_lookahead. See module
    docstring for the walk-forward-safe definition.
    """
    schedule_by_team_week: dict[tuple, str] = {}
    for _, g in games.iterrows():
        schedule_by_team_week[(g["season"], g["week"], g["home_team"])] = g["away_team"]
        schedule_by_team_week[(g["season"], g["week"], g["away_team"])] = g["home_team"]

    def overall_rating(team: str, season: int, week: int) -> float | None:
        r = ratings_lookup.get((season, week, team))
        return (r["off_rating"] + r["def_rating"]) if r is not None else None

    def scores_for(team: str, season: int, week: int) -> tuple[float, float]:
        cur_opp = schedule_by_team_week.get((season, week, team))
        prev_opp = schedule_by_team_week.get((season, week - 1, team))
        next_opp = schedule_by_team_week.get((season, week + 1, team))

        cur_opp_rating = overall_rating(cur_opp, season, week) if cur_opp else None
        prev_opp_rating = overall_rating(prev_opp, season, week) if prev_opp else None  # prev opp rated as of week W too (their most recent entering-week value we have)
        next_opp_rating = overall_rating(next_opp, season, week) if next_opp else None  # deliberately week W, not W+1 -- see docstring

        letdown = (prev_opp_rating - cur_opp_rating) if prev_opp_rating is not None and cur_opp_rating is not None else np.nan
        lookahead = (next_opp_rating - cur_opp_rating) if next_opp_rating is not None and cur_opp_rating is not None else np.nan
        return letdown, lookahead

    home_letdown, home_lookahead, away_letdown, away_lookahead = [], [], [], []
    for _, g in games.iterrows():
        hl, hla = scores_for(g["home_team"], g["season"], g["week"])
        al, ala = scores_for(g["away_team"], g["season"], g["week"])
        home_letdown.append(hl); home_lookahead.append(hla)
        away_letdown.append(al); away_lookahead.append(ala)

    games = games.copy()
    games["home_letdown"] = home_letdown
    games["home_lookahead"] = home_lookahead
    games["away_letdown"] = away_letdown
    games["away_lookahead"] = away_lookahead
    return games
