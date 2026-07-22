-- Two tables added alongside the original five, per decisions made during
-- Step 1 (data pipeline) build-out. See scripts/README.md for rationale.

-- Historical full-game schedules + lines, sourced from
-- nfl_data_py.import_schedules(). Small (~7-8k rows for 1999-2025) and
-- reused on every backtest run, so unlike raw PBP it's worth persisting
-- rather than re-pulling each time.
--
-- spread_total_backtest_safe / moneyline_backtest_safe encode the coverage
-- gap found in scripts/schedule_coverage.py: spread_line/total_line are
-- 100% populated for every season 1999-2025, but moneyline and vig-side
-- odds (spread/total juice) are 0% before 2006, noisy 2006-2009, and 100%
-- from 2010 on (one exception: a single 2017 game). These flags are set at
-- population time from season-level coverage, so backtest code checks a
-- column instead of re-deriving the coverage cutoff from memory.
create table if not exists historical_games (
    id bigint generated always as identity primary key,
    game_id text unique not null,
    season int not null,
    week int not null,
    game_type text not null,
    gameday date,
    home_team text not null,
    away_team text not null,
    home_score int,
    away_score int,
    spread_line numeric,
    total_line numeric,
    away_moneyline int,
    home_moneyline int,
    away_spread_odds int,
    home_spread_odds int,
    under_odds int,
    over_odds int,
    div_game boolean,
    roof text,
    surface text,
    temp numeric,
    wind numeric,
    away_rest int,
    home_rest int,
    spread_total_backtest_safe boolean not null default true,
    moneyline_backtest_safe boolean not null default false,
    created_at timestamptz not null default now()
);

create index if not exists idx_historical_games_season_week
    on historical_games (season, week);

-- Raw multi-book Odds API snapshots. lines_edges stays as the model's
-- computed/alerted output (fair_line vs best available line); this table
-- is the underlying per-book, per-pull-time archive that makes line
-- movement/steam detection (Layer 4) and book-specific CLV grading
-- possible later. Not deduplicated on write — every pull is a new row,
-- since the point is the time series.
create table if not exists odds_snapshots (
    id bigint generated always as identity primary key,
    odds_api_event_id text not null,
    game_id text,
    home_team text not null,
    away_team text not null,
    commence_time timestamptz,
    book text not null,
    market_type text not null,
    side text,
    line numeric,
    price int,
    pulled_at timestamptz not null default now()
);

create index if not exists idx_odds_snapshots_event
    on odds_snapshots (odds_api_event_id, book, market_type, pulled_at);
