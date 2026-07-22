# scripts/

Pipeline and model code lives here.

## Step 1 — data pipeline (done)

- `db.py` — shared Supabase connections: `get_pg_connection()` (session
  pooler, for DDL — direct/non-pooler connections are IPv6-only on this
  project tier and unreachable from this network) and
  `get_supabase_client()` (PostgREST, for normal CRUD).
- `pbp.py` — source 1. `load_pbp(seasons, weeks)` pulls play-by-play via
  nfl_data_py. Never persisted (see CLAUDE.md storage policy) — the
  backtest calls this directly each run.
- `schedule_coverage.py` — reports odds-column fill rate per season from
  `nfl_data_py.import_schedules()`. Read-only, no writes.
- `load_historical_games.py` — source 2. Builds/upserts `historical_games`
  rows, tagging each with `spread_total_backtest_safe` /
  `moneyline_backtest_safe` from the coverage check above.
- `odds_api.py` — source 3 client. `BOOKS` = bovada, draftkings, fanduel,
  williamhill_us (Caesars), betmgm, espnbet (Barstool → ESPN BET →
  theScore Bet lineage, same Penn Entertainment operator — see module
  docstring). Live full-game odds batch across all games in one call;
  halves/quarters require the per-event endpoint and only populate near
  kickoff. Historical endpoints work but cost ~10 credits/market/call and
  only go back to ~Aug 2020 — deliberately not backfilled yet (Step 5,
  not Step 1; see CLAUDE.md build sequence).
- `pull_odds_snapshot.py` — pulls one live snapshot into `odds_snapshots`
  (append-only time series). Run regularly so a free historical archive
  accumulates before the Step 5 halves/quarters backfill decision.
- `run_pipeline.py` — unified entrypoint: refreshes `historical_games`
  for the target season, pulls a live odds snapshot, and (optionally,
  via `--pbp-season`) smoke-tests the PBP loader. This is what GitHub
  Actions cron calls.
- `schema_new_tables.sql` — DDL for `historical_games` and
  `odds_snapshots`, added alongside the original 5 tables during Step 1.

## Planned (later build steps)

- `power_ratings.py` — Layer 1 (base ratings, Bayesian weekly update)
- `matchup_adjustments.py` — Layer 2
- `situational.py` — Layer 3
- `market.py` — Layer 4 (vig removal, steam detection)
- `halves_quarters.py` — distributional decomposition; historical odds
  backfill decision revisited here, not before
- `backtest.py` — walk-forward validation harness
- `alerts.py` — threshold-gated notification logic (email/SMS)
- `dashboard_export.py` — writes summary JSON consumed by docs/
