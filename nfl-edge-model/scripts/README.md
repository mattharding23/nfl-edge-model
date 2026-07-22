# scripts/

Pipeline and model code lives here. Planned modules (per CLAUDE.md
build sequence):

- `data_pipeline.py` — pull PBP (nfl_data_py), schedules (historical
  lines), Odds API (halves/quarters, live multi-book) → Supabase
- `power_ratings.py` — Layer 1 (base ratings, Bayesian weekly update)
- `matchup_adjustments.py` — Layer 2
- `situational.py` — Layer 3
- `market.py` — Layer 4 (vig removal, steam detection)
- `halves_quarters.py` — distributional decomposition
- `backtest.py` — walk-forward validation harness
- `alerts.py` — threshold-gated notification logic (email/SMS)
- `dashboard_export.py` — writes summary JSON consumed by docs/
