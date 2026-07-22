# NFL Betting Edge Model — Project Context

## What this is

A quantitative model that generates independent fair-value NFL lines
(spreads, totals, moneylines, and their halves/quarters equivalents),
compares them against live sportsbook prices, and surfaces only
high-confidence, mechanism-backed edges via text/email alerts and a
tracking dashboard.

**Primary validation metric: CLV (closing line value), not win/loss.**

## Architecture (four layers, full game first)

1. **Base power ratings** — Bayesian, weekly updating. EPA/play,
   success rate by down/situation, pace, opponent-adjusted
   (SRS/Sagarin-style), preseason prior blending last year's
   end-of-season ratings + roster turnover + Vegas win total, plus a
   separate QB adjustment layer.
2. **Matchup adjustments** — O-line vs D-line deltas, pace
   interaction effects for totals.
3. **Situational/contextual** — rest, travel, divisional flag,
   primetime, lookahead/letdown spots, weather (wind primary for
   totals), in-week injury updates (Thu practice report + ~90 min
   pre-kickoff pass for inactives).
4. **Market layer** — open vs current vs our fair number, line
   movement/steam detection, vig-removed implied probability.

**Halves/quarters are a distributional decomposition on top of the
full-game model** — not independent models. Full game projection +
team-specific scoring-share priors (shrunk toward league average) →
half/quarter fair lines.

International games, flexed games, and bye-week return spots are
first-class inputs from day one. Player props are explicitly
deferred to a later phase — do not build toward them yet.

## Edge definition

Fair value = no-vig consensus across books. Edge is measured and
alerted against the **best available line** across: Bovada,
DraftKings, FanDuel, Caesars, BetMGM, theScore (all via Odds API —
note Bovada/theScore may need a different `region` param, e.g. `us2`,
verify this early).

Confidence tiers are **backtest-derived per mechanism** (e.g.
"rest+travel spread edges" graded on how that mechanism performed
historically), not ensemble-derived. Every alert needs a "why" tag
(e.g. `rest+travel`, `wind_total`, `injury_adjusted_oline`) and must
clear both a minimum edge threshold AND a confidence floor — no
threshold clearance, no alert.

## Data sources

- **Play-by-play & team stats:** `nfl_data_py` (Python) — free, deep
  history.
- **Historical full-game lines:** `nfl_data_py.import_schedules()` —
  keyed to same game_id as PBP, back to 1999. **Verify odds coverage
  per season before trusting backtest depth — some seasons may have
  partial gaps.**
- **Halves/quarters lines + live multi-book odds + line movement:**
  Odds API. Shorter lookback and patchier book coverage than
  full-game — backtest with appropriately lower confidence.
- **Weather:** NWS/hourly, pulled close to kickoff.
- **Injury reports:** official NFL injury report feed.

## Data storage policy

**Raw historical play-by-play is never stored in Supabase.** It's
re-pulled fresh from `nfl_data_py`/nflverse each time the backtest
runs (occasional pre-season job, not recurring cost). Only summary
results (power ratings, current week's lines/edges, CLV/results
ledger, model version snapshots, alert history) get written to
Postgres. This keeps the dataset small and within the free tier
long-term.

## Hosting / compute pattern

Same no-server pattern as the existing `mlb-hr-model` repo:

- **GitHub Actions** (cron, UTC — needs DST-aware handling, same as
  baseball workflows) for scheduled compute. Schedule: Tue/Wed
  initial line → Thu evening re-run (injury reports) → final pass
  ~90 min pre-kickoff (weather/inactives).
- **GitHub Pages** (`docs/`) for the dashboard — full model board
  (not just alerted picks) plus running CLV/results ledger segmented
  by market type. Auto-publishes on push to `main` touching the
  Pages source.
- **Supabase** (free-tier Postgres) as the data layer.
- **Alerts** via Gmail SMTP (`smtplib`, app password) — text alerts
  route through carrier SMS-to-email gateways (e.g. `@vtext.com`),
  not a dedicated SMS API.

## Backtesting rules

- **Walk-forward only.** Train weeks 1–N, predict N+1, roll forward.
  Never leak future weeks into historical predictions.
- Grade against closing line (CLV) and straight result, segmented by
  market type and confidence tier.
- Kill underperforming market segments/mechanisms before the season
  starts.
- Stress-test against 1–2 historically "weird" seasons (heavy
  injury/upset years).
- **No paper-trading window.** Backtest + CLV validation is the gate
  — live alerts start Week 1.

## In-season change management (build alongside the dashboard, not later)

- Config-driven parameters (thresholds, confidence cutoffs, feature
  weights, mechanism on/off flags) — never hardcoded.
- **Shadow mode** before any change goes live: new logic runs in
  parallel, logs predictions, doesn't fire alerts, until it clears
  whatever sample size *that specific change* needs (variable, not a
  fixed calendar window).
- Versioned model snapshots (config + weights + go-live date) with
  rollback.
- Drift flagging on the CLV ledger — flagged for review, never
  auto-disabled.
- Minimum sample size gates defined upfront per mechanism.
- Change log: date, what changed, rationale/observation that
  prompted it.
- Review cadence: scheduled monthly (or bye-week-aligned) + ad hoc
  whenever drift monitoring trips.

## Secrets

Two-tier, matching where code runs:

- **Production (GitHub Actions):** GitHub repo secrets, referenced
  as `${{ secrets.ODDS_API_KEY }}` etc.
- **Local dev:** Doppler (free tier), already set up and connected to
  GitHub. Run scripts via `doppler run -- <command>` — never call
  scripts directly when testing locally.
- Application code reads secrets identically in both environments via
  `os.environ.get(...)`. See `.env.example` for the full key list.
- No real secrets ever committed. `.env` is gitignored; only
  `.env.example` (names, no values) is tracked.

## Repo structure

```
.github/workflows/   — cron-triggered Actions (data pull, model run, alerts)
docs/                — GitHub Pages dashboard source
data/                — summary/results data (NOT raw PBP — see storage policy above)
scripts/             — pipeline & model code
CLAUDE.md            — this file
requirements.txt
.env.example
.gitignore
```

## Build sequence

1. Data pipeline (PBP + schedules + Odds API → Supabase schema)
2. Layer 1 & 2 (base ratings + matchup adjustments)
3. Walk-forward backtest of core full-game model — validate before
   adding complexity
4. Layer 3 & 4 (situational + market), re-backtest, confirm
   incremental value
5. Halves/quarters decomposition, backtest with scaled confidence
6. Notification system (threshold-gated, backtest-derived confidence)
7. Dashboard + CLV/results tracking by market segment
8. Final pipeline-timing check before Week 1
9. In-season change-management infra — build alongside step 7, not
   mid-season

## Open decisions (confirm during build, don't block on them)

- Exact edge threshold / confidence-tier cutoffs (start conservative)
- Which weather/injury feeds specifically (free vs paid tiers)
- Odds API `region`/`bookmakers` handling for Bovada and theScore
