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

## Recency weighting (design requirement for Layers 1–4, not yet built)

Separate from walk-forward validation (which handles chronology/leakage)
— this is about weighting *within* the training window, given rule
changes, roster/franchise shifts, and a drifting league-wide scoring
environment over a 16–26 year lookback.

**In model coefficients:**
- Layer 1 (power ratings) self-solves this via sequential Bayesian
  updating — no extra work needed, but when built, confirm the
  implementation actually decays old evidence rather than treating all
  historical weeks as equally informative forever.
- Layers 2–4 (matchup deltas, situational/weather/rest coefficients) are
  likely fit via regression over the historical window, so they need an
  explicit recency weighting scheme (e.g. exponential decay by season)
  rather than equal weighting across all years.
- Structural changes (team relocations, dome vs. outdoor) belong as
  explicit context features (e.g. venue/roof type as an input), not
  something recency weighting alone fixes — decaying old data doesn't
  undo a model that's implicitly learned stale team-level assumptions.
- Decay rate must be tunable **per mechanism**, not one global setting.
  Thin-sample mechanisms (international games, rest+travel, unusual
  weather spots) risk falling below minimum sample size gates if decay
  is too aggressive — revisit explicitly when the situational layer is
  designed.

**In confidence-tier calibration:**
- Don't average CLV performance equally across the full backtest window
  — a mechanism strong in 2010–2018 but decayed since (books get sharper
  over time) should get a lower tier reflecting current reality, not the
  inflated historical average.
- Segment each mechanism's backtest CLV by era (e.g. thirds or halves of
  the window) and check the trend — stable, improving, decaying — rather
  than computing one all-time average.
- Use a **gentler/longer decay for tier calibration than for model
  coefficients**. Coefficients tolerate more aggressive decay because
  Bayesian updating smooths outliers; tiers are more sensitive to
  small-sample noise and risk the same "3-game cold streak" problem the
  minimum sample size gates already guard against.
- Conceptually the same tool as the in-season drift monitoring below,
  just applied at a multi-year backtest timescale instead of a
  weekly/monthly live timescale — design these as one coherent
  decay/drift framework rather than two unrelated mechanisms if a clean
  way to unify them surfaces during that build.

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

## Vegas win total for the preseason prior (follow-up, not built yet)

Odds API has no season-long win totals market for NFL — confirmed
empirically (no separate outrights/futures sport entry the way there is
for Super Bowl winner) and via their own docs (`team_totals` /
`alternate_team_totals` are single-game markets only). The preseason
prior currently blends only last year's end-of-season rating + roster
turnover (see Layer 1 in `scripts/power_ratings.py`).

Planned approximation, once Layer 1 has real backtest results to compare
against — **Bradley-Terry, not a full season/playoff simulator**:

1. Pull each team's Super Bowl winner odds from Odds API
   (`americanfootball_nfl_super_bowl_winner`, confirmed available —
   `has_outrights: true`).
2. Log-odds transform into an implied relative team-strength rating.
3. Using the actual regular-season schedule for that year, sum each
   team's per-game win probability against each opponent's implied
   strength (`team_strength / (team_strength + opponent_strength)`)
   across the season — no playoff bracket simulation.
4. This should naturally reflect schedule strength (a team with a weak
   schedule shows a higher expected win total than raw SB odds alone
   would suggest) without modeling conference/division effects
   separately.

This is a **derived approximation, not a true market-based win total** —
treat it as lower-confidence than the other two preseason-prior
components (last year's rating, roster turnover) once it's built. Compare
Layer 1's preseason-week accuracy with vs. without it once real backtest
results exist; don't assume it helps.

## Step 3 backtest findings (Layers 1+2 only, 2013-2024, 3,084 games)

Walk-forward backtest (`scripts/backtest.py`) of the core model using
only power ratings + matchup adjustments, before situational/market
layers existed. Full diagnostic methodology (pooled + season-by-season
significance tests, ATS slice analysis, moneyline reliability curve) is
in the script itself; key findings to carry forward:

- **Mechanics validated**: home-field-advantage recalibrated per season
  organically detected the real, documented COVID-era HFA drop (~2.4-2.7
  pts pre-2020 → ~1.3-2.1 pts 2020+) — evidence the walk-forward
  calibration is doing real work, not just bookkeeping.
- **No aggregate edge yet**: spread/total signal correlation with the
  closing line is ~0 and flips sign season to season (classic noise
  signature). Moneyline Brier/log-loss worse than the market's own
  de-vigged probabilities in every single graded season — but the
  reliability curve shows our probabilities track the market's closely
  bin-by-bin, so this reads as **missing information, not a broken
  probability-conversion mechanism**.
- **A specific, non-uniform bias was found**: ATS performance is
  statistically indistinguishable from 50% when we disagree with the
  market toward the *underdog* or only mildly (≤4pts) — but
  significantly *below* 50% specifically when we pick the **favorite**
  (44.5% win rate, n=512, p=0.013) or disagree **strongly** (4pts+:
  46.8%, n=974, p=0.047). Working hypothesis: EPA-based ratings likely
  overrate strong favorites (garbage-time/blowout inflation,
  insufficient shrinkage at the rating extremes) in a way the market
  already discounts via context Layers 1+2 don't have (injuries,
  trap-game awareness, opponent depth) — situational context (Layer 3)
  is the natural candidate to correct this specifically. **Check this
  explicitly with the same slice methodology whenever Layer 3 (and
  later Layer 4) get re-backtested — don't just report aggregate
  numbers.**

## Data feed decisions (resolved)

- **Weather**: `meteostat` (free) for historical backtest pull; NWS
  `api.weather.gov` (free) for live in-season forecasts.
- **Injuries**: `nfl_data_py.import_injuries()` for historical backtest;
  self-scraped official NFL injury report for live in-season updates
  (Thu practice report pass + ~90 min pre-kickoff final pass).
- **No paid data vendor for v1** — Sportradar/SportsDataIO priced out at
  $500+/month sales-gated contracts, not justified before the model
  proves CLV. Revisit later only if the scrape proves unreliable.

## Open decisions (confirm during build, don't block on them)

- Exact edge threshold / confidence-tier cutoffs (start conservative)
