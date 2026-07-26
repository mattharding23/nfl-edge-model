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

## Layer 3 backtest findings (Layers 1+2+3, 2013-2024) — negative result

Situational/injury layer (`scripts/situational.py`, `scripts/player_value.py`)
was built and re-backtested specifically to test whether it narrowed the
Step 3 favorite-side/large-disagreement bias above. **It didn't.**
Picked-favorite win rate went 44.5% → 44.0% (slightly worse, not
better); aggregate ATS/O-U/Brier/log-loss flat to marginally worse. A
pooled coefficient check of all 15 Layer 3 features combined showed
~0 R² against the Layer 1+2 residual, with only one nominally
significant coefficient (`letdown_diff`, p=0.022) — not treated as a
real finding given 15 near-arbitrary tests make ~0.75 false positives
expected by chance. None of the four injury mechanisms (QB-specific,
skill-specific, OL-coarse, DEF-coarse) showed detectable signal. Full
methodology and numbers in `scripts/backtest.py`'s `--layer3` path and
the `compare_bias_narrowing()` function.

## Garbage-time rating-inflation hypothesis — rejected

Follow-up diagnostic (not committed as pipeline code, exploratory only)
tested whether the Step 3 bias originates in Layer 1's power ratings via
garbage-time EPA inflating teams coming off blowouts, rather than
missing situational context. Definition used: `wp` outside 5-95% AND
(|score_differential|≥13 in Q4, OR ≥21 anytime in the 2nd half) — 10.8%
of plays (2010-2024) flagged. Computed power ratings two ways (all-snap
vs competitive-only) and compared the gap for the specific teams behind
the Step 3 bad bets against the league-average gap over the same window:

- **Favorite-slice bad bets (n=284, the stronger/cleaner of the two
  original signals)**: gap = -0.0215 vs league -0.0029, p=0.0001,
  Cohen's d=-0.28 — statistically real, but in the **opposite direction**
  from the hypothesis. Garbage time *deflates* these teams' all-snap
  rating relative to competitive-only, not inflates it (plausible
  mechanism: teams winning big shift to conservative clock-killing
  offense in garbage time, which drags down all-snap efficiency more
  than it helps).
- **Large-disagreement bad bets (n=518, the weaker original signal)**:
  gap = +0.0048 vs league -0.0029, p=0.019, Cohen's d=+0.11 — correct
  direction but small and inconsistent (off/def sub-components
  individually not significant, p=0.09/0.06).

**Conclusion: hypothesis rejected**, not confirmed-but-weak. The
stronger signal points the wrong way; the weaker one is too small/
inconsistent to lean on. Did not proceed to a shrinkage/downweighting
fix per the diagnostic's own stopping rule. **The favorite-side bias
most likely originates somewhere else in Layer 1/2's rating construction
or is a market-side (Layer 4) phenomenon** — candidates for the next
diagnostic: the Kalman filter's process/observation variance calibration
(general insufficient shrinkage, not garbage-time-specific), the
SRS-style opponent-adjustment mechanism, Layer 2's matchup-delta terms,
or public-money line movement on favorites that a ratings-only model
can't see. Re-check this bias explicitly whenever Layer 4 is backtested,
same slice methodology as Step 3.

## Kalman filter tail-miscalibration hypothesis — rejected

Follow-up diagnostic (exploratory only, not committed pipeline code)
tested whether the favorite-side/large-disagreement bias is a variance-
calibration problem: posterior rating confidence growing faster than
actual predictive accuracy, so the model's most extreme/confident
predictions are systematically less reliable than moderate ones (which
would explain both slices at once, since both select for extreme model
output rather than a team characteristic). Used the full 2013-2024
graded sample (3,084 games), not just the bad-bet slices.

Binned games by |predicted margin| and separately by the Kalman
filter's own posterior variance (`off_var`/`def_var` summed across the
four ratings feeding the margin signal) into deciles:

- **RMSE is flat across both binnings** — 12.8-14.7 pts regardless of
  prediction magnitude (corr with squared error = 0.03), 12.0-15.0 pts
  regardless of stated posterior confidence (corr = 0.002). No tail
  degradation in either direction.
- **Mean (signed) error is also flat** across magnitude bins (bounces
  between -1.2 and +1.3, including -0.15 in the most extreme decile) —
  no directional drift at the tails either.
- **Ratio test**: the Kalman filter's own variance says the
  most-confident decile should have ~65% the error-std of the
  least-confident decile (0.646); observed RMSE ratio was 0.966 —
  essentially no accuracy improvement despite much higher stated
  confidence.

**Conclusion: hypothesis rejected as an explanation for this bias.**
Error doesn't get worse at the tails (ruling out the specific
"confident predictions are less reliable" claim), and — logically, even
before the numbers — a symmetric/uniform variance-calibration problem
wouldn't naturally produce an *asymmetric* bias (favorites only, not
underdogs), which is what Step 3 actually found. Did not proceed to
isolating process vs. observation variance or building a shrinkage
schedule, per the diagnostic's own stopping rule.

**Separate, distinct finding worth keeping** (not a fix for this bias,
but relevant to future confidence-tier/probability work): the ratio
test shows the Kalman posterior variance carries close to zero real
predictive information about actual error — not a tail-specific
problem, a uniform one. Worth a dedicated look whenever confidence-tier
calibration or probability outputs are built (CLAUDE.md's own
backtest-derived confidence-tier requirement depends on this kind of
signal being real), but it is not what's driving 44.5%/46.8%.

**Two of three specific mechanisms now rejected for this bias**
(garbage-time EPA, Kalman tail miscalibration); Layer 3 situational/
injury context also showed no effect. Remaining candidates: the
SRS-style opponent-adjustment mechanism itself, Layer 2's matchup-delta
terms, or a Layer 4/market-side (public money on favorites) explanation
that a ratings-only model structurally can't see.

## Market-wide favorite bias — confirmed real, but too small to be the whole story

Follow-up diagnostic (exploratory only) tested whether the favorite-side
bias is inherited from a known market-wide phenomenon (public money
skewing lines toward favorites — the "favorite-longshot bias" documented
in sports-betting literature generally) rather than something our model
introduces. Used the full closing-line dataset from `historical_games`
(7,050 non-pickem games), independent of whether our model ever weighed
in on the game.

- **Market-wide favorite ATS cover rate: ~48.7%**, remarkably stable
  across every window tested — full history 1999-2025 (n=7,050, 48.72%,
  p=0.032 vs 50%), 2010-2024 (n=3,965, 48.73%, p=0.109), and 2013-2024
  matching Step 3's exact graded range (n=3,185, 48.73%, p=0.151). The
  same point estimate recurring across three different windows, with
  significance purely tracking sample size, is the signature of a real,
  stable small effect, not noise — and lines up with the established
  favorite-longshot literature. **Confirms the premise: a real
  market-wide favorite deficit exists**, independent of our model.
- **But it's the wrong size.** Market-wide deficit from 50%: 1.3
  percentage points. Our model's favorite-slice bad-bet deficit from
  50%: 5.5 points (44.5%). Our model's bias is **~4.3x larger** than the
  raw market-wide effect — not "roughly the same size" as the
  hypothesis's step-2 branch would need for a clean confirmation.

**Conclusion: partially confirmed, not a clean accept or reject.** The
market-wide phenomenon is real and our model isn't currently correcting
for it — that part is worth a genuine Layer 4 design item eventually.
But it only accounts for a fraction of what we're seeing; something in
our own model construction is independently contributing the
remaining, larger share. Neither "it's all market inheritance" nor "the
market has nothing to do with it" is accurate — don't force this into
either bucket later.

**Conceptual sketch for the eventual Layer 4 item** (not built, per this
session's scope): a favorite-side discount belongs in the edge/
confidence-tier step, not the fair-value line itself — the fair line
should stay an unbiased independent estimate; blending a "fade
favorites" prior into it would make it market-derivative and defeat the
point of having an independent rating system. This fits naturally into
CLAUDE.md's existing backtest-derived confidence-tier design (a
mechanism-specific discount/tier for favorite-side spread edges, tagged
accordingly) — but since it only closes ~1.3 of the 5.5-point gap,
finding the remaining model-specific component (opponent-adjustment
mechanism or Layer 2 matchup deltas — still open) matters at least as
much as this market-side item.

## Opponent-adjustment convergence hypothesis — rejected

Follow-up diagnostic (exploratory only) tested whether the remaining
~75% of the favorite-side bias (the share the market-wide effect above
doesn't explain) comes from residual strength-of-schedule distortion:
teams built up against a soft early-season schedule before Layer 1's
opponent-adjustment had enough games to correct, producing lasting
inflation. (Note: this repo's Layer 1 is a sequential Kalman filter, not
a classic offline iterative SRS solve — see `power_ratings.py`'s own
docstring on why — so this tested the analogous signature: elevated
posterior uncertainty / incomplete stabilization for bias-driving teams
at the time of their bad bet.)

Compared the 284 favorite-slice bad-bet games (same games as prior
sessions) against the league-wide sample, 2013-2024:

- **Posterior variance — the filter's own formal measure of convergence
  completeness — goes the wrong way.** Bias-driving teams: 0.0218 vs.
  league 0.0271 (p<0.0001, Cohen's d=-0.19). Their ratings were *more*
  confident/stable at bad-bet time, not less — directly contradicting
  "incomplete convergence."
- Two adjacent measures *did* move in the hypothesized direction: recent
  week-over-week rating volatility was higher (0.124 vs 0.105, d=0.27),
  and bad bets concentrated earlier in the season (mean week 8.6 vs 9.6;
  43.0% in weeks 1-6 vs league's 33.8%, χ²=12.55, p=0.0057).

**Conclusion: rejected as stated.** The most direct, literal test of
"convergence completeness" contradicts the hypothesis. The early-week
concentration and volatility instead point to a different, more precise
candidate: teams with an unusually decisive/convincing early-season hot
streak cause the Kalman filter to shrink variance *quickly* (correctly,
by its own math, in response to extreme observations) before enough of
the season has unfolded to know whether that start is representative —
"overconfident from a hot start," not "hasn't converged yet." Not
confirmed or investigated further this session; flagged for whoever
picks this up next, distinct from today's rejected hypothesis.

**Three of four specific mechanisms addressed for this bias**
(garbage-time EPA rejected, Kalman tail miscalibration rejected,
opponent-adjustment convergence rejected — this entry); market-wide
favorite-longshot bias confirmed but only ~25% of the gap. Remaining
candidates: Layer 2's matchup-delta terms (next up), the "hot start
overconfidence" variant noted above, or further Layer 4/market-side
investigation.

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
