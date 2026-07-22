# nfl-edge-model

Quantitative NFL betting edge model. Generates independent fair-value
lines (spreads, totals, moneylines, halves/quarters), compares them
against live sportsbook prices across six books, and alerts on
high-confidence, mechanism-backed edges. Validated primarily on CLV
(closing line value), backtested walk-forward before going live.

See [`CLAUDE.md`](./CLAUDE.md) for full project context, architecture,
and build sequence.

## Stack

- **Compute:** GitHub Actions (scheduled/cron)
- **Data:** Supabase (Postgres, free tier)
- **Dashboard:** GitHub Pages (`docs/`)
- **Alerts:** Gmail SMTP → email + SMS gateway
- **Secrets:** Doppler (local dev) / GitHub Actions secrets (prod)

## Local setup

```bash
git clone <this repo>
cd nfl-edge-model
doppler setup          # link to the nfl-edge-model Doppler project
pip install -r requirements.txt
doppler run -- python scripts/<script>.py
```

Never run pipeline scripts without `doppler run --` locally — secrets
are injected as env vars for that process only, nothing touches disk.

## Status

🚧 Pre-build. See `CLAUDE.md` build sequence — currently on Step 1
(data pipeline).
