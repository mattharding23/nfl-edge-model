"""One-off script to confirm Doppler is injecting expected secrets.

Run with: doppler run -- python scripts/verify_env.py
"""
import os

KEYS = [
    "ODDS_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_DB_PASSWORD",
]

for key in KEYS:
    print(f"{key}: {os.environ.get(key) is not None and os.environ.get(key) != ''}")
