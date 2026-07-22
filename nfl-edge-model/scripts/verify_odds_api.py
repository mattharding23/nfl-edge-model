"""One-off script to confirm Odds API connectivity (no quota cost).

Run with: doppler run -- python scripts/verify_odds_api.py
"""
import os

import requests

key = os.environ["ODDS_API_KEY"]

response = requests.get(
    "https://api.the-odds-api.com/v4/sports",
    params={"apiKey": key},
    timeout=10,
)

print(f"Status: {response.status_code}")
if response.ok:
    sports = response.json()
    print(f"Sports returned: {len(sports)}")
    print(f"Requests remaining: {response.headers.get('x-requests-remaining')}")
else:
    print(f"Body: {response.text}")
