"""One-off script to confirm Supabase connectivity via the service key.

Run with: doppler run -- python scripts/verify_supabase.py
"""
import os

from supabase import create_client

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SERVICE_KEY"]

client = create_client(url, key)

response = client.table("power_ratings").select("*", count="exact").limit(0).execute()

print(f"Query succeeded. Row count: {response.count}")
