"""Shared Supabase connection helpers.

Two ways in:
- get_pg_connection(): direct psycopg2 connection via Supabase's session
  pooler. Needed for DDL (CREATE TABLE) since PostgREST only exposes CRUD
  on existing tables/views. Direct (non-pooler) connections are IPv6-only
  on this project tier, which isn't reachable from this network, so we go
  through the Supavisor pooler instead.
- get_supabase_client(): the supabase-py client, for normal CRUD against
  PostgREST once tables exist.
"""
import os

import psycopg2
from supabase import create_client


def _project_ref() -> str:
    return os.environ["SUPABASE_URL"].replace("https://", "").split(".")[0]


def get_pg_connection():
    return psycopg2.connect(
        host="aws-0-us-east-1.pooler.supabase.com",
        port=5432,
        dbname="postgres",
        user=f"postgres.{_project_ref()}",
        password=os.environ["SUPABASE_DB_PASSWORD"],
        connect_timeout=10,
    )


def get_supabase_client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
