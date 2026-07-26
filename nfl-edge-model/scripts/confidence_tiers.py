"""Confidence-tier tag/downgrade infrastructure for the alert-decision
step described in CLAUDE.md ("every alert needs a why tag... must clear
both a minimum edge threshold AND a confidence floor").

Scope note: that alert-decision step itself is Step 6 in the build
sequence and hasn't been built yet -- there is no live alert loop in
this repo to attach to today. This module is standalone, reusable
infrastructure, ready to wire in once Step 6 starts, built now because
the underlying evidence (five rejected/non-explanatory mechanism
hypotheses for the favorite-side bias, see CLAUDE.md) is already in
hand.

Tags are downgrade-only
--------------------------
favorite_side_pick and large_disagreement_pick can only demote a
confidence tier, never raise one or create an alert on their own. The
fair-value line (Layers 1-3/4) is completely untouched by this module --
it operates purely on the already-computed predicted_margin vs. the
market spread_line.

Why the severities aren't frozen at 44.5%/46.8%
--------------------------------------------------
The 2013-2024 discovery sample's point estimates should NOT be treated
as confirmed effect sizes: the 2011-2012 out-of-sample holdout did not
cleanly reconfirm them (2011 alone showed both slices *above* 50%,
the opposite direction). DEFAULT_TAG_CONFIG's severities are therefore
a conservative single-tier downgrade, not a value tuned to match the
historical win rates -- see CLAUDE.md's proposed (pending sign-off)
sample-size gates for how live data should inform any future change to
these severities.

Combination rule: max, not stacked
--------------------------------------
When both tags fire on the same pick, the downgrade applied is the
LARGER of the two configured severities, never the sum -- deliberately
conservative given the holdout instability (see CLAUDE.md).
"""
import numpy as np

TIER_ORDER = ["HIGH", "MEDIUM", "LOW", "NO_ALERT"]

# Proposed defaults -- single-tier downgrade for either tag, matching the
# design agreed before this build. Loaded from model_snapshots.config at
# runtime (see load_tag_config); this is only the fallback for a
# first-run / no-active-snapshot case.
DEFAULT_TAG_CONFIG = {
    "favorite_side_pick": {"enabled": True, "downgrade_severity": 1},
    "large_disagreement_pick": {"enabled": True, "downgrade_severity": 1, "threshold_points": 4.0},
}


def compute_tags(predicted_margin: float, spread_line: float, tag_config: dict | None = None) -> dict:
    """Determines which tags fire for a single pick, purely from
    predicted_margin vs. spread_line -- the identical definitions used
    across all five diagnostic sessions (backtest.py's
    ats_slice_analysis): favorite_side_pick = the side we'd bet is the
    market favorite; large_disagreement_pick = |disagreement| clears the
    configured point threshold (default 4.0).
    """
    tag_config = tag_config or DEFAULT_TAG_CONFIG
    edge = predicted_margin - spread_line
    bet_home = edge > 0
    picked_favorite = (spread_line > 0) if bet_home else (spread_line < 0)
    edge_abs = abs(edge)

    tags = set()
    if tag_config.get("favorite_side_pick", {}).get("enabled") and picked_favorite:
        tags.add("favorite_side_pick")
    large_cfg = tag_config.get("large_disagreement_pick", {})
    threshold = large_cfg.get("threshold_points", 4.0)
    if large_cfg.get("enabled") and edge_abs >= threshold:
        tags.add("large_disagreement_pick")

    return {
        "tags": tags,
        "bet_home": bool(bet_home),
        "picked_favorite": bool(picked_favorite),
        "edge_abs": float(edge_abs),
    }


def apply_tier_downgrade(base_tier: str, tags: set, tag_config: dict | None = None) -> dict:
    """Max-of-the-two combination rule: the downgrade applied is the
    largest severity among triggered tags, never the sum, even when
    both tags fire on the same pick.
    """
    tag_config = tag_config or DEFAULT_TAG_CONFIG
    if base_tier not in TIER_ORDER:
        raise ValueError(f"Unknown tier {base_tier!r}, expected one of {TIER_ORDER}")
    if not tags:
        return {"tier": base_tier, "downgraded": False, "triggered_tags": [], "severity_applied": 0}

    severity = max((tag_config.get(t, {}).get("downgrade_severity", 0) for t in tags), default=0)
    idx = TIER_ORDER.index(base_tier)
    new_idx = min(idx + severity, len(TIER_ORDER) - 1)
    new_tier = TIER_ORDER[new_idx]
    return {
        "tier": new_tier,
        "downgraded": new_tier != base_tier,
        "triggered_tags": sorted(tags),
        "severity_applied": severity,
    }


def evaluate_pick(base_tier: str, predicted_margin: float, spread_line: float, tag_config: dict | None = None) -> dict:
    """Convenience wrapper: tag + downgrade in one call, for the
    eventual Step 6 alert loop to invoke directly."""
    tag_result = compute_tags(predicted_margin, spread_line, tag_config)
    downgrade_result = apply_tier_downgrade(base_tier, tag_result["tags"], tag_config)
    return {**tag_result, **downgrade_result}


def load_tag_config() -> dict:
    """Loads confidence_tags config from the active model_snapshot,
    falling back to DEFAULT_TAG_CONFIG if none exists yet. This is what
    makes severities live-adjustable (per CLAUDE.md's sample-size-gated
    review process) without a code change -- editing/inserting a new
    model_snapshot row changes behavior on next load.
    """
    from db import get_supabase_client
    client = get_supabase_client()
    resp = (
        client.table("model_snapshots")
        .select("config")
        .eq("is_active", True)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data and "confidence_tags" in (resp.data[0].get("config") or {}):
        return resp.data[0]["config"]["confidence_tags"]
    return DEFAULT_TAG_CONFIG


if __name__ == "__main__":
    import argparse
    import fastparquet
    import pandas as pd

    parser = argparse.ArgumentParser(description="Sanity-check tag logic against cached backtest data.")
    parser.add_argument("--backtest-parquet", type=str, required=True,
                         help="Path to a cached backtest_l3_results.parquet-style file (must have "
                              "predicted_margin_l12_only, spread_line, home_score, away_score).")
    args = parser.parse_args()

    pf = fastparquet.ParquetFile(args.backtest_parquet)
    cols = [c for c in pf.columns if c != "gameday"]
    df = pf.to_pandas(columns=cols)
    df = df.dropna(subset=["predicted_margin_l12_only", "spread_line", "home_score", "away_score"])

    results = [compute_tags(row.predicted_margin_l12_only, row.spread_line) for row in df.itertuples()]
    df = df.reset_index(drop=True)
    df["tags"] = [r["tags"] for r in results]
    df["edge_abs"] = [r["edge_abs"] for r in results]

    threshold = 1.0
    disagreement = df[df["edge_abs"] > threshold]
    favorite_n = disagreement["tags"].apply(lambda t: "favorite_side_pick" in t).sum()
    large_n = disagreement["tags"].apply(lambda t: "large_disagreement_pick" in t).sum()
    overlap_n = disagreement["tags"].apply(lambda t: {"favorite_side_pick", "large_disagreement_pick"}.issubset(t)).sum()

    print(f"favorite_side_pick (|disagreement|>{threshold}pt): n={favorite_n}  (expected 512)")
    print(f"large_disagreement_pick (>=4pt): n={large_n}  (expected 974)")
    print(f"both tags fire: n={overlap_n}  (expected 97)")

    print("\n=== Tier downgrade demo ===")
    for tier in TIER_ORDER[:-1]:
        r = apply_tier_downgrade(tier, {"favorite_side_pick"})
        print(f"{tier} + favorite_side_pick -> {r['tier']} (downgraded={r['downgraded']})")
        r2 = apply_tier_downgrade(tier, {"favorite_side_pick", "large_disagreement_pick"})
        print(f"{tier} + both tags          -> {r2['tier']} (severity_applied={r2['severity_applied']}, "
              f"NOT double-downgraded)")
