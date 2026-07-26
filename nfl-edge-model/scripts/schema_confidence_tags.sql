-- Adds favorite_side_pick / large_disagreement_pick / tier_downgraded
-- columns so these two tags are tracked as their own segments,
-- independent of the singular why_tag (which continues to represent
-- the PRIMARY edge-generating mechanism, e.g. "wind_total"). Both tags
-- can co-occur with any why_tag and with each other.
--
-- Extended to lines_edges and alert_history too, not just clv_ledger:
-- a downgraded-tier pick needs to be traceable at every pipeline stage
-- (computed in lines_edges, decided whether to alert, tag recorded in
-- alert_history, later graded in clv_ledger) -- not just at grading time.

alter table clv_ledger
    add column if not exists favorite_side_pick boolean not null default false,
    add column if not exists large_disagreement_pick boolean not null default false,
    add column if not exists tier_downgraded boolean not null default false;

alter table lines_edges
    add column if not exists favorite_side_pick boolean not null default false,
    add column if not exists large_disagreement_pick boolean not null default false,
    add column if not exists tier_downgraded boolean not null default false;

alter table alert_history
    add column if not exists favorite_side_pick boolean not null default false,
    add column if not exists large_disagreement_pick boolean not null default false,
    add column if not exists tier_downgraded boolean not null default false;

create index if not exists idx_clv_ledger_favorite_side_pick on clv_ledger (favorite_side_pick) where favorite_side_pick;
create index if not exists idx_clv_ledger_large_disagreement_pick on clv_ledger (large_disagreement_pick) where large_disagreement_pick;
