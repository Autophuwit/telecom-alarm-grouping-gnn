"""How good is a grouping?

Two views:
  ALARMS   B-cubed precision / recall against the true incidents (per alarm: the share of its group that truly
           belongs with it, and the share of its true incident that it was grouped with), plus the adjusted Rand index.
  TICKETS  the business view. Each ticket stands in its home group (the one holding most of its alarms); two tickets
           in one group are a merge. Precision: merges of two tickets that truly share an incident. Recall: true
           same-incident ticket pairs that were merged.

`resolution_agreement` needs no ground truth: the share of merged ticket pairs whose crews wrote the same
tier-2 resolution. It is what selects the threshold, as it would in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def bcubed(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    f = pd.DataFrame({"p": pred, "t": true})
    both = f.groupby(["p", "t"]).size()
    sp, st = f.groupby("p").size(), f.groupby("t").size()
    k = both.to_numpy(float)
    p_idx, t_idx = both.index.get_level_values(0), both.index.get_level_values(1)
    prec = float((k * k / sp.reindex(p_idx).to_numpy()).sum() / len(f))
    rec = float((k * k / st.reindex(t_idx).to_numpy()).sum() / len(f))
    return prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def ticket_homes(group, node_ticket: pd.DataFrame) -> pd.Series:
    f = node_ticket.assign(g=np.asarray(group)[node_ticket.node.to_numpy()])
    k = f.groupby(["ticket_id", "g"]).size().rename("k").reset_index()
    return k.sort_values(["ticket_id", "k", "g"], ascending=[True, False, True]).drop_duplicates("ticket_id").set_index("ticket_id").g


def _pairs(groups: pd.Series) -> set:
    out = set()
    for _, members in groups.groupby(groups).groups.items():
        m = sorted(members)
        out.update((m[x], m[y]) for x in range(len(m)) for y in range(x + 1, len(m)))
    return out


def ticket_pairs(group, node_ticket, true_group=None, tickets: pd.DataFrame | None = None) -> dict:
    home = ticket_homes(group, node_ticket)
    merged = _pairs(home)
    out = {"tickets": len(home), "merged_pairs": len(merged)}
    if tickets is not None and merged:
        res = tickets.set_index("ticket_id").res_tier2
        out["resolution_agreement"] = float(np.mean([res.get(a) == res.get(b) for a, b in merged]))
    if true_group is not None:
        truth = _pairs(ticket_homes(true_group, node_ticket))
        hit = len(merged & truth)
        out.update(true_pairs=len(truth), right_pairs=hit, pair_precision=hit / len(merged) if merged else np.nan,
                   pair_recall=hit / len(truth) if truth else np.nan)
    return out


def score_grouping(group, true_group, node_ticket, tickets) -> dict:
    p, r, f1 = bcubed(group, true_group)
    out = dict(groups=int(len(np.unique(group))), largest=int(np.bincount(np.unique(group, return_inverse=True)[1]).max()),
               b3_precision=p, b3_recall=r, b3_f1=f1, ari=float(adjusted_rand_score(true_group, group)))
    out.update(ticket_pairs(group, node_ticket, true_group, tickets))
    return out
