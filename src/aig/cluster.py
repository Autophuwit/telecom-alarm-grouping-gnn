"""From pair scores to incidents.

WHY NOT "LINK EVERY PAIR ABOVE THE THRESHOLD". Single linkage chains: one strong pair joins two groups of any
size, and on a busy day groups chain end to end into giant clusters.

AVERAGE LINKAGE WITH MUTUAL BEST. Every alarm starts alone. Two groups merge when the mean score over the
pairs between them reaches the threshold AND each is the other's best match; repeated until nothing
qualifies. The mean runs over every scored pair between the groups, the weak ones too, so a few strong
links across a border of weak ones no longer carry it.

THE PAIRS NEVER SCORED. Most pairs between two groups were never candidates, either because they are far
apart (the honest score is 0) or because they were crowded out of an alarm's k nearest (unknown). `benefit`
is the benefit of the doubt such a pair gets:

    mean = sum of scores / (scored pairs + (1 - benefit) x unscored pairs)

TICKETS JOIN CLUSTERS INTO INCIDENTS, by the same average: every pair of one ticket's alarms in two
clusters counts as a pair scored 1, so a ticket joins two clusters when its alarms are a good part of
both, not when 5 of its alarms touch two clusters of 100.

TICKET HOME. A ticket's alarms can still land in several incidents. Its home is the incident holding most
of them; the rest move there one by one (an alarm on two tickets stays put). Each ticket ends up in
exactly one incident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def components(n, i, j, keep=None):
    keep = np.ones(len(i), bool) if keep is None else keep
    return connected_components(coo_matrix((np.ones(int(keep.sum())), (i[keep], j[keep])), shape=(n, n)), directed=False)[1]


def single_link(n, i, j, score, thr):
    """The chaining baseline: connected pieces of the pairs at or above thr."""
    return components(n, np.asarray(i), np.asarray(j), np.asarray(score) >= thr)


def _merge(n, a, b, s, c, size, thr, benefit, max_rounds):
    """Rounds of mutual-best merges. Pair k between groups a[k] < b[k] holds c[k] scored pairs summing to s[k]."""
    size = np.asarray(size, np.float64).copy()
    grp, rounds = np.arange(n), 0
    while len(a) and rounds < max_rounds:
        val = s / (c + (1.0 - benefit) * (size[a] * size[b] - c))
        ok = val >= thr
        if not ok.any():
            break
        rounds += 1
        rows, cols, v = np.r_[a[ok], b[ok]], np.r_[b[ok], a[ok]], np.r_[val[ok], val[ok]]
        o = np.lexsort((cols, -v, rows))
        first = np.r_[True, rows[o][1:] != rows[o][:-1]]
        g, best = rows[o][first], cols[o][first]
        best_of = np.full(n, -1, np.int64)
        best_of[g] = best
        mutual = (best_of[best] == g) & (g < best)
        ga, gb = g[mutual], best[mutual]
        rep = np.arange(n)
        rep[gb] = ga
        np.add.at(size, ga, size[gb])
        grp = rep[grp]
        a, b = rep[a], rep[b]
        keep = a != b
        key = np.minimum(a[keep], b[keep]) * n + np.maximum(a[keep], b[keep])
        o = np.argsort(key, kind="stable")
        key = key[o]
        st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.empty(0, np.int64)
        s = np.add.reduceat(s[keep][o], st) if len(st) else np.empty(0)
        c = np.add.reduceat(c[keep][o], st) if len(st) else np.empty(0)
        a, b = key[st] // n, key[st] % n
    return grp, rounds


def average_clusters(n, i, j, score, thr, benefit=0.5, max_rounds=2000):
    """Cluster label per alarm. Every cluster lies inside one single-link piece at the same threshold:
    average linkage only ever splits what single linkage joined."""
    i, j, score = np.asarray(i, np.int64), np.asarray(j, np.int64), np.asarray(score, np.float64)
    comp = single_link(n, i, j, score, thr)
    inside = (comp[i] == comp[j]) & (i != j)
    a, b = np.minimum(i[inside], j[inside]), np.maximum(i[inside], j[inside])
    grp, _ = _merge(n, a, b, score[inside], np.ones(int(inside.sum())), np.ones(n), thr, benefit, max_rounds)
    return np.unique(grp, return_inverse=True)[1]


def incident_clusters(lab, i, j, score, node_ticket: pd.DataFrame, thr, benefit=0.5, bond=1.0, max_rounds=2000):
    """Join clusters through the tickets that span them, by the average. node_ticket: (node, ticket_id) rows."""
    lab = np.asarray(lab, np.int64)
    N, L = len(lab), int(lab.max()) + 1 if len(lab) else 0
    keys = []
    for _, g in node_ticket.groupby("ticket_id").node:
        x = np.unique(g.to_numpy())
        if len(x) < 2 or len(np.unique(lab[x])) < 2:
            continue
        p, q = np.triu_indices(len(x), 1)
        d = lab[x[p]] != lab[x[q]]
        keys.append(np.minimum(x[p[d]], x[q[d]]) * N + np.maximum(x[p[d]], x[q[d]]))
    bkey = np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)
    i, j = np.asarray(i, np.int64), np.asarray(j, np.int64)
    m = lab[i] != lab[j]
    mkey = np.minimum(i[m], j[m]) * N + np.maximum(i[m], j[m])
    ms = np.where(np.isin(mkey, bkey), bond, np.asarray(score, np.float64)[m])
    allkey = np.r_[mkey, bkey[~np.isin(bkey, mkey)]]
    alls = np.r_[ms, np.full(len(allkey) - len(mkey), bond)]
    bonded = np.isin(allkey, bkey)
    ga, gb = lab[allkey // N], lab[allkey % N]
    gk = np.minimum(ga, gb) * L + np.maximum(ga, gb)
    o = np.argsort(gk, kind="stable")
    gk, alls, bonded = gk[o], alls[o], bonded[o]
    if not len(gk):
        return lab.copy()
    st = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1]])
    s, c = np.add.reduceat(alls, st), np.diff(np.r_[st, len(gk)]).astype(np.float64)
    tb = np.maximum.reduceat(bonded.astype(np.int8), st).astype(bool)
    a, b = gk[st] // L, gk[st] % L
    size = np.bincount(lab, minlength=L).astype(np.float64)
    # only clusters a ticket links can end in one incident
    ok = tb & (s / c >= thr)
    comp = components(L, a, b, ok)
    inside = comp[a] == comp[b]
    grp, _ = _merge(L, a[inside], b[inside], s[inside], c[inside], size, thr, benefit, max_rounds)
    return np.unique(grp[lab], return_inverse=True)[1]


def ticket_home(inc, node_ticket: pd.DataFrame):
    """Move each ticket's stray alarms into its home incident (alarms on 2+ tickets stay)."""
    inc = np.asarray(inc, np.int64).copy()
    nt = node_ticket.drop_duplicates()
    f = nt.assign(g=inc[nt.node.to_numpy()])
    size = np.bincount(inc)
    k = f.groupby(["ticket_id", "g"]).size().rename("k").reset_index()
    k["size"] = size[k.g.to_numpy()]
    home = k.sort_values(["ticket_id", "k", "size", "g"], ascending=[True, False, False, True]).drop_duplicates("ticket_id")
    f = f.merge(home[["ticket_id", "g"]].rename(columns={"g": "home"}), on="ticket_id")
    move = f[(f.node.map(nt.node.value_counts()) == 1) & (f.g != f.home)]
    inc[move.node.to_numpy()] = move.home.to_numpy()
    return np.unique(inc, return_inverse=True)[1]


def rule_baseline(nodes: pd.DataFrame, E: dict, gap_sec: float = 300.0):
    """The rule an operator might write: alarms at one site, or at directly linked sites, that start within
    gap_sec of each other belong together (chained)."""
    keep = (E["hop"] >= 0) & (E["hop"] <= 1) & (E["dt"] <= gap_sec)
    return components(len(nodes), E["i"], E["j"], keep)
