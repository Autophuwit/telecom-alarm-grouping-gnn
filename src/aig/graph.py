"""Candidate alarm pairs for one day, and the features of each pair.

A day has far too many alarm pairs to score them all, so the graph only decides who MAY be compared.
Whether two alarms belong together is left entirely to the model. Four sources of candidates:

    1  near in time and space    k nearest by (km / radius, seconds / window), both within limits
    2  no place                  k nearest in time within one domain, when either end has no site
    4  linked in the topology    k partners at sites 1..max_hop hops away: fewest hops first, then nearest onset
    8  one site, longer          the next k alarms at the same site within an hour (battery drains after mains fail)

Pair features: seconds apart, km (or none), same site / device / domain / name, topology hops, and the
source bits. Never the clear time, the ticket, or raw coordinates (a model that knows coordinates
learns regions instead of relations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

R_KM = 6371.0
HOPS = (-1, 0, 1, 2, 3, 4, 5, 9)        # -1: an end off the topology; 9: further, or not linked
EDGE_FEATURES = (["dt", "log_km", "no_km", "same_site", "same_device", "same_domain", "same_name"]
                 + [f"hop_{h}" for h in HOPS] + ["src_space", "src_noplace", "src_topo", "src_site"])


@dataclass
class GraphConfig:
    window_sec: float = 900.0
    radius_km: float = 50.0
    k_space: int = 64
    k_time: int = 16
    max_hop: int = 5
    k_hop: int = 16
    site_window_sec: float = 3600.0
    k_site: int = 16


def haversine(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    s = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(s, 0, 1)))


def site_graph(links: pd.DataFrame):
    """Sites and their adjacency (fibre, microwave and X2 links alike) as a sparse matrix."""
    names = pd.Index(pd.unique(np.r_[links.a_site_id.to_numpy(), links.b_site_id.to_numpy()]))
    a, b = names.get_indexer(links.a_site_id), names.get_indexer(links.b_site_id)
    g = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(len(names), len(names))).tocsr()
    return names, g


def _next_k(key, t, k, max_gap, extra=None):
    """Pairs (p, q) of the next k nodes after each one within one key, at most max_gap seconds later."""
    order = np.lexsort((t, key))
    P, Q = [], []
    for off in range(1, k + 1):
        p, q = order[:-off], order[off:]
        ok = (key[p] == key[q]) & (key[p] >= 0) & (t[q] - t[p] <= max_gap)
        if extra is not None:
            ok &= extra(p, q)
        P.append(p[ok])
        Q.append(q[ok])
    return (np.concatenate(P), np.concatenate(Q)) if P else (np.empty(0, np.int64),) * 2


def build_day(nodes: pd.DataFrame, links: pd.DataFrame, cfg: GraphConfig = GraphConfig()) -> dict:
    """Candidate pairs (i < j in onset order) with their raw attributes."""
    N = len(nodes)
    t = nodes.t.to_numpy(float)
    lat, lon = nodes.lat.to_numpy(float), nodes.lon.to_numpy(float)
    located = nodes.located.to_numpy(bool)
    W, R = cfg.window_sec, cfg.radius_km

    # 1: near in time and space
    li = np.flatnonzero(located)
    si = sj = np.empty(0, np.int64)
    if len(li) > 1:
        lat0 = float(np.nanmean(lat[li]))
        P = np.column_stack([lon[li] * 111.32 * np.cos(np.radians(lat0)) / R, lat[li] * 110.57 / R, t[li] / W])
        dist, nb = cKDTree(P).query(P, k=min(cfg.k_space + 1, len(li)), distance_upper_bound=np.sqrt(2) + 1e-9)
        rows = np.repeat(np.arange(len(li)), nb.shape[1])
        nb = nb.ravel()
        ok = (nb < len(li)) & (nb != rows)
        si, sj = li[rows[ok]], li[nb[ok]]
        keep = (np.abs(t[sj] - t[si]) <= W) & (haversine(lat[si], lon[si], lat[sj], lon[sj]) <= R)
        si, sj = si[keep], sj[keep]

    # 2: an end with no place, near in time within one domain
    dom = pd.factorize(nodes.domain)[0]
    ti, tj = _next_k(dom, t, cfg.k_time, W, extra=lambda p, q: ~located[p] | ~located[q])

    # 4: sites linked in the topology
    names, G = site_graph(links)
    code = names.get_indexer(nodes.site.where(nodes.site != "", None))
    hi_, hj_ = [], []
    if cfg.max_hop > 0:
        lo_w, hi_w = np.searchsorted(t, t - W), np.searchsorted(t, t + W, "right")
        on = np.flatnonzero(code >= 0)
        by_site = pd.Series(on).groupby(code[on]).apply(lambda g: g.to_numpy())
        for s, members in by_site.items():
            row = dijkstra(G, directed=False, indices=int(s), unweighted=True, limit=cfg.max_hop)
            for x in members:
                c_ = code[lo_w[x]:hi_w[x]]
                d_ = np.where(c_ >= 0, row[c_], np.inf)
                idx = np.flatnonzero((d_ >= 1) & (d_ <= cfg.max_hop))
                if len(idx) > cfg.k_hop:
                    rank = d_[idx] * 1e6 + np.abs(t[lo_w[x] + idx] - t[x])
                    idx = idx[np.argpartition(rank, cfg.k_hop)[:cfg.k_hop]]
                hi_.append(np.full(len(idx), x))
                hj_.append(lo_w[x] + idx)
    hpi = np.concatenate(hi_).astype(np.int64) if hi_ else np.empty(0, np.int64)
    hpj = np.concatenate(hj_).astype(np.int64) if hj_ else np.empty(0, np.int64)

    # 8: one site over a longer time
    scode = pd.factorize(nodes.site)[0]
    scode[nodes.site.to_numpy() == ""] = -1
    qi, qj = _next_k(scode, t, cfg.k_site, cfg.site_window_sec)

    # one row per pair, i < j, its sources OR-ed together
    ai = np.concatenate([si, ti, hpi, qi]).astype(np.int64)
    aj = np.concatenate([sj, tj, hpj, qj]).astype(np.int64)
    src = np.concatenate([np.full(len(si), 1), np.full(len(ti), 2), np.full(len(hpi), 4), np.full(len(qi), 8)]).astype(np.int8)
    key = np.minimum(ai, aj) * N + np.maximum(ai, aj)
    o = np.argsort(key, kind="mergesort")
    key, src = key[o], src[o]
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.empty(0, np.int64)
    src = np.bitwise_or.reduceat(src, first) if len(key) else src
    key = key[first]
    i, j = key // N, key % N

    # hops between the two sites
    hop = np.full(len(i), -1, np.int64)
    both = (code[i] >= 0) & (code[j] >= 0)
    if both.any():
        us = np.unique(code[i][both])
        D = dijkstra(G, directed=False, indices=us, unweighted=True, limit=cfg.max_hop)
        h = D[np.searchsorted(us, code[i][both]), code[j][both]]
        hop[both] = np.where(np.isfinite(h), h, 9).astype(np.int64)
    site = nodes.site.to_numpy(object)
    same_site = (site[i] == site[j]) & (site[i] != "")
    hop[same_site] = 0
    km = np.where(located[i] & located[j], haversine(lat[i], lon[i], lat[j], lon[j]), np.nan)
    dev, nm = nodes.device.to_numpy(object), nodes.name.to_numpy(object)
    return dict(i=i, j=j, dt=np.abs(t[j] - t[i]), km=km, same_site=same_site,
                same_device=(dev[i] == dev[j]) & (dev[i] != ""), same_domain=dom[i] == dom[j],
                same_name=nm[i] == nm[j], hop=hop, src=src)


def edge_features(E: dict) -> np.ndarray:
    km = E["km"]
    F = {"dt": np.clip(E["dt"] / 900.0, 0, 4),
         "log_km": np.where(np.isfinite(km), np.log1p(np.nan_to_num(km)) / np.log1p(50.0), 0.0),
         "no_km": (~np.isfinite(km)).astype(float),
         "same_site": E["same_site"], "same_device": E["same_device"],
         "same_domain": E["same_domain"], "same_name": E["same_name"]}
    F.update({f"hop_{h}": E["hop"] == h for h in HOPS})
    F.update({"src_space": (E["src"] & 1) > 0, "src_noplace": (E["src"] & 2) > 0,
              "src_topo": (E["src"] & 4) > 0, "src_site": (E["src"] & 8) > 0})
    return np.column_stack([np.asarray(F[k], np.float32) for k in EDGE_FEATURES])
