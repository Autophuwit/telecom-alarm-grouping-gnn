"""Weak pair labels from clear times.

Alarms caused by one fault tend to come back with one repair, so their clear times are close.

TOLERANCE BY ALARM NAME. One window does not fit every kind: the cells of a site come back over minutes,
a link in seconds. For each name, the tolerance is measured on the TRAINING days only, from alarm pairs that
the ticketing system put on one ticket: the smallest step that holds `coverage` of them. A name with too
few such pairs gets `default_tol` (or, with None, its pairs get no label).

    positive   clears within the larger of the two names' tolerances
    negative   clears further apart than max(neg_sec, neg_factor x tolerance)
    no label   in between, a missing clear, or a name with no figure

WEIGHTED BY HOW LIKELY A POSITIVE IS REAL. Two unrelated alarms can clear together by chance. Chance is
measured on pairs that started within 15 minutes but stand more than 200 km apart (they cannot share a
cause), per (same domain, tolerance). Each positive weighs

    1 - chance / (positive rate of its distance band, same domain, tolerance)

so a kind of pair whose co-clears are no better than chance teaches nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .graph import haversine

STEPS = np.array([60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0])
BANDS = ((5.0, "<= 5 km"), (20.0, "5-20 km"), (50.0, "20-50 km"), (np.inf, "> 50 km"))


@dataclass
class LabelConfig:
    coverage: float = 0.5
    min_pairs: int = 30
    default_tol: float | None = 60.0   # a name with no figure; None: its pairs get no label
    neg_sec: float = 600.0
    neg_factor: float = 2.0
    far_km: float = 200.0
    chance_sample: int = 5000
    chance_floor: int = 100


def fit_tolerances(days: list[tuple[pd.DataFrame, pd.DataFrame]], cfg: LabelConfig = LabelConfig()) -> dict:
    """{alarm name: seconds} from (nodes, node-ticket rows) of the training days."""
    gaps = []
    for nodes, nt in days:
        m = nt.merge(nt, on="ticket_id")
        m = m[m.node_x < m.node_y]
        t, c, nm = nodes.t.to_numpy(), nodes.clear.to_numpy(), nodes.name.to_numpy(object)
        x, y = m.node_x.to_numpy(), m.node_y.to_numpy()
        ok = (np.abs(t[x] - t[y]) <= 900) & np.isfinite(c[x]) & np.isfinite(c[y])
        g = np.abs(c[x] - c[y])[ok]
        gaps.append(pd.DataFrame({"name": np.r_[nm[x][ok], nm[y][ok]], "gap": np.r_[g, g]}))
    gaps = pd.concat(gaps)
    out = {}
    for name, g in gaps.groupby("name").gap:
        if len(g) >= cfg.min_pairs:
            share = (g.to_numpy()[:, None] <= STEPS[None]).mean(0)
            hit = np.flatnonzero(share >= cfg.coverage)
            if len(hit):
                out[name] = float(STEPS[hit[0]])
    return out


def band_of(km, same_site):
    b = np.full(len(km), "no place", dtype=object)
    prev = -1.0
    for edge, label in BANDS:
        b[np.isfinite(km) & (km > prev) & (km <= edge)] = label
        prev = edge
    b[same_site] = "same site"
    return b


def _chance(nodes, tolv, rng, cfg):
    """Co-clear rate of far-apart pairs by (same domain, tolerance step), and the overall rate per domain kind."""
    c, t = nodes.clear.to_numpy(float), nodes.t.to_numpy(float)
    lat, lon = nodes.lat.to_numpy(float), nodes.lon.to_numpy(float)
    dom = pd.factorize(nodes.domain)[0]
    pool = np.flatnonzero(np.isfinite(c) & np.isfinite(lat) & np.isfinite(tolv))
    by = np.zeros((2, len(STEPS)))
    if len(pool) < 2:
        return by
    tp = t[pool]
    I, J = [], []
    for s in rng.choice(len(pool), min(cfg.chance_sample, len(pool)), replace=False):
        lo, hi = np.searchsorted(tp, tp[s] - 900), np.searchsorted(tp, tp[s] + 900, "right")
        cand = np.r_[lo:s, s + 1:hi]
        if len(cand):
            take = cand if len(cand) <= 50 else rng.choice(cand, 50, replace=False)
            I.append(np.full(len(take), pool[s]))
            J.append(pool[take])
    if not I:
        return by
    I, J = np.concatenate(I), np.concatenate(J)
    far = haversine(lat[I], lon[I], lat[J], lon[J]) > cfg.far_km
    I, J = I[far], J[far]
    pt = np.maximum(tolv[I], tolv[J])
    hit = (np.abs(c[I] - c[J]) <= pt).astype(float)
    same = (dom[I] == dom[J]).astype(int)
    cls = np.searchsorted(STEPS, pt)
    for e in (0, 1):
        m = same == e
        overall = hit[m].mean() if m.any() else 0.0
        cnt = np.bincount(cls[m], minlength=len(STEPS))
        hits = np.bincount(cls[m], weights=hit[m], minlength=len(STEPS))
        by[e] = np.where(cnt >= cfg.chance_floor, hits / np.maximum(cnt, 1), overall)
    return by


def label_day(nodes: pd.DataFrame, E: dict, tol: dict, rng, cfg: LabelConfig = LabelConfig()):
    """(labelled mask, target 0/1, weight) per candidate pair."""
    i, j = E["i"], E["j"]
    clear = nodes.clear.to_numpy(float)
    tolv = nodes.name.map(tol).to_numpy(float)
    if cfg.default_tol is not None:
        tolv = np.where(np.isnan(tolv), cfg.default_tol, tolv)
    pt = np.maximum(tolv[i], tolv[j])
    dc = np.abs(clear[i] - clear[j])
    ok = np.isfinite(dc) & np.isfinite(pt)
    pt = np.nan_to_num(pt, nan=0.0)
    pos = ok & (dc <= pt)
    neg = ok & (dc > np.maximum(cfg.neg_sec, cfg.neg_factor * pt))

    ch = _chance(nodes, tolv, rng, cfg)
    same = E["same_domain"].astype(np.int64)
    cls = np.minimum(np.searchsorted(STEPS, pt), len(STEPS) - 1)
    sub = same * len(STEPS) + cls
    flat = pd.factorize(band_of(E["km"], E["same_site"]))[0] * (2 * len(STEPS)) + sub
    n_lab = np.bincount(flat[pos | neg], minlength=flat.max() + 1 if len(flat) else 1).astype(float)
    n_pos = np.bincount(flat[pos], minlength=len(n_lab)).astype(float)
    rate = np.divide(n_pos, n_lab, out=np.zeros_like(n_lab), where=n_lab > 0)
    keep = np.clip(1 - np.divide(ch.ravel()[sub], rate[flat], out=np.ones(len(flat)), where=rate[flat] > 0), 0, 1)
    w = np.ones(len(i), np.float32)
    w[pos] = keep[pos]
    return pos | neg, pos.astype(np.float32), w
