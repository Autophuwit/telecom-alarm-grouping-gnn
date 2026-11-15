"""The pair model: a GATv2 encoder over the candidate graph and a symmetric head that scores each pair.

    node features -> Linear -> h0
    2 x [GATv2Conv(edge features) + ReLU, residual, LayerNorm]      messages pass along candidate pairs
    pair (i, j) -> MLP([h_i + h_j, h_i * h_j, |h_i - h_j|, edge features]) -> logit

The head is symmetric: swapping i and j gives the same score.

A day is too large for one pass, so it is cut into time slices. Each slice carries context on both sides,
long enough for every candidate pair, and scores the pairs whose earlier alarm falls inside it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from .graph import edge_features

SEVERITIES = ("warning", "minor", "major", "critical")


class Vocab:
    """Alarm names and domains seen in training; index 0 is 'unknown'."""

    def __init__(self, frames: list[pd.DataFrame], min_count: int = 1):
        allf = pd.concat(frames)
        self.name = {v: k + 1 for k, v in enumerate(sorted(n for n, c in allf.name.value_counts().items() if c >= min_count))}
        self.domain = {v: k + 1 for k, v in enumerate(sorted(allf.domain.unique()))}
        self.severity = {v: k + 1 for k, v in enumerate(SEVERITIES)}

    def sizes(self):
        return len(self.name) + 1, len(self.domain) + 1, len(self.severity) + 1


class DayTensors:
    """One day's nodes and candidate pairs as tensors, pairs sorted by their earlier alarm."""

    def __init__(self, nodes: pd.DataFrame, E: dict, vocab: Vocab, context_sec: float = 3600.0):
        self.N, self.t, self.H = len(nodes), nodes.t.to_numpy(float), context_sec
        enc = lambda col, voc: np.array([voc.get(v, 0) for v in nodes[col]], np.int64)  # noqa: E731
        self.x_cat = torch.from_numpy(np.column_stack([enc("name", vocab.name), enc("domain", vocab.domain),
                                                       enc("severity", vocab.severity)]))
        hour = (self.t % 86400) / 86400 * 2 * np.pi
        self.x_num = torch.from_numpy(np.column_stack([nodes.located.to_numpy(float), np.sin(hour), np.cos(hour)])
                                      .astype(np.float32))
        o = np.argsort(E["i"], kind="mergesort")
        self.order = o
        self.i, self.j = E["i"][o], E["j"][o]
        self.ef = torch.from_numpy(edge_features(E)[o])

    def windows(self, W: float):
        if not self.N:
            return
        s = math.floor(self.t.min() / W) * W
        while s <= self.t.max():
            a, b = np.searchsorted(self.t, s - self.H), np.searchsorted(self.t, s + W + self.H)
            ca, cb = np.searchsorted(self.t, s), np.searchsorted(self.t, s + W)
            if cb > ca:
                yield a, b, ca, cb
            s += W

    def batch(self, a, b, ca, cb):
        lo, hi = np.searchsorted(self.i, a), np.searchsorted(self.i, b)
        k = np.arange(lo, hi)
        k = k[self.j[k] < b]
        li, lj = self.i[k] - a, self.j[k] - a
        mp = torch.from_numpy(np.stack([np.r_[li, lj], np.r_[lj, li]]))
        mef = self.ef[torch.from_numpy(np.r_[k, k])]
        core = k[(self.i[k] >= ca) & (self.i[k] < cb)]
        pairs = torch.from_numpy(np.stack([self.i[core] - a, self.j[core] - a]))
        return self.x_cat[a:b], self.x_num[a:b], mp, mef, pairs, self.ef[torch.from_numpy(core)], core


class PairGNN(nn.Module):
    def __init__(self, n_name, n_domain, n_sev, n_num, e_dim, hid=64, layers=2, heads=2):
        super().__init__()
        self.name, self.domain, self.sev = nn.Embedding(n_name, 32), nn.Embedding(n_domain, 8), nn.Embedding(n_sev, 4)
        self.inp = nn.Linear(32 + 8 + 4 + n_num, hid)
        self.convs = nn.ModuleList([GATv2Conv(hid, hid, heads=heads, concat=False, edge_dim=e_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(3 * hid + e_dim, hid), nn.ReLU(), nn.Linear(hid, 1))

    def encode(self, x_cat, x_num, mp, mef):
        h = torch.relu(self.inp(torch.cat([self.name(x_cat[:, 0]), self.domain(x_cat[:, 1]), self.sev(x_cat[:, 2]), x_num], 1)))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + torch.relu(conv(h, mp, mef)))
        return h

    def forward(self, x_cat, x_num, mp, mef, pairs, pef):
        h = self.encode(x_cat, x_num, mp, mef)
        hi, hj = h[pairs[0]], h[pairs[1]]
        return self.head(torch.cat([hi + hj, hi * hj, (hi - hj).abs(), pef], 1)).squeeze(-1)


def train(model, days, window_sec=1800.0, epochs=5, lr=1e-3, seed=0, log=print):
    """days: list of (DayTensors, labelled, target, weight) with arrays in the graph's pair order."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        tot, steps = 0.0, 0
        for k in rng.permutation(len(days)):
            d, lab, y, w = days[k]
            lab, y, w = lab[d.order], y[d.order], w[d.order]
            wins = list(d.windows(window_sec))
            for q in rng.permutation(len(wins)):
                xc, xn, mp, mef, pairs, pef, core = d.batch(*wins[q])
                m = lab[core]
                if not m.any():
                    continue
                logit = model(xc, xn, mp, mef, pairs, pef)[torch.from_numpy(m)]
                wt = torch.from_numpy(w[core][m])
                loss = (nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[core][m]), reduction="none")
                        * wt).sum() / wt.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot, steps = tot + loss.item(), steps + 1
        log(f"epoch {ep}: {steps} slices, loss {tot / max(steps, 1):.4f}")
    return model


@torch.no_grad()
def score(model, d: DayTensors, window_sec=1800.0) -> np.ndarray:
    """Score per candidate pair, in the graph's original pair order."""
    model.eval()
    out = np.full(len(d.i), np.nan, np.float32)
    for a, b, ca, cb in d.windows(window_sec):
        xc, xn, mp, mef, pairs, pef, core = d.batch(a, b, ca, cb)
        if len(core):
            out[core] = torch.sigmoid(model(xc, xn, mp, mef, pairs, pef)).numpy()
    back = np.empty_like(out)
    back[d.order] = out
    return back
