"""End to end: build the daily graphs, label, train, pick a threshold, group the test days, evaluate.

    python -m aig.run --data data/synthetic --out runs/demo

Days are split in time order: the first --train-days train, the next --val-days pick the threshold, the rest
are test. The threshold is chosen WITHOUT ground truth: the largest number of merged ticket pairs whose
resolution codes still agree at least --min-agreement of the time. Ground truth is read only to report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import cluster
from .data import day_nodes, day_tickets, load_corpus
from .evaluate import score_grouping, ticket_pairs
from .graph import EDGE_FEATURES, GraphConfig, build_day
from .labels import LabelConfig, fit_tolerances, label_day
from .model import DayTensors, PairGNN, Vocab, score, train


def group_day(n, E, s, nt, thr, inc_gap=0.05, benefit=0.5):
    """Clusters by average linkage, incidents through tickets, then each ticket in its home."""
    lab = cluster.average_clusters(n, E["i"], E["j"], s, thr, benefit)
    inc = cluster.incident_clusters(lab, E["i"], E["j"], s, nt, thr - inc_gap, benefit)
    return lab, cluster.ticket_home(inc, nt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--out", type=Path, default=Path("runs/demo"))
    p.add_argument("--train-days", type=int, default=5)
    p.add_argument("--val-days", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    # Resolution codes are noisy: with 10% of tickets coded wrong, even perfect merges agree only ~0.9 x 0.9 = 81%
    # of the time. The bar has to sit just under that ceiling, or it only admits a handful of lucky merges.
    p.add_argument("--min-agreement", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t0:6.1f}s] {m}", flush=True)  # noqa: E731
    rng = np.random.default_rng(a.seed)

    c = load_corpus(a.data)
    days = c.days
    tr, va, te = days[:a.train_days], days[a.train_days:a.train_days + a.val_days], days[a.train_days + a.val_days:]
    log(f"{len(c.alarms):,} alarms, {len(days)} days: train {len(tr)}, val {len(va)}, test {len(te)}")

    gcfg, lcfg = GraphConfig(), LabelConfig()
    D = {}
    for d in days:
        n = day_nodes(c, d)
        D[d] = dict(n=n, nt=day_tickets(c, n), E=build_day(n, c.links, gcfg))
    log(f"graphs built: {sum(len(v['E']['i']) for v in D.values()):,} candidate pairs")

    tol = fit_tolerances([(D[d]["n"], D[d]["nt"]) for d in tr], lcfg)
    log(f"tolerances from training tickets: {len(tol)} names; " + ", ".join(f"{k} {v:g}s" for k, v in sorted(tol.items())))

    vocab = Vocab([D[d]["n"] for d in tr])
    batches = []
    for d in tr:
        lab, y, w = label_day(D[d]["n"], D[d]["E"], tol, rng, lcfg)
        batches.append((DayTensors(D[d]["n"], D[d]["E"], vocab), lab, y, w))
    n_lab = sum(int(b[1].sum()) for b in batches)
    n_pos = sum(int((b[1] & (b[2] > 0)).sum()) for b in batches)
    log(f"labels: {n_lab:,} labelled pairs, {n_pos / max(n_lab, 1):.0%} positive")

    torch.manual_seed(a.seed)
    model = PairGNN(*vocab.sizes(), n_num=3, e_dim=len(EDGE_FEATURES))
    train(model, batches, epochs=a.epochs, seed=a.seed, log=log)
    for d in va + te:
        D[d]["s"] = score(model, DayTensors(D[d]["n"], D[d]["E"], vocab), 1800.0)

    # threshold on the validation days, by resolution agreement only
    sweep = []
    for thr in np.round(np.arange(0.05, 0.91, 0.05), 2):
        merged = agree = 0.0
        for d in va:
            v = D[d]
            _, inc = group_day(len(v["n"]), v["E"], v["s"], v["nt"], thr)
            tp = ticket_pairs(inc, v["nt"], tickets=c.tickets)
            merged += tp["merged_pairs"]
            agree += tp.get("resolution_agreement", 0) * tp["merged_pairs"]
        sweep.append(dict(threshold=float(thr), merged_pairs=int(merged), resolution_agreement=agree / merged if merged else np.nan))
    sweep = pd.DataFrame(sweep)
    ok = sweep[sweep.resolution_agreement >= a.min_agreement]
    thr = float(ok.sort_values(["merged_pairs", "threshold"], ascending=[False, False]).threshold.iloc[0] if len(ok)
                else sweep.sort_values("resolution_agreement", ascending=False).threshold.iloc[0])
    log(f"threshold {thr:.2f} (validation sweep in sweep.csv)")
    sweep.to_csv(a.out / "sweep.csv", index=False)

    # test days: the full method against simpler groupings
    rows, preds = [], []
    truth = c.truth.set_index("alarm_id").incident_id
    for d in te:
        v = D[d]
        n, E, s, nt = v["n"], v["E"], v["s"], v["nt"]
        true = pd.factorize(n.alarm_id.map(truth))[0]
        lab, inc = group_day(len(n), E, s, nt, thr)
        methods = {
            "each alarm alone": np.arange(len(n)),
            "rule: linked sites, 5 min": cluster.rule_baseline(n, E),
            "GNN + single link": cluster.single_link(len(n), E["i"], E["j"], s, thr),
            "GNN + average linkage": lab,
            "GNN + average + tickets": inc,
        }
        for name, g in methods.items():
            rows.append(dict(day=str(d.date()), method=name, **score_grouping(g, true, nt, c.tickets)))
        preds.append(pd.DataFrame({"alarm_id": n.alarm_id, "cluster": lab, "incident": inc, "day": str(d.date())}))
    # the whole trade-off on the test days, for the report (ground truth: never used to choose anything)
    curve = []
    for t_ in np.round(np.arange(0.05, 0.91, 0.05), 2):
        for d in te:
            v = D[d]
            true = pd.factorize(v["n"].alarm_id.map(truth))[0]
            _, inc = group_day(len(v["n"]), v["E"], v["s"], v["nt"], t_)
            for name, g in [("GNN + single link", cluster.single_link(len(v["n"]), v["E"]["i"], v["E"]["j"], v["s"], t_)),
                            ("GNN + average + tickets", inc)]:
                curve.append(dict(threshold=float(t_), day=str(d.date()), method=name, **ticket_pairs(g, v["nt"], true)))
    curve = pd.DataFrame(curve).groupby(["method", "threshold"])[["merged_pairs", "true_pairs", "right_pairs"]].sum()
    curve["pair_precision"] = curve.right_pairs / curve.merged_pairs.where(curve.merged_pairs > 0)
    curve["pair_recall"] = curve.right_pairs / curve.true_pairs
    curve.round(3).to_csv(a.out / "test_curve.csv")
    res = pd.DataFrame(rows)
    summary = res.drop(columns="day").groupby("method", sort=False).mean().round(3)
    pd.concat(preds).to_csv(a.out / "predictions.csv", index=False)
    res.to_csv(a.out / "results_by_day.csv", index=False)
    summary.to_csv(a.out / "results.csv")
    torch.save({"state": model.state_dict(), "vocab": vocab.__dict__, "threshold": thr, "tolerances": tol}, a.out / "model.pt")
    (a.out / "run.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "threshold": thr,
                                                "train": [str(x.date()) for x in tr], "val": [str(x.date()) for x in va],
                                                "test": [str(x.date()) for x in te]}, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["groups", "largest", "b3_precision", "b3_recall", "b3_f1", "ari",
                       "merged_pairs", "pair_precision", "pair_recall"]].to_string())
    log(f"written to {a.out}")


if __name__ == "__main__":
    main()
