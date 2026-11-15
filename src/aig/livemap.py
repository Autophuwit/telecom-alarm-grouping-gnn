"""The live incident map, replayed: one day as an operations map refreshed every 5 minutes would have shown it.

    python -m aig.livemap --data data/synthetic --run runs/demo --out docs/live

Runs the live engine (aig.live) over a test day and writes what the map page (web/live_map.html) reads:

    ticks/HHMM.json    the map's list at that tick, in the live service's latest.json format: every incident on the
                       map with its state (just in / ongoing / quiet 1 h+), alarms, sites, tickets, alarm types,
                       where it split from and what merged into it
    ticks/index.json   the ticks written, and the busiest one to open on
    view_ids.json      {incident id: its 3-D file}
    inc/<g>.json       each incident's 3-D layers (sites, device stack, alarms, model pairs, tickets) as it ended the
                       day; an incident merged away opens the one it merged into
    index.html         the page

A live deployment writes the same files every tick and the page polls them; GitHub Pages cannot run the engine, so
the page finds ticks/index.json and plays the day back instead.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import day_nodes, day_tickets, load_corpus
from .graph import build_day
from .live import Display, FixedScores, LiveEngine, LiveScorer, Tracker
from .model import DayTensors, score
from .view import FAMILY, TEMPLATE, build_payload, load_model

PAGE = TEMPLATE.parent / "live_map.html"


def display_fields(n):
    """extra(members) for Tracker.snapshot: what the map's list and side panel show of an incident."""
    site, name = n.site.to_numpy(object), n.name.to_numpy(object)
    lat, lon = n.lat.to_numpy(float), n.lon.to_numpy(float)
    sev = n.severity.str.upper().to_numpy(object)
    fam = np.array([FAMILY.get(x, "OTHER") for x in name], object)

    def fn(m):
        sites = sorted({s for s in site[m] if s})
        ok = np.isfinite(lat[m])
        pts = []
        if ok.any():
            d = pd.DataFrame({"s": site[m][ok], "la": lat[m][ok], "lo": lon[m][ok]}).groupby("s")
            agg = d.agg(la=("la", "mean"), lo=("lo", "mean"), n=("la", "size")).sort_values("n", ascending=False)
            pts = [[s, round(r.la, 5), round(r.lo, 5), int(r.n)] for s, r in agg.head(60).iterrows()]
        return {"n_sites": len(sites), "sites": sites[:30], "site_pts": pts, "conf": None, "stay_sites": len(sites),
                "groups_now": 1, "in_graph": int(len(m)),
                "lat": round(float(lat[m][ok].mean()), 5) if ok.any() else None,
                "lon": round(float(lon[m][ok].mean()), 5) if ok.any() else None,
                "severity": dict(Counter(sev[m])), "families": dict(Counter(fam[m])),
                "top_alarms": [[k, v] for k, v in Counter(name[m]).most_common(3)], "n_in_window": int(len(m)),
                "leaving": []}
    return fn


def incident_files(G, out_dir, max_edges=4000):
    """One small JSON per incident of a view payload: its layers cut down to it, with indices of its own.
        sites    [name, lat, lon, via]     via 1: a quiet site between two of the incident's sites
        alarms   [site, onset epoch, name, family, repeats, device (-1 none), tickets, synthetic 0/1, leaving 0]
        nes      [name, kind, site, parent (-1), own 0/1]
        nelinks  [ne, ne, 0 link | 1 X2 neighbours]
        edges    [alarm, alarm, score, onset gap s, hops, km]
        topo     [site, site]
        tickets  [id, site, [alarms], synthetic alarm name]"""
    A, S, E = G["alarms"], G["sites"], G.get("edges", [])
    T, NE, NL, strs = G.get("tickets", []), G.get("nes", []), G.get("nelinks", []), G["strings"]
    transit = set(G.get("transit", []))
    adj = defaultdict(set)
    for p, q in G.get("topo", []):
        adj[p].add(q)
        adj[q].add(p)
    by_g, e_by_g, tk_of = defaultdict(list), defaultdict(list), defaultdict(list)
    for k, r in enumerate(A):
        by_g[r["g"]].append(k)
    for e in E:
        if A[e["i"]]["g"] == A[e["j"]]["g"]:
            e_by_g[A[e["i"]]["g"]].append(e)
    for ti, t in enumerate(T):
        for k in t.get("al", []):
            tk_of[k].append(ti)
    own_by_site, nl_by = defaultdict(list), defaultdict(list)
    for r, ne in enumerate(NE):
        if ne.get("st"):
            own_by_site[ne["s"]].append(r)
    for p, q, ty in NL:
        nl_by[p].append((q, ty))
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for g, al in by_g.items():
        aset = set(al)
        sites = {A[k]["s"] for k in al}
        tks = sorted({ti for k in al for ti in tk_of[k]})
        sites |= {T[ti]["s"] for ti in tks}
        used = {A[k]["ne"] for k in al if A[k].get("ne") is not None}
        used |= {NE[r]["p"] for r in list(used) if NE[r].get("p") is not None}
        used |= {r for s in sites for r in own_by_site.get(s, ())}
        sites |= {NE[r]["s"] for r in used}
        via = {t for t in transit if len(adj.get(t, set()) & sites) >= 2}
        allsite = sorted(sites | via)
        sl = {s: i for i, s in enumerate(allsite)}
        nl = {r: i for i, r in enumerate(sorted(used))}
        alo = {k: i for i, k in enumerate(al)}

        def tk_text(r):
            lb = r.get("l") or {}
            return strs[lb["TICKETID"]] if "TICKETID" in lb else ""
        es = sorted(e_by_g.get(g, []), key=lambda e: -e.get("s", 0))[:max_edges]
        doc = {
            "g": g,
            "sites": [[str(S[s]["n"]), S[s]["lat"], S[s]["lon"], 1 if s in via else 0] for s in allsite],
            "alarms": [[sl[A[k]["s"]], A[k]["t"], strs[A[k]["n"]], strs[A[k]["c"]], A[k].get("o", 1),
                        nl.get(A[k].get("ne"), -1), tk_text(A[k]), 0, 0] for k in al],
            "leaving": [],
            "nes": [[strs[NE[r]["n"]], NE[r]["k"], sl[NE[r]["s"]], nl.get(NE[r].get("p"), -1), int(bool(NE[r].get("st")))]
                    for r in sorted(used)],
            "nelinks": [[nl[p], nl[q], ty] for p in used for q, ty in nl_by.get(p, ()) if q in used],
            "edges": [[alo[e["i"]], alo[e["j"]], e["s"], e.get("g"), e.get("h"), e.get("d")] for e in es],
            "topo": [[sl[p], sl[q]] for p in allsite for q in adj[p] if q in sl and p < q],
            "tickets": [[str(T[ti]["id"]), sl[T[ti]["s"]], [alo[k] for k in T[ti].get("al", []) if k in aset], ""]
                        for ti in tks],
        }
        (out_dir / f"{g}.json").write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        done.add(int(g))
    return done


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--run", type=Path, default=Path("runs/demo"))
    p.add_argument("--day", help="YYYY-MM-DD; default the run's last test day")
    p.add_argument("--out", type=Path, default=Path("docs/live"))
    p.add_argument("--tick-min", type=float, default=5)
    p.add_argument("--write-every", type=int, default=3, help="write the map every this many ticks [3: 15 min]")
    p.add_argument("--lookback-h", type=float, default=3)
    p.add_argument("--settle-min", type=float, default=60)
    p.add_argument("--hold", type=int, default=2)
    p.add_argument("--scorer", choices=("live", "fixed"), default="live")
    a = p.parse_args(argv)

    c = load_corpus(a.data, with_truth=False)
    model, vocab, thr, tol = load_model(a.run)
    day = pd.Timestamp(a.day or json.loads((a.run / "run.json").read_text())["test"][-1])
    ds = day.strftime("%Y%m%d")
    n = day_nodes(c, day)
    nt, E = day_tickets(c, n), build_day(n, c.links)
    t = n.t.to_numpy(float)
    first_tk = nt.sort_values("ticket_id").drop_duplicates("node").set_index("node").ticket_id
    tt = np.array([first_tk.get(x, "") for x in range(len(n))], object)
    n = n.assign(severity=c.alarms.set_index("alarm_id").severity.reindex(n.alarm_id).to_numpy())

    tr = Tracker(ds, t, tt, n.alarm_id.to_numpy(object))
    disp = Display(len(t), a.hold)
    if a.scorer == "live":
        scorer = LiveScorer(n, c.links, model, vocab)
    else:
        scorer = FixedScores(E["i"], E["j"], score(model, DayTensors(n, E, vocab)))
    eng = LiveEngine(t, scorer, thr, tr, lookback=a.lookback_h * 3600, settle=a.settle_min * 60, displays=[disp])
    extra = display_fields(n)

    ticks_dir = a.out / "ticks"
    if a.out.exists():
        shutil.rmtree(a.out)
    ticks_dir.mkdir(parents=True)
    names, busy = [], []
    step = a.tick_min * 60
    for k, now in enumerate(np.arange(step, 86400 + step, step)):
        eng.tick(float(now))
        if (k + 1) % a.write_every:
            continue
        snap = tr.snapshot(float(now), labels=disp.shown, extra=extra)
        stamp = day + pd.Timedelta(seconds=float(now))
        top = {"generated_at": stamp.strftime("%Y-%m-%d %H:%M:%S"), "day": ds, "run": "GATv2 pair model",
               "threshold": round(thr, 2), "engine": "average", "hold": a.hold, "tick_min": a.tick_min,
               "data_last_onset": stamp.strftime("%Y-%m-%d %H:%M:%S"), "data_lag_min": 0.0, "stale": False,
               "itsm": "synthetic", "states": dict(Counter(x["state"] for x in snap["incidents"])), **snap}
        name = f"{int(now // 3600):02d}{int(now % 3600 // 60):02d}"
        (ticks_dir / f"{name}.json").write_text(json.dumps(top, separators=(",", ":"), default=float), encoding="utf-8")
        names.append(name)
        busy.append(sum(1 for x in snap["incidents"] if x["state"] != "SETTLED" and x["n_alarms"] >= 5))
    start = int(np.argmax(busy))
    (ticks_dir / "index.json").write_text(json.dumps({"day": str(day.date()), "ticks": names, "start": start}))

    # the 3-D files: each incident as it ended the day
    final = tr.final_labels()
    codes = pd.factorize(final)[0]
    payload, info = build_payload(c, day, model, vocab, thr, tol, max_incidents=10**6, labels=(codes, codes),
                                  min_tickets=0)
    written = incident_files(payload, a.out / "inc")
    code_of = dict(zip(final.tolist(), codes.tolist()))
    groups = {}
    for h, inc in enumerate(tr.inc):
        g = code_of.get(h)
        if g is not None and g in written:
            groups[inc.id] = g
    for inc in tr.inc:                                          # merged away: open the incident it merged into
        if inc.merged_into and tr.resolve(inc.id) in groups:
            groups[inc.id] = groups[tr.resolve(inc.id)]
    (a.out / "view_ids.json").write_text(json.dumps({"generated_at": names[-1], "day": ds, "groups": groups}))
    shutil.copy2(PAGE, a.out / "index.html")
    size = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file()) / 1e6
    print(f"{day.date()}: {len(names)} map ticks, {len(written)} incident 3-D files, {size:.1f} MB in {a.out}; "
          f"opens at {names[start]}")


if __name__ == "__main__":
    main()
