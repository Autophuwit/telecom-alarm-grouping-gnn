"""Render one day's incidents as an interactive page (web/template.html): a 3-D view of each incident, floor by
floor (sites and topology, the device stack, alarms with the model's pairs, tickets, resolution codes), a map, and a
replay of the alarms as they arrived.

    python -m aig.view --data data/synthetic --run runs/demo --day 2026-01-18 --out web/demo.html

The page is a single self-contained HTML file: the data is embedded as JSON in place of __PAGE_DATA__.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import day_nodes, day_tickets, load_corpus
from .graph import EDGE_FEATURES, build_day
from .model import DayTensors, PairGNN, Vocab, score
from .run import group_day

TEMPLATE = Path(__file__).resolve().parents[2] / "web" / "template.html"
MARKER = "__PAGE_DATA__"
HOP_LABELS = ["same site", "1 hop", "2 hops", "3 hops", "4 hops", "5 hops", "further or not linked", "not in the topology"]
FAR_H = len(HOP_LABELS) - 2
FAMILY = {"MAINS_FAIL": "POWER", "BATTERY_LOW": "POWER", "RECTIFIER_FAULT": "POWER",
          "NE_DISCONNECTED": "SITE DOWN", "CELL_DOWN": "CELL DOWN", "X2_LINK_FAIL": "RAN X2",
          "CPRI_LINK_FAIL": "RF HARDWARE", "VSWR_HIGH": "RF HARDWARE",
          "LOS": "LINK DOWN", "LINK_DOWN": "LINK DOWN", "MW_LINK_DOWN": "LINK DOWN", "MW_LINK_DEGRADED": "LINK DOWN",
          "BFD_DOWN": "IP ROUTING", "DOOR_OPEN": "ENVIRONMENT", "HIGH_TEMP": "ENVIRONMENT"}
NE_KIND = {"aggregation_router": "AGG", "cell_site_router": "CPE", "baseband": "RAN", "microwave": "OTHER",
           "rectifier": "OTHER"}
STACK_ORDER = ["AGG", "AC", "CPE", "IPRAN", "DWDM", "BB", "OTHER", "RAN"]
UNINFORMATIVE = "No fault found|Self-cleared"


def load_model(run: Path):
    ck = torch.load(run / "model.pt", weights_only=False)
    vocab = Vocab.__new__(Vocab)
    vocab.__dict__.update(ck["vocab"])
    model = PairGNN(*vocab.sizes(), n_num=3, e_dim=len(EDGE_FEATURES))
    model.load_state_dict(ck["state"])
    return model, vocab, ck["threshold"], ck["tolerances"]


def keep_links(i, j, s, per_alarm):
    """Each alarm's best `per_alarm` pairs, the union of them."""
    k = np.arange(len(i))
    ends = pd.DataFrame({"k": np.r_[k, k], "a": np.r_[i, j], "s": np.r_[s, s]})
    top = ends.sort_values(["a", "s"], ascending=[True, False]).groupby("a").head(per_alarm)
    return np.unique(top.k.to_numpy())


def build_payload(c, day, model, vocab, thr, tol, max_incidents=60, per_alarm=6):
    n = day_nodes(c, day)
    nt, E = day_tickets(c, n), build_day(n, c.links)
    s = score(model, DayTensors(n, E, vocab))
    lab, inc = group_day(len(n), E, s, nt, thr)
    N = len(n)

    # which incidents to draw: those that merge tickets, most tickets first
    tk_inc = nt.assign(g=inc[nt.node.to_numpy()])
    per = pd.DataFrame({"alarms": np.bincount(inc)})
    per["tickets"] = tk_inc.groupby("g").ticket_id.nunique().reindex(per.index, fill_value=0)
    per["sites"] = pd.Series(n.site.to_numpy()).groupby(inc).agg(lambda x: len(set(x) - {""})).reindex(per.index)
    pick = per[(per.alarms >= 2) & (per.tickets >= 2)].sort_values(["tickets", "alarms"], ascending=False).head(max_incidents)
    chosen = np.zeros(len(per), bool)
    chosen[pick.index.to_numpy()] = True

    i, j = E["i"], E["j"]
    m = (s >= thr) & chosen[inc[i]] & (inc[i] == inc[j])
    pi, pj, ps, ek = i[m], j[m], s[m], np.flatnonzero(m)
    kp = keep_links(pi, pj, ps, per_alarm)
    kp = kp[np.argsort(-ps[kp], kind="mergesort")]
    pi, pj, ps, ek = pi[kp], pj[kp], ps[kp], ek[kp]

    epoch0 = int((day - pd.Timestamp("1970-01-01")).total_seconds())
    t, clear = n.t.to_numpy(float), n.clear.to_numpy(float)
    site, dev = n.site.to_numpy(object), n.device.to_numpy(object)
    lat, lon = n.lat.to_numpy(float), n.lon.to_numpy(float)
    name, dom = n.name.to_numpy(object), n.domain.to_numpy(object)
    tick_of = nt.groupby("node").ticket_id.apply(list).to_dict()
    tolv = n.name.map(tol).to_numpy(float)

    strings, str_id = [], {}

    def sid(v):
        if v not in str_id:
            str_id[v] = len(strings)
            strings.append(v)
        return str_id[v]

    site_rows, site_id = [], {}

    def site_ref(key, la=None, lo=None, count=True):
        if key not in site_id:
            site_id[key] = len(site_rows)
            ok = la is not None and la == la
            site_rows.append({"n": key, "lat": round(float(la), 5) if ok else None,
                              "lon": round(float(lo), 5) if ok else None, "a": 0, "p": "", "r": None})
        if count:
            site_rows[site_id[key]]["a"] += 1
        return site_id[key]

    alarm_rows, alarm_id = [], {}

    def alarm_ref(x):
        x = int(x)
        if x not in alarm_id:
            alarm_id[x] = len(alarm_rows)
            lb = {"EMS": sid(dom[x])}
            if x in tick_of:
                lb["TICKETID"] = sid(" ".join(tick_of[x]))
            row = {"s": site_ref(site[x] or dev[x] or f"(no name {x})", lat[x], lon[x]), "t": int(epoch0 + t[x]),
                   "n": sid(name[x]), "c": sid(FAMILY.get(name[x], "OTHER")), "o": 1, "l": lb, "dr": None,
                   "ct": int(epoch0 + clear[x]) if clear[x] == clear[x] else None, "g": int(inc[x]), "mc": int(lab[x]),
                   "tol": int(tolv[x]) if tolv[x] == tolv[x] else None}
            alarm_rows.append(row)
        return alarm_id[x]

    edges = []
    for x, y, sc, k in zip(pi.tolist(), pj.tolist(), ps.tolist(), ek.tolist()):
        h = int(E["hop"][k])
        dc = abs(clear[x] - clear[y])
        e = {"i": alarm_ref(x), "j": alarm_ref(y), "s": round(float(sc), 3), "g": int(round(E["dt"][k])),
             "h": FAR_H if h == 9 else h, "d": round(float(E["km"][k]), 2) if E["km"][k] == E["km"][k] else None,
             "v": 0, "r": 0, "dc": int(round(dc)) if dc == dc else None}
        if e["dc"] is not None:
            pt = max(tolv[x], tolv[y]) if tolv[x] == tolv[x] and tolv[y] == tolv[y] else None
            e.update({"tol": int(pt)} if pt is not None else {"nj": 1})
        edges.append(e)
    # every alarm of a drawn incident is on the page, linked or not
    for x in np.flatnonzero(chosen[inc]).tolist():
        alarm_ref(x)

    # topology among the page's sites: transport links, and the quiet site in the middle of a 2-hop pair
    sites = c.sites.set_index("site_id")
    tl = c.links[c.links.kind != "x2"]
    adj = {}
    for a_, b_ in zip(tl.a_site_id, tl.b_site_id):
        adj.setdefault(a_, set()).add(b_)
        adj.setdefault(b_, set()).add(a_)
    alarming = set(site_id)
    quiet = set()
    for e in edges:
        if e["h"] == 2:
            sa, sb = site_rows[alarm_rows[e["i"]]["s"]]["n"], site_rows[alarm_rows[e["j"]]["s"]]["n"]
            common = adj.get(sa, set()) & adj.get(sb, set())
            quiet |= {x for x in common if x not in alarming}
    for q in sorted(quiet):
        site_ref(q, sites.lat.get(q), sites.lon.get(q), count=False)
    transit = sorted(site_id[q] for q in quiet)
    topo, seen = [], set()
    for a_ in site_id:
        for b_ in adj.get(a_, ()):
            key = (min(a_, b_), max(a_, b_))
            if b_ in site_id and key not in seen:
                seen.add(key)
                topo.append([site_id[key[0]], site_id[key[1]]])

    # the device stack: the NE or cell each alarm is raised on, and each alarming site's own NEs
    nes_t = c.nes[["ne_id", "site_id"]]
    ne_type = dict(zip(c.nes.ne_id, c.nes.ne_type))
    cell_ne = dict(zip(c.cells.cell_id, c.cells.ne_id))
    rows, idx = [], {}

    def add(key, nm, kind, s_idx, parent=None, static=False):
        if key not in idx:
            idx[key] = len(rows)
            rows.append({"n": sid(nm), "k": kind, "s": int(s_idx)})
            if parent is not None:
                rows[-1]["p"] = int(parent)
        if static:
            rows[idx[key]]["st"] = 1
        return idx[key]

    anchor = c.alarms.set_index("alarm_id").anchor_id
    for x, r in alarm_id.items():
        aid = anchor.get(n.alarm_id.iat[x])
        if not isinstance(aid, str):
            continue
        s_idx = alarm_rows[r]["s"]
        if aid in cell_ne:
            ne = cell_ne[aid]
            par = add(("N", ne), ne, NE_KIND.get(ne_type.get(ne), "OTHER"), s_idx)
            k = add(("C", aid), aid, "CELL", s_idx, par)
        elif aid in c.device_of:
            ne = c.device_of[aid]
            k = add(("N", ne), ne, NE_KIND.get(ne_type.get(ne), "OTHER"), s_idx)
        else:
            continue
        alarm_rows[r]["ne"] = k
    rank = {k: q for q, k in enumerate(STACK_ORDER)}
    for s_, g in nes_t.groupby("site_id"):
        if s_ in site_id and site_rows[site_id[s_]]["a"]:
            for ne in sorted(g.ne_id, key=lambda v: (rank.get(NE_KIND.get(ne_type.get(v), "OTHER"), 99), v))[:8]:
                add(("N", ne), ne, NE_KIND.get(ne_type.get(ne), "OTHER"), site_id[s_], static=True)
    nelinks = _ne_links(c, idx)

    # tickets carried by an alarm on the page
    in_cl = np.bincount(lab)[lab] > 1
    na = nt.groupby("ticket_id").size()
    nz = nt[~in_cl[nt.node.to_numpy()]].groupby("ticket_id").size()
    res = c.tickets.set_index("ticket_id")[["res_tier1", "res_tier2", "res_tier3"]]
    tickets = []
    for tk, g in nt[nt.node.isin(list(alarm_id))].groupby("ticket_id"):
        own = sorted(g.node.tolist(), key=lambda x: (t[x], x))
        row = {"id": tk, "s": alarm_rows[alarm_id[own[0]]]["s"], "t": int(epoch0 + t[own[0]]),
               "al": sorted(alarm_id[x] for x in own), "na": int(na[tk]), "nz": int(nz.get(tk, 0))}
        if tk in res.index:
            rc = [str(v) for v in res.loc[tk]]
            row["rc"] = rc
            if not pd.Series([rc[1]]).str.contains(UNINFORMATIVE).iat[0]:
                row["rj"] = rc[1]
        tickets.append(row)

    meta = {"label": str(day.date()), "corpus": "synthetic network", "by_incident": True,
            "model_tag": "GATv2 pair model", "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_exported": len(edges), "clustering": "average linkage (benefit 0.5) + tickets",
            "cross_site": sum(1 for e in edges if alarm_rows[e["i"]]["s"] != alarm_rows[e["j"]]["s"]),
            "hop_labels": HOP_LABELS, "score_cut": thr, "pair_window_sec": 900, "lines": "bb",
            "min_cluster": 2, "max_cluster_spread_sec": None, "max_cluster_onset_sec": None,
            "max_cluster_no_topo_km": None, "max_cluster_all_km": None,
            "coclear_tol_sec": 60, "coclear_neg_sec": 60, "coclear_default": False,
            "coclear_by_kind": "learned per alarm name", "via_hops": 0,
            "topology": {"folder": "synthetic inventory", "transit_devices": False},
            "ticket_match": {"by": "id", "time_col": "its first alarm on this page", "synthetic_below": False},
            "resolution": {"by": "res_tier2", "uninformative": UNINFORMATIVE,
                           "found": sum(1 for r in tickets if "rc" in r), "of": len(tickets)}}
    payload = {"meta": meta, "strings": strings, "sites": site_rows, "alarms": alarm_rows, "edges": edges,
               "topo": topo, "transit": transit, "tickets": tickets}
    if rows:
        payload["nes"], payload["nelinks"] = rows, nelinks
    return payload, {"incidents": int(chosen.sum()), "alarms": len(alarm_rows), "links": len(edges), "tickets": len(tickets)}


def _ne_links(c, idx):
    """[row, row, 0] for physical links between NEs on the page, [row, row, 1] for X2 neighbours."""
    have = {nm: k for (kind, nm), k in idx.items() if kind == "N"}
    by_site = c.nes.groupby("site_id").ne_id.apply(list).to_dict()
    out = set()

    def link(a, b, kind):
        if a in have and b in have and a != b:
            out.add((min(have[a], have[b]), max(have[a], have[b]), kind))

    for s_, ne_list in by_site.items():
        csr = next((x for x in ne_list if x.endswith("-CSR")), None)
        for x in ne_list:
            if csr and x != csr and not x.endswith("-PWR"):
                link(x, csr, 0)
    sites = c.sites.set_index("site_id")
    for s_, p_ in sites.parent_id.dropna().items():
        up = sites.uplink.get(s_)
        if up == "fiber":
            link(f"{s_}-CSR", f"{p_}-AGG", 0)
        elif up == "microwave":
            link(f"{s_}-MW", f"{p_}-MW", 0)
    for a_, b_ in c.links[c.links.kind == "x2"][["a_site_id", "b_site_id"]].itertuples(index=False):
        link(f"{a_}-BBU1", f"{b_}-BBU1", 1)
    return [list(x) for x in sorted(out)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--run", type=Path, default=Path("runs/demo"))
    p.add_argument("--day", help="YYYY-MM-DD; default the run's last test day")
    p.add_argument("--out", type=Path, default=Path("web/demo.html"))
    p.add_argument("--max-incidents", type=int, default=60)
    a = p.parse_args(argv)
    c = load_corpus(a.data)
    model, vocab, thr, tol = load_model(a.run)
    day = pd.Timestamp(a.day or json.loads((a.run / "run.json").read_text())["test"][-1])
    payload, info = build_payload(c, day, model, vocab, thr, tol, a.max_incidents)
    tpl = TEMPLATE.read_text(encoding="utf-8")
    if tpl.count(MARKER) != 1:
        raise SystemExit(f"{TEMPLATE} must hold {MARKER} exactly once")
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    html = tpl.replace(MARKER, blob, 1).replace("<h1>Alarm Incidents</h1>", f"<h1>Alarm Incidents · {day.date()}</h1>", 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html, encoding="utf-8")
    print(f"wrote {a.out} ({len(blob) / 1e6:.1f} MB): {info}")


if __name__ == "__main__":
    main()
