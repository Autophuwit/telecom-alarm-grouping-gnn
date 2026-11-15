"""Running live: re-cluster every few minutes, keep incident IDs stable, freeze incidents that have gone quiet.

    python -m aig.live --data data/synthetic --run runs/demo --day 2026-01-18

A batch run sees the whole day at once. Live, an operations map is refreshed every `tick` minutes from the alarms
known so far, and three things break that batch never meets:

  * a cluster label means nothing from one refresh to the next, so incidents need IDs that survive re-clustering,
    with defined outcomes when two incidents join or one breaks in two;
  * the recent window must be bounded, or every tick re-clusters the whole day;
  * re-running average linkage as alarms arrive reshuffles borderline alarms, and the map flickers.

ONE TICK at time `now`:
  1. pool      alarms with onset in the lookback window that are not frozen, plus every alarm of an incident still
               live however old (an incident is never cut by the window)
  2. score     the pool's candidate pairs: FixedScores (the batch scores, to isolate windowing and tracking) or
               LiveScorer (graph built and scored by the GNN from the pool alone, as production would)
  3. cluster   average linkage over the pool
  4. track     Tracker: carry IDs, record births, merges, splits and dissolves
  5. settle    an incident with no new alarm for `settle` seconds freezes and leaves the pool for good

The replay compares the final live incidents with the batch clustering of the same day.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import cluster
from .data import day_nodes, day_tickets, load_corpus
from .graph import build_day
from .model import DayTensors, score

NEW, ACTIVE, SETTLED = "NEW", "ACTIVE", "SETTLED"


def clock(day: str, t: float) -> str:
    """'YYYYMMDDHHMM' of t seconds after midnight of day 'YYYYMMDD'."""
    return (dt.datetime.strptime(day, "%Y%m%d") + dt.timedelta(seconds=float(t))).strftime("%Y%m%d%H%M")


# ---------------------------------------------------------------------------------------------------- incident state

@dataclass
class Incident:
    id: str
    born: float                 # tick of birth
    first_t: float              # first onset
    last_t: float               # latest onset
    members: np.ndarray         # alarm indices at its last update
    state: str = NEW
    seen: int = 1
    merged_into: str = ""
    split_from: str = ""
    why_frozen: str = ""
    changed: bool = True        # changed since the last save()


class Tracker:
    """Stable incident IDs from tick to tick.

    born      a group whose alarms belonged to no incident gets an ID minted once and never changed:
              YYYYMMDDHHMM_<first ticket>, or YYYYMMDDHHMM_A<first alarm> while no alarm has a ticket
    heir      each old incident's heir is the new group holding most of its alarms (ties: the larger group)
    continue  a group that is the heir of exactly one old incident keeps that incident's ID
    merge     a group that is the heir of several keeps the OLDEST; the others record merged_into, so old links resolve
    split     a group that is nobody's heir but holds alarms of an old incident is born with split_from
    dissolve  an old incident with no heir whose alarms are all still in the pool fell apart into singles
    settle    no new alarm for settle seconds: SETTLED and frozen; a frozen incident never changes again
    capped    onsets spanning more than max_span: frozen anyway (a guard against chains)
    """

    def __init__(self, day: str, t, ticket, key, min_size=2):
        self.day = day
        self.t = np.asarray(t, np.float64)
        self.ticket = np.asarray(ticket, object)
        self.key = np.asarray(key, object)
        self.min_size = int(min_size)
        N = len(self.t)
        self.cur = np.full(N, -1, np.int64)          # handle of each alarm's incident now (-1: alone)
        self.prev = np.full(N, -1, np.int64)         # the handle before its last change
        self.first_seen = np.full(N, np.nan)         # first tick the alarm sat in an incident
        self.frozen = np.zeros(N, bool)
        self.inc: list[Incident] = []
        self.live: set[int] = set()
        self.by_id: dict[str, int] = {}
        self.events: list[tuple] = []                # (tick, kind, id, other)

    def _mint(self, g):
        first = g[np.argmin(self.t[g])]
        tg = self.ticket[g]
        has = np.flatnonzero(tg != "")
        tail = str(tg[has[np.argmin(self.t[g][has])]]) if len(has) else f"A{self.key[first]}"
        stem = f"{clock(self.day, self.t[first])}_{tail}"
        id_, k = stem, 2
        while id_ in self.by_id:
            id_, k = f"{stem}~{k}", k + 1
        return id_

    def _new(self, now, g, split_from=""):
        h = len(self.inc)
        inc = Incident(id=self._mint(g), born=now, first_t=float(self.t[g].min()), last_t=float(self.t[g].max()),
                       members=g, split_from=split_from)
        self.inc.append(inc)
        self.by_id[inc.id] = h
        self.live.add(h)
        self.events.append((now, "split" if split_from else "born", inc.id, split_from))
        return h

    def resolve(self, id_):
        """The ID an old link should open: follows merged_into to the survivor."""
        seen = set()
        while id_ in self.by_id and self.inc[self.by_id[id_]].merged_into and id_ not in seen:
            seen.add(id_)
            id_ = self.inc[self.by_id[id_]].merged_into
        return id_

    def update(self, now, pool, lab):
        """pool: alarms recomputed this tick (never frozen ones); lab: their cluster labels. Returns this tick's events."""
        pool, lab = np.asarray(pool, np.int64), np.asarray(lab, np.int64)
        n_ev = len(self.events)
        if len(pool) and self.frozen[pool].any():
            raise ValueError("frozen alarms in the pool")
        if len(lab):
            lab = np.unique(lab, return_inverse=True)[1]
        size = np.bincount(lab) if len(lab) else np.zeros(0, np.int64)
        keep = size[lab] >= self.min_size if len(lab) else np.zeros(0, bool)
        order = np.argsort(lab[keep], kind="stable")
        ka, kl = pool[keep][order], lab[keep][order]
        gid, start = np.unique(kl, return_index=True)
        groups = np.split(ka, start[1:]) if len(ka) else []

        # overlap of each new group with the old incidents
        old = self.cur[ka]
        has = old >= 0
        gi = np.searchsorted(gid, kl[has])
        if has.any():
            pair, pc = np.unique(np.stack([old[has], gi], 1), axis=0, return_counts=True)
            po, pg = pair[:, 0], pair[:, 1]
        else:
            po = pg = pc = np.zeros(0, np.int64)
        gsize = np.array([len(g) for g in groups], np.int64)
        heir, most = {}, {}
        if len(po):
            o = np.lexsort((pg, -gsize[pg], -pc, po))           # per old incident: most alarms, then larger group
            f = np.r_[True, po[o][1:] != po[o][:-1]]
            heir = dict(zip(po[o][f].tolist(), pg[o][f].tolist()))
            o = np.lexsort((po, -pc, pg))                       # per group: the old incident it took most from
            f = np.r_[True, pg[o][1:] != pg[o][:-1]]
            most = dict(zip(pg[o][f].tolist(), po[o][f].tolist()))
        claims = {}
        for h, g in heir.items():
            claims.setdefault(g, []).append(h)

        in_pool = np.zeros(len(self.cur), bool)
        in_pool[pool] = True
        new_cur = self.cur.copy()
        new_cur[pool] = -1

        def outside(x):                                         # x's alarms that left the window (still x's)
            m = self.inc[x].members
            return m[~in_pool[m] & (self.cur[m] == x)]

        for k, g in enumerate(groups):
            cl = claims.get(k, [])
            if not cl:
                new_cur[g] = self._new(now, g, split_from=self.inc[most[k]].id if k in most else "")
                continue
            cl.sort(key=lambda x: (self.inc[x].first_t, self.inc[x].born, x))
            h, keep_ = cl[0], [outside(cl[0])]
            for x in cl[1:]:
                keep_.append(outside(x))
                self.inc[x].merged_into, self.inc[x].state, self.inc[x].changed = self.inc[h].id, "MERGED", True
                self.live.discard(x)
                self.events.append((now, "merge", self.inc[x].id, self.inc[h].id))
            inc = self.inc[h]
            m = np.unique(np.concatenate(keep_ + [g]))
            same = len(inc.members) == len(m) and np.array_equal(np.sort(inc.members), m)
            inc.members, inc.first_t, inc.last_t = m, float(self.t[m].min()), float(self.t[m].max())
            inc.seen += 1
            if inc.state == NEW:
                inc.state, inc.changed = ACTIVE, True
            inc.changed |= not same
            new_cur[m] = h

        moved = (new_cur != self.cur) & (self.cur >= 0) & (new_cur >= 0)
        self.prev[moved] = self.cur[moved]
        fresh = (new_cur >= 0) & np.isnan(self.first_seen)
        self.first_seen[fresh] = now
        self.cur = new_cur

        for h in sorted(self.live):                             # old incidents with no heir
            inc = self.inc[h]
            if h in heir or inc.born == now:
                continue
            rest = inc.members[self.cur[inc.members] == h]
            if len(rest) >= self.min_size or (len(rest) and not in_pool[inc.members].any()):
                inc.members = rest
                self._freeze(now, h, "aged")
            else:
                self.cur[rest] = -1
                self.live.discard(h)
                inc.state, inc.changed = "DISSOLVED", True
                self.events.append((now, "dissolve", inc.id, ""))
        return self.events[n_ev:]

    def _freeze(self, now, h, why):
        inc = self.inc[h]
        inc.state, inc.why_frozen, inc.changed = SETTLED, why, True
        self.live.discard(h)
        self.frozen[inc.members[self.cur[inc.members] == h]] = True
        self.events.append((now, "settle", inc.id, why))

    def settle(self, now, settle_sec, max_span=None):
        n = 0
        for h in list(self.live):
            inc = self.inc[h]
            if now - inc.last_t >= settle_sec:
                self._freeze(now, h, "quiet")
                n += 1
            elif max_span is not None and inc.last_t - inc.first_t > max_span:
                self._freeze(now, h, "capped")
                n += 1
        return n

    def live_members(self):
        hs = sorted(self.live)
        return np.concatenate([self.inc[h].members for h in hs]) if hs else np.zeros(0, np.int64)

    def final_labels(self):
        N = len(self.cur)
        return np.where(self.cur >= 0, self.cur, N + np.arange(N))

    def snapshot(self, now, labels=None):
        """The incidents a map would draw at this tick, as plain dicts."""
        labels = self.cur if labels is None else np.asarray(labels)
        idx = np.flatnonzero(labels >= 0)
        idx = idx[np.argsort(labels[idx], kind="stable")]
        hs, start = np.unique(labels[idx], return_index=True)
        out = []
        for h, m in zip(hs.tolist(), np.split(idx, start[1:]) if len(idx) else []):
            inc = self.inc[h]
            if len(m) < self.min_size:
                continue
            out.append({"id": inc.id, "state": inc.state, "first_onset": clock(self.day, inc.first_t),
                        "last_onset": clock(self.day, inc.last_t), "n_alarms": int(len(m)),
                        "tickets": sorted({str(x) for x in self.ticket[m] if x}), "split_from": inc.split_from,
                        "why_frozen": inc.why_frozen})
        return {"tick": clock(self.day, now), "n_incidents": len(out), "incidents": out}


class Display:
    """What the map shows: the tracker's placement, held back against churn. An alarm keeps the ID it is shown with
    until a NEW placement has held for `hold` ticks in a row; an alarm placed for the first time shows at once.
    hold=1 shows the tracker as it is. Only the drawing is held back; the tracker is untouched."""

    def __init__(self, n, hold=2):
        self.hold = int(hold)
        self.shown = np.full(n, -1, np.int64)
        self.pend = np.full(n, -1, np.int64)
        self.pend_n = np.zeros(n, np.int64)
        self.before = np.full(n, -1, np.int64)
        self.moves = 0                               # a shown alarm moved from one ID to another
        self.flickers = 0                            # ... and back to the one before (A -> B -> A)

    def observe(self, cur):
        s = self.shown
        same = cur == s
        self.pend[same], self.pend_n[same] = -1, 0
        new = (s == -1) & (cur >= 0)
        diff = ~same & ~new
        cont = diff & (cur == self.pend)
        self.pend_n[cont] += 1
        start = diff & ~cont
        self.pend[start], self.pend_n[start] = cur[start], 1
        go = new | (diff & (self.pend_n >= self.hold))
        mv = go & (s >= 0) & (cur >= 0)
        if mv.any():
            self.moves += int(mv.sum())
            self.flickers += int((cur[mv] == self.before[mv]).sum())
            self.before[mv] = s[mv]
        s[go] = cur[go]
        self.pend[go], self.pend_n[go] = -1, 0


class LiveDB:
    """SQLite store of the tracker's state; save() writes only what changed since the last save."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, state TEXT, born REAL, first_t REAL, last_t REAL,
        n_alarms INTEGER, merged_into TEXT, split_from TEXT, why_frozen TEXT, updated REAL);
    CREATE TABLE IF NOT EXISTS members (id TEXT, alarm TEXT, PRIMARY KEY (id, alarm));
    CREATE TABLE IF NOT EXISTS events (tick REAL, kind TEXT, id TEXT, other TEXT);
    CREATE TABLE IF NOT EXISTS ticks (tick REAL PRIMARY KEY, stats TEXT);
    """

    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(self.SCHEMA)
        self.n_events = 0

    def save(self, tr: Tracker, now, stats=None):
        c = self.db.cursor()
        for h, inc in enumerate(tr.inc):
            if not inc.changed:
                continue
            m = inc.members[tr.cur[inc.members] == h]
            c.execute("INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (inc.id, inc.state, inc.born, inc.first_t, inc.last_t, int(len(m)), inc.merged_into,
                       inc.split_from, inc.why_frozen, now))
            c.execute("DELETE FROM members WHERE id = ?", (inc.id,))
            c.executemany("INSERT INTO members VALUES (?,?)", [(inc.id, str(tr.key[a])) for a in m])
            inc.changed = False
        c.executemany("INSERT INTO events VALUES (?,?,?,?)", tr.events[self.n_events:])
        self.n_events = len(tr.events)
        c.execute("INSERT OR REPLACE INTO ticks VALUES (?,?)", (now, json.dumps(stats or {})))
        self.db.commit()


# ---------------------------------------------------------------------------------------------------- scorers

class FixedScores:
    """Pairs scored once over the whole day; a pair is usable when both alarms are in the pool."""

    def __init__(self, i, j, s):
        self.i, self.j, self.s = (np.asarray(x) for x in (i, j, s))

    def pairs(self, pool_mask, now):
        m = pool_mask[self.i] & pool_mask[self.j]
        return self.i[m], self.j[m], self.s[m]


class LiveScorer:
    """As production would: the candidate graph is built from the pool's alarms alone and scored by the GNN, so the
    most recent alarms are scored with less context than the batch run gave them."""

    def __init__(self, nodes, links, model, vocab, gcfg=None):
        self.nodes, self.links, self.model, self.vocab = nodes, links, model, vocab
        self.kw = {} if gcfg is None else {"cfg": gcfg}

    def pairs(self, pool_mask, now):
        pool = np.flatnonzero(pool_mask)
        if len(pool) < 2:
            return (np.zeros(0, np.int64),) * 2 + (np.zeros(0),)
        sub = self.nodes.iloc[pool].reset_index(drop=True)
        E = build_day(sub, self.links, **self.kw)
        s = score(self.model, DayTensors(sub, E, self.vocab))
        return pool[E["i"]], pool[E["j"]], s


# ---------------------------------------------------------------------------------------------------- engine

class LiveEngine:
    def __init__(self, t, scorer, thr, tracker, lookback, settle=3600.0, freeze=True, max_span=12 * 3600.0,
                 benefit=0.5, displays=()):
        self.t = np.asarray(t, np.float64)
        self.scorer, self.thr, self.benefit, self.tr = scorer, thr, benefit, tracker
        self.lookback, self.settle, self.freeze, self.max_span = float(lookback), float(settle), bool(freeze), max_span
        self.displays = list(displays)
        self.stats = []

    def tick(self, now):
        w0 = time.perf_counter()
        pool_mask = (self.t <= now) & (self.t > now - self.lookback) & ~self.tr.frozen
        lm = self.tr.live_members()
        pool_mask[lm[~self.tr.frozen[lm]]] = True
        pool = np.flatnonzero(pool_mask)
        gi, gj, s = self.scorer.pairs(pool_mask, now)
        w1 = time.perf_counter()
        loc = np.full(len(self.t), -1, np.int64)
        loc[pool] = np.arange(len(pool))
        lab = cluster.average_clusters(len(pool), loc[gi], loc[gj], s, self.thr, self.benefit) if len(pool) else np.zeros(0, np.int64)
        ev = self.tr.update(now, pool, lab)
        n_frozen = self.tr.settle(now, self.settle if self.freeze else np.inf, self.max_span)
        for d in self.displays:
            d.observe(self.tr.cur)
        st = {"tick": float(now), "pool": int(len(pool)), "pairs": int(len(gi)), "live": len(self.tr.live),
              "events": len(ev), "froze": int(n_frozen), "sec_score": w1 - w0, "sec": time.perf_counter() - w0}
        self.stats.append(st)
        return st


# ---------------------------------------------------------------------------------------------------- replay

def _pieces(a, b):
    """Per label of a with 2+ alarms: how many b labels it spans, its size, and its largest b piece."""
    m = np.bincount(a)[a] >= 2
    c = pd.DataFrame({"a": a[m], "b": b[m]}).groupby(["a", "b"]).size().rename("n").reset_index()
    return c.groupby("a")["n"].agg(["size", "max", "sum"])


def _ticket_pairs(lab, tt):
    m = tt != ""
    c = pd.DataFrame({"tt": tt[m], "g": lab[m]}).groupby(["tt", "g"]).size().rename("n").reset_index()
    home = c.sort_values(["tt", "n", "g"], ascending=[True, False, True], kind="mergesort").drop_duplicates("tt")
    pairs = set()
    for _, s in home.groupby("g")["tt"]:
        pairs.update(itertools.combinations(sorted(s), 2))
    return pairs


def compare(batch, live, tt, t, first_seen):
    """Live incidents against the batch clustering of the same alarms."""
    from sklearn.metrics import adjusted_rand_score
    per, per_l = _pieces(batch, live), _pieces(live, batch)
    inc = (np.bincount(batch)[batch] >= 2) | (np.bincount(live)[live] >= 2)
    pb, pl = _ticket_pairs(batch, tt), _ticket_pairs(live, tt)
    out = {"batch incidents": int(len(per)), "live incidents": int(len(per_l)),
           "split %": 100 * float((per["size"] > 1).mean()), "over-merged %": 100 * float((per_l["size"] > 1).mean()),
           "ARI vs batch": float(adjusted_rand_score(batch[inc], live[inc])),
           "ticket pairs Jaccard": len(pb & pl) / len(pb | pl) if pb | pl else float("nan")}
    # latency: from each batch incident's 2nd onset to the first tick it was on the map
    big = np.bincount(batch)[batch] >= 2
    df = pd.DataFrame({"g": batch[big], "t": t[big], "seen": first_seen[big]}).sort_values(["g", "t"], kind="mergesort")
    df["r"] = df.groupby("g").cumcount()
    lat = ((df.groupby("g").seen.min() - df[df.r == 1].set_index("g").t).clip(lower=0) / 60).dropna()
    out["latency p50 min"], out["latency p90 min"] = float(lat.median()), float(lat.quantile(.9))
    return out


def replay(day_str, n, nt, E, s_batch, thr, scorer, tick_min=5, lookback_h=3, settle_min=60, freeze=True,
           holds=(1, 2, 3), name="", db=None):
    t = n.t.to_numpy(float)
    first_tk = nt.sort_values("ticket_id").drop_duplicates("node").set_index("node").ticket_id
    tt = np.array([first_tk.get(x, "") for x in range(len(n))], object)
    batch = cluster.average_clusters(len(n), E["i"], E["j"], s_batch, thr)
    tr = Tracker(day_str, t, tt, n.alarm_id.to_numpy(object))
    disp = [Display(len(t), h) for h in holds]
    eng = LiveEngine(t, scorer, thr, tr, lookback=lookback_h * 3600, settle=settle_min * 60, freeze=freeze, displays=disp)
    store = LiveDB(db) if db else None
    ticks = list(np.arange(tick_min * 60, 86400 + tick_min * 60, tick_min * 60, dtype=float))
    for now in ticks:
        st = eng.tick(now)
        if store:
            store.save(tr, now, st)
    row = {"setting": name}
    row.update(compare(batch, tr.final_labels(), tt, t, tr.first_seen))
    n_final = max(row["live incidents"], 1)
    for d in disp:
        row[f"moves / incident (hold {d.hold})"] = d.moves / n_final
        row[f"flickers (hold {d.hold})"] = int(d.flickers)
    kinds = pd.Series([e[1] for e in tr.events], dtype=object).value_counts()
    for k in ("born", "merge", "split", "dissolve", "settle"):
        row[k] = int(kinds.get(k, 0))
    sec = np.array([x["sec"] for x in eng.stats])
    row["tick p95 sec"] = float(np.percentile(sec, 95))
    return row, tr, eng


def main(argv=None):
    from .view import load_model
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--run", type=Path, default=Path("runs/demo"))
    p.add_argument("--day", help="YYYY-MM-DD; default the run's last test day")
    p.add_argument("--tick-min", type=float, default=5)
    p.add_argument("--lookback-h", type=float, default=3)
    p.add_argument("--settle-min", type=float, default=60)
    p.add_argument("--out", type=Path, default=None, help="CSV of the settings compared (default <run>/live_<day>.csv)")
    p.add_argument("--db", type=Path, default=None, help="SQLite file the live-GNN run writes its state to")
    a = p.parse_args(argv)

    c = load_corpus(a.data)
    model, vocab, thr, _ = load_model(a.run)
    day = pd.Timestamp(a.day or json.loads((a.run / "run.json").read_text())["test"][-1])
    n = day_nodes(c, day)
    nt, E = day_tickets(c, n), build_day(n, c.links)
    s = score(model, DayTensors(n, E, vocab))
    ds = day.strftime("%Y%m%d")
    print(f"{day.date()}: {len(n):,} alarms, {len(E['i']):,} candidate pairs, threshold {thr:.2f}")

    fixed = FixedScores(E["i"], E["j"], s)
    settings = [
        ("disjoint 30-min blocks", fixed, dict(tick_min=30, lookback_h=0.5, freeze=False)),
        (f"{a.tick_min:g}-min ticks, batch scores", fixed, dict(tick_min=a.tick_min, lookback_h=a.lookback_h)),
        (f"{a.tick_min:g}-min ticks, live GNN", LiveScorer(n, c.links, model, vocab),
         dict(tick_min=a.tick_min, lookback_h=a.lookback_h, db=a.db)),
    ]
    from .evaluate import ticket_pairs
    true = pd.factorize(n.alarm_id.map(c.truth.set_index("alarm_id").incident_id))[0]

    def vs_truth(lab):
        tp = ticket_pairs(lab, nt, true)
        return {"ticket-pair precision (truth)": tp["pair_precision"], "ticket-pair recall (truth)": tp["pair_recall"]}

    batch = cluster.average_clusters(len(n), E["i"], E["j"], s, thr)
    rows = [{"setting": "batch: whole day at once", **vs_truth(batch)}]
    for name, scorer, kw in settings:
        w = time.perf_counter()
        row, tr, eng = replay(ds, n, nt, E, s, thr, scorer, settle_min=a.settle_min, name=name, **kw)
        row.update(vs_truth(tr.final_labels()))
        row["wall sec"] = time.perf_counter() - w
        rows.append(row)
        print(f"  {name}: done in {row['wall sec']:.0f}s", flush=True)
    df = pd.DataFrame(rows).set_index("setting")
    out = a.out or a.run / f"live_{day.date()}.csv"
    df.to_csv(out)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.2f}".format):
        print(df.T.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
