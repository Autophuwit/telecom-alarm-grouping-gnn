#!/usr/bin/env python
"""
Synthetic telecom network: topology, alarms, tickets and ground truth.

Builds a small mobile network (regions -> aggregation hubs -> sites, with
microwave chains and X2 neighbour relations), simulates failures on it, and
records what an operations centre would actually observe.

Scenarios (each one is a ground-truth incident):
  site_power  mains failure; the battery drains, the site and its microwave
              children drop, and neighbours raise X2 alarms
  fiber_cut   a cut behind an aggregation hub takes several sites down at once
  area_power  a grid outage across a neighbourhood; batteries drain at different
              rates and crews restore downed sites one by one ("site waves")
  radio_wave  a configuration push ripples across sites over a few minutes; it
              often lands minutes after an area_power in the same region
              (a trap that tempts a model to merge two incidents)
  hw_fault    a single board fails, may flap, and may recur hours later
              (the source of duplicate tickets)
  x2_storm    neighbour-link alarms across a large part of the network
              (a trap that tempts a model to build one giant cluster)
  noise       unrelated minor alarms, some flapping

Artefacts of observation:
  * some network elements are missing from inventory, so their alarms have no
    site, only a legacy node alias
  * the ticketing system opens one ticket per site and attaches only alarms near
    the ticket's first alarm, so later alarms of the same incident are missed
  * part of the alarm clear times are bulk-stamped from the ticket close time
  * resolution categories carry label noise; duplicate tickets are only
    sometimes marked as such

Everything under truth/ is for evaluation only and must never be a model input.

Usage:
    python generate.py --out ../data/synthetic --days 3 --seed 7
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# name: (domain, anchor_type, severity)
CATALOG = {
    "MAINS_FAIL": ("power", "ne", "major"),
    "BATTERY_LOW": ("power", "ne", "critical"),
    "RECTIFIER_FAULT": ("power", "ne", "minor"),
    "NE_DISCONNECTED": ("ran", "ne", "critical"),
    "CELL_DOWN": ("ran", "cell", "major"),
    "CPRI_LINK_FAIL": ("ran", "ne", "major"),
    "X2_LINK_FAIL": ("ran", "ne", "minor"),
    "VSWR_HIGH": ("ran", "cell", "minor"),
    "LOS": ("transport", "port", "critical"),
    "LINK_DOWN": ("transport", "port", "major"),
    "MW_LINK_DOWN": ("transport", "ne", "major"),
    "MW_LINK_DEGRADED": ("transport", "ne", "minor"),
    "BFD_DOWN": ("transport", "ne", "minor"),
    "DOOR_OPEN": ("environment", "site", "warning"),
    "HIGH_TEMP": ("environment", "site", "minor"),
}
MAJOR = {"major", "critical"}

# incident kind -> (tier 1, tier 2, tier 3) resolution written by the field crew
RESOLUTION = {
    "site_power": ("Power", "Commercial power", "Mains failure"),
    "area_power": ("Power", "Commercial power", "Grid outage"),
    "fiber_cut": ("Transport", "Fiber", "Fiber cut repaired"),
    "radio_wave": ("RAN", "Software", "Configuration rollback"),
    "hw_fault": ("RAN", "Hardware", "Board replaced"),
    "x2_storm": ("Core", "Signalling", "Neighbour relation"),
    "noise": ("Other", "No fault found", "Self-cleared"),
}


@dataclass
class Config:
    seed: int = 7
    start: str = "2026-01-05 00:00:00"
    days: float = 3.0
    # topology
    n_regions: int = 4
    hubs_per_region: int = 3
    sites_per_region: int = 60
    mw_prob: float = 0.35            # chance a site hangs off another site by microwave
    max_mw_depth: int = 3
    x2_neighbours: int = 4
    ne_unplaced_frac: float = 0.4    # NEs missing from inventory
    # scenarios per day across the whole network
    rate_site_power: float = 10.0
    rate_fiber_cut: float = 2.0
    rate_area_power: float = 2.0
    rate_radio_wave: float = 2.0
    rate_hw_fault: float = 10.0
    rate_x2_storm: float = 0.5
    noise_per_site_day: float = 6.0
    flap_prob: float = 0.15
    coincident_wave_prob: float = 0.4
    recur_prob: float = 0.3
    x2_on_site_down_prob: float = 0.6
    late_symptom_prob: float = 0.3   # symptom raised 10-60 min late (retries, OSS resync)
    # ticketing and observation
    ticket_attach_window_min: float = 8.0
    dup_ticket_prob: float = 0.15
    dup_marked_prob: float = 0.3     # duplicates the crew actually marks as such
    bulk_clear_prob: float = 0.35
    missing_clear_prob: float = 0.05
    label_noise: float = 0.1


class Topology:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.sites: dict[str, dict] = {}
        self.nes: dict[str, dict] = {}
        self.cells: dict[str, str] = {}          # cell -> NE
        self.ports: dict[str, str] = {}          # port -> NE
        self.links: list[tuple[str, str, str]] = []
        self.ran = defaultdict(list)             # site -> RAN NEs
        self.ne_cells = defaultdict(list)
        self.tx: dict[str, str] = {}
        self.pwr: dict[str, str] = {}
        self.mw: dict[str, str] = {}
        self.uplink_port: dict[str, str] = {}    # site's own uplink port
        self.hub_port: dict[str, str] = {}       # hub-side port facing a fibre site
        self.children = defaultdict(list)
        self.neighbours = defaultdict(set)
        self.region_sites = defaultdict(list)
        self.region_hubs = defaultdict(list)
        self.unplaced: set[str] = set()
        self.alias: dict[str, str] = {}
        self._build()

    def xy(self, s: str) -> np.ndarray:
        return np.array([self.sites[s]["lon"], self.sites[s]["lat"]])

    def _add_ne(self, ne, site, domain, ne_type):
        self.nes[ne] = dict(ne_id=ne, site_id=site, domain=domain, ne_type=ne_type)

    def _build(self):
        c, rng = self.cfg, self.rng
        n = 0
        for r in range(c.n_regions):
            region = f"R{r + 1}"
            cx, cy = 100.0 + 1.5 * r, 14.0 + rng.uniform(-1, 1)
            for k in range(c.hubs_per_region):
                hub = f"H{r + 1}{k + 1:02d}"
                self.sites[hub] = dict(site_id=hub, region=region, kind="hub",
                                       lon=cx + rng.normal(0, .15), lat=cy + rng.normal(0, .15),
                                       uplink=None, parent_id=None, depth=0)
                self._add_ne(f"{hub}-AGG", hub, "transport", "aggregation_router")
                self.tx[hub] = f"{hub}-AGG"
                self.region_hubs[region].append(hub)
            new = []
            for _ in range(c.sites_per_region):
                n += 1
                s = f"S{n:04d}"
                self.sites[s] = dict(site_id=s, region=region, kind="site",
                                     lon=cx + rng.normal(0, .35), lat=cy + rng.normal(0, .35),
                                     uplink=None, parent_id=None, depth=None)
                for i in range(1 + int(rng.random() < .3)):
                    ne = f"{s}-BBU{i + 1}"
                    self._add_ne(ne, s, "ran", "baseband")
                    self.ran[s].append(ne)
                    for j in range(int(rng.integers(3, 7))):
                        cell = f"{ne}-C{j + 1}"
                        self.cells[cell] = ne
                        self.ne_cells[ne].append(cell)
                self._add_ne(f"{s}-CSR", s, "transport", "cell_site_router")
                self.tx[s] = f"{s}-CSR"
                self.uplink_port[s] = f"{s}-CSR/P1"
                self.ports[self.uplink_port[s]] = self.tx[s]
                self._add_ne(f"{s}-PWR", s, "power", "rectifier")
                self.pwr[s] = f"{s}-PWR"
                self.region_sites[region].append(s)
                new.append(s)
            self._wire(region, new, cx, cy)

        for ne, row in self.nes.items():
            if self.sites[row["site_id"]]["kind"] == "site" and rng.random() < c.ne_unplaced_frac:
                self.unplaced.add(ne)
                self.alias[ne] = f"NE{rng.integers(100000, 999999)}"
            else:
                self.alias[ne] = ne

    def _wire(self, region, sites, cx, cy):
        c, rng = self.cfg, self.rng
        hubs = self.region_hubs[region]
        order = sorted(sites, key=lambda s: np.hypot(self.sites[s]["lon"] - cx, self.sites[s]["lat"] - cy))
        done: list[str] = []
        for s in order:
            p_xy = self.xy(s)
            cands = [p for p in done if self.sites[p]["depth"] < c.max_mw_depth
                     and np.linalg.norm(self.xy(p) - p_xy) < .4]
            if cands and rng.random() < c.mw_prob:
                p = min(cands, key=lambda q: np.linalg.norm(self.xy(q) - p_xy))
                kind = "microwave"
                for x in (s, p):
                    if x not in self.mw:
                        self.mw[x] = f"{x}-MW"
                        self._add_ne(self.mw[x], x, "transport", "microwave")
            else:
                p = min(hubs, key=lambda h: np.linalg.norm(self.xy(h) - p_xy))
                kind = "fiber"
                port = f"{self.tx[p]}/P{len(self.children[p]) + 1}"
                self.ports[port] = self.tx[p]
                self.hub_port[s] = port
            self.sites[s].update(parent_id=p, uplink=kind, depth=self.sites[p]["depth"] + 1)
            self.children[p].append(s)
            self.links.append((s, p, kind))
            done.append(s)

        pts = np.array([self.xy(s) for s in sites])
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        for i, s in enumerate(sites):
            for j in np.argsort(d[i])[1:c.x2_neighbours + 1]:
                t = sites[j]
                if t not in self.neighbours[s]:
                    self.neighbours[s].add(t)
                    self.neighbours[t].add(s)
                    self.links.append((s, t, "x2"))

    def descendants(self, s: str) -> list[str]:
        out, stack = [], list(self.children[s])
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(self.children[x])
        return out

    def tables(self) -> dict[str, pd.DataFrame]:
        """Inventory as the operator sees it: unplaced NEs are absent."""
        keep = lambda ne: ne not in self.unplaced  # noqa: E731
        return dict(
            sites=pd.DataFrame(self.sites.values()).drop(columns="depth"),
            nes=pd.DataFrame([r for ne, r in self.nes.items() if keep(ne)]),
            cells=pd.DataFrame([(x, ne) for x, ne in self.cells.items() if keep(ne)], columns=["cell_id", "ne_id"]),
            ports=pd.DataFrame([(x, ne) for x, ne in self.ports.items() if keep(ne)], columns=["port_id", "ne_id"]),
            links=pd.DataFrame(self.links, columns=["a_site_id", "b_site_id", "kind"]),
        )


class Simulator:
    def __init__(self, cfg: Config, topo: Topology, rng: np.random.Generator):
        self.c, self.t, self.rng = cfg, topo, rng
        self.horizon = cfg.days * 1440.0            # minutes
        self.alarms: list[dict] = []
        self.incidents: dict[str, dict] = {}
        self.regions = list(topo.region_sites)
        self.all_sites = [s for r in self.regions for s in topo.region_sites[r]]

    # helpers -----------------------------------------------------------
    def u(self, a, b) -> float:
        return float(self.rng.uniform(a, b))

    def late(self) -> float:
        return self.u(10, 60) if self.rng.random() < self.c.late_symptom_prob else 0.0

    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def sample(self, seq, k):
        k = min(len(seq), k)
        return [seq[i] for i in self.rng.choice(len(seq), k, replace=False)]

    def new_incident(self, kind, recurrence_of=None) -> str:
        inc = f"I{len(self.incidents) + 1:06d}"
        self.incidents[inc] = dict(incident_id=inc, kind=kind, recurrence_of=recurrence_of)
        return inc

    def emit(self, name, anchor, onset, clear, inc, role="symptom"):
        domain, atype, sev = CATALOG[name]
        ne = {"cell": self.t.cells.get(anchor), "port": self.t.ports.get(anchor),
              "ne": anchor, "site": None}[atype]
        site = self.t.nes[ne]["site_id"] if ne else anchor
        self.alarms.append(dict(onset=onset, clear=max(clear, onset + .2), name=name, domain=domain,
                                severity=sev, anchor_type=atype, anchor=anchor, ne=ne, site=site,
                                incident_id=inc, role=role))

    def site_down(self, s, t_down, t_up, inc, skip=frozenset()):
        """A site loses service: its RAN, its microwave children, and X2 at its neighbours."""
        u, late = self.u, self.late
        for ne in self.t.ran[s]:
            self.emit("NE_DISCONNECTED", ne, t_down + u(1, 3), t_up + u(0, 3), inc)
            for cell in self.t.ne_cells[ne]:
                self.emit("CELL_DOWN", cell, t_down + u(0, .5) + late(), t_up + u(0, 3), inc)
        for nb in self.t.neighbours[s]:
            if nb not in skip and self.rng.random() < self.c.x2_on_site_down_prob:
                self.emit("X2_LINK_FAIL", self.t.ran[nb][0], t_down + u(0, 2) + late(), t_up + u(0, 5), inc)
        for d in self.t.descendants(s):
            if d in skip:
                continue
            lag = u(1, 8)
            self.emit("MW_LINK_DOWN", self.t.mw[d], t_down + u(0, .5), t_up + lag, inc)
            for ne in self.t.ran[d]:
                self.emit("NE_DISCONNECTED", ne, t_down + u(1, 3), t_up + lag + u(0, 3), inc)
                for cell in self.t.ne_cells[ne]:
                    self.emit("CELL_DOWN", cell, t_down + u(.2, 1) + late(), t_up + lag + u(0, 3), inc)

    # scenarios ---------------------------------------------------------
    def site_power(self, t0):
        s, inc = self.pick(self.all_sites), self.new_incident("site_power")
        dur, batt = float(self.rng.lognormal(np.log(90), .8)), self.u(60, 240)
        self.emit("MAINS_FAIL", self.t.pwr[s], t0, t0 + dur, inc, "root")
        if dur > .7 * batt:
            self.emit("BATTERY_LOW", self.t.pwr[s], t0 + .7 * batt, t0 + dur + self.u(1, 5), inc)
        if dur > batt:
            self.site_down(s, t0 + batt, t0 + dur + self.u(2, 15), inc)

    def fiber_cut(self, t0):
        region = self.pick(self.regions)
        hub = self.pick([h for h in self.t.region_hubs[region] if self.t.children[h]])
        hit = self.sample(self.t.children[hub], int(self.rng.integers(2, 9)))
        inc, t_up = self.new_incident("fiber_cut"), t0 + self.u(60, 360)
        self.emit("LOS", self.t.hub_port[hit[0]], t0, t_up + self.u(0, 1), inc, "root")
        for s in hit:
            self.emit("LINK_DOWN", self.t.uplink_port[s], t0 + self.u(0, .3), t_up + self.u(0, 2), inc)
            self.site_down(s, t0 + self.u(.5, 2), t_up + self.u(1, 6), inc, skip=set(hit))

    def area_power(self, t0):
        region = self.pick(self.regions)
        pool = self.t.region_sites[region]
        centre = self.t.xy(self.pick(pool))
        hit = sorted(pool, key=lambda s: np.linalg.norm(self.t.xy(s) - centre))[:int(self.rng.integers(8, 36))]
        inc = self.new_incident("area_power")
        grid_back = t0 + float(self.rng.lognormal(np.log(180), .6))
        onset = {s: t0 + self.u(0, 3) for s in hit}
        root = min(onset, key=onset.get)
        down = []
        for s in hit:
            on, batt = onset[s], self.u(45, 300)
            self.emit("MAINS_FAIL", self.t.pwr[s], on, grid_back + self.u(0, 5), inc,
                      "root" if s == root else "symptom")
            if on + .7 * batt < grid_back:
                self.emit("BATTERY_LOW", self.t.pwr[s], on + .7 * batt, grid_back + self.u(1, 5), inc)
            if on + batt < grid_back:
                down.append((s, on + batt))
        crews = [grid_back, grid_back]
        self.rng.shuffle(down)
        for s, t_down in down:
            if self.rng.random() < .5:                  # comes back by itself with the grid
                t_up = grid_back + self.u(5, 20)
            else:                                       # needs a crew: restored site by site
                k = int(np.argmin(crews))
                crews[k] += self.u(15, 45)
                t_up = crews[k]
            self.site_down(s, t_down, t_up, inc, skip=set(hit))
        if self.rng.random() < self.c.coincident_wave_prob:
            self.radio_wave(t0 + self.u(5, 20), region)

    def radio_wave(self, t0, region=None):
        pool = self.t.region_sites[region or self.pick(self.regions)]
        hit, inc, t = self.sample(pool, int(self.rng.integers(5, 26))), self.new_incident("radio_wave"), t0
        for i, s in enumerate(hit):
            ne, dur = self.t.ran[s][0], self.u(3, 15)
            self.emit("CPRI_LINK_FAIL", ne, t, t + dur, inc, "root" if i == 0 else "symptom")
            for cell in self.t.ne_cells[ne]:
                if self.rng.random() < .7:
                    self.emit("CELL_DOWN", cell, t + self.u(0, .3), t + dur + self.u(0, 1), inc)
            t += self.u(.5, 2)

    def hw_fault(self, t0, site=None, ne=None, recurrence_of=None):
        s = site or self.pick(self.all_sites)
        ne = ne or self.pick(self.t.ran[s])
        inc, t, role = self.new_incident("hw_fault", recurrence_of), t0, "root"
        if self.rng.random() < 2 * self.c.flap_prob:
            for _ in range(int(self.rng.integers(2, 6))):
                d = self.u(.5, 4)
                self.emit("CPRI_LINK_FAIL", ne, t, t + d, inc, role)
                role, t = "symptom", t + d + self.u(1, 6)
        dur = float(self.rng.lognormal(np.log(240), .7))
        self.emit("CPRI_LINK_FAIL", ne, t, t + dur, inc, role)
        for cell in self.sample(self.t.ne_cells[ne], int(self.rng.integers(1, 4))):
            self.emit("CELL_DOWN", cell, t + self.u(0, .5) + self.late(), t + dur + self.u(0, 2), inc)
        if recurrence_of is None and self.rng.random() < self.c.recur_prob:
            self.hw_fault(t + dur + self.u(60, 480), s, ne, inc)

    def x2_storm(self, t0):
        regions = self.sample(self.regions, int(self.rng.integers(1, 3)))
        pool = [s for r in regions for s in self.t.region_sites[r]]
        hit = self.sample(pool, int(len(pool) * self.u(.5, .9)))
        inc = self.new_incident("x2_storm")
        hub = self.pick(self.t.region_hubs[regions[0]])
        self.emit("BFD_DOWN", self.t.tx[hub], t0, t0 + self.u(30, 120), inc, "root")
        for s in hit:
            on = t0 + self.u(0, 60)
            self.emit("X2_LINK_FAIL", self.t.ran[s][0], on, on + self.u(2, 90), inc)

    def noise(self):
        names = ["DOOR_OPEN", "HIGH_TEMP", "VSWR_HIGH", "BFD_DOWN", "MW_LINK_DEGRADED", "RECTIFIER_FAULT"]
        n = self.rng.poisson(self.c.noise_per_site_day * self.c.days * len(self.all_sites))
        for _ in range(n):
            s, t, name = self.pick(self.all_sites), self.u(0, self.horizon), self.pick(names)
            if name == "MW_LINK_DEGRADED" and s not in self.t.mw:
                name = "BFD_DOWN"
            anchor = {"DOOR_OPEN": s, "HIGH_TEMP": s, "BFD_DOWN": self.t.tx[s],
                      "MW_LINK_DEGRADED": self.t.mw.get(s), "RECTIFIER_FAULT": self.t.pwr[s],
                      "VSWR_HIGH": self.pick(self.t.ne_cells[self.pick(self.t.ran[s])])}[name]
            inc, role = self.new_incident("noise"), "root"
            reps = int(self.rng.integers(2, 7)) if self.rng.random() < self.c.flap_prob else 1
            for _ in range(reps):
                d = float(self.rng.lognormal(np.log(20), 1)) if reps == 1 else self.u(.5, 5)
                self.emit(name, anchor, t, t + d, inc, role)
                role, t = "symptom", t + d + self.u(2, 10)

    def run(self) -> pd.DataFrame:
        c = self.c
        for fn, rate in [(self.site_power, c.rate_site_power), (self.fiber_cut, c.rate_fiber_cut),
                         (self.area_power, c.rate_area_power), (self.radio_wave, c.rate_radio_wave),
                         (self.hw_fault, c.rate_hw_fault), (self.x2_storm, c.rate_x2_storm)]:
            for t0 in np.sort(self.rng.uniform(0, self.horizon, self.rng.poisson(rate * c.days))):
                fn(float(t0))
        self.noise()
        df = pd.DataFrame(self.alarms)
        df = df[df.onset < self.horizon].sort_values("onset", kind="stable").reset_index(drop=True)
        df.insert(0, "alarm_id", [f"A{i:07d}" for i in range(len(df))])
        df["incident_kind"] = df.incident_id.map(lambda i: self.incidents[i]["kind"])
        return df


def make_tickets(a: pd.DataFrame, cfg: Config, rng: np.random.Generator):
    """One ticket per site (or per legacy alias when the site is unknown), opened by the first
    major alarm; only alarms within the attach window join it."""
    u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    kinds = list(RESOLUTION)
    tickets, links = [], []

    def close_of(restore):
        if rng.random() < .2:                       # closed in bulk at shift end (08:00 / 20:00)
            return np.ceil((restore - 480) / 720) * 720 + 480
        return np.ceil((restore + u(10, 14)) / 5) * 5

    def open_ticket(key, placed, t_open, restore, ids, dup):
        kind = a.loc[ids, "incident_kind"].value_counts().idxmax()
        if rng.random() < cfg.label_noise:
            kind = kinds[int(rng.integers(len(kinds)))]
        tid = f"T{len(tickets) + 1:06d}"
        tickets.append(dict(
            ticket_id=tid, open_time=t_open, site_id=key if placed else None,
            node_alias=None if placed else key, restore_time=restore, close_time=close_of(restore),
            res_tier1=RESOLUTION[kind][0], res_tier2=RESOLUTION[kind][1], res_tier3=RESOLUTION[kind][2],
            resolution_note="DUPLICATE JOB" if dup and rng.random() < cfg.dup_marked_prob else None))
        links.extend((tid, aid, dup) for aid in a.loc[ids, "alarm_id"])

    W = cfg.ticket_attach_window_min
    for key, g in a.groupby("ticket_key", sort=False):
        on, cl = g.onset.to_numpy(), g.clear.to_numpy()
        major, ids, placed = g.severity.isin(MAJOR).to_numpy(), g.index.to_numpy(), bool(g.placed.iloc[0])
        i, n = 0, len(g)
        while i < n:
            if not major[i]:
                i += 1
                continue
            m = (on >= on[i] - 2) & (on <= on[i] + W)
            restore = cl[m].max()
            open_ticket(key, placed, on[i] + u(2, 20), restore, ids[m], dup=False)
            j = int(np.searchsorted(on, on[i] + W, side="right"))
            while j < n and on[j] < restore:            # the site is still being worked on
                if major[j] and rng.random() < cfg.dup_ticket_prob:
                    m2 = (on >= on[j]) & (on <= on[j] + 5)
                    open_ticket(key, placed, on[j] + u(2, 20), cl[m2].max(), ids[m2], dup=True)
                j += 1
            i = j
    t = pd.DataFrame(tickets)
    ta = pd.DataFrame(links, columns=["ticket_id", "alarm_id", "is_dup"])
    return t, ta


def observe(alarms: pd.DataFrame, topo: Topology, cfg: Config, rng: np.random.Generator):
    a = alarms.copy()
    a["placed"] = [ne is None or ne not in topo.unplaced for ne in a["ne"]]
    a["node_alias"] = [anc if p else topo.alias[ne] + anc[len(ne):]
                       for anc, ne, p in zip(a.anchor, a["ne"], a.placed)]
    a["ticket_key"] = [s if p else topo.alias[ne] for s, ne, p in zip(a.site, a["ne"], a.placed)]

    tickets, ta = make_tickets(a, cfg, rng)

    # observed clear: bulk-stamped from the first ticket's close time, sometimes missing
    first = ta[~ta.is_dup].drop_duplicates("alarm_id").merge(tickets[["ticket_id", "close_time"]], on="ticket_id")
    bulk = first[rng.random(len(first)) < cfg.bulk_clear_prob].set_index("alarm_id").close_time
    obs_clear = a.clear.copy()
    is_bulk = a.alarm_id.isin(bulk.index)
    obs_clear[is_bulk] = a.loc[is_bulk, "alarm_id"].map(bulk).to_numpy()
    obs_clear[rng.random(len(a)) < cfg.missing_clear_prob] = np.nan

    observed = pd.DataFrame(dict(
        alarm_id=a.alarm_id, onset=a.onset, clear=obs_clear, alarm_name=a["name"], domain=a.domain,
        severity=a.severity, anchor_type=a.anchor_type.where(a.placed),
        anchor_id=a.anchor.where(a.placed), node_alias=a.node_alias, site_id=a.site.where(a.placed)))
    truth = pd.DataFrame(dict(
        alarm_id=a.alarm_id, incident_id=a.incident_id, role=a.role, true_site_id=a.site,
        true_ne_id=a["ne"], true_clear=a.clear, clear_is_bulk=is_bulk))
    return observed, tickets, ta.drop(columns="is_dup"), truth


def to_time(start: pd.Timestamp, minutes: pd.Series) -> pd.Series:
    return start + pd.to_timedelta(np.round(minutes.astype(float) * 60), unit="s")


def generate(cfg: Config, out: Path):
    rng = np.random.default_rng(cfg.seed)
    topo = Topology(cfg, rng)
    sim = Simulator(cfg, topo, rng)
    raw = sim.run()
    alarms, tickets, ticket_alarms, truth = observe(raw, topo, cfg, rng)

    start = pd.Timestamp(cfg.start)
    for df, cols in [(alarms, ["onset", "clear"]), (truth, ["true_clear"]),
                     (tickets, ["open_time", "restore_time", "close_time"])]:
        for col in cols:
            df[col] = to_time(start, df[col])
    alarms = alarms.rename(columns={"onset": "onset_time", "clear": "clear_time"})
    truth = truth.rename(columns={"true_clear": "true_clear_time"})

    g = truth.merge(alarms[["alarm_id", "onset_time"]], on="alarm_id").groupby("incident_id")
    incidents = pd.DataFrame(sim.incidents.values()).set_index("incident_id").join(pd.DataFrame(dict(
        start_time=g.onset_time.min(), end_time=g.true_clear_time.max(), n_alarms=g.size(),
        n_sites=g.true_site_id.nunique(),
        root_alarm_id=truth[truth.role == "root"].groupby("incident_id").alarm_id.first())), how="inner")

    (out / "inventory").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for name, df in topo.tables().items():
        df.to_csv(out / "inventory" / f"{name}.csv", index=False)
    alarms.to_csv(out / "alarms.csv", index=False)
    tickets.to_csv(out / "tickets.csv", index=False)
    ticket_alarms.to_csv(out / "ticket_alarms.csv", index=False)
    truth.to_csv(out / "truth" / "alarms.csv", index=False)
    incidents.reset_index().to_csv(out / "truth" / "incidents.csv", index=False)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    summarise(alarms, tickets, ticket_alarms, truth, incidents)


def summarise(alarms, tickets, ticket_alarms, truth, incidents):
    print(f"alarms {len(alarms):,} | no site {alarms.site_id.isna().mean():.0%} | "
          f"bulk-stamped clear {truth.clear_is_bulk.mean():.0%} | missing clear {alarms.clear_time.isna().mean():.0%}")
    k = incidents.groupby("kind").agg(incidents=("n_alarms", "size"), median_alarms=("n_alarms", "median"),
                                      max_alarms=("n_alarms", "max"), max_sites=("n_sites", "max"))
    print(k.to_string())
    ticketed = truth[truth.incident_id.isin(
        truth[truth.alarm_id.isin(ticket_alarms.alarm_id)].incident_id)]
    recall = ticketed.alarm_id.isin(ticket_alarms.alarm_id).mean()
    print(f"tickets {len(tickets):,} | placeless {tickets.site_id.isna().mean():.0%} | "
          f"marked DUPLICATE JOB {tickets.resolution_note.notna().sum()} | "
          f"alarms of ticketed incidents that sit on a ticket {recall:.0%}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "synthetic")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--days", type=float, default=Config.days)
    p.add_argument("--regions", type=int, default=Config.n_regions)
    p.add_argument("--sites-per-region", type=int, default=Config.sites_per_region)
    args = p.parse_args()
    cfg = Config(seed=args.seed, days=args.days, n_regions=args.regions, sites_per_region=args.sites_per_region)
    generate(cfg, args.out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
