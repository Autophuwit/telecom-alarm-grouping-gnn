"""Read the corpus (synthetic/generate.py's schema) and cut it into days of alarm nodes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Corpus:
    alarms: pd.DataFrame          # one row per alarm, onset order
    tickets: pd.DataFrame
    ticket_alarms: pd.DataFrame   # (ticket_id, alarm_id)
    sites: pd.DataFrame
    links: pd.DataFrame
    device_of: dict               # anchor id -> network element
    truth: pd.DataFrame | None    # evaluation only; never a model input
    nes: pd.DataFrame | None = None
    cells: pd.DataFrame | None = None

    @property
    def days(self) -> list[pd.Timestamp]:
        return sorted(self.alarms.onset_time.dt.normalize().unique())


def load_corpus(root: str | Path, with_truth: bool = True) -> Corpus:
    root = Path(root)
    inv = root / "inventory"
    alarms = (pd.read_csv(root / "alarms.csv", parse_dates=["onset_time", "clear_time"])
              .sort_values(["onset_time", "alarm_id"]).reset_index(drop=True))
    cells = pd.read_csv(inv / "cells.csv")
    ports = pd.read_csv(inv / "ports.csv")
    nes = pd.read_csv(inv / "nes.csv")
    device_of = {**dict(zip(cells.cell_id, cells.ne_id)), **dict(zip(ports.port_id, ports.ne_id)),
                 **dict(zip(nes.ne_id, nes.ne_id))}
    truth = pd.read_csv(root / "truth" / "alarms.csv") if with_truth and (root / "truth").exists() else None
    return Corpus(
        alarms=alarms,
        tickets=pd.read_csv(root / "tickets.csv", parse_dates=["open_time", "restore_time", "close_time"]),
        ticket_alarms=pd.read_csv(root / "ticket_alarms.csv"),
        sites=pd.read_csv(inv / "sites.csv"),
        links=pd.read_csv(inv / "links.csv"),
        device_of=device_of,
        truth=truth,
        nes=nes,
        cells=cells,
    )


def day_nodes(c: Corpus, day: pd.Timestamp) -> pd.DataFrame:
    """The alarms that start on `day`, one node each, in onset order.

    t and clear are seconds from the day's midnight. `clear` and `ticket` are carried for labels and
    evaluation only: the model never reads them (most alarms have not cleared when it runs)."""
    a = c.alarms[c.alarms.onset_time.dt.normalize() == day].reset_index(drop=True)
    n = pd.DataFrame({"alarm_id": a.alarm_id, "name": a.alarm_name, "domain": a.domain, "severity": a.severity})
    n["t"] = (a.onset_time - day).dt.total_seconds().to_numpy()
    n["clear"] = (a.clear_time - day).dt.total_seconds().to_numpy()
    n["site"] = a.site_id.fillna("")
    # the device the alarm is raised on: from the inventory, or the root of its legacy alias
    dev = a.anchor_id.map(c.device_of)
    dev = dev.where(a.anchor_type != "site", a.anchor_id)
    n["device"] = dev.fillna(a.node_alias.str.extract(r"^([^-/]+)", expand=False)).fillna("")
    xy = c.sites.set_index("site_id")[["lat", "lon"]]
    n["lat"] = n.site.map(xy.lat)
    n["lon"] = n.site.map(xy.lon)
    n["located"] = n.lat.notna()
    return n


def day_tickets(c: Corpus, nodes: pd.DataFrame) -> pd.DataFrame:
    """One row per (node, ticket) for the day's alarms."""
    ta = c.ticket_alarms.merge(pd.DataFrame({"alarm_id": nodes.alarm_id, "node": np.arange(len(nodes))}), on="alarm_id")
    return ta[["node", "ticket_id"]].reset_index(drop=True)
