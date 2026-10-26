from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from .cost import CostTracker
from .models import Candidate
from .sources import MessageSource
from .stages import Stage
from .storage import Store


@dataclass
class RunSummary:
    total: int = 0
    stages: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    leads: list = field(default_factory=list)
    labelled: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    seconds: float = 0.0
    run_id: Optional[int] = None


class Pipeline:
    def __init__(self, stages: list[Stage], tracker: CostTracker, lead_threshold: float = 0.5,
                 store: Optional[Store] = None):
        self.stages, self.tracker, self.lead_threshold, self.store = stages, tracker, lead_threshold, store

    def process(self, c: Candidate, summary: Optional[RunSummary] = None) -> Candidate:
        """Run one message through the funnel. Usable standalone for live/streaming sources."""
        for st in self.stages:
            t0 = time.perf_counter()
            keep = st.process(c)
            if summary is not None:
                row = summary.stages.setdefault(st.name, {"in": 0, "out": 0, "seconds": 0.0,
                                                          "leads_lost": 0, "noise_dropped": 0})
                row["in"] += 1
                row["out"] += int(keep)
                row["seconds"] += time.perf_counter() - t0
                lab = c.message.label
                if not keep and lab is not None:
                    row["leads_lost" if lab == 1 else "noise_dropped"] += 1
            if not keep:
                c.dropped_at = st.name
                break
        c.is_lead = c.dropped_at is None and (c.intent_score or 0.0) >= self.lead_threshold
        return c

    def run(self, source: MessageSource, source_name: str = "") -> RunSummary:
        from .models import Candidate as C

        summary = RunSummary()
        t0 = time.perf_counter()
        run_id = self.store.start_run(source_name) if self.store else None
        summary.run_id = run_id
        for msg in source.messages():
            c = self.process(C(message=msg), summary)
            summary.total += 1
            if c.is_lead:
                summary.leads.append(c)
            if msg.label is not None:
                summary.labelled += 1
                summary.tp += int(c.is_lead and msg.label == 1)
                summary.fp += int(c.is_lead and msg.label == 0)
                summary.fn += int((not c.is_lead) and msg.label == 1)
            if self.store:
                self.store.save(run_id, c)
                if summary.total % 500 == 0:
                    self.store.commit()
        if self.store:
            self.store.commit()
        summary.seconds = time.perf_counter() - t0
        summary.leads.sort(key=lambda c: (c.intent_score or 0), reverse=True)
        return summary
