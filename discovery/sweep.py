"""Measure stages 1 and 2 on labelled data: which real leads survive, and how much noise comes along.

python -m discovery.sweep --csv data/synthetic.csv --backend ollama
python -m discovery.sweep --jsonl data/review_chat.jsonl --labels data/review_labels.jsonl --backend ollama

Embeds each message once, then tries a grid of thresholds for the two stage-2 paths
(with / without a buying keyword). "noise kept" is what stage 3 would have to process.
"""
from __future__ import annotations

import argparse
import json

from .backends import HashingEmbedder, OllamaEmbedder
from .models import Candidate
from .settings import DEFAULT_LIBRARY
from .sources import CsvSource, JsonlSource
from .stages import EmbeddingStage, PreFilterStage


def collect(messages, lead_ids, embedder, library=DEFAULT_LIBRARY, min_len=12):
    """Returns (total_leads, noise_dropped_leads, rows) with rows = (keyword_hit, similarity, is_lead)."""
    pre, emb = PreFilterStage(min_len), EmbeddingStage(embedder, library, -1.0)
    total, lost, rows = 0, 0, []
    for m in messages:
        is_lead = (m.label == 1) if lead_ids is None else (int(m.id) in lead_ids)
        total += is_lead
        c = Candidate(m)
        if not pre.process(c):
            lost += is_lead
            continue
        emb.process(c)
        rows.append((c.keyword_hit, c.similarity, is_lead))
    return total, lost, rows


def grid(base: float, step: float):
    return [round(base - step, 3), base, round(base + step, 3)], [round(base + i * step, 3) for i in range(0, 5)]


def table(total, lost, rows, kw_grid, nk_grid) -> str:
    kw = [r for r in rows if r[0]]
    nk = [r for r in rows if not r[0]]
    out = [f"{total} real leads. Noise filter dropped {lost} of them; "
           f"{len(rows)} messages reach stage 2 ({len(kw)} with a buying keyword, {len(nk)} without)."]
    out.append(f"\n{'kw thr':>7} {'no-kw thr':>10} {'leads kept':>11} {'recall':>7} {'noise kept':>11}   "
               f"{'(of which no-kw)':>16}")
    for t1 in kw_grid:
        kl = sum(1 for _, s, l in kw if l and s >= t1)
        kn = sum(1 for _, s, l in kw if not l and s >= t1)
        out.append(f"{t1:>7} {'gate':>10} {kl:>11} {kl / total:>7.2f} {kn:>11}   {'(old behaviour)':>16}")
        for t2 in nk_grid:
            nl = sum(1 for _, s, l in nk if l and s >= t2)
            nn = sum(1 for _, s, l in nk if not l and s >= t2)
            out.append(f"{t1:>7} {t2:>10} {kl + nl:>11} {(kl + nl) / total:>7.2f} {kn + nn:>11}   {nn:>16}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="discovery.sweep")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--jsonl")
    p.add_argument("--labels", help="JSONL of labelled leads (lead_id); otherwise the source's label column is used")
    p.add_argument("--backend", choices=["offline", "ollama"], default="offline")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--limit", type=int)
    a = p.parse_args()

    embedder = OllamaEmbedder(a.ollama_url, a.embed_model) if a.backend == "ollama" else HashingEmbedder()
    source = CsvSource(a.csv, a.limit) if a.csv else JsonlSource(a.jsonl, a.limit)
    lead_ids = None
    if a.labels:
        lead_ids = {json.loads(l)["lead_id"] for l in open(a.labels, encoding="utf-8") if l.strip()}
    total, lost, rows = collect(source.messages(), lead_ids, embedder)
    kw_grid, nk_grid = grid(embedder.default_threshold, 0.05)
    print(table(total, lost, rows, kw_grid, nk_grid))


if __name__ == "__main__":
    main()
