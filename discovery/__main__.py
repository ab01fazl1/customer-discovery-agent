"""python -m discovery --csv data/synthetic.csv
   python -m discovery --jsonl data/review_chat.jsonl --review
   python -m discovery --telegram @mygroup --api-id 123 --api-hash abc --backend ollama --review --review-backend ollama
"""
from __future__ import annotations

import argparse
import os

from .factory import build_pipeline, build_review_backend
from .report import render
from .review_step import RecordingSource, render_reviews, review_leads
from .settings import Settings
from .sources import CsvSource, JsonlSource, TelegramSource


def main() -> None:
    p = argparse.ArgumentParser(prog="discovery")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--jsonl")
    src.add_argument("--telegram", help="group username or id")
    p.add_argument("--api-id", type=int, default=int(os.environ.get("TG_API_ID", 0)))
    p.add_argument("--api-hash", default=os.environ.get("TG_API_HASH", ""))
    p.add_argument("--backend", choices=["offline", "ollama"], default="offline")
    p.add_argument("--no-cloud", action="store_true", help="disable stage-4 Claude fallback")
    p.add_argument("--limit", type=int)
    p.add_argument("--db", default="leads.db", help="SQLite path ('' to disable)")
    p.add_argument("--embed-threshold", type=float, help="stage 2 threshold for messages with a buying keyword")
    p.add_argument("--embed-threshold-no-keyword", type=float, help="stricter stage 2 threshold for the rest")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--review", action="store_true", help="run the reviewer agent on the leads (alert/queue/drop)")
    p.add_argument("--review-backend", choices=["policy", "ollama"], default="policy",
                   help="policy = fixed procedure, no model; ollama = local model")
    p.add_argument("--review-model", default="qwen2.5:7b")
    p.add_argument("--review-max", type=int, default=50, help="review at most this many top leads")
    a = p.parse_args()

    s = Settings.from_env(backend=a.backend, db_path=a.db, embed_threshold=a.embed_threshold,
                          embed_threshold_no_keyword=a.embed_threshold_no_keyword,
                          review_backend=a.review_backend, review_model=a.review_model)
    pipe = build_pipeline(s, use_cloud=False if a.no_cloud else None)

    if a.csv:
        source, name = CsvSource(a.csv, a.limit), f"csv:{a.csv}"
    elif a.jsonl:
        source, name = JsonlSource(a.jsonl, a.limit), f"jsonl:{a.jsonl}"
    else:
        source = TelegramSource(a.telegram, a.api_id, a.api_hash, limit=a.limit)
        name = f"telegram:{a.telegram}"

    recorder = RecordingSource(source) if a.review else None
    summary = pipe.run(recorder or source, name)
    print(render(summary, pipe.tracker, a.top))

    if a.review:
        pairs = review_leads(summary.leads, recorder.seen, build_review_backend(s), pipe.tracker,
                             s.review_max_steps, a.review_max)
        print(render_reviews(pairs, len(summary.leads), a.top))
        if pipe.store:
            for c, r in pairs:
                pipe.store.save_review(summary.run_id, c, r)
            pipe.store.commit()
    if pipe.store:
        pipe.store.close()


if __name__ == "__main__":
    main()
