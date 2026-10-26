"""The four funnel stages. Each one: process(candidate) -> True to keep, False to drop.

A stage knows nothing about Telegram, files, the database or other stages.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from .backends import Embedder, Scorer, ScoreResult, cosine
from .cost import CostTracker
from .models import Candidate


class Stage(ABC):
    name: str

    @abstractmethod
    def process(self, c: Candidate) -> bool: ...


# ------------------------------------------------------------------ stage 1

INTENT_KEYWORDS = re.compile(
    r"\b(buy|buying|purchase|purchasing|order|price|prices|pricing|cost|costs|how much|quote|budget|"
    r"discount|invoice|supplier|vendor|wholesale|bulk|in stock|available|recommend|selling|sells?|"
    r"looking for|looking to|where can i|deliver|delivery|shipping)\b", re.I)
URL_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$", re.I)
HAS_WORD = re.compile(r"[^\W\d_]{2,}")   # two or more letters in any script


class PreFilterStage(Stage):
    """Drops only obvious noise (too short, bare links, no words). Everything else goes on to
    stage 2; a buying keyword just earns the looser stage-2 threshold."""

    name = "1_prefilter"

    def __init__(self, min_len: int = 12):
        self.min_len = min_len

    def process(self, c: Candidate) -> bool:
        t = c.message.text.strip()
        if len(t) < self.min_len or URL_ONLY.match(t) or not HAS_WORD.search(t):
            return False
        c.keyword_hit = bool(INTENT_KEYWORDS.search(t))
        return True


# ------------------------------------------------------------------ stage 2


class EmbeddingStage(Stage):
    name = "2_embedding"

    def __init__(self, embedder: Embedder, library_path: str, threshold: float,
                 threshold_no_keyword: Optional[float] = None):
        self.embedder, self.threshold = embedder, threshold
        self.threshold_no_keyword = threshold if threshold_no_keyword is None else threshold_no_keyword
        with open(library_path, encoding="utf-8") as f:
            self.library = [embedder.embed(t) for t in json.load(f)]

    def process(self, c: Candidate) -> bool:
        v = self.embedder.embed(c.message.text)
        c.similarity = max(cosine(v, ref) for ref in self.library)
        return c.similarity >= (self.threshold if c.keyword_hit else self.threshold_no_keyword)


# ------------------------------------------------------------------ stage 3


def _apply(c: Candidate, r: ScoreResult, scorer_name: str) -> None:
    c.intent_score, c.urgency, c.deal_size = r.intent_score, r.urgency, r.deal_size
    c.confidence, c.scored_by = r.confidence, scorer_name


class LocalScorerStage(Stage):
    name = "3_local_llm"

    def __init__(self, scorer: Scorer, tracker: CostTracker, keep_above: float, fallback_below_conf: float):
        self.scorer, self.tracker = scorer, tracker
        self.keep_above, self.fb_conf = keep_above, fallback_below_conf

    def process(self, c: Candidate) -> bool:
        r = self.scorer.score(c.message.text)
        _apply(c, r, self.scorer.name)
        self.tracker.record(c, self.name, r.model, r.tokens_in, r.tokens_out)
        c.needs_fallback = r.confidence < self.fb_conf
        return r.intent_score >= self.keep_above


# ------------------------------------------------------------------ stage 4


class CloudFallbackStage(Stage):
    """Re-scores only the ambiguous ones. With scorer=None it is a pass-through that
    leaves them flagged, so a run without an API key still completes."""

    name = "4_cloud_fallback"

    def __init__(self, scorer: Optional[Scorer], tracker: CostTracker, lead_threshold: float):
        self.scorer, self.tracker, self.lead_threshold = scorer, tracker, lead_threshold
        self.skipped = 0

    def process(self, c: Candidate) -> bool:
        if not c.needs_fallback:
            return True
        if self.scorer is None:
            self.skipped += 1
            c.trace.append("fallback_wanted_but_unavailable")
            return True
        r = self.scorer.score(c.message.text)
        _apply(c, r, self.scorer.name)
        self.tracker.record(c, self.name, r.model, r.tokens_in, r.tokens_out)
        c.needs_fallback = False
        return r.intent_score >= self.lead_threshold
