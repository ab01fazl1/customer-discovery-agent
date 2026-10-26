"""Wires settings -> backends -> stages -> pipeline. The only place that picks implementations."""
from __future__ import annotations

import os
from typing import Optional

from .backends import ClaudeScorer, HashingEmbedder, HeuristicScorer, OllamaEmbedder, OllamaScorer
from .cost import CostTracker
from .pipeline import Pipeline
from .settings import Settings
from .stages import CloudFallbackStage, EmbeddingStage, LocalScorerStage, PreFilterStage
from .storage import Store


def build_review_backend(s: Settings):
    if s.review_backend == "ollama":
        from .ollama_llm import OllamaChat
        return OllamaChat(s.ollama_url, s.review_model)
    if s.review_backend == "policy":
        from .policy_llm import PolicyBackend
        return PolicyBackend()
    raise ValueError(f"unknown review backend {s.review_backend!r}")


def build_pipeline(s: Settings, use_cloud: Optional[bool] = None) -> Pipeline:
    tracker = CostTracker()

    if s.backend == "ollama":
        embedder, local_scorer = OllamaEmbedder(s.ollama_url, s.embed_model), OllamaScorer(s.ollama_url, s.llm_model)
    elif s.backend == "offline":
        embedder, local_scorer = HashingEmbedder(), HeuristicScorer()
    else:
        raise ValueError(f"unknown backend {s.backend!r}")

    threshold = s.embed_threshold if s.embed_threshold is not None else embedder.default_threshold
    threshold_nk = (s.embed_threshold_no_keyword if s.embed_threshold_no_keyword is not None
                    else embedder.default_threshold_no_keyword)

    if use_cloud is None:
        use_cloud = bool(os.environ.get("ANTHROPIC_API_KEY"))
    cloud = ClaudeScorer(s.cloud_model) if use_cloud else None

    stages = [
        PreFilterStage(s.prefilter_min_len),
        EmbeddingStage(embedder, s.intent_library, threshold, threshold_nk),
        LocalScorerStage(local_scorer, tracker, s.keep_after_local, s.fallback_below_confidence),
        CloudFallbackStage(cloud, tracker, s.lead_threshold),
    ]
    store = Store(s.db_path) if s.db_path else None
    return Pipeline(stages, tracker, s.lead_threshold, store)
