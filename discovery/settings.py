from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LIBRARY = str(Path(__file__).parent / "intent_library.json")


@dataclass
class Settings:
    backend: str = "offline"            # offline | ollama
    prefilter_min_len: int = 12
    embed_threshold: float | None = None  # None -> backend default (messages with a buying keyword)
    embed_threshold_no_keyword: float | None = None  # None -> backend default (stricter, no keyword)
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    cloud_model: str = "claude-sonnet-4-6"
    lead_threshold: float = 0.5          # final intent_score needed to count as a lead
    keep_after_local: float = 0.3        # stage 3 drops anything below this
    fallback_below_confidence: float = 0.4  # stage 3 flags these for stage 4
    db_path: str = "leads.db"
    intent_library: str = DEFAULT_LIBRARY
    review_backend: str = "policy"       # policy (no model) | ollama
    review_model: str = "qwen2.5:7b"
    review_max_steps: int = 6

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        env = os.environ
        s = cls(
            ollama_url=env.get("OLLAMA_URL", cls.ollama_url),
            embed_model=env.get("EMBED_MODEL", cls.embed_model),
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            cloud_model=env.get("CLAUDE_MODEL", cls.cloud_model),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        return s
