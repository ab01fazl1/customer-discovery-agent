"""Swappable engines behind the stages.

offline : HashingEmbedder + HeuristicScorer (no network, no models; for tests/dev)
ollama  : OllamaEmbedder + OllamaScorer     (real local models)
cloud   : ClaudeScorer                      (stage 4 fallback)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol

# ----------------------------------------------------------------- embedders


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


_STOP = set("a an the i we you he she it is are was were be to of and or in on at for my our your this that "
            "with so just im me can do does did have has had if but not no yes ok".split())


class HashingEmbedder:
    """Dependency-free stand-in: signed hashed bag of words + bigrams."""

    default_threshold = 0.20
    default_threshold_no_keyword = 0.30

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        toks = [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v


class OllamaEmbedder:
    default_threshold = 0.55             # measured with nomic-embed-text on synthetic data
    default_threshold_no_keyword = 0.60  # measured: recovers vague leads with no extra noise

    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/"), model

    def embed(self, text: str) -> list[float]:
        return _post(f"{self.url}/api/embeddings", {"model": self.model, "prompt": text})["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ------------------------------------------------------------------- scorers


@dataclass
class ScoreResult:
    intent_score: float
    urgency: str = "low"        # low | medium | high
    deal_size: str = "unknown"  # unknown | small | medium | large
    confidence: float = 1.0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class Scorer(Protocol):
    name: str

    def score(self, text: str) -> ScoreResult: ...


SCORING_PROMPT = (
    "You score chat messages for BUYING INTENT (someone who wants to purchase a product or service).\n"
    "Reply with JSON only: "
    '{"intent_score": 0.0-1.0, "urgency": "low|medium|high", "deal_size": "unknown|small|medium|large", '
    '"confidence": 0.0-1.0}.\n'
    "Casual price complaints, jokes, past purchases and off-topic questions are NOT buying intent. "
    "Lower confidence when the message is ambiguous."
)


def _parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0)) if m else {}


def _result(d: dict, tin: int, tout: int, model: str) -> ScoreResult:
    def clamp(x, default):
        try:
            return max(0.0, min(1.0, float(x)))
        except (TypeError, ValueError):
            return default

    urg = d.get("urgency", "low")
    size = d.get("deal_size", "unknown")
    return ScoreResult(
        intent_score=clamp(d.get("intent_score"), 0.0),
        urgency=urg if urg in ("low", "medium", "high") else "low",
        deal_size=size if size in ("unknown", "small", "medium", "large") else "unknown",
        confidence=clamp(d.get("confidence"), 0.0),  # unparseable -> zero confidence -> goes to fallback
        tokens_in=tin, tokens_out=tout, model=model,
    )


class OllamaScorer:
    name = "ollama"

    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/"), model

    def score(self, text: str) -> ScoreResult:
        r = _post(f"{self.url}/api/chat", {
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SCORING_PROMPT},
                         {"role": "user", "content": text}],
        })
        try:
            d = _parse_json(r["message"]["content"])
        except (ValueError, KeyError):
            d = {}
        return _result(d, r.get("prompt_eval_count", 0), r.get("eval_count", 0), self.model)


class ClaudeScorer:
    name = "claude"

    def __init__(self, model: str):
        import anthropic  # lazy: optional dependency

        self.client, self.model = anthropic.Anthropic(), model

    def score(self, text: str) -> ScoreResult:
        resp = self.client.messages.create(
            model=self.model, max_tokens=150,
            system=[{"type": "text", "text": SCORING_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": text}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        try:
            d = _parse_json(raw)
        except ValueError:
            d = {}
        return _result(d, resp.usage.input_tokens, resp.usage.output_tokens, self.model)


# ---- offline heuristic scorer (dev/test stand-in for the LLM) ----

_STRONG = re.compile(
    r"looking to (buy|get|order|purchase)|want to (buy|order|purchase)|need to (buy|order)|ready to (buy|order|purchase)|"
    r"where can i (buy|get|order)|need a quote|send (me )?(a )?(quote|price|prices|invoice)|who sells|"
    r"can (anyone|someone) recommend (a )?(supplier|vendor|seller)|(i'll|i will) take it|looking for a (vendor|supplier)|interested in buying|wholesale|in bulk", re.I)
_MEDIUM = re.compile(r"how much (does|do|for|is|would)|\bprice\b|\bcost\b|in stock|available|recommend|\bquote\b|\bbudget\b|\bdiscount\b|\border\b", re.I)
_NEGATIVE = re.compile(r"\blol\b|haha|\bjoke\b|crazy|insane|\bmeme\b|i bought|i got (a|an|my)|last (year|month|week)|\bused to\b|"
                       r"dinner|lunch|movie|pizza|groceries|coffee|living|\bsalary\b|\brent\b", re.I)
_HIGH = re.compile(r"today|asap|urgent|right now|tomorrow|this week|immediately", re.I)
_MED = re.compile(r"next week|soon|this month|next month", re.I)
_LARGE = re.compile(r"\b(\d{2,})\s*(units|pcs|pieces|licenses|seats|boxes|desks|panels|laptops)\b|bulk|wholesale|company|whole team|office|\$\s?\d{4,}", re.I)
_SMALL = re.compile(r"\b(one|1|a single|a couple|2|3)\b", re.I)


class HeuristicScorer:
    name = "heuristic"

    def score(self, text: str) -> ScoreResult:
        s = 0.0
        strong, medium, neg = bool(_STRONG.search(text)), bool(_MEDIUM.search(text)), bool(_NEGATIVE.search(text))
        s += 0.55 if strong else 0.0
        s += 0.25 if medium else 0.0
        s += 0.10 if (strong or medium) and "?" in text else 0.0
        s += 0.10 if _HIGH.search(text) and (strong or medium) else 0.0
        s -= 0.40 if neg else 0.0
        s = max(0.0, min(1.0, s))
        urgency = "high" if _HIGH.search(text) else "medium" if _MED.search(text) else "low"
        deal = "large" if _LARGE.search(text) else "small" if _SMALL.search(text) else "unknown"
        # confidence: far from 0.5 = sure. Mixed signals (positive AND negative cues) cap it.
        conf = min(1.0, abs(s - 0.5) * 2 + 0.05)
        if neg and (strong or medium):
            conf = min(conf, 0.3)
        return ScoreResult(s, urgency, deal, conf, model="heuristic")


# ------------------------------------------------------------------- helpers


def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
