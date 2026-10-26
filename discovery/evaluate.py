"""Score the reviewer agent against labelled leads.

python -m discovery.evaluate --backend policy
python -m discovery.evaluate --backend ollama --model qwen2.5:7b --save data/eval_results.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from .agent import DECISIONS, QUEUE, ReviewAgent
from .cost import CostTracker
from .policy_llm import LARGE, URGENT
from .scenarios import SCENARIOS
from .tools import ChatLog, ToolBox


def baseline_rules(text: str) -> str:
    """Text-only baseline, no tools and no model: alert if it sounds urgent and large, else queue."""
    return "alert" if URGENT.search(text) and LARGE.search(text) else QUEUE


def run_eval(agent: ReviewAgent, log: ChatLog, labels: list[dict], limit: int | None = None,
             verbose: bool = False) -> list[dict]:
    rows = []
    for lab in labels[:limit] if limit else labels:
        lead = log.by_id[lab["lead_id"]]
        try:
            r = agent.review(lead)
            row = {"decision": r.decision, "reason": r.reason, "finished": r.finished, "steps": r.steps,
                   "tool_calls": len(r.tool_calls), "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
                   "cost_usd": r.cost_usd, "seconds": r.seconds, "forced": r.forced, "facts": r.facts,
                   "trace": r.trace}
        except Exception as e:  # one bad call (timeout, server error) must not end a long run
            row = {"decision": QUEUE, "reason": f"error: {e}", "finished": False, "steps": 0, "tool_calls": 0,
                   "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "seconds": 0.0, "forced": False,
                   "facts": {}, "trace": [f"error: {e}"]}
        row.update(lead_id=lab["lead_id"], scenario=lab["scenario"], expected=lab["decision"],
                   text=lead["text"], baseline=baseline_rules(lead["text"]))
        rows.append(row)
        if verbose:
            ok = "ok " if row["decision"] == row["expected"] else "MISS"
            print(f"{ok} {row['scenario']:<16} expected {row['expected']:<5} got {row['decision']:<5} "
                  f"steps {row['steps']} {row['seconds']:.1f}s", flush=True)
    return rows


def _acc(rows, key="decision") -> float:
    return sum(r[key] == r["expected"] for r in rows) / len(rows) if rows else 0.0


def render(rows: list[dict]) -> str:
    n = len(rows)
    out = [f"\nREVIEWER EVALUATION ({n} labelled leads)"]
    majority = Counter(r["expected"] for r in rows).most_common(1)[0]
    out.append(f"  agent accuracy          {_acc(rows):.2f}")
    out.append(f"  text-only rule baseline {_acc(rows, 'baseline'):.2f}   (no tools, no model)")
    out.append(f"  always-'{majority[0]}' baseline  {majority[1] / n:.2f}")
    out.append(f"  finished a decision     {sum(r['finished'] for r in rows) / n:.2f}   "
               f"(unfinished default to queue)")
    out.append(f"  forced to submit at the step cap  {sum(r['forced'] for r in rows) / n:.2f}")
    out.append(f"  avg steps {sum(r['steps'] for r in rows) / n:.1f}   avg tool calls "
               f"{sum(r['tool_calls'] for r in rows) / n:.1f}")

    out.append("\nCONFUSION (rows = correct, columns = agent said)")
    out.append(f"  {'':<8}" + "".join(f"{d:>7}" for d in DECISIONS))
    for exp in DECISIONS:
        counts = Counter(r["decision"] for r in rows if r["expected"] == exp)
        out.append(f"  {exp:<8}" + "".join(f"{counts[d]:>7}" for d in DECISIONS))

    out.append("\nBY SCENARIO (agent vs text-only baseline)")
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    for sc in SCENARIOS:
        g = by.get(sc.name, [])
        if g:
            out.append(f"  {sc.name:<16} agent {_acc(g):.2f}   baseline {_acc(g, 'baseline'):.2f}   (n={len(g)})")

    tin, tout = sum(r["tokens_in"] for r in rows), sum(r["tokens_out"] for r in rows)
    cost, secs = sum(r["cost_usd"] for r in rows), sum(r["seconds"] for r in rows)
    out.append("\nCOST")
    out.append(f"  tokens in/out {tin}/{tout}   ${cost:.4f} total   ${cost / n:.5f} per lead")
    out.append(f"  time {secs:.1f}s total   {secs / n:.1f}s per lead")

    misses = [r for r in rows if r["decision"] != r["expected"]]
    out.append(f"\nMISTAKES ({len(misses)}), first 8")
    for r in misses[:8]:
        out.append(f"  [{r['scenario']}] expected {r['expected']}, got {r['decision']}"
                   f"{'' if r['finished'] else ' (unfinished)'}: {r['text'][:70]}")
        out.append(f"      agent said: {r['reason'][:110]}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="discovery.evaluate")
    p.add_argument("--chat", default="data/review_chat.jsonl")
    p.add_argument("--labels", default="data/review_labels.jsonl")
    p.add_argument("--backend", choices=["policy", "ollama"], default="policy")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--save", help="write per-lead results to this JSONL file")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    if a.backend == "ollama":
        from .ollama_llm import OllamaChat
        backend = OllamaChat(a.ollama_url, a.model)
    else:
        from .policy_llm import PolicyBackend
        backend = PolicyBackend()

    log = ChatLog.from_jsonl(a.chat)
    labels = [json.loads(line) for line in open(a.labels, encoding="utf-8") if line.strip()]
    tracker = CostTracker()
    rows = run_eval(ReviewAgent(backend, ToolBox(log), a.max_steps, tracker), log, labels, a.limit, a.verbose)
    print(render(rows))
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


if __name__ == "__main__":
    main()
