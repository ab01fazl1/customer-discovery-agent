from __future__ import annotations

from .cost import CostTracker
from .pipeline import RunSummary


def render(s: RunSummary, tracker: CostTracker, top: int = 15) -> str:
    out = []
    out.append(f"\nProcessed {s.total} messages in {s.seconds:.1f}s")
    out.append("\nFUNNEL")
    out.append(f"  {'stage':<18}{'in':>7}{'out':>7}{'kept':>8}{'sec':>8}")
    for name, r in s.stages.items():
        pct = f"{100 * r['out'] / r['in']:.0f}%" if r["in"] else "-"
        out.append(f"  {name:<18}{r['in']:>7}{r['out']:>7}{pct:>8}{r['seconds']:>8.2f}")

    if s.labelled:
        out.append("\nACCURACY (against ground-truth labels)")
        prec = s.tp / (s.tp + s.fp) if (s.tp + s.fp) else 0.0
        rec = s.tp / (s.tp + s.fn) if (s.tp + s.fn) else 0.0
        out.append(f"  precision {prec:.2f}   recall {rec:.2f}   (TP {s.tp}, FP {s.fp}, FN {s.fn})")
        out.append("  where real leads were lost / noise removed, per stage:")
        for name, r in s.stages.items():
            out.append(f"    {name:<18} leads lost {r['leads_lost']:>4}   noise dropped {r['noise_dropped']:>5}")

    out.append("\nCOST")
    for name, r in tracker.by_stage.items():
        out.append(f"  {name:<18} calls {r['calls']:>5}  tokens in/out {r['tokens_in']}/{r['tokens_out']}  ${r['usd']:.4f}")
    out.append(f"  total ${tracker.total_usd:.4f}")

    out.append(f"\nTOP LEADS ({len(s.leads)} total)")
    for c in s.leads[:top]:
        text = c.message.text if len(c.message.text) <= 90 else c.message.text[:87] + "..."
        flag = " ?" if "fallback_wanted_but_unavailable" in c.trace else ""
        out.append(f"  {c.intent_score:.2f} {c.urgency:<6} {c.deal_size:<7} {text}{flag}")
    if any("fallback_wanted_but_unavailable" in c.trace for c in s.leads):
        out.append("  (? = ambiguous; cloud fallback not configured, set ANTHROPIC_API_KEY)")
    return "\n".join(out)
