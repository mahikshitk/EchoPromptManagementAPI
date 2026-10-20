"""Validate LLM items, dedup against the do-not-repeat list, compute groundingUsed, assemble output."""

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher

from promptgen import config
from promptgen.schema import OutAudience, OutPrompt, OutTopic, Output, Payload

_DOSE = re.compile(config.DOSE_PATTERN, re.I)
_PRELAUNCH = re.compile(config.PRELAUNCH_BLOCK_PATTERN, re.I)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())


class Deduper:
    """Do-not-repeat list: existingPrompts plus everything accepted so far."""

    def __init__(self, texts: list[str] = ()):
        self.texts: list[str] = []
        self._norm: list[str] = []
        for t in texts:
            self.add(t)

    def is_dup(self, text: str) -> bool:
        n = normalize(text)
        for other in self._norm:
            m = SequenceMatcher(None, n, other)
            # cheap upper bounds first; ratio() is the expensive one
            if m.real_quick_ratio() >= config.DEDUP_THRESHOLD and m.quick_ratio() >= config.DEDUP_THRESHOLD \
                    and m.ratio() >= config.DEDUP_THRESHOLD:
                return True
        return False

    def add(self, text: str) -> None:
        self.texts.append(text)
        self._norm.append(normalize(text))


def validate(item, p: Payload, audience: str) -> tuple[dict | None, str]:
    """-> (clean item, "") or (None, reject reason). audience: code, selects the audience's rules."""
    prof = config.AUDIENCE_PROFILES[audience]
    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
        return None, "malformed"
    text = " ".join(item["text"].split())
    if not text.endswith("?"):
        return None, "not a question"
    if not config.MIN_LEN <= len(text) <= config.MAX_LEN:
        return None, "length"
    if prof["block_doses"] and _DOSE.search(text):
        return None, "dose"
    if p.grounding.stage.lower() != "launched" and _PRELAUNCH.search(text):
        return None, "implies availability"
    intent = item.get("intent")
    if intent not in prof["intents"]:
        intent = config.DEFAULT_INTENT
    competitor = None
    if p.options.includeCompetitorPrompts and isinstance(item.get("competitor"), str):
        by_lower = {c.lower(): c for c in p.grounding.competitors}
        competitor = by_lower.get(item["competitor"].strip().lower())
    return {"text": text, "intent": intent, "competitor": competitor}, ""


@dataclass
class BatchResult:
    accepted: list[dict] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)  # reason -> count
    duplicates: int = 0
    surplus: int = 0  # valid and unique, but over quota


def filter_batch(items, p: Payload, audience: str, dedup: Deduper, quota: int) -> BatchResult:
    """Validate and dedup LLM items, accepting up to quota. Accepted texts are added to dedup."""
    res = BatchResult()
    for raw in items if isinstance(items, list) else []:
        item, reason = validate(raw, p, audience)
        if item is None:
            res.rejected[reason] += 1
        elif dedup.is_dup(item["text"]):
            res.duplicates += 1
        elif len(res.accepted) >= quota:
            res.surplus += 1
        else:
            dedup.add(item["text"])
            res.accepted.append(item)
    return res


def _mentions(text: str, term: str, prefix_only: bool = False) -> bool:
    end = "" if prefix_only else r"\b"
    return re.search(rf"\b{re.escape(term)}{end}", text, re.I) is not None


def grounding_used(text: str, p: Payload) -> list[str]:
    g = p.grounding
    out = ["brand"] if _mentions(text, g.brand) else []
    out += [f"competitors[{i}]" for i, c in enumerate(g.competitors) if _mentions(text, c)]
    # a disease also matches its last word ("psoriasis", "headaches")
    out += [f"diseaseAreas[{i}]" for i, d in enumerate(g.diseaseAreas)
            if _mentions(text, d) or _mentions(text, d.split()[-1], prefix_only=True)]
    return out


def assemble(p: Payload, records: list[dict], now: datetime | None = None) -> Output:
    """records: {"text", "intent", "competitor", "audience": code, "topic": code, ...debug keys ignored}."""
    audiences = {a.code: a for a in p.active_audiences()}
    topics = {t.code: t for t in p.active_topics()}
    now = now or datetime.now(timezone.utc)
    return Output(
        requestId=p.requestId,
        runId=p.runId,
        generatedAt=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        prompts=[
            OutPrompt(
                tempId=f"g_{i:03d}",
                text=r["text"],
                audience=OutAudience(code=r["audience"], name=audiences[r["audience"]].name),
                topic=OutTopic(code=r["topic"], name=topics[r["topic"]].name,
                               categoryId=topics[r["topic"]].categoryId),
                competitor=r["competitor"],
                intent=r["intent"],
                groundingUsed=grounding_used(r["text"], p),
            )
            for i, r in enumerate(records, 1)
        ],
    )


if __name__ == "__main__":
    import json

    from promptgen.schema import load_cli

    p, a = load_cli("Run made-up LLM items through validation, dedup and assembly",
                    lambda ap: ap.add_argument("--stage", help="override grounding.stage, e.g. Pre-launch"))
    if a.stage:
        p = p.model_copy(update={"grounding": p.grounding.model_copy(update={"stage": a.stage})})
    b, c0 = p.grounding.brand, (p.grounding.competitors or ["Other"])[0]
    items = [
        {"text": f"Is {b} better than {c0} at clearing my skin?", "intent": "comparison", "competitor": c0},
        {"text": f"Can I drink alcohol while taking {b}?", "intent": "side_effects", "competitor": None},
        {"text": f"Can I drink alcohol while I'm taking {b}?", "intent": "side_effects", "competitor": None},
        {"text": f"What are the side effects of {b}?", "intent": "side_effects", "competitor": None},
        {"text": f"Is 40mg of {b} a normal dose?", "intent": "informational", "competitor": None},
        {"text": f"My GP prescribed {b}, how long until it works?", "intent": "experience", "competitor": None},
        {"text": f"How much will {b} cost me", "intent": "cost_access", "competitor": None},
        {"text": f"Will {b} help my flare-ups?", "intent": "made_up_intent", "competitor": "Nope"},
        "garbage",
    ]
    d = Deduper(p.do_not_repeat("patient"))
    r = filter_batch(items, p, "patient", d, quota=10)
    print(f"stage={p.grounding.stage}  accepted={len(r.accepted)}  rejected={dict(r.rejected)}  "
          f"duplicates={r.duplicates}\n")
    print(json.dumps(assemble(p, [{**x, "audience": "patient", "topic": "safety"} for x in r.accepted]).model_dump(), indent=2))
