"""Reference pool (harvested questions tagged by topic) and the per-batch example sampler."""

import random
import re
from collections import Counter

from promptgen import config
from promptgen.schema import Payload

_PATTERNS = {
    topic: re.compile("|".join(rf"\b{re.escape(k)}" for k in kws), re.I)
    for topic, kws in config.TOPIC_KEYWORDS.items()
}


def tag(text: str) -> str:
    return next((t for t, rx in _PATTERNS.items() if rx.search(text)), config.GENERAL)


def is_relevant(text: str, p: Payload) -> bool:
    """Mentions the brand, a competitor, or a disease-area word (drops Trends noise)."""
    g = p.grounding
    words = [g.brand, *g.competitors] + [w for d in g.diseaseAreas for w in d.split() if len(w) > 3]
    t = text.lower()
    return any(w.lower() in t for w in words)


def build_pool(questions: list[dict], p: Payload) -> list[dict]:
    """Tag relevant harvested questions, then add fallback examples: all of them if the pool is
    under MIN_POOL, otherwise only those for topics with fewer than MIN_PER_TOPIC examples."""
    pool = [{**q, "topic": tag(q["text"])} for q in questions if is_relevant(q["text"], p)]
    counts = topic_counts(pool)
    short = {t for t in config.TOPIC_KEYWORDS if counts[t] < config.MIN_PER_TOPIC}
    disease = p.grounding.diseaseAreas[0].lower() if p.grounding.diseaseAreas else "my condition"
    seen = {q["text"].lower() for q in pool}
    small = len(pool) < config.MIN_POOL
    for tmpl in config.FALLBACK_EXAMPLES:
        text = tmpl.format(brand=p.grounding.brand, disease=disease)
        topic = tag(text)
        if (small or topic in short) and text.lower() not in seen:
            seen.add(text.lower())
            pool.append({"text": text, "source": "fallback", "seed": "", "topic": topic})
    return pool


def topic_counts(pool: list[dict]) -> Counter:
    return Counter(q["topic"] for q in pool)


class Sampler:
    """Draws examples without replacement per bucket; a bucket reshuffles only once exhausted,
    so consecutive batches see different examples."""

    ANY = "__any__"

    def __init__(self, pool: list[dict], seed: int | None = None):
        self.rng = random.Random(seed)
        self.buckets: dict[str, list[dict]] = {self.ANY: pool}
        for q in pool:
            self.buckets.setdefault(q["topic"], []).append(q)
        self.decks: dict[str, list[dict]] = {}

    def _draw(self, bucket: str, k: int, taken: set[str]) -> list[dict]:
        items = self.buckets.get(bucket, [])
        k = min(k, sum(q["text"] not in taken for q in items))
        out = []
        while len(out) < k:
            if not self.decks.get(bucket):
                self.decks[bucket] = self.rng.sample(items, len(items))
            q = self.decks[bucket].pop()
            if q["text"] not in taken:
                taken.add(q["text"])
                out.append(q)
        return out

    def pick(self, topic: str) -> list[dict]:
        """~6 topic-matched + general to make 9; short buckets are filled from the rest of the pool."""
        total = config.FEWSHOT_TOPIC + config.FEWSHOT_GENERAL
        taken: set[str] = set()
        ex = self._draw(topic, config.FEWSHOT_TOPIC, taken)
        ex += self._draw(config.GENERAL, total - len(ex), taken)
        ex += self._draw(self.ANY, total - len(ex), taken)
        return ex


if __name__ == "__main__":
    from promptgen.harvest import harvest
    from promptgen.schema import load_cli

    p, a = load_cli("Show the tagged reference pool and sample batches", lambda ap: ap.add_argument(
        "--no-sources", action="store_true", help="skip Autocomplete/Trends (fallback only)"))
    off = False if a.no_sources else None
    pool = build_pool(harvest(p, off, off).questions, p)
    print(f"brand={p.grounding.brand}  pool={len(pool)}  by topic={dict(topic_counts(pool))}")
    print(f"sources={dict(Counter(q['source'] for q in pool))}\n")
    for topic in config.TOPIC_KEYWORDS:
        print(f"== {topic} ==")
        for q in pool:
            if q["topic"] == topic:
                print("   ", q["text"])
    s = Sampler(pool, seed=0)
    for topic in config.TOPIC_KEYWORDS:
        for i in range(2):
            print(f"\n-- sample batch: {topic} #{i + 1} --")
            for q in s.pick(topic):
                print(f"  [{q['topic']:11}] {q['text']}")
