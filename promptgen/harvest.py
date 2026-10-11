"""Harvest real patient search questions (Google Autocomplete, Google Trends), cached in SQLite."""

import json
import os
import sqlite3
import time
import warnings
from dataclasses import dataclass, field

import httpx

from promptgen import config
from promptgen.schema import Payload, load_cli

AC_URL = "https://suggestqueries.google.com/complete/search"


class Cache:
    """(source, query, lang, geo) -> list[str]. Expired rows are kept as a fallback when a fetch fails."""

    def __init__(self, path: str = config.CACHE_PATH, ttl: int = config.CACHE_TTL_S):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.ttl = ttl
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS harvest (source TEXT, query TEXT, lang TEXT, geo TEXT,"
            " fetched_at REAL, results TEXT, PRIMARY KEY (source, query, lang, geo))"
        )

    def get(self, key: tuple, stale_ok: bool = False) -> list[str] | None:
        row = self.db.execute(
            "SELECT fetched_at, results FROM harvest WHERE source=? AND query=? AND lang=? AND geo=?", key
        ).fetchone()
        if row and (stale_ok or time.time() - row[0] < self.ttl):
            return json.loads(row[1])
        return None

    def put(self, key: tuple, results: list[str]) -> None:
        self.db.execute("INSERT OR REPLACE INTO harvest VALUES (?,?,?,?,?,?)",
                        (*key, time.time(), json.dumps(results)))
        self.db.commit()


@dataclass
class HarvestResult:
    questions: list[dict] = field(default_factory=list)  # {"text", "source", "seed"}
    stats: dict = field(default_factory=lambda: {"cache_hits": 0, "fetched": 0, "stale": 0, "failed": 0})
    errors: list[str] = field(default_factory=list)


def is_question(s: str) -> bool:
    s = s.strip().lower()
    return s.endswith("?") or s.split(" ", 1)[0] in config.QUESTION_WORDS


def seeds(p: Payload) -> list[str]:
    out = []
    for term in [p.grounding.brand, *p.grounding.diseaseAreas]:
        t = term.lower()
        out += [f"{pre} {t}" for pre in config.AC_PREFIXES] + [f"{t} {suf}" for suf in config.AC_SUFFIXES]
    return list(dict.fromkeys(out))[: config.AC_MAX_SEEDS]


def fetch_autocomplete(q: str, lang: str, geo: str) -> list[str]:
    r = httpx.get(AC_URL, params={"client": "firefox", "q": q, "hl": lang, "gl": geo}, timeout=10)
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8", "replace"))[1]


def fetch_trends(term: str, lang: str, geo: str) -> list[str]:
    from pytrends.exceptions import TooManyRequestsError
    from pytrends.request import TrendReq

    for attempt in range(2):  # single retry on 429
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = TrendReq(hl=lang, tz=0)
                t.build_payload([term], geo=geo, timeframe="today 12-m")
                rq = t.related_queries().get(term) or {}
            return [q for k in ("top", "rising") if rq.get(k) is not None for q in rq[k]["query"]]
        except TooManyRequestsError:
            if attempt:
                raise
            time.sleep(config.TRENDS_SLEEP_S * 5)
    return []


def _cached(cache: Cache, res: HarvestResult, key: tuple, fetch, sleep: float) -> list[str]:
    hit = cache.get(key)
    if hit is not None:
        res.stats["cache_hits"] += 1
        return hit
    try:
        time.sleep(sleep)
        out = fetch(*key[1:])
        cache.put(key, out)
        res.stats["fetched"] += 1
        return out
    except Exception as e:
        stale = cache.get(key, stale_ok=True)
        res.stats["stale" if stale is not None else "failed"] += 1
        res.errors.append(f"{key[0]} '{key[1]}': {type(e).__name__}: {str(e)[:120]}")
        return stale or []


def harvest(p: Payload, use_autocomplete: bool | None = None, use_trends: bool | None = None,
            cache: Cache | None = None) -> HarvestResult:
    """Defaults come from payload options; the UI can override them."""
    if use_autocomplete is None:
        use_autocomplete = p.options.includeAutocomplete
    if use_trends is None:
        use_trends = p.options.includeGoogleTrends
    cache = cache or Cache()
    lang, geo = p.options.language, p.grounding.market.isoCode
    res = HarvestResult()
    seen: set[str] = set()

    def add(texts, source, seed):
        for t in texts:
            k = t.strip().lower().rstrip("?")
            if k not in seen and k != seed.lower() and is_question(t):  # skip the seed echoed back
                seen.add(k)
                res.questions.append({"text": t.strip(), "source": source, "seed": seed})

    if use_autocomplete:
        for s in seeds(p):
            add(_cached(cache, res, ("autocomplete", s, lang, geo), fetch_autocomplete, config.AC_SLEEP_S),
                "autocomplete", s)
    if use_trends:
        for term in [p.grounding.brand, *p.grounding.diseaseAreas]:
            add(_cached(cache, res, ("trends", term.lower(), lang, geo), fetch_trends, config.TRENDS_SLEEP_S),
                "trends", term)
    return res


if __name__ == "__main__":
    p, _ = load_cli("Harvest patient search questions")
    print(f"brand={p.grounding.brand} diseaseAreas={p.grounding.diseaseAreas}\n")
    t = time.time()
    r = harvest(p)
    for q in r.questions:
        print(f"[{q['source']:12}] {q['text']}")
    print(f"\n{len(r.questions)} questions, {r.stats}, {time.time() - t:.1f}s")
    for e in r.errors:
        print("ERROR:", e)
