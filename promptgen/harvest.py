"""Harvest real patient search questions (Google Autocomplete, Google Trends) into an append-only store.

Every result ever fetched is kept, keyed (source, query, lang, geo): a refresh adds new texts and never
deletes, so the reference pool only grows, and when a fetch fails or is rate-limited everything stored
for the key is still served. The store is AtrionDB on AWS (through the database Lambda) and a local
SQLite file otherwise. A run makes one batched read before fetching and one batched write after.
Store interface: load(keys) -> [{"key", "fetchedAt" (epoch s or None), "texts" (oldest first)}];
save([{"key", "texts"}]) records texts and marks the keys fetched now.
"""

import json
import os
import sqlite3
import time
import warnings
from dataclasses import dataclass, field

import httpx

from promptgen import config, db
from promptgen.schema import Payload

AC_URL = "https://suggestqueries.google.com/complete/search"


class SqliteHarvestStore:
    """Local store for CLI runs and tests: a SQLite file at path, or ":memory:"."""

    def __init__(self, path: str = config.CACHE_PATH):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS fetches (source TEXT, query TEXT, lang TEXT, geo TEXT, fetched_at REAL,"
            " PRIMARY KEY (source, query, lang, geo));"
            "CREATE TABLE IF NOT EXISTS results (source TEXT, query TEXT, lang TEXT, geo TEXT, text TEXT,"
            " first_seen REAL, PRIMARY KEY (source, query, lang, geo, text));"
        )

    def load(self, keys: list[list[str]]) -> list[dict]:
        out = []
        for k in keys:
            row = self.db.execute(
                "SELECT fetched_at FROM fetches WHERE source=? AND query=? AND lang=? AND geo=?", k).fetchone()
            texts = [r[0] for r in self.db.execute(
                "SELECT text FROM results WHERE source=? AND query=? AND lang=? AND geo=? ORDER BY first_seen, rowid",
                k)]
            out.append({"key": list(k), "fetchedAt": row[0] if row else None, "texts": texts})
        return out

    def save(self, entries: list[dict]) -> None:
        now = time.time()
        for e in entries:
            k = tuple(e["key"])
            self.db.executemany("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?)",
                                [(*k, t, now) for t in e["texts"]])
            self.db.execute("INSERT OR REPLACE INTO fetches VALUES (?,?,?,?,?)", (*k, now))
        self.db.commit()


FETCH_TABLE = "public.atrion_prompt_harvest_fetch"
RESULT_TABLE = "public.atrion_prompt_harvest_result"


def _key_arrays(keys: list[list[str]]) -> dict:
    return {"s": [k[0] for k in keys], "q": [k[1] for k in keys], "l": [k[2] for k in keys], "g": [k[3] for k in keys]}


class PostgresHarvestStore:
    """AtrionDB tables; INSERT ... ON CONFLICT keeps concurrent workers from losing each other's rows."""

    def __init__(self, database: db.Database):
        self._execute = database.execute

    def load(self, keys: list[list[str]]) -> list[dict]:
        rows = self._execute(
            "SELECT k.source, k.query, k.lang, k.geo, extract(epoch FROM f.fetchedat)::float8 AS fetchedat,"
            f" ARRAY(SELECT r.text FROM {RESULT_TABLE} r"
            "        WHERE (r.source, r.query, r.lang, r.geo) = (k.source, k.query, k.lang, k.geo)"
            "        ORDER BY r.firstseen, r.resultid) AS texts"
            " FROM unnest(%(s)s::text[], %(q)s::text[], %(l)s::text[], %(g)s::text[]) AS k(source, query, lang, geo)"
            f" LEFT JOIN {FETCH_TABLE} f ON (f.source, f.query, f.lang, f.geo) = (k.source, k.query, k.lang, k.geo)",
            _key_arrays(keys)).fetchall()
        return [{"key": [r["source"], r["query"], r["lang"], r["geo"]], "fetchedAt": r["fetchedat"],
                 "texts": r["texts"]} for r in rows]

    def save(self, entries: list[dict]) -> None:
        rows = [(*e["key"], t) for e in entries for t in dict.fromkeys(e["texts"])]
        # one statement, so results and fetch times land together
        self._execute(
            f"WITH added AS (INSERT INTO {RESULT_TABLE} (source, query, lang, geo, text)"
            "   SELECT * FROM unnest(%(rs)s::text[], %(rq)s::text[], %(rl)s::text[], %(rg)s::text[], %(rt)s::text[])"
            "   ON CONFLICT DO NOTHING)"
            f" INSERT INTO {FETCH_TABLE} (source, query, lang, geo, fetchedat)"
            " SELECT *, now() FROM unnest(%(s)s::text[], %(q)s::text[], %(l)s::text[], %(g)s::text[])"
            " ON CONFLICT (source, query, lang, geo) DO UPDATE SET fetchedat = excluded.fetchedat",
            {**_key_arrays([e["key"] for e in entries]),
             "rs": [r[0] for r in rows], "rq": [r[1] for r in rows], "rl": [r[2] for r in rows],
             "rg": [r[3] for r in rows], "rt": [r[4] for r in rows]})


class LambdaHarvestStore:
    """Forwards load/save to the VPC database Lambda (worker Lambda, outside the VPC)."""

    def __init__(self, function: db.DatabaseFunction):
        self._call = function.call

    def load(self, keys: list[list[str]]) -> list[dict]:
        return self._call("harvest_load", keys=keys)

    def save(self, entries: list[dict]) -> None:
        self._call("harvest_save", entries=entries)


def default_store():
    """AtrionDB per db.backend(); otherwise the SQLite file at HARVEST_DB_PATH (default .cache/harvest.sqlite)."""
    b = db.backend()
    if isinstance(b, db.DatabaseFunction):
        return LambdaHarvestStore(b)
    if isinstance(b, db.Database):
        return PostgresHarvestStore(b)
    return SqliteHarvestStore(os.getenv("HARVEST_DB_PATH") or config.CACHE_PATH)


@dataclass
class HarvestResult:
    questions: list[dict] = field(default_factory=list)  # {"text", "source", "seed"}
    stats: dict = field(default_factory=lambda: {"cache_hits": 0, "fetched": 0, "stale": 0, "failed": 0})
    errors: list[str] = field(default_factory=list)
    stored: int = 0  # texts stored for this payload's searches after this run


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


def _lookup(known: dict, fetched: list, res: HarvestResult, key: tuple, fetch, sleep: float, offline: bool,
            ttl: float) -> list[str]:
    """Stored texts for key, refreshed from Google when older than ttl. New results are queued in
    fetched for the one batched save at the end of the run."""
    entry = known.get(key) or {"fetchedAt": None, "texts": []}
    old = entry["texts"]
    if offline or (entry["fetchedAt"] is not None and time.time() - entry["fetchedAt"] < ttl):
        res.stats["cache_hits"] += 1
        return old
    try:
        time.sleep(sleep)
        new = list(dict.fromkeys(t for t in fetch(*key[1:]) if t and len(t) <= config.HARVEST_MAX_TEXT))
        res.stats["fetched"] += 1
    except Exception as e:
        res.errors.append(f"{key[0]} '{key[1]}': {type(e).__name__}: {str(e)[:120]}")
        res.stats["stale" if old else "failed"] += 1
        return old
    fetched.append({"key": list(key), "texts": new})
    seen = set(old)
    return old + [t for t in new if t not in seen]


def harvest(p: Payload, use_autocomplete: bool | None = None, use_trends: bool | None = None,
            store=None, offline: bool = False, ttl: float = config.CACHE_TTL_S) -> HarvestResult:
    """Defaults come from payload options. offline: serve only what is stored, never fetch.
    Store failures are recorded in errors and never fail the run."""
    if use_autocomplete is None:
        use_autocomplete = p.options.includeAutocomplete
    if use_trends is None:
        use_trends = p.options.includeGoogleTrends
    lang, geo = p.options.language, p.grounding.market.isoCode
    res = HarvestResult()
    if not (use_autocomplete or use_trends):
        return res
    searches = []  # (source, seed shown in the pool, key, fetch, sleep)
    if use_autocomplete:
        searches += [("autocomplete", s, ("autocomplete", s, lang, geo), fetch_autocomplete, config.AC_SLEEP_S)
                     for s in seeds(p)]
    if use_trends:
        terms: dict[str, str] = {}  # one search per term, first spelling wins
        for t in [p.grounding.brand, *p.grounding.diseaseAreas]:
            terms.setdefault(t.lower(), t)
        searches += [("trends", t, ("trends", key, lang, geo), fetch_trends, config.TRENDS_SLEEP_S)
                     for key, t in terms.items()]
    store = store or default_store()
    try:
        known = {tuple(e["key"]): e for e in store.load([list(s[2]) for s in searches])}
    except Exception as e:
        res.errors.append(f"store load: {type(e).__name__}: {str(e)[:120]}")
        known = {}
    seen: set[str] = set()

    def add(texts, source, seed):
        for t in texts:
            k = t.strip().lower().rstrip("?")
            if k not in seen and k != seed.lower() and is_question(t):  # skip the seed echoed back
                seen.add(k)
                res.questions.append({"text": t.strip(), "source": source, "seed": seed})

    fetched: list[dict] = []
    for source, seed, key, fetch, sleep in searches:
        texts = _lookup(known, fetched, res, key, fetch, sleep, offline, ttl)
        res.stored += len(texts)
        add(texts, source, seed)
    if fetched:
        try:
            store.save(fetched)
        except Exception as e:
            res.errors.append(f"store save: {type(e).__name__}: {str(e)[:120]}")
    return res
