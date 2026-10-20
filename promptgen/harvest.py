"""Harvest real patient search questions (Google Autocomplete, Google Trends) into an append-only SQLite store."""

import json
import os
import sqlite3
import time
import warnings
from dataclasses import dataclass, field

import httpx

from promptgen import config
from promptgen.schema import Payload

AC_URL = "https://suggestqueries.google.com/complete/search"


class Cache:
    """Append-only store of every harvested result, keyed (source, query, lang, geo). A refresh adds
    new texts and never deletes, so the reference pool only grows, and when a fetch fails or is
    rate-limited everything ever stored for the key is still served. Optionally synced with an
    S3 copy of the file (s3://bucket/key) so the store survives Lambda's ephemeral /tmp."""

    def __init__(self, path: str = config.CACHE_PATH, ttl: int = config.CACHE_TTL_S, s3_uri: str | None = None):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path, self.ttl, self.s3_uri = path, ttl, s3_uri
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS fetches (source TEXT, query TEXT, lang TEXT, geo TEXT, fetched_at REAL,"
            " PRIMARY KEY (source, query, lang, geo));"
            "CREATE TABLE IF NOT EXISTS results (source TEXT, query TEXT, lang TEXT, geo TEXT, text TEXT,"
            " first_seen REAL, PRIMARY KEY (source, query, lang, geo, text));"
        )

    def fresh(self, key: tuple) -> bool:
        row = self.db.execute(
            "SELECT fetched_at FROM fetches WHERE source=? AND query=? AND lang=? AND geo=?", key).fetchone()
        return row is not None and time.time() - row[0] < self.ttl

    def get(self, key: tuple) -> list[str]:
        """Every text ever stored for the key, oldest first."""
        return [r[0] for r in self.db.execute(
            "SELECT text FROM results WHERE source=? AND query=? AND lang=? AND geo=? ORDER BY first_seen, rowid",
            key)]

    def put(self, key: tuple, results: list[str]) -> None:
        now = time.time()
        self.db.executemany("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?)",
                            [(*key, t, now) for t in results])
        self.db.execute("INSERT OR REPLACE INTO fetches VALUES (?,?,?,?,?)", (*key, now))
        self.db.commit()

    def size(self) -> int:
        return self.db.execute("SELECT count(*) FROM results").fetchone()[0]

    # ---- S3 sync (no-ops without s3_uri) ----

    def _s3(self):
        import boto3

        bucket, _, key = self.s3_uri.removeprefix("s3://").partition("/")
        return boto3.client("s3"), bucket, key

    def pull(self) -> None:
        """Merge the S3 copy into the local file. A missing object (first run) is not an error."""
        if not self.s3_uri:
            return
        from botocore.exceptions import ClientError

        client, bucket, key = self._s3()
        tmp = f"{self.path}.remote"
        try:
            client.download_file(bucket, key, tmp)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return
            raise
        try:
            self.db.execute("ATTACH DATABASE ? AS remote", (tmp,))
            self.db.execute("INSERT OR IGNORE INTO results SELECT * FROM remote.results")
            self.db.execute("INSERT INTO fetches SELECT * FROM remote.fetches WHERE true ON CONFLICT"
                            " DO UPDATE SET fetched_at = max(fetched_at, excluded.fetched_at)")
            self.db.commit()
            self.db.execute("DETACH DATABASE remote")
        finally:
            os.remove(tmp)

    def push(self) -> None:
        """Re-merge the latest S3 copy (another run may have written since our pull), then upload.
        ponytail: no locking, two runs pushing in the same instant can drop one's new rows; they come
        back on that key's next refresh. Use a database if concurrent writers become common."""
        if not self.s3_uri:
            return
        self.pull()
        client, bucket, key = self._s3()
        client.upload_file(self.path, bucket, key)


@dataclass
class HarvestResult:
    questions: list[dict] = field(default_factory=list)  # {"text", "source", "seed"}
    stats: dict = field(default_factory=lambda: {"cache_hits": 0, "fetched": 0, "stale": 0, "failed": 0})
    errors: list[str] = field(default_factory=list)
    stored: int = 0  # rows in the store after this run


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


def _cached(cache: Cache, res: HarvestResult, key: tuple, fetch, sleep: float, offline: bool) -> list[str]:
    if offline or cache.fresh(key):
        res.stats["cache_hits"] += 1
        return cache.get(key)
    try:
        time.sleep(sleep)
        cache.put(key, fetch(*key[1:]))
        res.stats["fetched"] += 1
    except Exception as e:
        res.errors.append(f"{key[0]} '{key[1]}': {type(e).__name__}: {str(e)[:120]}")
        res.stats["stale" if cache.get(key) else "failed"] += 1
    return cache.get(key)


def default_cache() -> Cache:
    """Local file at HARVEST_DB_PATH (set to /tmp/... on Lambda), synced to HARVEST_DB_S3_URI if set."""
    return Cache(os.getenv("HARVEST_DB_PATH") or config.CACHE_PATH, s3_uri=os.getenv("HARVEST_DB_S3_URI"))


def harvest(p: Payload, use_autocomplete: bool | None = None, use_trends: bool | None = None,
            cache: Cache | None = None, offline: bool = False) -> HarvestResult:
    """Defaults come from payload options. offline: serve only what is stored, never fetch.
    S3 sync failures are recorded in errors and never fail the run."""
    if use_autocomplete is None:
        use_autocomplete = p.options.includeAutocomplete
    if use_trends is None:
        use_trends = p.options.includeGoogleTrends
    cache = cache or default_cache()
    lang, geo = p.options.language, p.grounding.market.isoCode
    res = HarvestResult()
    if not (use_autocomplete or use_trends):
        return res
    try:
        cache.pull()
    except Exception as e:
        res.errors.append(f"store pull from {cache.s3_uri}: {type(e).__name__}: {str(e)[:120]}")
    seen: set[str] = set()

    def add(texts, source, seed):
        for t in texts:
            k = t.strip().lower().rstrip("?")
            if k not in seen and k != seed.lower() and is_question(t):  # skip the seed echoed back
                seen.add(k)
                res.questions.append({"text": t.strip(), "source": source, "seed": seed})

    if use_autocomplete:
        for s in seeds(p):
            add(_cached(cache, res, ("autocomplete", s, lang, geo), fetch_autocomplete, config.AC_SLEEP_S,
                        offline), "autocomplete", s)
    if use_trends:
        for term in [p.grounding.brand, *p.grounding.diseaseAreas]:
            add(_cached(cache, res, ("trends", term.lower(), lang, geo), fetch_trends, config.TRENDS_SLEEP_S,
                        offline), "trends", term)
    if res.stats["fetched"]:
        try:
            cache.push()
        except Exception as e:
            res.errors.append(f"store push to {cache.s3_uri}: {type(e).__name__}: {str(e)[:120]}")
    res.stored = cache.size()
    return res
