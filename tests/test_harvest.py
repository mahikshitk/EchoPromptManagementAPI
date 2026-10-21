import json
import os
import time

import pytest

import database
from promptgen import db, harvest as h
from promptgen.schema import Payload


@pytest.fixture
def payload():
    return Payload.model_validate(json.load(open("sample_payload.json")))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(h.time, "sleep", lambda s: None)


def test_is_question():
    assert h.is_question("how long does it last")
    assert h.is_question("brandex for kids?")
    assert not h.is_question("plaque psoriasis treatment")
    assert not h.is_question("isotretinoin dose")  # word match, not prefix match


def test_seeds(payload):
    s = h.seeds(payload)
    assert "is brandex" in s and "plaque psoriasis cost" in s
    assert len(s) == len(set(s)) <= 30


def test_seeds_follow_brand_override(payload):
    s = h.seeds(payload.model_copy(update={"grounding": payload.grounding.model_copy(update={"brand": "Tylenol", "diseaseAreas": ["Headache"]})}))
    assert "is tylenol" in s and "headache cost" in s
    assert not any("brandex" in x or "psoriasis" in x for x in s)


# ---- stores ----

KEY = ["autocomplete", "q", "en-GB", "GB"]
TRENDS_BRANDEX = ["trends", "brandex", "en-GB", "GB"]


def mem():
    return h.SqliteHarvestStore(":memory:")


def store_contract(store):
    """Every store behaves the same: unknown keys load empty, saves append, fetch times are recorded."""
    other = ["trends", "x", "en-GB", "GB"]
    assert store.load([KEY, other]) == [{"key": KEY, "fetchedAt": None, "texts": []},
                                        {"key": other, "fetchedAt": None, "texts": []}]
    t0 = time.time()
    store.save([{"key": KEY, "texts": ["a", "b"]}, {"key": other, "texts": []}])
    store.save([{"key": KEY, "texts": ["b", "c"]}])  # a refresh that no longer returns "a"
    got = {tuple(e["key"]): e for e in store.load([KEY, other])}
    assert got[tuple(KEY)]["texts"] == ["a", "b", "c"]  # append-only, oldest first
    assert got[tuple(other)]["texts"] == [] and got[tuple(other)]["fetchedAt"] >= t0 - 1  # empty fetch recorded
    assert abs(got[tuple(KEY)]["fetchedAt"] - time.time()) < 60


def test_sqlite_store():
    store_contract(mem())


def test_sqlite_store_persists(tmp_path):
    path = str(tmp_path / "x" / "h.sqlite")
    h.SqliteHarvestStore(path).save([{"key": KEY, "texts": ["a"]}])
    assert h.SqliteHarvestStore(path).load([KEY])[0]["texts"] == ["a"]


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to a scratch PostgreSQL")
def test_postgres_store():
    import psycopg

    with psycopg.connect(os.environ["TEST_DATABASE_URL"], autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {h.RESULT_TABLE}, {h.FETCH_TABLE}")
        conn.execute(open("database/prerequisites/002_prompt_harvest.sql").read())
    store_contract(h.PostgresHarvestStore(db.Database(os.environ["TEST_DATABASE_URL"])))


def test_lambda_store_round_trips_through_database_handler(monkeypatch):
    from tests.test_db import FakeLambda

    monkeypatch.setattr(database, "_stores", {"harvest": mem()})  # SQLite stands in for PostgresHarvestStore
    fake = FakeLambda()
    store_contract(h.LambdaHarvestStore(db.DatabaseFunction("fn", fake)))
    assert {e["operation"] for _, e in fake.calls} == {"harvest_load", "harvest_save"}


def test_default_store_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_backend", None)
    monkeypatch.delenv("DATABASE_FUNCTION", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("HARVEST_DB_PATH", str(tmp_path / "h.sqlite"))
    assert isinstance(h.default_store(), h.SqliteHarvestStore) and (tmp_path / "h.sqlite").exists()
    monkeypatch.setenv("DATABASE_FUNCTION", "prompt-management-dev-database")
    assert isinstance(h.default_store(), h.LambdaHarvestStore)


# ---- harvest ----

class CountingStore:
    def __init__(self, inner):
        self.inner, self.loads, self.saves = inner, [], []

    def load(self, keys):
        self.loads.append(keys)
        return self.inner.load(keys)

    def save(self, entries):
        self.saves.append(entries)
        self.inner.save(entries)


def test_harvest_filters_dedups_and_caches(payload, monkeypatch):
    calls = []

    def fake_ac(q, lang, geo):
        calls.append(q)
        return [q, "Is it safe?", "is it safe", "brandex tablets", f"how long does {q} last"]

    monkeypatch.setattr(h, "fetch_autocomplete", fake_ac)
    store = CountingStore(mem())
    r = h.harvest(payload, use_trends=False, store=store)
    texts = [q["text"] for q in r.questions]
    assert "Is it safe?" in texts and "is it safe" not in texts  # deduped
    assert not any(t in h.seeds(payload) for t in texts)        # seed echo dropped
    assert "brandex tablets" not in texts                 # not question-shaped
    assert r.stats["fetched"] == len(calls)
    assert len(store.loads) == 1 and len(store.loads[0]) == len(calls)  # one batched read
    assert len(store.saves) == 1 and len(store.saves[0]) == len(calls)  # one batched write

    r2 = h.harvest(payload, use_trends=False, store=store)
    assert r2.stats == {"cache_hits": len(calls), "fetched": 0, "stale": 0, "failed": 0}
    assert len(r2.questions) == len(r.questions) and r2.stored == r.stored > 0
    assert len(store.saves) == 1                          # all fresh: nothing written


def test_refresh_grows_the_pool(payload, monkeypatch):
    store = mem()
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: [f"what is {term}"])
    h.harvest(payload, use_autocomplete=False, store=store, ttl=0)
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: [f"is {term} safe"])
    r = h.harvest(payload, use_autocomplete=False, store=store, ttl=0)
    assert {"what is brandex", "is brandex safe"} <= {q["text"] for q in r.questions}


def test_offline_serves_store_without_fetching(payload, monkeypatch):
    store = mem()
    store.save([{"key": TRENDS_BRANDEX, "texts": ["what is brandex"]}])
    monkeypatch.setattr(h, "fetch_trends", lambda *a: pytest.fail("fetched"))
    monkeypatch.setattr(h, "fetch_autocomplete", lambda *a: pytest.fail("fetched"))
    r = h.harvest(payload, store=store, offline=True, ttl=0)
    assert [q["text"] for q in r.questions] == ["what is brandex"] and r.stats["fetched"] == 0


def test_failure_falls_back_to_stored_then_empty(payload, monkeypatch):
    store = mem()
    store.save([{"key": TRENDS_BRANDEX, "texts": ["what is brandex"]}])

    def boom(*a):
        raise RuntimeError("429")

    monkeypatch.setattr(h, "fetch_trends", boom)
    r = h.harvest(payload, use_autocomplete=False, use_trends=True, store=store, ttl=0)
    assert [q["text"] for q in r.questions] == ["what is brandex"]
    assert r.stats["stale"] == 1 and r.stats["failed"] == 1 and len(r.errors) == 2


def test_store_errors_never_fail_the_run(payload, monkeypatch):
    class Down:
        def load(self, keys):
            raise db.DatabaseError("harvest_load: database operation harvest_load failed (OperationalError)")

        def save(self, entries):
            raise db.DatabaseError("harvest_save: database operation harvest_save failed (OperationalError)")

    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: [f"what is {term}"])
    r = h.harvest(payload, use_autocomplete=False, store=Down())
    assert r.questions and any("store load" in e for e in r.errors) and any("store save" in e for e in r.errors)


def test_long_results_are_dropped(payload, monkeypatch):
    store = mem()
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: ["what is " + "x" * 600, f"what is {term}"])
    h.harvest(payload, use_autocomplete=False, store=store)
    assert store.load([TRENDS_BRANDEX])[0]["texts"] == ["what is brandex"]


def test_trends_search_each_term_once(payload, monkeypatch):
    calls = []
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: calls.append(term) or [])
    p = payload.model_copy(update={"grounding": payload.grounding.model_copy(
        update={"diseaseAreas": ["BRANDEX", "Plaque psoriasis", "plaque psoriasis"]})})
    h.harvest(p, use_autocomplete=False, store=mem())
    assert calls == ["brandex", "plaque psoriasis"]  # queried lower-cased, once per term


def test_all_sources_off(payload, monkeypatch):
    monkeypatch.setattr(h, "fetch_autocomplete", lambda *a: pytest.fail("called"))
    r = h.harvest(payload, use_autocomplete=False, use_trends=False, store=mem())
    assert r.questions == []
