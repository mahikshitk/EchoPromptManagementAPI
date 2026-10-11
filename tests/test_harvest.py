import json

import pytest

from promptgen import harvest as h
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


def test_cache_ttl_and_stale(tmp_path):
    c = h.Cache(str(tmp_path / "c.sqlite"), ttl=0)
    key = ("autocomplete", "q", "en-GB", "GB")
    c.put(key, ["a"])
    assert c.get(key) is None  # expired
    assert c.get(key, stale_ok=True) == ["a"]


def test_harvest_filters_dedups_and_caches(payload, monkeypatch):
    calls = []

    def fake_ac(q, lang, geo):
        calls.append(q)
        return [q, "Is it safe?", "is it safe", "brandex tablets", f"how long does {q} last"]

    monkeypatch.setattr(h, "fetch_autocomplete", fake_ac)
    cache = h.Cache(":memory:")
    r = h.harvest(payload, use_trends=False, cache=cache)
    texts = [q["text"] for q in r.questions]
    assert "Is it safe?" in texts and "is it safe" not in texts  # deduped
    assert not any(t in h.seeds(payload) for t in texts)        # seed echo dropped
    assert "brandex tablets" not in texts                 # not question-shaped
    assert r.stats["fetched"] == len(calls)

    r2 = h.harvest(payload, use_trends=False, cache=cache)
    assert r2.stats == {"cache_hits": len(calls), "fetched": 0, "stale": 0, "failed": 0}
    assert len(r2.questions) == len(r.questions)


def test_failure_falls_back_to_stale_then_empty(payload, monkeypatch):
    cache = h.Cache(":memory:", ttl=0)
    cache.put(("trends", "brandex", "en-GB", "GB"), ["what is brandex"])

    def boom(*a):
        raise RuntimeError("429")

    monkeypatch.setattr(h, "fetch_trends", boom)
    r = h.harvest(payload, use_autocomplete=False, use_trends=True, cache=cache)
    assert [q["text"] for q in r.questions] == ["what is brandex"]
    assert r.stats["stale"] == 1 and r.stats["failed"] == 1 and len(r.errors) == 2


def test_all_sources_off(payload, monkeypatch):
    monkeypatch.setattr(h, "fetch_autocomplete", lambda *a: pytest.fail("called"))
    r = h.harvest(payload, use_autocomplete=False, use_trends=False, cache=h.Cache(":memory:"))
    assert r.questions == []


def test_seeds_follow_brand_override(payload):
    s = h.seeds(payload.with_grounding(brand="Tylenol", disease_areas=["Headache"]))
    assert "is tylenol" in s and "headache cost" in s
    assert not any("brandex" in x or "psoriasis" in x for x in s)
