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


KEY = ("autocomplete", "q", "en-GB", "GB")


def test_cache_ttl(tmp_path):
    c = h.Cache(str(tmp_path / "c.sqlite"), ttl=0)
    assert not c.fresh(KEY) and c.get(KEY) == []
    c.put(KEY, ["a"])
    assert not c.fresh(KEY) and c.get(KEY) == ["a"]  # expired, still served
    assert h.Cache(":memory:").fresh(KEY) is False


def test_cache_is_append_only():
    c = h.Cache(":memory:")
    c.put(KEY, ["a", "b"])
    c.put(KEY, ["b", "c"])  # a refresh that no longer returns "a"
    assert c.get(KEY) == ["a", "b", "c"] and c.size() == 3
    assert c.fresh(KEY)


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
    assert len(r2.questions) == len(r.questions) and r2.stored == r.stored > 0


def test_refresh_grows_the_pool(payload, monkeypatch):
    cache = h.Cache(":memory:", ttl=0)
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: [f"what is {term}"])
    h.harvest(payload, use_autocomplete=False, cache=cache)
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: [f"is {term} safe"])
    r = h.harvest(payload, use_autocomplete=False, cache=cache)
    assert {"what is brandex", "is brandex safe"} <= {q["text"] for q in r.questions}


def test_offline_serves_store_without_fetching(payload, monkeypatch):
    cache = h.Cache(":memory:", ttl=0)
    cache.put(("trends", "brandex", "en-GB", "GB"), ["what is brandex"])
    monkeypatch.setattr(h, "fetch_trends", lambda *a: pytest.fail("fetched"))
    monkeypatch.setattr(h, "fetch_autocomplete", lambda *a: pytest.fail("fetched"))
    r = h.harvest(payload, cache=cache, offline=True)
    assert [q["text"] for q in r.questions] == ["what is brandex"] and r.stats["fetched"] == 0


def test_failure_falls_back_to_stored_then_empty(payload, monkeypatch):
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
    s = h.seeds(payload.model_copy(update={"grounding": payload.grounding.model_copy(update={"brand": "Tylenol", "diseaseAreas": ["Headache"]})}))
    assert "is tylenol" in s and "headache cost" in s
    assert not any("brandex" in x or "psoriasis" in x for x in s)


# ---- S3 sync ----

class FakeS3:
    """download_file/upload_file against an in-memory bucket."""

    def __init__(self, fail=False):
        self.objects, self.fail, self.uploads = {}, fail, 0

    def download_file(self, bucket, key, path):
        from botocore.exceptions import ClientError
        if self.fail:
            raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject")
        if (bucket, key) not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        open(path, "wb").write(self.objects[bucket, key])

    def upload_file(self, path, bucket, key):
        self.uploads += 1
        self.objects[bucket, key] = open(path, "rb").read()


@pytest.fixture
def s3(monkeypatch):
    import boto3
    fake = FakeS3()
    monkeypatch.setattr(boto3, "client", lambda name: fake)
    return fake


URI = "s3://bucket/harvest.sqlite"


def test_s3_first_run_then_shared_between_instances(tmp_path, s3):
    a = h.Cache(str(tmp_path / "a.sqlite"), s3_uri=URI)
    a.pull()  # nothing in S3 yet: fine
    a.put(KEY, ["a"])
    a.push()
    b = h.Cache(str(tmp_path / "b.sqlite"), s3_uri=URI)  # e.g. a fresh Lambda container
    b.pull()
    assert b.get(KEY) == ["a"] and b.fresh(KEY)


def test_s3_push_merges_concurrent_writes(tmp_path, s3):
    a = h.Cache(str(tmp_path / "a.sqlite"), s3_uri=URI)
    b = h.Cache(str(tmp_path / "b.sqlite"), s3_uri=URI)
    a.pull(), b.pull()
    a.put(KEY, ["from a"])
    b.put(KEY, ["from b"])
    a.push()
    b.push()  # re-pulls a's rows before uploading
    c = h.Cache(str(tmp_path / "c.sqlite"), s3_uri=URI)
    c.pull()
    assert sorted(c.get(KEY)) == ["from a", "from b"]
    assert not list(tmp_path.glob("*.remote"))  # temp download cleaned up


def test_harvest_pushes_only_after_new_fetches(payload, tmp_path, s3, monkeypatch):
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: [f"what is {term}"])
    cache = h.Cache(str(tmp_path / "a.sqlite"), s3_uri=URI)
    h.harvest(payload, use_autocomplete=False, cache=cache)
    h.harvest(payload, use_autocomplete=False, cache=cache)  # all fresh: nothing to push
    assert s3.uploads == 1


def test_s3_errors_never_fail_the_run(payload, tmp_path, monkeypatch):
    import boto3
    monkeypatch.setattr(boto3, "client", lambda name: FakeS3(fail=True))
    monkeypatch.setattr(h, "fetch_trends", lambda term, *a: [f"what is {term}"])
    r = h.harvest(payload, use_autocomplete=False, cache=h.Cache(str(tmp_path / "a.sqlite"), s3_uri=URI))
    assert r.questions and any("store pull" in e for e in r.errors) and any("store push" in e for e in r.errors)


def test_default_cache_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HARVEST_DB_PATH", str(tmp_path / "x" / "h.sqlite"))
    monkeypatch.setenv("HARVEST_DB_S3_URI", URI)
    c = h.default_cache()
    assert c.path == str(tmp_path / "x" / "h.sqlite") and c.s3_uri == URI
