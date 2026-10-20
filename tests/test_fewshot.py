import json
from collections import Counter

import pytest

from promptgen import config
from promptgen.fewshot import Sampler, build_pool, is_relevant, tag
from promptgen.schema import Payload


@pytest.fixture
def payload():
    return Payload.model_validate(json.load(open("sample_payload.json")))


def q(text):
    return {"text": text, "source": "autocomplete", "seed": ""}


@pytest.mark.parametrize("text,topic", [
    ("is brandex safe during pregnancy", "safety"),
    ("brandex side effects", "safety"),
    ("how long does brandex take to work", "efficacy"),
    ("can i get it on the nhs", "access_cost"),
    ("can dogs get plaque psoriasis", "general"),   # "get" alone is not access_cost
    ("is psoriasis contagious", "disease"),
    ("what causes plaque psoriasis", "disease"),
    ("can children use brandex", "population"),
    ("who makes brandex", "corporate"),
    ("is brandex a biologic", "moa"),
    ("does homework help", "general"),              # keyword must start a word
])
def test_tag(text, topic):
    assert tag(text) == topic


def test_relevance(payload):
    assert is_relevant("Is BRANDEX safe", payload)
    assert is_relevant("can psoriasis spread", payload)          # disease word
    assert is_relevant("competitor a vs brandex", payload)
    assert not is_relevant("who is tim ruckman", payload)


def test_all_sources_off_uses_all_fallbacks(payload):
    pool = build_pool([], payload)
    assert len(pool) == len(config.FALLBACK_EXAMPLES)
    assert all(x["source"] == "fallback" for x in pool)
    assert "how long does Brandex take to work" in [x["text"] for x in pool]
    assert "does Brandex work for plaque psoriasis" in [x["text"] for x in pool]
    assert all(Counter(x["topic"] for x in pool)[t] >= 2 for t in ("safety", "efficacy", "access_cost"))


def test_fallback_without_disease_areas(payload):
    pool = build_pool([], payload.model_copy(update={"grounding": payload.grounding.model_copy(update={"diseaseAreas": []})}))
    assert "does Brandex work for my condition" in [x["text"] for x in pool]


def test_short_topics_topped_up_only(payload):
    harvested = [q(f"is plaque psoriasis thing {i}") for i in range(20)] + [
        q("brandex side effects"), q("is brandex safe"),          # safety has 2: no top-up
    ]
    pool = build_pool(harvested, payload)
    fb = [x for x in pool if x["source"] == "fallback"]
    assert fb and "safety" not in {x["topic"] for x in fb}


def make_pool():
    pool = [{"text": f"{t} {i}", "topic": t} for t in ("safety", "general") for i in range(12)]
    return pool + [{"text": f"efficacy {i}", "topic": "efficacy"} for i in range(2)]


def test_sampler_mix_and_no_repeats():
    s = Sampler(make_pool(), seed=1)
    batch = s.pick("safety")
    topics = [x["topic"] for x in batch]
    assert len(batch) == 9 and topics.count("safety") == 6 and topics.count("general") == 3
    assert len({x["text"] for x in batch}) == 9


def test_sampler_rotates_across_batches():
    s = Sampler(make_pool(), seed=1)
    a = {x["text"] for x in s.pick("safety") if x["topic"] == "safety"}
    b = {x["text"] for x in s.pick("safety") if x["topic"] == "safety"}
    assert not a & b  # 12 safety examples: two batches of 6 never overlap


def test_sampler_fills_short_topic():
    batch = Sampler(make_pool(), seed=1).pick("access_cost")  # no access_cost examples at all
    assert len(batch) == 9 and len({x["text"] for x in batch}) == 9


def test_sampler_tiny_pool_does_not_hang():
    pool = [{"text": "a", "topic": "safety"}, {"text": "b", "topic": "general"}]
    s = Sampler(pool, seed=1)
    for _ in range(5):
        assert sorted(x["text"] for x in s.pick("safety")) == ["a", "b"]
