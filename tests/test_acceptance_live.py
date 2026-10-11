"""Design §7 acceptance criteria against the real model. Costs a few cents and ~1 minute.

    RUN_LIVE=1 pytest tests/test_acceptance_live.py -v -s
"""

import itertools
import json
import os
import re
from difflib import SequenceMatcher

import pytest

from promptgen import config
from promptgen.generator import generate
from promptgen.harvest import harvest
from promptgen.postprocess import normalize
from promptgen.schema import Output, Payload

pytestmark = pytest.mark.skipif(
    not (os.getenv("RUN_LIVE") and os.getenv("OPENROUTER_API_KEY")),
    reason="live: set RUN_LIVE=1 and OPENROUTER_API_KEY",
)
EXISTING = "What are the side effects of Brandex?"


def quiet(msg):
    pass


@pytest.fixture(scope="module")
def payload():
    return Payload.model_validate(json.load(open("sample_payload.json")))


@pytest.fixture(scope="module")
def run47(payload):
    return generate(payload, on_event=quiet)


@pytest.fixture(scope="module")
def run100(payload):
    harvest(payload)  # warm the cache first: the criterion is about a warm run
    return generate(payload, n=100, on_event=quiet)


def texts(run):
    return [p.text for p in run.output.prompts]


def test_1_n47_returns_47_split_19_14_14(run47):
    out = run47.output.model_dump()
    Output.model_validate(out)
    by = {c: sum(p["topic"]["code"] == c for p in out["prompts"]) for c in config.TOPIC_WEIGHTS}
    assert len(out["prompts"]) == 47
    assert by == {"safety": 19, "efficacy": 14, "access_cost": 14}


@pytest.mark.parametrize("run_name", ["run47", "run100"])
def test_2_no_near_duplicates(run_name, request):
    ts = [normalize(t) for t in texts(request.getfixturevalue(run_name))]
    existing = normalize(EXISTING)
    assert all(SequenceMatcher(None, t, existing).ratio() < config.DEDUP_THRESHOLD for t in ts)
    for a, b in itertools.combinations(ts, 2):
        assert SequenceMatcher(None, a, b).ratio() < config.DEDUP_THRESHOLD, (a, b)


@pytest.mark.parametrize("run_name", ["run47", "run100"])
def test_3_topics_and_category_ids(run_name, request, payload):
    cats = {t.code: t.categoryId for t in payload.topics}
    for p in request.getfixturevalue(run_name).output.prompts:
        assert p.topic.code in config.TOPIC_WEIGHTS
        assert p.topic.categoryId == cats[p.topic.code]


def test_4_all_sources_off_uses_fallback(payload):
    r = generate(payload, n=15, use_autocomplete=False, use_trends=False, on_event=quiet)
    assert r.harvest.questions == []
    assert r.pool and all(q["source"] == "fallback" for q in r.pool)
    assert len(r.records) == 15


def test_5_n100_warm_under_90s(run100):
    print(f"\nn=100: {len(run100.records)} prompts, {run100.calls} calls, {run100.seconds:.1f}s")
    assert len(run100.records) == 100
    assert run100.harvest.stats["fetched"] == 0  # warm
    assert run100.seconds < 90


def test_6_prelaunch_has_no_availability_wording(payload):
    pre = payload.model_copy(update={"grounding": payload.grounding.model_copy(update={"stage": "Pre-launch"})})
    r = generate(pre, n=30, on_event=quiet)
    # wider than the in-pipeline filter, so this checks the model's behaviour, not just the filter
    implies = re.compile(
        r"\b(prescribed|available now|now available|is available|currently available|on the nhs now|"
        r"(?:is|been|was|got|now) approved|licensed for|i(?:'m| am| was) (?:on|taking|using) brandex|"
        r"since (?:starting|taking) brandex)\b", re.I)
    hits = [t for t in texts(r) if implies.search(t)]
    rejected = sum(b.get("rejected", {}).get("implies availability", 0) for b in r.batches)
    print(f"\npre-launch: {len(r.records)} prompts, filter rejected {rejected}; spot check:")
    for t in texts(r)[::5]:
        print("   ", t)
    assert len(r.records) == 30
    assert hits == []
