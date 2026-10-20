"""Acceptance criteria against the real model. Costs a few cents and a few minutes.

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


def with_options(payload, **opts) -> Payload:
    return Payload.model_validate({**payload.model_dump(), "options": {**payload.options.model_dump(), **opts}})


@pytest.fixture(scope="module")
def run47(payload):
    return generate(payload, seed=0, on_event=quiet)


@pytest.fixture(scope="module")
def run100(payload):
    harvest(payload)  # warm the cache first: the criterion is about a warm run
    return generate(with_options(payload, maxPrompts=100, promptsPerPair=4), on_event=quiet)


def texts(run):
    return [p.text for p in run.output.prompts]


def by_pair(run):
    got = {}
    for p in run.output.prompts:
        got[p.audience.code, p.topic.code] = got.get((p.audience.code, p.topic.code), 0) + 1
    return got


def test_1_n47_fills_every_pair_quota(run47, payload):
    Output.model_validate(run47.output.model_dump())
    assert len(run47.records) == 47
    assert by_pair(run47) == {k: v for k, v in payload.allocate(seed=0).items() if v}


@pytest.mark.parametrize("run_name", ["run47", "run100"])
def test_2_no_near_duplicates_within_an_audience(run_name, request):
    run = request.getfixturevalue(run_name)
    existing = normalize(EXISTING)
    for aud in {p.audience.code for p in run.output.prompts}:
        ts = [normalize(p.text) for p in run.output.prompts if p.audience.code == aud]
        assert all(SequenceMatcher(None, t, existing).ratio() < config.DEDUP_THRESHOLD for t in ts)
        for a, b in itertools.combinations(ts, 2):
            assert SequenceMatcher(None, a, b).ratio() < config.DEDUP_THRESHOLD, (a, b)


@pytest.mark.parametrize("run_name", ["run47", "run100"])
def test_3_codes_ids_and_intents(run_name, request, payload):
    cats = {t.code: t.categoryId for t in payload.topics}
    for p in request.getfixturevalue(run_name).output.prompts:
        assert p.topic.categoryId == cats[p.topic.code]
        assert p.intent in config.AUDIENCE_PROFILES[p.audience.code]["intents"]


def test_4_all_sources_off_uses_fallback(payload):
    r = generate(with_options(payload, maxPrompts=15), use_autocomplete=False, use_trends=False, on_event=quiet)
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
    r = generate(with_options(pre, maxPrompts=30), on_event=quiet)
    # wider than the in-pipeline filter, so this checks the model's behaviour, not just the filter
    implies = re.compile(
        r"\b(prescribed|available now|now available|is available|currently available|on the nhs now|"
        r"(?:is|been|was|got|now) approved|licensed for|i(?:'m| am| was) (?:on|taking|using) brandex|"
        r"since (?:starting|taking) brandex)\b", re.I)
    hits = [t for t in texts(r) if implies.search(t)]
    rejected = sum(b.get("rejected", {}).get("implies availability", 0) for b in r.batches)
    print(f"\npre-launch: {len(r.records)} prompts, filter rejected {rejected}; spot check:")
    for t in texts(r)[::3]:
        print("   ", t)
    assert len(r.records) == 30
    assert hits == []


def test_7_audiences_sound_different(run47):
    print()
    for p in run47.output.prompts:
        print(f"    {p.audience.code:15} {p.topic.code:12} {p.intent:22} {p.text}")
    patient = [p.text for p in run47.output.prompts if p.audience.code == "patient"]
    assert sum(bool(re.search(r"\b(i|my|me|i'm)\b", t, re.I)) for t in patient) >= len(patient) / 2
