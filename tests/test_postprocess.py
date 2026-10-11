import json
from datetime import datetime, timezone

import pytest

from promptgen import config
from promptgen.postprocess import Deduper, assemble, filter_batch, grounding_used, normalize, validate
from promptgen.schema import Output, Payload

EXISTING = "What are the side effects of Brandex?"


@pytest.fixture
def payload():
    return Payload.model_validate(json.load(open("sample_payload.json")))


def item(text, intent="informational", competitor=None):
    return {"text": text, "intent": intent, "competitor": competitor}


# ---- dedup threshold (0.85) ----

def test_normalize():
    assert normalize("  What's   THE cost?! ") == "whats the cost"


@pytest.mark.parametrize("text,dup", [
    ("What are the side effects of Brandex?", True),               # 1.0
    ("what are the side-effects of brandex", True),                # 0.986
    ("What are the side effects of Brandex for my kids?", True),   # 0.857, just above
    ("What are the rare side effects of Brandex tablets?", False), # 0.847, just below
    ("Does Brandex have side effects?", False),                    # paraphrase: left to the LLM rule
])
def test_dedup_threshold(text, dup):
    assert config.DEDUP_THRESHOLD == 0.85
    assert Deduper([EXISTING]).is_dup(text) is dup


def test_dedup_within_batch_and_quota(payload):
    d = Deduper([EXISTING])
    items = [item("Can I drink alcohol on Brandex?"), item("Can I drink alcohol on Brandex??"),
             item(EXISTING), item("Is Brandex safe if I'm pregnant?"), item("How long until Brandex works?")]
    r = filter_batch(items, payload, d, quota=2)
    assert [x["text"] for x in r.accepted] == ["Can I drink alcohol on Brandex?", "Is Brandex safe if I'm pregnant?"]
    assert (r.duplicates, r.surplus) == (2, 1)
    assert d.texts == [EXISTING, *[x["text"] for x in r.accepted]]


# ---- validation ----

@pytest.mark.parametrize("raw,reason", [
    ("not a dict", "malformed"),
    ({"text": 5}, "malformed"),
    (item("Is Brandex safe"), "not a question"),
    (item("Brandex ok?"), "length"),
    (item("Is Brandex " + "really " * 30 + "safe?"), "length"),
    (item("Is 50mg of Brandex too much for me?"), "dose"),
    (item("Can I take Brandex with 2 ml of alcohol?"), "dose"),
])
def test_validate_rejects(payload, raw, reason):
    assert validate(raw, payload) == (None, reason)


def test_validate_fixes_intent_and_competitor(payload):
    out, _ = validate(item("  Is Brandex better than   competitor a?", "made_up", " COMPETITOR A "), payload)
    assert out == {"text": "Is Brandex better than competitor a?", "intent": "informational",
                   "competitor": "Competitor A"}
    out, _ = validate(item("Is Brandex better than Humira?", "comparison", "Humira"), payload)
    assert out["competitor"] is None  # not in payload list


def test_competitor_dropped_when_option_off(payload):
    payload.options.includeCompetitorPrompts = False
    out, _ = validate(item("Is Brandex better than Competitor A?", "comparison", "Competitor A"), payload)
    assert out["competitor"] is None


@pytest.mark.parametrize("text,blocked", [
    ("My GP prescribed Brandex, is it safe with alcohol?", True),
    ("Is Brandex available on the NHS now?", True),
    ("Now that Brandex is approved, can I ask for it?", True),
    ("Is Brandex now available in the UK?", True),
    ("When will Brandex be available on the NHS?", False),
    ("Will Brandex work for my psoriasis?", False),
])
def test_prelaunch_wording(payload, text, blocked):
    assert validate(item(text), payload)[0] is not None  # launched: all allowed
    pre = payload.model_copy(update={"grounding": payload.grounding.model_copy(update={"stage": "Pre-launch"})})
    out, reason = validate(item(text), pre)
    assert (out is None) is blocked
    if blocked:
        assert reason == "implies availability"


# ---- groundingUsed ----

@pytest.mark.parametrize("text,expected", [
    ("Is Brandex better than Competitor A at clearing my psoriasis?", ["brand", "competitors[0]", "diseaseAreas[0]"]),
    ("Will brandex clear my plaque psoriasis?", ["brand", "diseaseAreas[0]"]),
    ("Is competitor b cheaper?", ["competitors[1]"]),
    ("Does my skin get better in summer?", []),
    ("Is Brandexa a thing?", []),  # whole-word brand match
])
def test_grounding_used(payload, text, expected):
    assert grounding_used(text, payload) == expected


def test_grounding_used_follows_brand_override(payload):
    p = payload.with_grounding(brand="Tylenol", disease_areas=["Headache"], competitors=["Advil"])
    assert grounding_used("Is Tylenol better than Advil for my headaches?", p) == \
        ["brand", "competitors[0]", "diseaseAreas[0]"]
    assert grounding_used("Is Brandex safe?", p) == []


# ---- assembly ----

def test_assemble(payload):
    records = [
        {"text": "Is Brandex better than Competitor A at clearing my psoriasis?", "intent": "comparison",
         "competitor": "Competitor A", "topic": "efficacy", "angle": "vs competitor", "batchId": "efficacy-1"},
        {"text": "Can I drink alcohol on Brandex?", "intent": "side_effects", "competitor": None, "topic": "safety"},
    ]
    out = assemble(payload, records, now=datetime(2026, 10, 3, 10, 15, tzinfo=timezone.utc))
    d = out.model_dump()
    Output.model_validate(d)
    assert d["generatedAt"] == "2026-10-03T10:15:00Z" and d["requestId"] == "pg_7f3c2a"
    first = d["prompts"][0]
    assert first == {
        "tempId": "g_001",
        "text": "Is Brandex better than Competitor A at clearing my psoriasis?",
        "audience": {"code": "patient", "name": "Patient"},
        "topic": {"code": "efficacy", "name": "Efficacy", "categoryId": 3},
        "competitor": "Competitor A",
        "intent": "comparison",
        "groundingUsed": ["brand", "competitors[0]", "diseaseAreas[0]"],
    }  # debug keys (angle, batchId) dropped
    assert d["prompts"][1]["tempId"] == "g_002" and d["prompts"][1]["topic"]["categoryId"] == 4
