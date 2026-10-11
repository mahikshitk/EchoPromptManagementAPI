import json

import pytest
from pydantic import ValidationError

from promptgen.schema import Output, Payload


def sample() -> dict:
    return json.load(open("sample_payload.json"))


def test_sample_parses():
    p = Payload.model_validate(sample())
    assert p.audience.code == "patient"
    assert [(t.code, t.categoryId) for t in p.enabled_topics()] == [
        ("safety", 4), ("efficacy", 3), ("access_cost", 8)
    ]
    assert p.do_not_repeat() == ["What are the side effects of Brandex?"]
    assert p.grounding.model_extra["brandId"] == 12  # extra="allow" keeps unknown fields


def test_warnings():
    w = " | ".join(Payload.model_validate(sample()).warnings())
    for word in ("hcp", "moa", "corporate", "includeReddit"):
        assert word in w


def test_include_autocomplete_defaults_true():
    d = sample()
    del d["options"]["includeAutocomplete"]
    assert Payload.model_validate(d).options.includeAutocomplete is True


def test_existing_prompt_for_other_audience_not_in_do_not_repeat():
    d = sample()
    d["existingPrompts"].append({"text": "HCP question?", "audience": "hcp"})
    assert "HCP question?" not in Payload.model_validate(d).do_not_repeat()


def test_rejects_no_patient_audience():
    d = sample()
    d["audiences"] = [a for a in d["audiences"] if a["code"] != "patient"]
    with pytest.raises(ValidationError, match="patient"):
        Payload.model_validate(d)


def test_rejects_n_over_100():
    d = sample()
    d["options"]["maxPrompts"] = 101
    with pytest.raises(ValidationError):
        Payload.model_validate(d)


def test_output_schema():
    prompt = {
        "tempId": "g_001",
        "text": "Is Brandex better than Competitor A at clearing my psoriasis?",
        "audience": {"code": "patient", "name": "Patient"},
        "topic": {"code": "efficacy", "name": "Efficacy", "categoryId": 3},
        "competitor": "Competitor A",
        "intent": "comparison",
        "groundingUsed": ["brand", "competitors[0]", "diseaseAreas[0]"],
    }
    doc = {"requestId": "r", "runId": "1", "generatedAt": "2026-10-03T10:15:00Z", "prompts": [prompt]}
    Output.model_validate(doc)
    with pytest.raises(ValidationError):
        Output.model_validate({**doc, "prompts": [{**prompt, "intent": "nope"}]})
    with pytest.raises(ValidationError):  # debug fields must never leak into API output
        Output.model_validate({**doc, "prompts": [{**prompt, "angle": "alcohol"}]})


def test_grounding_override():
    p = Payload.model_validate(sample())
    q = p.with_grounding(" Tylenol ", ["Headache", " "], [])
    assert (q.grounding.brand, q.grounding.diseaseAreas, q.grounding.competitors) == ("Tylenol", ["Headache"], [])
    assert q.grounding.stage == "Launched" and q.grounding.model_extra["brandId"] == 12
    assert p.grounding.brand == "Brandex"  # original untouched
    assert p.with_grounding().grounding == p.grounding


def test_grounding_override_rejects_empty_brand():
    with pytest.raises(ValueError):
        Payload.model_validate(sample()).with_grounding(brand="  ")
