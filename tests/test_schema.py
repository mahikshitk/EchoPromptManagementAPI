import json

import pytest
from pydantic import ValidationError

from promptgen.schema import Output, Payload


def sample() -> dict:
    return json.load(open("sample_payload.json"))


def test_sample_parses():
    p = Payload.model_validate(sample())
    assert [(t.code, t.categoryId) for t in p.active_topics()][:2] == [("efficacy", 3), ("safety", 4)]
    assert p.grounding.websites.corporate == "https://pharmaco.com"
    assert p.do_not_repeat("patient") == ["What are the side effects of Brandex?"]
    assert p.grounding.model_extra["brandId"] == 12  # extra="allow" keeps unknown fields


def test_all_audiences_and_topics_active():
    p = Payload.model_validate(sample())
    assert [a.code for a in p.active_audiences()] == ["patient", "hcp", "policy_payer", "media_investor"]
    assert [t.code for t in p.active_topics()] == [
        "efficacy", "safety", "moa", "population", "disease", "access_cost", "corporate"]
    assert len(p.pairs()) == 28


def test_unknown_codes_ignored_with_warning():
    d = sample()
    d["audiences"].append({"code": "pharmacist", "name": "Pharmacist"})
    d["topics"].append({"code": "pricing", "name": "Pricing", "categoryId": 99})
    d["audiences"].append(d["audiences"][0])  # duplicate code counted once
    p = Payload.model_validate(d)
    assert len(p.pairs()) == 28
    w = " | ".join(p.warnings())
    assert "pharmacist" in w and "pricing" in w


def test_warnings():
    w = " | ".join(Payload.model_validate(sample()).warnings())
    for word in ("caps 28 pairs x 2 = 56", "includeReddit", "benchmark"):
        assert word in w


def test_benchmark_parsed():
    b = Payload.model_validate(sample()).benchmark
    assert b.hasBenchmark and [d.documentId for d in b.documents] == [901, 902]
    d = sample()
    del d["benchmark"]
    assert Payload.model_validate(d).benchmark is None


def test_include_autocomplete_defaults_true():
    assert Payload.model_validate(sample()).options.includeAutocomplete is True


def test_do_not_repeat_per_audience():
    d = sample()
    d["existingPrompts"].append({"text": "HCP question?", "audience": "hcp"})
    p = Payload.model_validate(d)
    assert "HCP question?" not in p.do_not_repeat("patient")
    assert p.do_not_repeat("hcp") == ["What are the side effects of Brandex?", "HCP question?"]


def test_rejects_no_known_audience():
    d = sample()
    d["audiences"] = [{"code": "pharmacist", "name": "Pharmacist"}]
    with pytest.raises(ValidationError, match="audiences"):
        Payload.model_validate(d)


def test_rejects_zero_prompts_per_pair():
    d = sample()
    d["options"]["promptsPerPair"] = 0
    with pytest.raises(ValidationError):
        Payload.model_validate(d)


def with_options(**opts) -> Payload:
    d = sample()
    d["options"].update(opts)
    return Payload.model_validate(d)


def test_allocate_capped_spreads_evenly():
    a = with_options(maxPrompts=47, promptsPerPair=2).allocate(seed=1)
    assert sum(a.values()) == 47 and len(a) == 28
    assert sorted(a.values()) == [1] * 9 + [2] * 19


def test_allocate_uncapped_gives_every_pair_its_target():
    assert set(with_options(maxPrompts=100, promptsPerPair=3).allocate().values()) == {3}  # 84 <= 100
    a = with_options(maxPrompts=100, promptsPerPair=4).allocate()  # 112 > 100
    assert sum(a.values()) == 100 and sorted(set(a.values())) == [3, 4]


def test_allocate_fewer_prompts_than_pairs():
    a = with_options(maxPrompts=20, promptsPerPair=2).allocate(seed=0)
    assert sum(a.values()) == 20 and sorted(a.values()) == [0] * 8 + [1] * 20
    assert "some get none" in " | ".join(with_options(maxPrompts=20).warnings())


def test_allocate_rotates_between_runs():
    p = with_options(maxPrompts=47, promptsPerPair=2)
    assert p.allocate(seed=1) == p.allocate(seed=1)
    assert len({tuple(p.allocate().values()) for _ in range(10)}) > 1


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
