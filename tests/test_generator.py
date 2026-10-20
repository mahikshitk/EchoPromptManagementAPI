import json
import random
import re
import string
import threading

import pytest

from promptgen import config, generator as g, llm
from promptgen.schema import Output, Payload


@pytest.fixture
def payload():
    return Payload.model_validate(json.load(open("sample_payload.json")))


def make(**opts):
    """Sample payload with option overrides, re-validated."""
    d = json.load(open("sample_payload.json"))
    d["options"].update(opts)
    return Payload.model_validate(d)


@pytest.mark.parametrize("q,sizes", [(19, [10, 9]), (14, [7, 7]), (12, [12]), (40, [10, 10, 10, 10]), (0, [])])
def test_batch_sizes(q, sizes):
    assert g.batch_sizes(q) == sizes


def test_angle_orders_rotate_between_runs(payload):
    alloc = payload.allocate(seed=0)
    a = g.angle_orders(payload, alloc, seed=1)
    assert a == g.angle_orders(payload, alloc, seed=1)
    assert a != g.angle_orders(payload, alloc, seed=2)
    assert sorted(a["patient", "safety"]) == sorted(g.angles_for("safety", "patient", payload))


def test_angles_competitors(payload):
    assert "vs Competitor A" in g.angles_for("efficacy", "hcp", payload)
    payload.options.includeCompetitorPrompts = False
    assert not any(a.startswith("vs") for a in g.angles_for("efficacy", "hcp", payload))


# ---- prompts ----

def audience(payload, code):
    return next(a for a in payload.active_audiences() if a.code == code)


def topic(payload, code):
    return next(t for t in payload.active_topics() if t.code == code)


def test_system_prompt_stage_rule(payload):
    patient = audience(payload, "patient")
    assert "not launched" not in g.system_prompt(payload, patient)
    pre = payload.model_copy(update={"grounding": payload.grounding.model_copy(update={"stage": "Pre-launch"})})
    assert "not launched yet (stage: Pre-launch)" in g.system_prompt(pre, patient)


def test_system_prompt_per_audience(payload):
    pat, hcp = (g.system_prompt(payload, audience(payload, c)) for c in ("patient", "hcp"))
    assert "First person" in pat and "EXAMPLE SEARCHES" in pat and "- guideline:" not in pat
    assert "clinician" in hcp and "EXAMPLE SEARCHES" not in hcp and "- guideline:" in hcp
    assert "clinical, guideline and endpoint terms" in hcp  # payload voice


def test_user_prompt_contents(payload):
    u = g.user_prompt(payload, topic(payload, "safety"), "vs Competitor B", 13, [{"text": "is it safe"}],
                      ["Old question?"])
    assert "WRITE: 13 questions" in u and "- is it safe" in u and "- Old question?" in u
    assert 'set "competitor" to "Competitor B"' in u and "Name Brandex in most questions" in u
    assert "https://pharmaco.com" in u


def test_user_prompt_disease_brand_optional_no_examples(payload):
    u = g.user_prompt(payload, topic(payload, "disease"), "symptoms and triggers", 3, [], [])
    assert "name Brandex in only some" in u and "EXAMPLE SEARCHES" not in u


def test_every_pair_has_angles(payload):
    for a, t in payload.pairs():
        assert g.angles_for(t.code, a.code, payload), (a.code, t.code)


# ---- end-to-end with a fake LLM ----

class FakeLLM:
    """Returns `count` unique questions per call; `short` makes every call return only that many."""

    def __init__(self, short=None, fail=False):
        self.n, self.short, self.fail, self.users, self.systems = 0, short, fail, [], []
        self.lock = threading.Lock()

    def __call__(self, system, user, model=None):
        with self.lock:  # called from worker threads
            self.users.append(user)
            self.systems.append(system)
        if self.fail:
            raise llm.LLMError("boom")
        count = int(re.search(r"WRITE: (\d+)", user).group(1))
        count = min(count, self.short) if self.short is not None else count
        out = []
        for _ in range(count):
            with self.lock:
                self.n += 1
                rnd = random.Random(self.n)  # random words: far below the 0.85 dedup threshold
            words = " ".join("".join(rnd.choices(string.ascii_lowercase, k=6)) for _ in range(5))
            out.append({"text": f"Is {words} ok with Brandex?", "intent": "informational", "competitor": None})
        return llm.LLMResult({"prompts": out}, 10, 20, "fake")


def run(payload, fake, monkeypatch, seed=0):
    monkeypatch.setattr(g.llm, "chat_json", fake)
    return g.generate(payload, False, False, seed=seed, on_event=lambda m: None)


def test_generate_all_pairs(payload, monkeypatch):
    fake = FakeLLM()
    r = run(payload, fake, monkeypatch)
    out = r.output.model_dump()
    Output.model_validate(out)
    assert len(out["prompts"]) == 47 and r.calls == 28  # one call per pair, all quotas <= BATCH_SIZE
    assert [p["tempId"] for p in out["prompts"]][:2] == ["g_001", "g_002"]
    got = {}
    for p in out["prompts"]:
        key = (p["audience"]["code"], p["topic"]["code"])
        got[key] = got.get(key, 0) + 1
    assert got == {k: v for k, v in payload.allocate(seed=0).items() if v}
    cats = {t.code: t.categoryId for t in payload.topics}
    names = {a.code: a.name for a in payload.audiences}
    assert all(p["topic"]["categoryId"] == cats[p["topic"]["code"]] for p in out["prompts"])
    assert all(p["audience"]["name"] == names[p["audience"]["code"]] for p in out["prompts"])
    assert all({"angle", "batchId", "referenceIds"} <= rec.keys() for rec in r.records)
    # existingPrompts (audience "") reach every pair; patient pairs get examples, others don't
    assert all("What are the side effects of Brandex?" in u for u in fake.users)
    assert all(("EXAMPLE SEARCHES" in u) == ("first person" in s.lower()) for s, u in zip(fake.systems, fake.users))
    # overgenerate 1.3x: quota 2 -> ask 3, quota 1 -> ask 2
    assert {re.search(r"WRITE: (\d+)", u).group(1) for u in fake.users} == {"2", "3"}


def test_dedup_is_per_audience(monkeypatch):
    d = json.load(open("sample_payload.json"))
    d["existingPrompts"].append({"text": "Is Brandex cost-effective for the NHS?", "audience": "policy_payer"})
    fake = FakeLLM()
    run(Payload.model_validate(d), fake, monkeypatch)
    payer = [u for s, u in zip(fake.systems, fake.users) if "health economist" in s]
    others = [u for s, u in zip(fake.systems, fake.users) if "health economist" not in s]
    assert all("Is Brandex cost-effective for the NHS?" in u for u in payer)
    assert not any("Is Brandex cost-effective for the NHS?" in u for u in others)


def test_patient_sources_off_uses_fallback(payload, monkeypatch):
    r = run(payload, FakeLLM(), monkeypatch)
    assert r.pool and all(q["source"] == "fallback" for q in r.pool)


def test_no_fewshot_audience_skips_harvest(monkeypatch):
    d = json.load(open("sample_payload.json"))
    d["audiences"] = [a for a in d["audiences"] if a["code"] != "patient"]
    monkeypatch.setattr(g, "harvest", lambda *a, **k: pytest.fail("harvest called"))
    fake = FakeLLM()
    r = run(Payload.model_validate(d), fake, monkeypatch)
    assert r.pool == [] and not any("EXAMPLE SEARCHES" in u for u in fake.users)
    assert {p.audience.code for p in r.output.prompts} == {"hcp", "policy_payer", "media_investor"}


def single_pair(**opts) -> Payload:
    d = json.load(open("sample_payload.json"))
    d["audiences"], d["topics"] = d["audiences"][:1], [t for t in d["topics"] if t["code"] == "safety"]
    d["options"].update(opts)
    return Payload.model_validate(d)


def test_topup_then_warn_never_pad(monkeypatch):
    fake = FakeLLM(short=1)  # every call yields 1 prompt
    r = run(single_pair(maxPrompts=5, promptsPerPair=5), fake, monkeypatch)  # 1 batch + 2 top-ups -> 3 of 5
    assert r.calls == 3 and len(r.records) == 3
    assert [b["batchId"] for b in r.batches] == ["patient/safety-1", "patient/safety-topup1", "patient/safety-topup2"]
    assert r.warnings[-1] == "patient/safety: returned 3 of 5 after top-up"
    assert len({b["angle"] for b in r.batches}) == 3  # top-ups move to unused angles


def test_llm_failure_is_a_warning(monkeypatch):
    r = run(single_pair(), FakeLLM(fail=True), monkeypatch)
    assert r.records == [] and r.output.prompts == []
    assert any("LLM call failed" in w for w in r.warnings)


def test_pairs_with_zero_quota_are_not_called(monkeypatch):
    fake = FakeLLM()
    r = run(make(maxPrompts=10), fake, monkeypatch)  # 10 prompts over 28 pairs
    assert r.calls == 10 and len(r.records) == 10
