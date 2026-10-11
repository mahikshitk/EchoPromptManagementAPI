import json
import random
import re
import string

import pytest

from promptgen import config, generator as g, llm
from promptgen.schema import Output, Payload


@pytest.fixture
def payload():
    return Payload.model_validate(json.load(open("sample_payload.json")))


CODES = ["safety", "efficacy", "access_cost"]


# ---- allocation (acceptance #7) ----

@pytest.mark.parametrize("n,expected", [
    (7, {"safety": 3, "efficacy": 2, "access_cost": 2}),
    (47, {"safety": 19, "efficacy": 14, "access_cost": 14}),
    (100, {"safety": 40, "efficacy": 30, "access_cost": 30}),
])
def test_allocate(n, expected):
    assert g.allocate(n, CODES, 2) == expected
    assert sum(expected.values()) == n


@pytest.mark.parametrize("n", range(1, 101))
def test_allocate_sums_and_floor(n):
    a = g.allocate(n, CODES, 2)
    assert sum(a.values()) == n
    assert min(a.values()) >= min(2, n // 3)


def test_allocate_floor_takes_from_largest():
    assert g.allocate(10, CODES, 4) == {"safety": 4, "efficacy": 3, "access_cost": 3}  # 4*3 > 10: floor 3
    assert g.allocate(12, CODES, 4) == {"safety": 4, "efficacy": 4, "access_cost": 4}


def test_allocate_subset_renormalises():
    assert g.allocate(10, ["efficacy", "access_cost"], 2) == {"efficacy": 5, "access_cost": 5}


@pytest.mark.parametrize("q,sizes", [(19, [10, 9]), (14, [7, 7]), (12, [12]), (40, [10, 10, 10, 10]), (0, [])])
def test_batch_sizes(q, sizes):
    assert g.batch_sizes(q) == sizes


def test_llm_call_counts():
    calls = lambda n: sum(len(g.batch_sizes(q)) for q in g.allocate(n, CODES, 2).values())
    assert (calls(47), calls(100)) == (6, 10)


def test_angles_competitors(payload):
    assert "vs Competitor A" in g.angles_for("efficacy", payload)
    payload.options.includeCompetitorPrompts = False
    assert not any(a.startswith("vs") for a in g.angles_for("efficacy", payload))


# ---- prompts ----

def test_system_prompt_stage_rule(payload):
    assert "not launched" not in g.system_prompt(payload)
    pre = payload.model_copy(update={"grounding": payload.grounding.model_copy(update={"stage": "Pre-launch"})})
    assert "not launched yet (stage: Pre-launch)" in g.system_prompt(pre)


def test_user_prompt_contents(payload):
    topic = payload.enabled_topics()[1]
    u = g.user_prompt(payload, topic, "vs Competitor B", 13, [{"text": "is it safe"}], ["Old question?"])
    assert "WRITE: 13 questions" in u and "- is it safe" in u and "- Old question?" in u
    assert 'set "competitor" to "Competitor B"' in u


# ---- end-to-end with a fake LLM ----

class FakeLLM:
    """Returns `count` unique questions per call; `short` makes every call return only that many."""

    def __init__(self, short=None, fail=False):
        self.n, self.short, self.fail, self.users = 0, short, fail, []

    def __call__(self, system, user, model=None):
        self.users.append(user)
        if self.fail:
            raise llm.LLMError("boom")
        count = int(re.search(r"WRITE: (\d+)", user).group(1))
        count = min(count, self.short) if self.short is not None else count
        out = []
        for _ in range(count):
            self.n += 1
            rnd = random.Random(self.n)  # random words: far below the 0.85 dedup threshold
            words = " ".join("".join(rnd.choices(string.ascii_lowercase, k=6)) for _ in range(5))
            out.append({"text": f"Is {words} ok with Brandex?", "intent": "informational", "competitor": None})
        return llm.LLMResult({"prompts": out}, 10, 20, "fake")


def run(payload, fake, monkeypatch, n=None):
    monkeypatch.setattr(g.llm, "chat_json", fake)
    return g.generate(payload, n, False, False, seed=0, on_event=lambda m: None)


def test_generate_47(payload, monkeypatch):
    fake = FakeLLM()
    r = run(payload, fake, monkeypatch)
    out = r.output.model_dump()
    Output.model_validate(out)
    assert len(out["prompts"]) == 47 and r.calls == 6
    assert [p["tempId"] for p in out["prompts"]][:2] == ["g_001", "g_002"]
    by = {}
    for p in out["prompts"]:
        by[p["topic"]["code"]] = by.get(p["topic"]["code"], 0) + 1
        assert p["topic"]["categoryId"] == {"safety": 4, "efficacy": 3, "access_cost": 8}[p["topic"]["code"]]
    assert by == {"safety": 19, "efficacy": 14, "access_cost": 14}
    assert all({"angle", "batchId", "referenceIds"} <= rec.keys() for rec in r.records)
    assert (r.prompt_tokens, r.completion_tokens) == (60, 120)
    # overgenerate 1.3x; do-not-repeat always includes existingPrompts
    assert all("What are the side effects of Brandex?" in u for u in fake.users)
    assert "WRITE: 13 questions" in fake.users[[i for i, u in enumerate(fake.users) if "TOPIC: Safety" in u][0]]


def test_generate_all_sources_off_uses_fallback(payload, monkeypatch):
    r = run(payload, FakeLLM(), monkeypatch, n=7)
    assert r.pool and all(q["source"] == "fallback" for q in r.pool)
    assert len(r.records) == 7


def test_topup_then_warn_never_pad(payload, monkeypatch):
    fake = FakeLLM(short=1)  # every call yields 1 prompt
    r = run(payload, fake, monkeypatch, n=15)  # 6/5/4, each topic: 1 batch + 2 top-ups -> 3 of quota
    assert r.calls == 9 and len(r.records) == 9
    assert any(b["batchId"] == "safety-topup2" for b in r.batches)
    assert sum("after top-up" in w for w in r.warnings) == 3


def test_llm_failure_is_a_warning(payload, monkeypatch):
    r = run(payload, FakeLLM(fail=True), monkeypatch, n=7)
    assert r.records == [] and r.output.prompts == []
    assert any("LLM call failed" in w for w in r.warnings)


def test_n_out_of_range(payload):
    with pytest.raises(ValueError):
        g.generate(payload, n=101)
