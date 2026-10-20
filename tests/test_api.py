import json

from fastapi.testclient import TestClient

import api
from promptgen import generator as g
from promptgen.harvest import HarvestResult
from tests.test_generator import FakeLLM

client = TestClient(api.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_generate(monkeypatch):
    monkeypatch.setattr(g.llm, "chat_json", FakeLLM())
    monkeypatch.setattr(g, "harvest", lambda *a, **k: HarvestResult())  # no Google calls
    d = json.load(open("sample_payload.json"))
    r = client.post("/generate", json=d)
    assert r.status_code == 200
    body = r.json()
    assert body["requestId"] == d["requestId"] and body["runId"] == d["runId"]
    assert body["prompts"] and set(body["prompts"][0]) == {
        "tempId", "text", "audience", "topic", "competitor", "intent", "groundingUsed"}


def test_bad_payload_is_422():
    d = json.load(open("sample_payload.json"))
    del d["grounding"]
    assert client.post("/generate", json=d).status_code == 422
    assert client.post("/generate", content="{not json", headers={"content-type": "application/json"}).status_code == 422
