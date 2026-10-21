import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import api
import worker
from promptgen import config, db, generator as g, jobs
from promptgen.harvest import HarvestResult
from tests.test_generator import FakeLLM
from tests.tokens import ADMIN_CLIENT, bearer, trust

client = TestClient(api.app)


@pytest.fixture
def store(monkeypatch):
    """Trusted test tokens, a fresh in-memory store, the worker run inline, fake LLM, no Google calls."""
    trust(monkeypatch)
    s = jobs.MemoryJobs()
    monkeypatch.setattr(jobs, "_store", s)
    monkeypatch.setattr(worker, "submit", lambda gid: worker.run(gid))
    monkeypatch.setattr(g.llm, "chat_json", FakeLLM())
    monkeypatch.setattr(g, "harvest", lambda *a, **k: HarvestResult())
    return s


def sample() -> dict:
    return json.load(open("sample_payload.json"))


def test_health_needs_no_token():
    assert client.get("/health").json() == {"status": "ok"}


def test_swagger_offers_bearer_auth_on_generation_routes_only():
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert spec["components"]["securitySchemes"]["HTTPBearer"] == {
        "type": "http", "scheme": "bearer", "description": "Cognito access token (not the ID token)"}
    assert spec["paths"]["/generate"]["post"]["security"] == [{"HTTPBearer": []}]
    assert spec["paths"]["/generate/{generationId}"]["get"]["security"] == [{"HTTPBearer": []}]
    assert "security" not in spec["paths"]["/health"]["get"]


def test_generate_then_poll(store):
    d = sample()
    r = client.post("/generate", json=d, headers=bearer())
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["requestId"] == d["requestId"] and body["runId"] == d["runId"]
    assert r.headers["location"] == body["statusUrl"] == f"/generate/{body['generationId']}"

    r = client.get(body["statusUrl"], headers=bearer())
    assert r.status_code == 200
    got = r.json()
    assert got["status"] == "completed" and got["error"] is None
    assert got["progress"]["pairsDone"] == got["progress"]["pairsTotal"] > 0
    result = got["result"]
    assert result["requestId"] == d["requestId"] and result["runId"] == d["runId"]
    assert result["prompts"] and set(result["prompts"][0]) == {
        "tempId", "text", "audience", "topic", "competitor", "intent", "groundingUsed"}


def test_running_job_has_no_result(store, monkeypatch):
    monkeypatch.setattr(worker, "submit", lambda gid: None)
    gid = client.post("/generate", json=sample(), headers=bearer()).json()["generationId"]
    store.start(gid)
    store.progress(gid, 3, 12)
    got = client.get(f"/generate/{gid}", headers=bearer()).json()
    assert got["status"] == "running" and got["progress"] == {"pairsDone": 3, "pairsTotal": 12}
    assert got["result"] is None and got["completedAt"] is None


def test_other_users_generation_is_404(store):
    gid = client.post("/generate", json=sample(), headers=bearer(sub="alice")).json()["generationId"]
    assert client.get(f"/generate/{gid}", headers=bearer(sub="bob")).status_code == 404
    assert client.get(f"/generate/{gid}", headers=bearer(sub="alice", client_id=ADMIN_CLIENT)).status_code == 200


def test_unknown_and_malformed_ids(store):
    assert client.get("/generate/00000000-0000-0000-0000-000000000000", headers=bearer()).status_code == 404
    assert client.get("/generate/not-a-uuid", headers=bearer()).status_code == 422


def test_failed_generation_hides_cause(store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(worker.generator, "generate", boom)
    gid = client.post("/generate", json=sample(), headers=bearer()).json()["generationId"]
    got = client.get(f"/generate/{gid}", headers=bearer()).json()
    assert got["status"] == "failed" and "secret" not in got["error"] and got["result"] is None
    assert "secret detail" in store.rows[gid]["error"]  # kept for the logs and the table


def test_stale_job_reported_failed(store, monkeypatch):
    monkeypatch.setattr(worker, "submit", lambda gid: None)
    gid = client.post("/generate", json=sample(), headers=bearer()).json()["generationId"]
    monkeypatch.setattr(jobs, "age_s", lambda iso: config.GENERATION_STALE_S + 1)
    got = client.get(f"/generate/{gid}", headers=bearer()).json()
    assert got["status"] == "failed" and "timed out" in got["error"]


def test_per_user_limit(store, monkeypatch):
    monkeypatch.setattr(worker, "submit", lambda gid: None)  # jobs stay queued
    for _ in range(config.MAX_ACTIVE_GENERATIONS):
        assert client.post("/generate", json=sample(), headers=bearer()).status_code == 202
    assert client.post("/generate", json=sample(), headers=bearer()).status_code == 429
    assert client.post("/generate", json=sample(), headers=bearer(sub="other")).status_code == 202


def test_worker_start_failure_is_503(store, monkeypatch):
    def down(gid):
        raise RuntimeError("lambda unavailable")

    monkeypatch.setattr(worker, "submit", down)
    assert client.post("/generate", json=sample(), headers=bearer()).status_code == 503
    assert [r["status"] for r in store.rows.values()] == ["failed"]  # does not count against the limit


def test_bad_payload_is_422(store):
    d = sample()
    del d["grounding"]
    assert client.post("/generate", json=d, headers=bearer()).status_code == 422
    assert client.post("/generate", content="{not json", headers={**bearer(), "content-type": "application/json"}
                       ).status_code == 422


# ---- auth: RunManagementAPI's rules ----

OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.mark.parametrize("headers,status", [
    ({}, 401),
    ({"Authorization": "Basic abc"}, 401),
    ({"Authorization": "Bearer not-a-jwt"}, 401),
    (bearer(key=OTHER_KEY), 401),                          # not signed by the pool
    (bearer(iss="https://cognito-idp.eu-west-2.amazonaws.com/other"), 401),
    (bearer(exp=int(time.time()) - 10), 401),
    (bearer(token_use="id"), 401),                         # ID tokens are not accepted
    (bearer(client_id="backend-client"), 401),             # not a UI client
    (bearer(client_id=None), 401),
    (bearer(scope="openid email"), 403),                   # authenticated, missing the scope
])
def test_token_rules(store, headers, status):
    assert client.post("/generate", json=sample(), headers=headers).status_code == status
    assert client.get("/generate/00000000-0000-0000-0000-000000000000", headers=headers).status_code == status


def test_auth_not_configured_fails_closed(store, monkeypatch):
    monkeypatch.delenv("AUTH_AUTHORITY")
    assert client.post("/generate", json=sample(), headers=bearer()).status_code == 503


def test_auth_disabled_for_local_runs(store, monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "1")
    gid = client.post("/generate", json=sample()).json()["generationId"]
    assert client.get(f"/generate/{gid}").json()["status"] == "completed"


def test_local_runs_use_a_background_thread(store, monkeypatch):
    monkeypatch.undo()  # the real submit, without WORKER_FUNCTION: a local thread
    trust(monkeypatch)
    monkeypatch.delenv("WORKER_FUNCTION", raising=False)
    monkeypatch.setattr(jobs, "_store", store)
    monkeypatch.setattr(g.llm, "chat_json", FakeLLM())
    monkeypatch.setattr(g, "harvest", lambda *a, **k: HarvestResult())
    gid = client.post("/generate", json=sample(), headers=bearer()).json()["generationId"]
    for _ in range(100):
        if (status := client.get(f"/generate/{gid}", headers=bearer()).json()["status"]) == "completed":
            break
        time.sleep(0.05)
    assert status == "completed"


def test_job_store_down_is_503(store, monkeypatch):
    def down(*a, **k):
        raise db.DatabaseError("get: database operation get failed (OperationalError)")

    monkeypatch.setattr(store, "create", down)
    monkeypatch.setattr(store, "get", down)
    assert client.post("/generate", json=sample(), headers=bearer()).status_code == 503
    assert client.get("/generate/00000000-0000-0000-0000-000000000000", headers=bearer()).status_code == 503
