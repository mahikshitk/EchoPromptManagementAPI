import os
import uuid

import pytest

import database
from promptgen import db, generator as g, jobs
from promptgen.harvest import HarvestResult
from tests.test_db import FakeLambda
from tests.test_generator import FakeLLM, make


def job(sub="user-1") -> dict:
    return {"generationId": str(uuid.uuid4()), "ownerSub": sub, "requestId": "pg_1", "runId": "42",
            "payload": {"requestId": "pg_1"}}


def lifecycle(store):
    """Shared by the memory and PostgreSQL stores: they must behave the same."""
    a, b, c = job(), job(), job()
    assert store.create(a, 2, 900) and store.create(b, 2, 900)
    assert not store.create(c, 2, 900)                      # limit reached for this user
    assert store.create(job("someone-else"), 2, 900)
    assert store.get(str(uuid.uuid4())) is None

    got = store.get(a["generationId"])
    assert got["status"] == "queued" and got["ownerSub"] == "user-1" and got["payload"] == a["payload"]
    assert got["createdAt"].endswith("Z") and got["startedAt"] is None and got["result"] is None

    assert store.start(a["generationId"]) and not store.start(a["generationId"])  # duplicate invoke
    store.progress(a["generationId"], 5, 10)
    store.progress(a["generationId"], 3, 10)                # out of order: never goes backwards
    assert (store.get(a["generationId"])["pairsDone"], store.get(a["generationId"])["pairsTotal"]) == (5, 10)

    store.complete(a["generationId"], {"prompts": [{"text": "q?"}]})
    got = store.get(a["generationId"])
    assert got["status"] == "completed" and got["result"] == {"prompts": [{"text": "q?"}]} and got["completedAt"]
    store.fail(a["generationId"], "late")                   # finished jobs stay finished
    assert store.get(a["generationId"])["status"] == "completed"
    assert store.create(c, 2, 900)                          # a finished job frees a slot

    store.fail(b["generationId"], "boom")
    assert store.get(b["generationId"])["status"] == "failed"
    assert store.get(b["generationId"])["error"] == "boom"


def test_memory_store():
    lifecycle(jobs.MemoryJobs())


def test_limit_ignores_old_unfinished_jobs():
    s = jobs.MemoryJobs()
    old = job()
    assert s.create(old, 1, 900)
    assert not s.create(job(), 1, 900)
    assert s.create(job(), 1, 0)                            # outside the window, the stuck job no longer counts


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to a scratch PostgreSQL")
def test_postgres_store():
    import psycopg

    with psycopg.connect(os.environ["TEST_DATABASE_URL"], autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS public.atrion_prompt_generation")
        conn.execute(open("database/prerequisites/001_prompt_generation.sql").read())
    lifecycle(jobs.PostgresJobs(db.Database(os.environ["TEST_DATABASE_URL"])))


def test_lambda_jobs_round_trip_through_database_handler(monkeypatch):
    monkeypatch.setattr(database, "_stores", {"jobs": jobs.MemoryJobs()})  # MemoryJobs stands in for PostgresJobs
    fake = FakeLambda()
    lifecycle(jobs.LambdaJobs(db.DatabaseFunction("prompt-management-dev-database", fake)))
    name, event = fake.calls[0]
    assert name == "prompt-management-dev-database" and event["operation"] == "create"


# ---- generator progress ----

def test_generator_reports_progress(monkeypatch):
    monkeypatch.setattr(g.llm, "chat_json", FakeLLM())
    monkeypatch.setattr(g, "harvest", lambda *a, **k: HarvestResult())
    seen = []
    g.generate(make(), on_event=lambda m: None, on_progress=lambda d, t: seen.append((d, t)), seed=0)
    total = seen[0][1]
    assert seen[0] == (0, total) and total > 0
    assert sorted(seen[1:]) == [(i, total) for i in range(1, total + 1)]
