"""Generation jobs: one row per POST /generate, owned by the caller's Cognito sub.

Three interchangeable stores with the same methods:
- PostgresJobs: the atrion_prompt_generation table in AtrionDB (runs in the VPC database Lambda)
- LambdaJobs:   forwards each call to that database Lambda (API and worker Lambdas, outside the VPC)
- MemoryJobs:   in-process dict for local runs and tests
"""

import threading
import time
from datetime import datetime, timezone

from promptgen import db

QUEUED, RUNNING, COMPLETED, FAILED = "queued", "running", "completed", "failed"
ACTIVE = (QUEUED, RUNNING)
OPERATIONS = ("create", "get", "start", "progress", "complete", "fail")  # what the database Lambda serves


def _iso(t: datetime | None) -> str | None:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryJobs:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.lock = threading.Lock()

    def create(self, job: dict, max_active: int, window_s: int) -> bool:
        """Insert a queued job unless the owner already has max_active unfinished jobs from the last window_s."""
        now = _now()
        with self.lock:
            active = [r for r in self.rows.values() if r["ownerSub"] == job["ownerSub"] and r["status"] in ACTIVE
                      and (now - r["_created"]).total_seconds() < window_s]
            if len(active) >= max_active:
                return False
            self.rows[job["generationId"]] = {
                **job, "status": QUEUED, "result": None, "error": None, "pairsDone": 0, "pairsTotal": 0,
                "_created": now, "_updated": now, "_started": None, "_completed": None}
            return True

    def get(self, generation_id: str) -> dict | None:
        with self.lock:
            r = self.rows.get(generation_id)
            if r is None:
                return None
            return {**{k: v for k, v in r.items() if not k.startswith("_")},
                    "createdAt": _iso(r["_created"]), "updatedAt": _iso(r["_updated"]),
                    "startedAt": _iso(r["_started"]), "completedAt": _iso(r["_completed"])}

    def _update(self, generation_id: str, allowed: tuple, **fields) -> bool:
        with self.lock:
            r = self.rows.get(generation_id)
            if r is None or r["status"] not in allowed:
                return False
            r.update(fields, _updated=_now())
            return True

    def start(self, generation_id: str) -> bool:
        """queued -> running. False if the job is missing or already picked up (duplicate invoke)."""
        return self._update(generation_id, (QUEUED,), status=RUNNING, _started=_now())

    def progress(self, generation_id: str, pairs_done: int, pairs_total: int) -> None:
        with self.lock:
            r = self.rows.get(generation_id)
            if r and r["status"] == RUNNING:
                r.update(pairsDone=max(r["pairsDone"], pairs_done), pairsTotal=pairs_total, _updated=_now())

    def complete(self, generation_id: str, result: dict) -> None:
        self._update(generation_id, (RUNNING,), status=COMPLETED, result=result, _completed=_now())

    def fail(self, generation_id: str, error: str) -> None:
        self._update(generation_id, ACTIVE, status=FAILED, error=error, _completed=_now())


# ---- PostgreSQL ----

TABLE = "public.atrion_prompt_generation"
COLUMNS = ("generationid, cognitosub, requestid, runid, status, payload, result, error, pairsdone, pairstotal,"
           " createdat, updatedat, startedat, completedat")


class PostgresJobs:
    def __init__(self, database: db.Database):
        self._execute = database.execute

    def create(self, job: dict, max_active: int, window_s: int) -> bool:
        from psycopg.types.json import Jsonb

        cur = self._execute(
            f"INSERT INTO {TABLE} (generationid, cognitosub, requestid, runid, status, payload)"
            " SELECT %(id)s, %(sub)s, %(requestId)s, %(runId)s, 'queued', %(payload)s"
            f" WHERE (SELECT count(*) FROM {TABLE} WHERE cognitosub = %(sub)s AND status IN ('queued', 'running')"
            "        AND createdat > now() - make_interval(secs => %(window)s)) < %(max)s",
            {"id": job["generationId"], "sub": job["ownerSub"], "requestId": job["requestId"], "runId": job["runId"],
             "payload": Jsonb(job["payload"]), "window": window_s, "max": max_active})
        return cur.rowcount == 1

    def get(self, generation_id: str) -> dict | None:
        r = self._execute(f"SELECT {COLUMNS} FROM {TABLE} WHERE generationid = %(id)s",
                          {"id": generation_id}).fetchone()
        if r is None:
            return None
        return {"generationId": str(r["generationid"]), "ownerSub": r["cognitosub"], "requestId": r["requestid"],
                "runId": r["runid"], "status": r["status"], "payload": r["payload"], "result": r["result"],
                "error": r["error"], "pairsDone": r["pairsdone"], "pairsTotal": r["pairstotal"],
                "createdAt": _iso(r["createdat"]), "updatedAt": _iso(r["updatedat"]),
                "startedAt": _iso(r["startedat"]), "completedAt": _iso(r["completedat"])}

    def start(self, generation_id: str) -> bool:
        cur = self._execute(f"UPDATE {TABLE} SET status = 'running', startedat = now(), updatedat = now()"
                            " WHERE generationid = %(id)s AND status = 'queued'", {"id": generation_id})
        return cur.rowcount == 1

    def progress(self, generation_id: str, pairs_done: int, pairs_total: int) -> None:
        # GREATEST: pairs finish on parallel threads, so updates can arrive out of order
        self._execute(f"UPDATE {TABLE} SET pairsdone = GREATEST(pairsdone, %(done)s), pairstotal = %(total)s,"
                      " updatedat = now() WHERE generationid = %(id)s AND status = 'running'",
                      {"id": generation_id, "done": pairs_done, "total": pairs_total})

    def complete(self, generation_id: str, result: dict) -> None:
        from psycopg.types.json import Jsonb

        self._execute(f"UPDATE {TABLE} SET status = 'completed', result = %(result)s, completedat = now(),"
                      " updatedat = now() WHERE generationid = %(id)s AND status = 'running'",
                      {"id": generation_id, "result": Jsonb(result)})

    def fail(self, generation_id: str, error: str) -> None:
        self._execute(f"UPDATE {TABLE} SET status = 'failed', error = %(error)s, completedat = now(),"
                      " updatedat = now() WHERE generationid = %(id)s AND status IN ('queued', 'running')",
                      {"id": generation_id, "error": error[:2000]})


# ---- database Lambda proxy ----

class LambdaJobs:
    """Forwards each call to the VPC database Lambda, so the API and worker never hold PostgreSQL credentials."""

    def __init__(self, function: db.DatabaseFunction):
        self._call = function.call

    def create(self, job: dict, max_active: int, window_s: int) -> bool:
        return self._call("create", job=job, max_active=max_active, window_s=window_s)

    def get(self, generation_id: str) -> dict | None:
        return self._call("get", generation_id=generation_id)

    def start(self, generation_id: str) -> bool:
        return self._call("start", generation_id=generation_id)

    def progress(self, generation_id: str, pairs_done: int, pairs_total: int) -> None:
        self._call("progress", generation_id=generation_id, pairs_done=pairs_done, pairs_total=pairs_total)

    def complete(self, generation_id: str, result: dict) -> None:
        self._call("complete", generation_id=generation_id, result=result)

    def fail(self, generation_id: str, error: str) -> None:
        self._call("fail", generation_id=generation_id, error=error)


_store = None
_store_lock = threading.Lock()


def store():
    """LambdaJobs or PostgresJobs per db.backend(); otherwise one shared MemoryJobs, so local runs need no database."""
    global _store
    with _store_lock:
        if _store is None:
            b = db.backend()
            _store = (LambdaJobs(b) if isinstance(b, db.DatabaseFunction) else
                      PostgresJobs(b) if isinstance(b, db.Database) else MemoryJobs())
        return _store


def age_s(iso: str) -> float:
    return time.time() - datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
