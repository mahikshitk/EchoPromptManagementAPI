"""HTTP API for the UI. POST /generate queues a generation and returns 202 with its id;
GET /generate/{generationId} reports progress and returns the prompts once it completes.
`handler` is the Lambda entry point."""

import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel

import worker
from promptgen import config, db, jobs
from promptgen.auth import require_scope
from promptgen.runtime import setup_logging
from promptgen.schema import Output, Payload

log = logging.getLogger("promptgen.api")
setup_logging()

app = FastAPI(title="Echo Prompt Generator")
SCOPE = os.getenv("AUTH_REQUIRED_SCOPE") or config.REQUIRED_SCOPE


class Progress(BaseModel):
    pairsDone: int
    pairsTotal: int


class GenerationAccepted(BaseModel):
    generationId: str
    requestId: str
    runId: str
    status: str
    statusUrl: str


class Generation(BaseModel):
    generationId: str
    requestId: str
    runId: str
    status: str  # queued | running | completed | failed
    progress: Progress
    createdAt: str
    updatedAt: str
    completedAt: str | None = None
    result: Output | None = None  # set once status is completed
    error: str | None = None      # set once status is failed


@app.exception_handler(db.DatabaseError)
def database_unavailable(request: Request, e: db.DatabaseError) -> JSONResponse:
    log.error("database unavailable: %s", e)
    return JSONResponse({"detail": "prompt generation is unavailable; try again"}, status_code=503)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", status_code=202, response_model=GenerationAccepted)
def generate(p: Payload, response: Response, claims: dict = require_scope(SCOPE)) -> GenerationAccepted:
    gid = str(uuid.uuid4())
    store = jobs.store()
    job = {"generationId": gid, "ownerSub": claims["sub"], "requestId": p.requestId, "runId": p.runId,
           "payload": p.model_dump(mode="json")}
    if not store.create(job, config.MAX_ACTIVE_GENERATIONS, config.GENERATION_STALE_S):
        raise HTTPException(429, f"at most {config.MAX_ACTIVE_GENERATIONS} generations can run at once; "
                                 "wait for one to finish")
    try:
        worker.submit(gid)
    except Exception:
        log.exception("%s: could not start worker for %s", p.requestId, gid)
        store.fail(gid, "worker could not be started")
        raise HTTPException(503, "prompt generation is unavailable; try again")
    log.info("%s: queued generation %s", p.requestId, gid)
    url = f"/generate/{gid}"
    response.headers["Location"] = url
    return GenerationAccepted(generationId=gid, requestId=p.requestId, runId=p.runId, status=jobs.QUEUED,
                              statusUrl=url)


@app.get("/generate/{generationId}", response_model=Generation)
def generation(generationId: uuid.UUID, claims: dict = require_scope(SCOPE)) -> Generation:
    job = jobs.store().get(str(generationId))
    if job is None or job["ownerSub"] != claims["sub"]:  # another user's job looks the same as a missing one
        raise HTTPException(404, "generation not found")
    status, error = job["status"], None
    if status in jobs.ACTIVE and jobs.age_s(job["updatedAt"]) > config.GENERATION_STALE_S:
        status, error = jobs.FAILED, "generation timed out; start a new one"  # the worker died or timed out
    elif status == jobs.FAILED:
        error = "generation failed; start a new one"  # the stored cause is for the logs, not the UI
    return Generation(
        generationId=job["generationId"], requestId=job["requestId"], runId=job["runId"], status=status,
        progress=Progress(pairsDone=job["pairsDone"], pairsTotal=job["pairsTotal"]),
        createdAt=job["createdAt"], updatedAt=job["updatedAt"], completedAt=job["completedAt"],
        result=job["result"] if status == jobs.COMPLETED else None, error=error)


handler = Mangum(app, lifespan="off")
