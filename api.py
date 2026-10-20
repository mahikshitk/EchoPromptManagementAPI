"""HTTP API: POST /generate runs the prompt agent; `handler` is the Lambda entry point."""

import logging

from fastapi import FastAPI
from mangum import Mangum

from promptgen import generator
from promptgen.schema import Output, Payload

log = logging.getLogger("promptgen.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Echo Prompt Generator")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=Output)
def generate(p: Payload) -> Output:  # sync def: FastAPI runs it in a worker thread
    r = generator.generate(p, on_event=log.info)
    for w in r.warnings:  # Output is the client's exact schema, so warnings go to the logs
        log.warning("%s: %s", p.requestId, w)
    log.info("%s: %d prompts, %d LLM calls, %.1fs", p.requestId, len(r.records), r.calls, r.seconds)
    return r.output


handler = Mangum(app, lifespan="off")
