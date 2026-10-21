"""Generation worker: runs one queued job and records progress and the result.
`handler` is the worker Lambda's entry point, invoked asynchronously by `submit`."""

import json
import logging
import os
import threading

from promptgen import generator, jobs
from promptgen.runtime import load_env_secret, setup_logging
from promptgen.schema import Payload

log = logging.getLogger("promptgen.worker")
setup_logging()


def submit(generation_id: str) -> None:
    """Start the worker for a job: async Lambda invoke when WORKER_FUNCTION is set, else a local thread."""
    if fn := os.getenv("WORKER_FUNCTION"):
        import boto3

        boto3.client("lambda").invoke(FunctionName=fn, InvocationType="Event",
                                      Payload=json.dumps({"generationId": generation_id}).encode())
    else:
        threading.Thread(target=run, args=(generation_id,), daemon=True).start()


def run(generation_id: str, store=None) -> None:
    store = store or jobs.store()
    job = store.get(generation_id)
    if job is None or not store.start(generation_id):  # missing, or a duplicate invoke already picked it up
        log.warning("%s: not queued, skipping", generation_id)
        return

    def progress(done: int, total: int) -> None:
        try:
            store.progress(generation_id, done, total)
        except Exception as e:  # progress is best effort; never fail the run over it
            log.warning("%s: progress update failed: %s", generation_id, e)

    rid = job["requestId"]
    try:
        r = generator.generate(Payload.model_validate(job["payload"]), on_event=lambda m: log.info("%s: %s", rid, m),
                               on_progress=progress)
        for w in r.warnings:  # Output is the client's exact schema, so warnings go to the logs
            log.warning("%s: %s", rid, w)
        log.info("%s: %d prompts, %d LLM calls, %.1fs", rid, len(r.records), r.calls, r.seconds)
        store.complete(generation_id, r.output.model_dump())
    except Exception as e:
        log.exception("%s: generation %s failed", rid, generation_id)
        store.fail(generation_id, f"{type(e).__name__}: {e}")


def handler(event, context):
    load_env_secret("OPENROUTER_API_KEY", "OPENROUTER_SECRET_ARN")
    run(event["generationId"])
