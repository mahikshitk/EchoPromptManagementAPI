"""Database Lambda: the only code that connects to AtrionDB. It runs in the VPC; the API and worker
Lambdas call it synchronously with {"operation": ..., "args": {...}} and get {"result": ...} or {"error": ...}."""

import logging

from promptgen import db, harvest, jobs
from promptgen.runtime import json_secret, setup_logging

log = logging.getLogger("promptgen.database")
setup_logging()

# operation -> (store, method)
OPERATIONS = {**{op: ("jobs", op) for op in jobs.OPERATIONS},
              "harvest_load": ("harvest", "load"), "harvest_save": ("harvest", "save")}
_stores: dict | None = None


def handler(event, context):
    global _stores
    op, args = event.get("operation"), event.get("args") or {}
    if op not in OPERATIONS:
        return {"error": f"unknown operation {op!r}"}
    try:
        if _stores is None:
            database = db.Database(db.conninfo_from_secret(json_secret("ATRION_DB_SECRET_ARN")))
            _stores = {"jobs": jobs.PostgresJobs(database), "harvest": harvest.PostgresHarvestStore(database)}
        store, method = OPERATIONS[op]
        return {"result": getattr(_stores[store], method)(**args)}
    except Exception as e:  # the caller gets the type only; details stay in this function's logs
        log.exception("database operation %s failed", op)
        return {"error": f"database operation {op} failed ({type(e).__name__})"}
