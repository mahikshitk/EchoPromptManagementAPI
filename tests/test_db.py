import io
import json

import pytest

import database
from promptgen import db


# ---- connection settings from the AtrionDB secret ----

def parse(conninfo: str) -> dict:
    from psycopg.conninfo import conninfo_to_dict

    return conninfo_to_dict(conninfo)


def test_conninfo_rds_shape_enforces_tls():
    c = parse(db.conninfo_from_secret(
        {"host": "db.internal", "port": 5432, "dbname": "atrion", "username": "app", "password": "p w"}))
    assert (c["host"], c["port"], c["dbname"], c["user"], c["password"], c["sslmode"]) == (
        "db.internal", "5432", "atrion", "app", "p w", "require")


@pytest.mark.parametrize("ssl,expected", [("Require", "require"), ("Disable", "require"), ("Prefer", "require"),
                                          ("VerifyFull", "verify-full"), ("VerifyCA", "verify-ca"), (None, "require")])
def test_conninfo_npgsql_shape(ssl, expected):
    s = "Host=db.internal;Port=5432;Database=atrion;Username=app;Password=secret"
    c = parse(db.conninfo_from_secret({"ConnectionStrings": {"AtrionDb": s + (f";SSL Mode={ssl}" if ssl else "")}}))
    assert (c["host"], c["dbname"], c["user"], c["password"], c["sslmode"]) == (
        "db.internal", "atrion", "app", "secret", expected)


def test_conninfo_missing_fields():
    with pytest.raises(db.DatabaseError):
        db.conninfo_from_secret({"host": "db.internal", "username": "app"})


# ---- database Lambda and its proxy ----

class FakeLambda:
    """boto3 Lambda client that runs database.handler in-process."""

    def __init__(self):
        self.calls = []

    def invoke(self, FunctionName, Payload):
        event = json.loads(Payload)
        self.calls.append((FunctionName, event))
        return {"Payload": io.BytesIO(json.dumps(database.handler(event, None)).encode())}


def test_handler_rejects_unknown_operation():
    assert "unknown operation" in database.handler({"operation": "drop", "args": {}}, None)["error"]


def test_handler_hides_details(monkeypatch):
    class Broken:
        def get(self, generation_id):
            raise RuntimeError("password=hunter2")

    monkeypatch.setattr(database, "_stores", {"jobs": Broken()})
    err = database.handler({"operation": "get", "args": {"generation_id": "x"}}, None)["error"]
    assert "hunter2" not in err and "RuntimeError" in err


def test_function_raises_on_error():
    class Failing:
        def invoke(self, **kw):
            return {"Payload": io.BytesIO(b'{"error": "database operation get failed (OperationalError)"}')}

    with pytest.raises(db.DatabaseError, match="OperationalError"):
        db.DatabaseFunction("fn", Failing()).call("get", generation_id="x")


def test_backend_from_env(monkeypatch):
    monkeypatch.setattr(db, "_backend", None)
    monkeypatch.delenv("DATABASE_FUNCTION", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db.backend() is None
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/atrion")
    assert isinstance(db.backend(), db.Database)
    monkeypatch.setattr(db, "_backend", None)
    monkeypatch.setenv("DATABASE_FUNCTION", "prompt-management-dev-database")  # wins over DATABASE_URL
    assert db.backend().function_name == "prompt-management-dev-database"
