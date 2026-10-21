"""AtrionDB access shared by the job store and the harvest store.

On AWS only the VPC database Lambda connects (Database); the API and worker call it through
DatabaseFunction, as RunManagementAPI's LambdaDatabaseGateway does. Locally DATABASE_URL connects directly.
"""

import json
import os
import threading


class DatabaseError(RuntimeError):
    pass


def conninfo_from_secret(secret: dict) -> str:
    """libpq conninfo from the AtrionDB secret, which RunManagementAPI reads in either the RDS shape
    (host, port, dbname, username, password) or {"ConnectionStrings": {"AtrionDb": "<Npgsql string>"}}.
    TLS is enforced as in RunManagementAPI: disable/allow/prefer become require."""
    from psycopg.conninfo import make_conninfo

    npgsql = (secret.get("ConnectionStrings") or {}).get("AtrionDb")
    if npgsql:
        parts = {k.strip().lower(): v.strip() for k, _, v in
                 (kv.partition("=") for kv in npgsql.split(";") if kv.strip())}
        kw = {"host": parts.get("host") or parts.get("server"), "port": parts.get("port"),
              "dbname": parts.get("database"), "user": parts.get("username") or parts.get("user id"),
              "password": parts.get("password"),
              "sslmode": (parts.get("ssl mode") or parts.get("sslmode") or "").lower().replace("verify", "verify-")}
    else:
        kw = {"host": secret.get("host"), "port": secret.get("port"), "dbname": secret.get("dbname"),
              "user": secret.get("username"), "password": secret.get("password"), "sslmode": secret.get("sslmode")}
    if not (kw["host"] and kw["user"] and kw["password"]):
        raise DatabaseError("the AtrionDB secret is missing host, username, or password")
    if kw["sslmode"] in (None, "", "disable", "allow", "prefer"):
        kw["sslmode"] = "require"
    return make_conninfo(**{k: str(v) for k, v in kw.items() if v not in (None, "")}, connect_timeout="10",
                         application_name="prompt-management")


class Database:
    """One autocommit connection per container, reopened once if it dropped between invocations.
    psycopg serializes calls on a connection, so threads can share it."""

    def __init__(self, conninfo: str):
        self.conninfo = conninfo
        self.conn = None

    def execute(self, sql: str, params: dict | None = None):
        import psycopg
        from psycopg.rows import dict_row

        for attempt in range(2):
            if self.conn is None or self.conn.closed:
                self.conn = psycopg.connect(self.conninfo, autocommit=True, row_factory=dict_row)
            try:
                return self.conn.execute(sql, params)
            except psycopg.OperationalError:
                self.conn = None
                if attempt:
                    raise


class DatabaseFunction:
    """Synchronous calls to the database Lambda: {"operation", "args"} -> {"result"} or {"error"}."""

    def __init__(self, function_name: str, client=None):
        self.function_name, self._client = function_name, client

    def call(self, operation: str, **args):
        if self._client is None:
            import boto3

            self._client = boto3.client("lambda")
        r = self._client.invoke(FunctionName=self.function_name,
                                Payload=json.dumps({"operation": operation, "args": args}).encode())
        body = json.loads(r["Payload"].read() or b"null") or {}
        if r.get("FunctionError") or body.get("error"):
            raise DatabaseError(f"{operation}: {body.get('error') or body.get('errorMessage') or r.get('FunctionError')}")
        return body.get("result")


_backend = None
_backend_lock = threading.Lock()


def backend() -> Database | DatabaseFunction | None:
    """DATABASE_FUNCTION set (API and worker Lambdas) -> DatabaseFunction; DATABASE_URL set (local
    PostgreSQL) -> Database; neither -> None, and callers fall back to local stores."""
    global _backend
    with _backend_lock:
        if _backend is None:
            if fn := os.getenv("DATABASE_FUNCTION"):
                _backend = DatabaseFunction(fn)
            elif url := os.getenv("DATABASE_URL"):
                _backend = Database(url)
        return _backend
