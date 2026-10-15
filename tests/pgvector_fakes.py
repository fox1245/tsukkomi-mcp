"""A fake psycopg driver for the pgvector backend: no psycopg, no server, no Docker.

``FakeServer`` keeps rows and the ``tsukkomi_meta`` rows in memory and answers the statements ``PgVectorIndex`` issues
by pattern (it is not a SQL engine), records every statement with its parameters, can fail the next statement of a
connection in a chosen way, and models a transaction (a rollback restores the rows). Real behaviour is the business of
the integration tests against a real server; this exists to pin SQL shapes, batching, validation and control flow.
"""
from __future__ import annotations

from contextlib import nullcontext
import struct
import threading
import time

import numpy as np

from self_directing_mcp.index.pgvector import PgVectorUnavailable


class FakeDbError(Exception):
    """What the fake driver raises; ``kind`` is what ``classify`` answers."""

    def __init__(self, kind: str, message: str = "boom", sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.kind, self.sqlstate = kind, sqlstate


class FakeVector:
    __slots__ = ("payload",)

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def array(self) -> np.ndarray:
        dim, unused = struct.unpack(">HH", self.payload[:4])
        assert unused == 0
        return np.frombuffer(self.payload[4:], dtype=">f4").astype(np.float64)


class FakeResult:
    def __init__(self, rows) -> None:
        self.rows = list(rows)
        self.description = [("x",)] if rows else None

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class FakeInfo:
    def __init__(self, host="127.0.0.1", port=5432, dbname="appdb", ssl=False, sslmode=None) -> None:
        self.host, self.port, self.dbname, self.ssl_in_use = host, port, dbname, ssl
        self._parameters = {"sslmode": sslmode} if sslmode else {}

    def get_parameters(self):
        return dict(self._parameters)


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, rows):
        self.connection.record(sql, rows)
        self.connection.before_statement(sql)
        for chunk_id, vector in rows:
            if "tsukkomi_meta" in sql:
                self.connection.server.meta.setdefault(chunk_id, vector)     # ON CONFLICT (key) DO NOTHING
            else:
                self.connection.server.write_row(self.connection, chunk_id, vector)
        self.connection.after_statement(sql)


class FakeConnection:
    def __init__(self, server: "FakeServer", info: FakeInfo) -> None:
        self.server, self.info = server, info
        self.closed = False
        self.broken = False
        self.autocommit = False
        self.adapters = self
        self.snapshot: dict | None = None
        self.in_transaction = False
        self.fail_on: list[tuple] = []          # (substring, FakeDbError) raised once by the first matching statement
        self.commits = 0
        self.rollbacks = 0

    # -- plumbing the backend uses ------------------------------------------------------------------------------------
    def register_dumper(self, *args):
        pass

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1
        self.server.log.append(("COMMIT", None))
        self.before_statement("COMMIT")
        self.snapshot, self.in_transaction = None, False

    def rollback(self):
        self.rollbacks += 1
        self.server.log.append(("ROLLBACK", None))
        if self.snapshot is not None:
            self.server.rows = self.snapshot
        self.snapshot, self.in_transaction = None, False

    def close(self):
        self.closed = True

    def record(self, sql, params):
        self.server.log.append((sql, params))

    def before_statement(self, sql):
        with self.server.guard:
            self.server.active += 1
            self.server.max_active = max(self.server.max_active, self.server.active)
        try:
            if self.server.delay:
                time.sleep(self.server.delay)
            for entry in list(self.fail_on):
                if entry[0] in sql:
                    self.fail_on.remove(entry)
                    if getattr(entry[1], "kind", None) == "connection":
                        self.broken = True
                    raise entry[1]
            if self.server.hook is not None:
                self.server.hook(self, sql)
        finally:
            with self.server.guard:
                self.server.active -= 1

    def after_statement(self, sql):
        if self.server.after_hook is not None:
            self.server.after_hook(self, sql)

    def execute(self, sql, params=None):
        self.record(sql, params)
        self.before_statement(sql)
        result = self.server.answer(self, sql, params)
        self.after_statement(sql)
        return result


class FakeServer:
    """One database: rows, meta, flags that shape the answers of the validation query, and the statement log."""

    def __init__(self, *, dim=4, extension="0.8.1", server_version="17.2", has_schema=True, has_meta=True, has_table=True,
                 has_index=False, meta=None, extension_installed=True, reltuples=-1, type_visible=True) -> None:
        self.rows: dict[str, FakeVector] = {}
        self.meta = {"schema_version": "1", "dim": str(dim), "vector_type": "vector"} if meta is None else dict(meta)
        self.extension = extension if extension_installed else None
        self.server_version, self.has_schema, self.has_meta = server_version, has_schema, has_meta
        self.has_table, self.has_index, self.type_visible = has_table, has_index, type_visible
        self.reltuples = reltuples
        self.log: list[tuple] = []
        self.guard = threading.Lock()
        self.active = self.max_active = 0
        self.delay = 0.0
        self.hook = None                    # called with (connection, sql) before every statement
        self.after_hook = None
        self.atttypmod = dim

    # -- statements -------------------------------------------------------------------------------------------------
    def statements(self, needle: str) -> list[tuple]:
        return [entry for entry in self.log if needle in entry[0]]

    def write_row(self, connection, chunk_id, vector) -> None:
        if not connection.in_transaction:
            connection.snapshot, connection.in_transaction = dict(self.rows), True
        self.rows[chunk_id] = vector

    def answer(self, connection, sql, params) -> FakeResult:
        if "set_config('statement_timeout'" in sql:
            return FakeResult([(None, None, None, None, self.reltuples if self.reltuples >= 0 else None)])
        if "FROM pg_extension" in sql:
            if self.extension is None:
                return FakeResult([])
            return FakeResult([(self.extension, "public", self.type_visible, self.has_schema, self.has_meta,
                                self.has_table, self.has_index, self.server_version, None)])
        if sql.startswith("SELECT key, value FROM"):
            return FakeResult(list(self.meta.items()))
        if "SELECT atttypmod" in sql:
            return FakeResult([(self.atttypmod,)])
        if sql.startswith("INSERT INTO") and "tsukkomi_meta" in sql:
            if "'policy'" in sql:
                self.meta.setdefault("policy", params[0])
            return FakeResult([])
        if sql.startswith("SELECT chunk_id FROM") and "ANY" in sql:
            wanted = set(params[0])
            return FakeResult([(cid,) for cid in sorted(self.rows) if cid in wanted])
        if sql.startswith("SELECT chunk_id FROM"):
            return FakeResult([(cid,) for cid in sorted(self.rows)])
        if sql.startswith("SELECT count(*)"):
            return FakeResult([(len(self.rows),)])
        if sql.startswith("DELETE FROM"):
            if not connection.in_transaction:
                connection.snapshot, connection.in_transaction = dict(self.rows), True
            for cid in params[0]:
                self.rows.pop(cid, None)
            return FakeResult([])
        if sql.startswith("TRUNCATE"):
            if not connection.in_transaction:
                connection.snapshot, connection.in_transaction = dict(self.rows), True
            self.rows = {}
            return FakeResult([])
        if sql.startswith("SELECT chunk_id, embedding <=> "):
            query = params["q"].array()
            allowed = None
            if "blob" in params:                                  # the filter: ids joined by chr(31), plus the odd ones
                allowed = set(filter(None, params["blob"].split("\x1f"))) | set(params["odd"])
            scored = []
            for cid, vector in self.rows.items():
                if allowed is not None and cid not in allowed:
                    continue
                v = vector.array()
                denominator = np.linalg.norm(v) * np.linalg.norm(query)
                scored.append((cid, float(1.0 - np.dot(v, query) / denominator) if denominator else float("nan")))
            scored.sort(key=lambda item: (item[1] if item[1] == item[1] else float("inf"), item[0]))     # distance, NaN last
            return FakeResult(scored[:params["k"]])
        return FakeResult([])


class FakeDriver:
    """Same four-plus methods as ``PsycopgDriver``; ``connections`` lists every connection it handed out."""

    def __init__(self, server: FakeServer | None = None, *, info: FakeInfo | None = None) -> None:
        self.server = server if server is not None else FakeServer()
        self.info = info
        self.connections: list[FakeConnection] = []
        self.connect_calls: list[tuple[str, dict]] = []
        self.connect_failures: list[Exception] = []     # raised by the next connect() calls, in order

    @staticmethod
    def parse(conninfo: str) -> dict:
        parsed = {}
        if conninfo.startswith(("postgresql://", "postgres://")):         # what conninfo_to_dict does with a URI
            from urllib.parse import parse_qsl, unquote, urlsplit
            parts = urlsplit(conninfo)
            for key, value in (("user", parts.username), ("password", parts.password), ("host", parts.hostname),
                               ("port", parts.port), ("dbname", parts.path.lstrip("/") or None)):
                if value is not None:
                    parsed[key] = unquote(str(value))
            parsed.update(parse_qsl(parts.query))
            return parsed
        for word in conninfo.split():
            if "=" not in word:
                raise ValueError("bad")
            key, value = word.split("=", 1)
            parsed[key] = value
        return parsed

    def connect(self, conninfo, **params):
        self.connect_calls.append((conninfo, dict(params)))
        if self.connect_failures:
            raise self.connect_failures.pop(0)
        connection = FakeConnection(self.server, self.info or FakeInfo())
        self.connections.append(connection)
        return connection

    @staticmethod
    def vector(payload: bytes) -> FakeVector:
        return FakeVector(payload)

    @staticmethod
    def pipeline(connection):
        return nullcontext()

    @staticmethod
    def classify(exc, connection) -> str:
        if isinstance(exc, FakeDbError):
            return exc.kind
        return "unrelated"

    @staticmethod
    def message(exc) -> str:
        return " ".join(str(exc).split())[:300]


__all__ = ["FakeConnection", "FakeDbError", "FakeDriver", "FakeInfo", "FakeServer", "PgVectorUnavailable"]
