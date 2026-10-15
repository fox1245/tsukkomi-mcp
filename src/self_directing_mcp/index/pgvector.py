"""PostgreSQL + pgvector as the dense index: the third ``VectorIndex`` backend, behind ``SELF_DIRECT_DENSE_BACKEND=pgvector``.

What it is. ``chunk_vectors(chunk_id text primary key, embedding vector(<dim>))`` in a schema of its own (default
``tsukkomi``) plus ``tsukkomi_meta(key, value)``, which records the schema version, the dimension and the embedding
policy tag (the prefix of the embedder's cache key, the one that names ``vectors-v2-*`` for the file backends). A process
whose dimension or policy differs from what the database holds gets an explicit error naming the repair, never vectors
of another model. Search is exact cosine, scored ``1 - cosine distance`` like ``SqliteVectorIndex`` (higher is better,
ties broken by ``chunk_id``), with ``allowed_ids`` pushed into the query. An optional HNSW index (off by default) serves
only unfiltered searches; filtered searches stay exact, because an approximate index cannot honour a filter without
returning fewer rows than asked for.

What it is not. It never stands in for another backend and another backend never stands in for it: an unreachable
server, a missing extension or a mismatched index is a ``DenseBackendError`` that names the repair. psycopg (the optional
``pgvector`` extra) is imported at the first query, not here and not when the index object is built, so the command
hooks and ``import self_directing_mcp`` never load it.

Requests. One connection per index object, opened lazily, protected by a lock. Every query is armed with a
``statement_timeout`` and ``lock_timeout`` equal to the time the request has left (``SET LOCAL`` inside its transaction),
so a slow server cannot outlive the request, and ``check_request()`` runs between batches. Each write call is one
transaction that commits inside ``publication()``, so a stopped request leaves nothing half-applied. A dropped
connection (and a deadlock or serialization failure) is retried once; every operation here is idempotent.

Credentials. The connection string is a secret and reaches this module only as the text of a private file or through the
standard libpq environment. It is never logged, repr'd or put into an exception: messages carry host, port and database
name from the parsed string and the server's own primary message with any password value removed.

Transport. A server that is not this machine (anything but ``localhost``, a loopback address or a socket directory) is
reached only with ``sslmode=verify-full``; ``disable``, ``allow``, ``prefer`` (libpq's default), ``require`` and
``verify-ca`` are refused before a connection is opened, so no credential is sent in the clear and no certificate goes
unchecked. A connection that names only a ``service`` hides its host, so it is connected with ``sslmode=verify-full``
as well, and every new connection is checked once more against what libpq actually used.
"""
from __future__ import annotations

from contextlib import nullcontext, suppress
import importlib.util
import math
import os
import random
import re
import struct
import threading
import time
from typing import Sequence

from self_directing_mcp._runtime import np
from self_directing_mcp.index.vector import DenseBackendError, VectorIndex
from self_directing_mcp.request_control import (IndexBusy, RequestStopped, check_request, current_request, publication,
                                                request_operation)
from self_directing_mcp.settings_core import DEFAULT_PGVECTOR_SCHEMA, check_pgvector_schema

SCHEMA_VERSION = 1
TABLE = "chunk_vectors"
META_TABLE = "tsukkomi_meta"
HNSW_INDEX = "chunk_vectors_embedding_hnsw"
MAX_DIM = 16000                 # the largest ``vector`` column
HNSW_MAX_DIM = 2000             # the largest vector dimension an HNSW index accepts
PLAIN_STORAGE_MAX_DIM = 1700    # up to here a row (vector, and an id of up to 1 kB) fits one 8 kB page: stored inline (_create)
MAX_ID_CHARS = 256              # at most 1 kB of UTF-8: keeps a row inside one page next to a 1700-dim vector
UPSERT_BATCH = 256              # rows per ``executemany`` round (all in the call's one transaction)
PRESENT_BATCH = 2000            # ids per ``present`` / ``delete_ids`` statement
NO_DEADLINE_TIMEOUT_MS = 300_000    # statement and lock timeout when the caller has no request deadline
CONNECT_TIMEOUT_SEC = 10
POSTFILTER_MIN_SHARE = 0.25     # a filter naming at least this share of the table is applied after an unfiltered scan (search)
POSTFILTER_ROUNDS = 3           # unfiltered scans a search may try, each asking for 4 times as many rows, before it sends the ids
POSTFILTER_GROWTH = 4
MIN_HNSW_EXTENSION = (0, 5, 0)  # the first pgvector release with HNSW indexes
SECURE_SSLMODE = "verify-full"
ID_SEPARATOR = "\x1f"                  # joins a large filter into one text parameter (see _filter_parameters)
IDLE_IN_TRANSACTION_MS = 120_000    # the server drops a session this long idle inside a transaction
APPLICATION_NAME = "tsukkomi-mcp"
ENVIRONMENT_HINT = ("SELF_DIRECT_PGVECTOR_DSN_FILE (a private file with one libpq connection string) or the libpq "
                    "environment (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSFILE, PGSERVICE)")
CONNECTION_ENVIRONMENT = ("PGHOST", "PGHOSTADDR", "PGPORT", "PGDATABASE", "PGUSER", "PGSERVICE")
INSTALL_HINT = "Install the optional driver with: pip install 'tsukkomi-mcp[pgvector]' (psycopg 3)."


class PgVectorError(DenseBackendError):
    """Any pgvector failure; the message names the repair and never contains a credential."""


class PgVectorUnavailable(PgVectorError):
    """The server could not be reached or the connection was lost and could not be re-established."""


class PgVectorConfigError(PgVectorError):
    """Unusable configuration or an index that does not match this process (extension, dimension, embedding policy)."""


class _VectorParam:
    """A vector already in pgvector's binary wire format (``dim:int16, 0:int16, float32 big-endian * dim``)."""

    __slots__ = ("payload",)

    def __init__(self, payload: bytes) -> None:
        self.payload = payload


def _is_loopback(host) -> bool:
    """Whether every host of a connection is this machine (a socket directory, ``localhost`` or a loopback address)."""
    import ipaddress

    for name in str(host or "").split(","):
        name = name.strip()
        if name in ("", "localhost") or name.startswith("/"):
            continue
        try:
            if ipaddress.ip_address(name).is_loopback:
                continue
        except ValueError:
            pass
        return False
    return True


def _version_tuple(text) -> tuple[int, ...]:
    """``'0.8.1'`` -> ``(0, 8, 1)``; anything unparseable sorts as the oldest version."""
    parts = []
    for piece in str(text).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


_REDACT = re.compile(r"'(?:\[[^']*\]|\{[^']*\})'(?:::\w+(?:\[\])?)?")     # the vector or the id list a plan prints as a literal


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def check_transport(parsed: dict, environ=None) -> dict:
    """Apply the TLS rule to a parsed connection string and the libpq environment, before anything is connected.

    Returns ``{"remote": bool, "unknown_host": bool, "sslmode": str | None}``: ``remote`` when any host named is not this
    machine, ``unknown_host`` when only a ``service`` could say where the server is (the caller then forces
    ``sslmode=verify-full`` on the connection). Raises ``PgVectorConfigError`` for a remote server whose sslmode is
    anything but ``verify-full`` (libpq's default, ``prefer``, included). Only the mode name is ever put in a message."""
    environ = os.environ if environ is None else environ
    hosts = [parsed["host"] if "host" in parsed else environ.get("PGHOST", ""),
             parsed["hostaddr"] if "hostaddr" in parsed else environ.get("PGHOSTADDR", "")]
    named = [value for value in hosts if value]
    service = parsed.get("service") or environ.get("PGSERVICE")
    unknown_host = not named and bool(service)
    remote = any(not _is_loopback(value) for value in named)
    sslmode = parsed.get("sslmode") or environ.get("PGSSLMODE") or None
    insecure = sslmode != SECURE_SSLMODE
    if (remote and insecure) or (unknown_host and sslmode is not None and insecure):
        shown = sslmode or "prefer (libpq's default)"
        raise PgVectorConfigError(
            f"The pgvector server is not on this machine and the connection uses sslmode={shown}. A remote server is "
            f"reached only with sslmode={SECURE_SSLMODE} (and a root certificate that verifies its host name): set it in "
            "the connection string of SELF_DIRECT_PGVECTOR_DSN_FILE or in PGSSLMODE. Vectors and chunk ids are sensitive "
            "and are never sent over an unverified connection.")
    return {"remote": remote, "unknown_host": unknown_host, "sslmode": sslmode}


# --- the driver: the only place that touches psycopg ----------------------------------------------------------------

class PsycopgDriver:
    """psycopg 3. Imported at the first connection; tests inject a fake with the same four methods."""

    def __init__(self) -> None:
        if importlib.util.find_spec("psycopg") is None:
            raise PgVectorConfigError("The pgvector dense backend needs psycopg 3, which is not installed. " + INSTALL_HINT)
        self._psycopg = None
        self._dumper = None

    def _module(self):
        if self._psycopg is None:
            try:
                import psycopg
            except ImportError:
                raise PgVectorConfigError("psycopg 3 could not be imported. " + INSTALL_HINT) from None
            self._psycopg = psycopg
        return self._psycopg

    def parse(self, conninfo: str) -> dict:
        """The keywords of a connection string; ``ValueError`` (without any of its text) when it is not valid."""
        self._module()
        from psycopg.conninfo import conninfo_to_dict

        try:
            return dict(conninfo_to_dict(conninfo))
        except Exception:
            raise ValueError("not a valid libpq connection string") from None

    def connect(self, conninfo: str, **params):
        psycopg = self._module()
        if self._dumper is None:
            from psycopg import pq
            from psycopg.adapt import Dumper

            class VectorDumper(Dumper):
                format = pq.Format.BINARY       # the type oid stays 0: statements cast the parameter, ``%b::vector``

                def dump(self, obj):
                    return obj.payload

            self._dumper = VectorDumper
        connection = psycopg.connect(conninfo, autocommit=False, prepare_threshold=None, **params)
        connection.adapters.register_dumper(_VectorParam, self._dumper)
        return connection

    def vector(self, payload: bytes) -> _VectorParam:
        return _VectorParam(payload)

    def pipeline(self, connection):
        """One network round trip for the statements queued inside (libpq 14+); plain sequential execution before."""
        psycopg = self._module()
        return connection.pipeline() if psycopg.Pipeline.is_supported() else nullcontext()

    def classify(self, exc: BaseException, connection) -> str:
        """``timeout`` | ``connection`` | ``transient`` | ``undefined`` | ``database`` | ``unrelated``."""
        psycopg = self._module()
        errors = psycopg.errors
        if not isinstance(exc, psycopg.Error):
            return "unrelated"
        if isinstance(exc, (errors.QueryCanceled, errors.LockNotAvailable)):
            return "timeout"
        state = exc.sqlstate or ""
        if getattr(connection, "closed", False) or getattr(connection, "broken", False):
            return "connection"
        if state.startswith("08") or state in ("57P01", "57P02", "57P03", "25P03"):
            return "connection"
        if isinstance(exc, (errors.DeadlockDetected, errors.SerializationFailure)):
            return "transient"
        if isinstance(exc, errors.UndefinedTable):
            return "undefined"
        if isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)) and not state:
            return "connection"
        return "database"

    def message(self, exc: BaseException) -> str:
        """The server's primary message (never its DETAIL, which can echo a row, vector included), at most 300 characters."""
        primary = getattr(getattr(exc, "diag", None), "message_primary", None)
        text = " ".join((primary or str(exc)).split())
        state = getattr(exc, "sqlstate", None)
        return f"{text[:300]} [{state}]" if state else text[:300]


# --- the index ---------------------------------------------------------------------------------------------------------

class PgVectorIndex(VectorIndex):
    """Exact cosine search in PostgreSQL. See the module docstring."""

    remote = True

    def __init__(self, dsn: str = "", dim: int = 1024, *, schema: str = DEFAULT_PGVECTOR_SCHEMA, hnsw: bool = False,
                 policy: str | None = None, driver=None, hnsw_ef_search: int | None = None) -> None:
        check_request()
        if not isinstance(dim, int) or isinstance(dim, bool) or not 1 <= dim <= MAX_DIM:
            raise ValueError(f"pgvector supports vector dimensions 1..{MAX_DIM}")
        if hnsw and dim > HNSW_MAX_DIM:
            raise PgVectorConfigError(f"An HNSW index supports at most {HNSW_MAX_DIM} dimensions (dimension is {dim}); "
                                      "unset SELF_DIRECT_PGVECTOR_HNSW")
        if policy is not None and (not isinstance(policy, str) or not policy or len(policy) > 128
                                   or not policy.isprintable()):
            raise ValueError("The embedding policy tag must be a short printable string")
        if hnsw_ef_search is not None and (not isinstance(hnsw_ef_search, int) or isinstance(hnsw_ef_search, bool)
                                           or not 1 <= hnsw_ef_search <= 1000):
            raise ValueError("hnsw_ef_search must be an integer 1..1000 (or None for max(100, 4 * top_k))")
        self.hnsw_ef_search = hnsw_ef_search       # programmatic only (the benchmark); not a setting
        self.dim = dim
        self.schema = check_pgvector_schema(schema)
        self.hnsw = bool(hnsw)
        self.policy = policy
        self._dsn = dsn or ""
        if not self._dsn and not any(os.environ.get(name) for name in CONNECTION_ENVIRONMENT):
            raise PgVectorConfigError("The pgvector dense backend has no server to connect to. Set " + ENVIRONMENT_HINT + ".")
        self._driver = driver if driver is not None else PsycopgDriver()
        self._lock = threading.RLock()
        self._conn = None
        self._peer: dict = {}           # host, port and database of the open connection, noted when it was made
        self.extension_version: str | None = None
        self.server_version: str | None = None
        self.compute_backend: str | None = None
        self._tls: bool | None = None
        self._endpoint_cache: dict | None = None
        self._table_rows: int | None = None     # the planner's row count, noted by every read
        self._secrets: tuple[str, ...] | None = None
        self._table = f"{_quote(self.schema)}.{_quote(TABLE)}"
        self._meta = f"{_quote(self.schema)}.{_quote(META_TABLE)}"

    def __repr__(self) -> str:
        return f"PgVectorIndex(schema={self.schema!r}, dim={self.dim}, hnsw={self.hnsw}, policy={self.policy!r})"

    # -- connection details, never secrets ------------------------------------------------------------------------

    def _parsed(self) -> dict:
        if self._endpoint_cache is None:
            parsed = {}
            if self._dsn:
                try:
                    parsed = self._driver.parse(self._dsn)
                except ValueError:
                    raise PgVectorConfigError("The pgvector connection string is not a valid libpq connection string "
                                              "(host=... dbname=... or postgresql://...); its content is not shown. "
                                              "Check SELF_DIRECT_PGVECTOR_DSN_FILE.") from None
            self._endpoint_cache = parsed
        return self._endpoint_cache

    def _endpoint(self) -> str:
        """``host:port/dbname`` of the configured server, for messages; no user, no password, no options."""
        try:
            parsed = self._parsed()
        except PgVectorConfigError:
            return "(unparseable connection string)"
        env = os.environ
        host = parsed.get("host") or parsed.get("hostaddr") or env.get("PGHOST") or env.get("PGHOSTADDR")
        port = parsed.get("port") or env.get("PGPORT") or "5432"
        database = parsed.get("dbname") or env.get("PGDATABASE") or parsed.get("user") or env.get("PGUSER")
        if host:
            return f"{host}:{port}/{database or '?'}"
        service = parsed.get("service") or env.get("PGSERVICE")
        return f"service {service}" if service else f"the default local socket, port {port}/{database or '?'}"

    def _scrub(self, text: str) -> str:
        """``text`` with every secret value (and the whole connection string) replaced, also in its URI-encoded form."""
        if self._secrets is None:
            from urllib.parse import quote

            values = [os.environ.get("PGPASSWORD", ""), self._dsn]
            with suppress(PgVectorConfigError):
                parsed = self._parsed()
                values += [parsed.get("password", ""), parsed.get("sslpassword", "")]
            values += [quote(value, safe="") for value in tuple(values) if value]
            self._secrets = tuple(sorted({value for value in values if value}, key=len, reverse=True))
        for value in self._secrets:
            text = text.replace(value, "***")
        return text

    # -- time budget ------------------------------------------------------------------------------------------------

    @staticmethod
    def _remaining() -> float | None:
        control = current_request()
        if control is None or control.deadline is None:
            return None
        return control.deadline - time.monotonic()

    def _timeout_ms(self, override: int | None = None) -> int:
        """The statement and lock timeout: what the request has left (at least 1 ms), else a fixed ceiling."""
        if override is not None:
            return override
        remaining = self._remaining()
        if remaining is None:
            return NO_DEADLINE_TIMEOUT_MS
        return max(1, min(int(math.ceil(remaining * 1000)), 2_147_483_647))

    ARM = ("SELECT set_config('statement_timeout', %s, %s), set_config('lock_timeout', %s, %s), "
           "(SELECT c.reltuples::bigint FROM pg_class c WHERE c.oid = to_regclass(%s))")

    def _arm_parameters(self, timeout_ms: int | None, local: bool) -> tuple:
        """The time the request has left, for both limits (``SET LOCAL`` in a transaction, the session for a read)."""
        value = str(self._timeout_ms(timeout_ms))
        return (value, local, value, local, self._table)

    def _noted(self, row) -> int | None:
        rows = row[2] if row is not None else None
        self._table_rows = rows if rows is not None and rows >= 0 else None
        return self._table_rows

    def _arm(self, connection, timeout_ms: int | None = None) -> int | None:
        """Bind the remaining request time to the statements of the open transaction (``SET LOCAL``). Returns the
        planner's row count of the table, ``None`` before its first ANALYZE."""
        return self._noted(connection.execute(self.ARM, self._arm_parameters(timeout_ms, True)).fetchone())

    def _read(self, connection, statement: str, params=None, *, before: Sequence = ()):
        """One read on an autocommit connection, armed with the time left, in a single network round trip.

        ``before`` are further statements (``(sql, params)``) queued ahead of it, in the same round trip."""
        with self._driver.pipeline(connection):
            armed = connection.execute(self.ARM, self._arm_parameters(None, False))
            for sql, values in before:
                connection.execute(sql, values)
            cursor = connection.execute(statement, params)
        self._noted(armed.fetchone())
        return cursor.fetchall()

    # -- connection life cycle -----------------------------------------------------------------------------------------

    def _connect(self):
        parsed = self._parsed()
        params: dict = {}
        env = os.environ
        transport = check_transport(parsed, env)                    # before any socket is opened
        if transport["unknown_host"]:
            params["sslmode"] = SECURE_SSLMODE                      # a service file hides the host: never in the clear
        remaining = self._remaining()
        if "connect_timeout" not in parsed and "PGCONNECT_TIMEOUT" not in env:
            budget = CONNECT_TIMEOUT_SEC if remaining is None else min(CONNECT_TIMEOUT_SEC, math.ceil(remaining))
            params["connect_timeout"] = max(2, int(budget))         # libpq's minimum is 2 seconds
        if "application_name" not in parsed and "PGAPPNAME" not in env:
            params["application_name"] = APPLICATION_NAME
        # A peer that vanishes (a cable pulled, a host that froze) must end the connection, not hold the one thread that
        # waits on it: keepalives find an idle dead peer in about a minute, tcp_user_timeout one with data in flight in 30 s.
        for key, value in (("keepalives_idle", 30), ("keepalives_interval", 10), ("keepalives_count", 3),
                           ("tcp_user_timeout", 30_000)):
            if key not in parsed:
                params[key] = value
        try:
            connection = self._driver.connect(self._dsn, **params)
        except PgVectorError:
            raise
        except Exception as exc:
            kind = self._driver.classify(exc, None)
            if kind == "unrelated":
                raise
            raise PgVectorUnavailable(
                f"Cannot connect to the pgvector server {self._endpoint()}: {self._scrub(self._driver.message(exc))}. "
                "Check that the server is reachable, the credentials in " + ENVIRONMENT_HINT +
                " are right, and (for a remote server) that sslmode=verify-full is set. No other dense backend is used.") from None
        try:
            self._verify_transport(connection)
        except BaseException:
            with suppress(Exception):
                connection.close()
            raise
        return connection

    def _verify_transport(self, connection) -> None:
        """What libpq actually connected to: a host that is not this machine must have been reached over verified TLS."""
        info = getattr(connection, "info", None)
        host = getattr(info, "host", None)
        if not host or _is_loopback(host):
            return
        try:
            sslmode = info.get_parameters().get("sslmode")
        except Exception:
            sslmode = None
        if not getattr(info, "ssl_in_use", False) or sslmode != SECURE_SSLMODE:
            raise PgVectorConfigError(
                f"The connection to the pgvector server {self._endpoint()} is not a verified TLS connection (the host is "
                f"not this machine). Set sslmode={SECURE_SSLMODE} in SELF_DIRECT_PGVECTOR_DSN_FILE (or PGSSLMODE).")

    def _connection(self):
        connection = self._conn
        if connection is not None and not (connection.closed or connection.broken):
            return connection
        self._conn = None
        connection = self._connect()
        try:
            self._validate(connection)
        except BaseException:
            with suppress(Exception):
                connection.close()
            raise
        info = getattr(connection, "info", None)
        self._peer = {"host": getattr(info, "host", None), "port": getattr(info, "port", None),
                      "dbname": getattr(info, "dbname", None)}
        self._conn = connection
        return connection

    def _discard(self, connection=None) -> None:
        connection = connection if connection is not None else self._conn
        if connection is self._conn:
            self._conn = None
        if connection is not None:
            with suppress(Exception):
                connection.close()

    @staticmethod
    def _rollback(connection) -> None:
        if connection is not None:
            with suppress(Exception):
                connection.rollback()

    def close(self) -> None:
        with self._lock:
            self._discard()

    # -- schema -------------------------------------------------------------------------------------------------------

    def _validate(self, connection) -> None:
        """Check the extension and the recorded dimension and policy; create the schema when it is absent."""
        self._arm(connection)
        row = connection.execute(
            "SELECT e.extversion, n.nspname, to_regtype('vector') IS NOT NULL, to_regnamespace(%s) IS NOT NULL, "
            "to_regclass(%s) IS NOT NULL, to_regclass(%s) IS NOT NULL, to_regclass(%s) IS NOT NULL, "
            "current_setting('server_version'), "
            "CASE WHEN current_setting('idle_in_transaction_session_timeout') = '0' "
            "THEN set_config('idle_in_transaction_session_timeout', %s, false) END "
            "FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector'",
            (_quote(self.schema), self._meta, self._table, f"{_quote(self.schema)}.{_quote(HNSW_INDEX)}",
             str(IDLE_IN_TRANSACTION_MS))).fetchone()
        if row is None:
            connection.rollback()
            raise PgVectorConfigError(
                f"The pgvector extension is not installed in the database of {self._endpoint()}. Ask an administrator to "
                "run CREATE EXTENSION vector; in that database (this process never installs it).")
        extension_version, extension_schema, type_visible, has_schema, has_meta, has_table, has_index, server_version = row[:8]
        if self.hnsw and _version_tuple(extension_version) < MIN_HNSW_EXTENSION:
            connection.rollback()
            raise PgVectorConfigError(
                f"The pgvector extension {extension_version} of {self._endpoint()} has no HNSW index (0.5.0 or newer is "
                "needed): upgrade it (ALTER EXTENSION vector UPDATE;) or unset SELF_DIRECT_PGVECTOR_HNSW.")
        if not type_visible:
            # An extension in another schema (for example "extensions"): make its type and operators resolvable.
            connection.execute("SELECT set_config('search_path', %s, false)",
                               (f"{_quote(extension_schema)}, pg_catalog, {_quote(self.schema)}",)).fetchall()
        self.extension_version, self.server_version = extension_version, server_version
        if has_meta and has_table and (has_index or not self.hnsw):
            meta = dict(connection.execute(f"SELECT key, value FROM {self._meta}").fetchall())
            self._check_meta(meta)
            if self.policy is not None and "policy" not in meta:
                # Built by a process that had no policy to record: the first one that has one claims the schema.
                connection.execute(f"INSERT INTO {self._meta}(key, value) VALUES ('policy', %s) ON CONFLICT (key) DO NOTHING",
                                   (self.policy,))
                self._check_meta(dict(connection.execute(f"SELECT key, value FROM {self._meta}").fetchall()))
            connection.commit()
        else:
            self._create(connection, has_schema)
        self._tls = bool(getattr(getattr(connection, "info", None), "ssl_in_use", False))

    def _check_meta(self, meta: dict) -> None:
        where = f"schema {self.schema!r} of {self._endpoint()}"
        version = meta.get("schema_version")
        if version != str(SCHEMA_VERSION):
            raise PgVectorConfigError(
                f"The pgvector {where} has index schema version {version!r}; this release uses {SCHEMA_VERSION}. "
                "Use a different SELF_DIRECT_PGVECTOR_SCHEMA or upgrade/downgrade tsukkomi-mcp to match.")
        if meta.get("dim") != str(self.dim):
            raise PgVectorConfigError(
                f"The pgvector {where} holds {meta.get('dim')}-dimensional vectors but this process uses {self.dim}. "
                "Use a different SELF_DIRECT_PGVECTOR_SCHEMA for the new embedding dimension, or drop the old schema "
                "(DROP SCHEMA ... CASCADE) and let it be rebuilt.")
        recorded = meta.get("policy")
        if self.policy is not None and recorded is not None and recorded != self.policy:
            raise PgVectorConfigError(
                f"The pgvector {where} was filled with embedding policy {recorded!r} but this process uses "
                f"{self.policy!r} (another embedding model, dimension or encoding): its vectors must not be mixed. Use a "
                "different SELF_DIRECT_PGVECTOR_SCHEMA, or drop the old schema and let it be rebuilt.")

    def _create(self, connection, has_schema: bool) -> None:
        """Create what is missing, under an advisory lock, in one transaction."""
        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                           ("tsukkomi-pgvector:" + self.schema,)).fetchall()
        if not has_schema:
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote(self.schema)}")
        connection.execute(f"CREATE TABLE IF NOT EXISTS {self._meta} (key text PRIMARY KEY, value text NOT NULL)")
        # COLLATE "C": byte order like SQLite's BINARY collation, whatever the database's locale (tie-breaks, parity).
        connection.execute(f"CREATE TABLE IF NOT EXISTS {self._table} "
                           f'(chunk_id text COLLATE "C" PRIMARY KEY, embedding vector({self.dim}) NOT NULL)')
        if self.dim <= PLAIN_STORAGE_MAX_DIM:
            # Inline storage. pgvector's own default for the type is EXTERNAL (out-of-line TOAST chunks, never
            # compressed): about a third less disk, but every row of an exact scan costs a second lookup (README,
            # "pgvector numbers"). The bound keeps a row of an id of at most 1 kB and the vector inside one page.
            connection.execute(f"ALTER TABLE {self._table} ALTER COLUMN embedding SET STORAGE PLAIN")
        stored = connection.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding' "
            "AND NOT attisdropped", (self._table,)).fetchone()
        if stored is None or stored[0] != self.dim:
            raise PgVectorConfigError(
                f"The table {self.schema}.{TABLE} of {self._endpoint()} exists with another shape "
                f"(embedding dimension {None if stored is None else stored[0]}, expected {self.dim}). Use a different "
                "SELF_DIRECT_PGVECTOR_SCHEMA.")
        wanted = [("schema_version", str(SCHEMA_VERSION)), ("dim", str(self.dim)), ("vector_type", "vector")]
        if self.policy is not None:
            wanted.append(("policy", self.policy))
        with connection.cursor() as cursor:
            cursor.executemany(f"INSERT INTO {self._meta}(key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING", wanted)
        self._check_meta(dict(connection.execute(f"SELECT key, value FROM {self._meta}").fetchall()))
        if self.hnsw:
            try:
                connection.execute(f"CREATE INDEX IF NOT EXISTS {_quote(HNSW_INDEX)} ON {self._table} "
                                   "USING hnsw (embedding vector_cosine_ops)")
            except Exception as exc:
                if self._driver.classify(exc, connection) != "timeout":
                    raise
                check_request()                                 # the request's own stop wins over the database's message
                raise PgVectorError(
                    f"Building the HNSW index on {self.schema}.{TABLE} did not finish inside this request. Build it once, "
                    "outside a request: PgVectorIndex(...).build_hnsw_index(maintenance_work_mem='2GB', timeout_ms=0), or "
                    f"CREATE INDEX {HNSW_INDEX} ON {self.schema}.{TABLE} USING hnsw (embedding vector_cosine_ops);") from None
        check_request()
        connection.commit()

    def build_hnsw_index(self, *, maintenance_work_mem: str | None = None, timeout_ms: int | None = None) -> None:
        """Create the HNSW index now (a large table can take minutes: run it outside a request with ``timeout_ms=0``)."""
        if self.dim > HNSW_MAX_DIM:
            raise PgVectorConfigError(f"An HNSW index supports at most {HNSW_MAX_DIM} dimensions.")

        def operation(connection):
            self._arm(connection, timeout_ms)
            if maintenance_work_mem is not None:
                connection.execute("SELECT set_config('maintenance_work_mem', %s, true)", (maintenance_work_mem,)).fetchall()
            connection.execute(f"CREATE INDEX IF NOT EXISTS {_quote(HNSW_INDEX)} ON {self._table} "
                               "USING hnsw (embedding vector_cosine_ops)")
        self._run(operation, write=True)

    # -- running one operation -----------------------------------------------------------------------------------------

    def _run(self, operation, *, write: bool = False):
        """Run ``operation(connection)``: a write in one transaction, a read as autocommit statements; one retry for a lost
        connection or a lock conflict."""
        with self._lock:
            retried = False
            while True:
                check_request()
                connection = None
                try:
                    connection = self._connection()
                    self._mode(connection, autocommit=not write)    # reads: one round trip each; writes: one transaction
                    result = operation(connection)
                    if write:
                        check_request()
                        with publication():         # admission: a stopped request never commits
                            connection.commit()
                    return result
                except (RequestStopped, IndexBusy):
                    self._rollback(connection)
                    raise
                except PgVectorError:
                    self._rollback(connection)
                    raise
                except Exception as exc:
                    kind = self._driver.classify(exc, connection)
                    if kind == "unrelated":
                        self._rollback(connection)
                        raise
                    self._rollback(connection)
                    if kind in ("connection", "transient", "undefined") and not retried:
                        retried = True
                        if kind == "connection":
                            self._discard(connection)
                        elif kind == "undefined":
                            self._discard(connection)       # the schema vanished: the next connection re-creates it
                        else:
                            time.sleep(random.uniform(0.01, 0.05))
                        continue
                    if kind == "connection":
                        self._discard(connection)
                    raise self._translate(exc, kind) from None
                except BaseException:
                    self._discard(connection)               # an interrupt leaves the connection in an unknown state
                    raise

    @staticmethod
    def _mode(connection, *, autocommit: bool) -> None:
        if connection.autocommit != autocommit:
            try:
                connection.autocommit = autocommit
            except Exception:                   # a transaction is still open: end it, then switch
                connection.rollback()
                connection.autocommit = autocommit

    def _translate(self, exc: BaseException, kind: str) -> PgVectorError | RequestStopped:
        detail = self._scrub(self._driver.message(exc))
        endpoint = self._endpoint()
        if kind == "timeout":
            check_request()                                 # the request's own deadline or cancellation wins
            remaining = self._remaining()
            if remaining is not None and remaining < 0.1:   # the server's timer is the request's own, to within a clock tick
                raise RequestStopped("request_deadline")
            return PgVectorError(f"A pgvector statement on {endpoint} exceeded its time limit: {detail}")
        if kind == "connection":
            return PgVectorUnavailable(
                f"The pgvector server {endpoint} became unreachable and the connection could not be re-established: "
                f"{detail}. Check the server and the network, then retry; no other dense backend is used.")
        if kind == "transient":
            return PgVectorError(f"A pgvector write on {endpoint} conflicted with another writer twice: {detail}")
        if kind == "undefined":
            return PgVectorError(f"The pgvector tables of schema {self.schema!r} on {endpoint} disappeared and could not "
                                 f"be re-created: {detail}")
        hint = ""
        if getattr(exc, "sqlstate", None) == "42501":
            hint = (f" The database role needs CREATE on the database to make schema {self.schema!r} (or an administrator "
                    "creates the schema and tables once) and INSERT/DELETE/SELECT/TRUNCATE on the tables.")
        return PgVectorError(f"A pgvector operation on {endpoint} failed: {detail}.{hint}")

    # -- validation ---------------------------------------------------------------------------------------------------

    @staticmethod
    def _storable(item_id) -> bool:
        """Whether ``item_id`` can be a stored id: a nonempty ``str`` of at most ``MAX_ID_CHARS`` characters, no NUL, valid UTF-8."""
        if not isinstance(item_id, str) or not item_id or len(item_id) > MAX_ID_CHARS or "\x00" in item_id:
            return False
        try:
            item_id.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True

    @classmethod
    def _item_id(cls, item_id) -> str:
        """An id to be written: anything that cannot be stored is an error (the write never reaches SQL)."""
        if not cls._storable(item_id):
            raise ValueError(f"A chunk id must be a nonempty UTF-8 string of at most {MAX_ID_CHARS} characters "
                             "without NUL characters")
        return item_id

    @classmethod
    def _lookup_ids(cls, ids) -> list[str]:
        """Ids to be looked up or deleted: the distinct ones that could be stored. An id that could not have been stored is
        simply absent (as in the other backends), not an error."""
        items = ids if isinstance(ids, (set, frozenset)) else dict.fromkeys(ids)           # distinct
        if not items:
            return []
        try:                                                    # the common case in a few C-speed passes, however large
            joined = "".join(items)                             # TypeError: an element that is not a str
            joined.encode("utf-8")                              # UnicodeEncodeError: a lone surrogate
            lengths = list(map(len, items))
        except (TypeError, UnicodeEncodeError):
            pass
        else:
            if "\x00" not in joined and min(lengths) >= 1 and max(lengths) <= MAX_ID_CHARS:
                check_request()
                return list(items)
        found: dict[str, None] = {}
        for position, item_id in enumerate(items):
            if not position & 1023:
                check_request()             # cooperative stop, without a check per id of a very large filter
            if cls._storable(item_id):
                found[item_id] = None
        return list(found)

    @staticmethod
    def _filter_parameters(allowed: list[str]) -> tuple[str, list[str]]:
        """A filter as ``(blob, odd)``: the ids joined by ``ID_SEPARATOR`` into ONE text parameter, which the server splits with
        ``string_to_array``, and the few ids that contain the separator, as an array. A 50 000-id filter costs about 12 ms to
        send this way and about 42 ms as a ``text[]`` parameter (psycopg dumps an array element by element)."""
        blob = ID_SEPARATOR.join(allowed)
        if blob.count(ID_SEPARATOR) == len(allowed) - 1:
            return blob, []
        plain = [item for item in allowed if ID_SEPARATOR not in item]
        return ID_SEPARATOR.join(plain), [item for item in allowed if ID_SEPARATOR in item]

    def _pack(self, vector) -> bytes:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self.dim,) or not np.isfinite(array).all():
            raise ValueError(f"Expected a finite vector with dimension {self.dim}")
        return struct.pack(">HH", self.dim, 0) + array.astype(">f4", copy=False).tobytes()

    # -- VectorIndex ----------------------------------------------------------------------------------------------------

    def refresh(self) -> None:
        check_request()             # reads go to the server and are always current

    @request_operation
    def ids(self) -> set[str]:
        def operation(connection):
            return {row[0] for row in self._read(connection, f"SELECT chunk_id FROM {self._table}")}
        out = self._run(operation)
        check_request()
        return out

    @request_operation
    def present(self, ids) -> set[str]:
        wanted = self._lookup_ids(ids)
        if not wanted:
            return set()

        def operation(connection):
            found: set[str] = set()
            for start in range(0, len(wanted), PRESENT_BATCH):
                check_request()
                rows = self._read(connection, f"SELECT chunk_id FROM {self._table} WHERE chunk_id = ANY(%b)",
                                  (wanted[start:start + PRESENT_BATCH],))
                found.update(row[0] for row in rows)
            return found
        return self._run(operation)

    def clear(self) -> None:
        check_request()

        def operation(connection):
            self._arm(connection)
            connection.execute(f"TRUNCATE {self._table}")   # transactional; reclaims the space at once
        self._run(operation, write=True)

    def upsert(self, item_id, vector) -> None:
        self.upsert_many([(item_id, vector)])

    def upsert_many(self, pairs) -> None:
        rows: dict[str, bytes] = {}
        for item_id, vector in pairs:
            check_request()
            rows[self._item_id(item_id)] = self._pack(vector)       # the last of a duplicate id wins
        if not rows:
            return
        ordered = sorted(rows.items())                  # one lock order for every writer: no deadlock between them
        make = self._driver.vector

        def operation(connection):
            with connection.cursor() as cursor:
                for offset in range(0, len(ordered), UPSERT_BATCH):
                    check_request()
                    self._arm(connection)
                    cursor.executemany(
                        f"INSERT INTO {self._table} AS t(chunk_id, embedding) VALUES (%s, %b::vector) "
                        "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding "
                        "WHERE t.embedding IS DISTINCT FROM EXCLUDED.embedding",     # an identical row is not rewritten
                        [(cid, make(payload)) for cid, payload in ordered[offset:offset + UPSERT_BATCH]])
        self._run(operation, write=True)

    def delete_ids(self, ids) -> None:
        check_request()
        doomed = sorted(self._lookup_ids(ids))
        if not doomed:
            return

        def operation(connection):
            for start in range(0, len(doomed), PRESENT_BATCH):
                check_request()
                self._arm(connection)
                connection.execute(f"DELETE FROM {self._table} WHERE chunk_id = ANY(%b)",
                                   (doomed[start:start + PRESENT_BATCH],))
        self._run(operation, write=True)

    def _plan(self, query, top_k: int, allowed_ids, *, exact: bool = False):
        """The statement a search runs: ``(statement, params, before, allowed)``; ``allowed`` is the sanitized filter
        (``None`` for no filter, an empty list when nothing can match). ``exact`` never uses the HNSW index."""
        allowed = None if allowed_ids is None else self._lookup_ids(allowed_ids)
        probe = self._driver.vector(self._pack(query))
        approximate = self.hnsw and allowed is None and not exact
        where = ("" if allowed is None else
                 "WHERE chunk_id = ANY(string_to_array(%(blob)s::text, chr(31)) || %(odd)b::text[]) ")
        # ``ORDER BY distance, chunk_id`` is the exact scan with a deterministic tie-break (it cannot use an HNSW index);
        # ``ORDER BY distance`` alone is what the approximate index serves. The distance is selected as it is sorted, so the
        # server computes it once per row (``1.0 - (a <=> b)`` in the select list made it twice, 16 % slower at 50 000 rows);
        # the score is ``1 - distance``, computed here.
        order = "distance" + ("" if approximate else ', chunk_id COLLATE "C"')
        # ``+ 0`` keeps an exact search off an HNSW index that happens to exist: with it the planner may order by the index
        # (it trusts it) and apply the id filter to what the index returns, which is approximate and can return fewer than
        # ``top_k`` rows. The sort key no longer matches the index's operator, so only the table and its primary key are used.
        distance = "embedding <=> %(q)b::vector" if approximate else "(embedding <=> %(q)b::vector) + 0"
        statement = (f"SELECT chunk_id, {distance} AS distance FROM {self._table} {where}"
                     f"ORDER BY {order} LIMIT %(k)s")
        params = {"q": probe, "k": int(top_k)}
        if allowed is not None:
            params["blob"], params["odd"] = self._filter_parameters(allowed)
        before = []
        if approximate:
            # pgvector's own default (40) gave a recall@20 of 0.80 at 300 000 clustered vectors, 100 gave 0.89 for 0.1 ms more
            # (README, "pgvector numbers")
            ef = self.hnsw_ef_search or min(1000, max(100, 4 * int(top_k)))
            before.append(("SELECT set_config('hnsw.ef_search', %s, false)", (str(ef),)))
        return statement, params, before, allowed

    @staticmethod
    def _scored(rows) -> list[tuple[str, float]]:
        out = []
        for item_id, distance in rows:
            check_request()
            # A zero vector has no cosine (pgvector: NaN, ordered last); SqliteVectorIndex and NumpyVectorIndex score it 0.
            out.append((item_id, 1.0 - float(distance) if distance is not None and math.isfinite(distance) else 0.0))
        return out

    def _search_after_scan(self, query, top_k: int, allowed_ids, table_rows: int):
        """A filter that names a large part of the table, applied after an unfiltered exact scan: the top ``m`` rows of the
        whole table are fetched and the ids that are not allowed are dropped here, with ``m`` grown until ``top_k`` allowed
        rows are in hand or the table is exhausted. That is exact (an allowed row ranked above the last one found is among
        the ``m`` rows) and it sends no ids. Returns ``None`` after ``POSTFILTER_ROUNDS`` scans that were not enough (the
        filter was not as large a part of the table as it looked): the caller then sends the ids.

        The ids are not sent for a filter that names a quarter of the table or more because sending them costs more than the
        scan: with 65-character ids a hashed filter made the scan 4 times slower at 100 000 rows, and looking 50 000 ids up
        took 146 ms against 72 ms for the scan (README, "pgvector numbers")."""
        allowed = allowed_ids if isinstance(allowed_ids, (set, frozenset)) else set(allowed_ids)
        share = max(0.05, min(1.0, len(allowed) / max(1, table_rows)))
        limit = int(top_k / share * 1.5) + 50
        for _round in range(POSTFILTER_ROUNDS):
            check_request()
            statement, params, _before, _allowed = self._plan(query, limit, None, exact=True)
            rows = self._run(lambda connection: self._read(connection, statement, params))
            hits = [row for row in rows if row[0] in allowed]
            if len(hits) >= top_k or len(rows) < limit:         # enough, or the whole table has been seen
                return self._scored(hits[:top_k])
            limit *= POSTFILTER_GROWTH
        return None

    @request_operation
    def search(self, query, top_k: int = 20, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        check_request()
        if top_k <= 0:
            return []
        if allowed_ids is not None:
            if not hasattr(allowed_ids, "__len__"):
                allowed_ids = set(allowed_ids)
            if len(allowed_ids) == 0:
                return []
            rows = self._table_rows                              # the planner's count, noted by the previous read
            if rows and len(allowed_ids) >= POSTFILTER_MIN_SHARE * rows:
                self._pack(query)                                # a bad vector is an error before any scan
                found = self._search_after_scan(query, int(top_k), allowed_ids, rows)
                if found is not None:
                    return found
        statement, params, before, allowed = self._plan(query, top_k, allowed_ids)
        if allowed is not None and not allowed:
            return []
        return self._scored(self._run(lambda connection: self._read(connection, statement, params, before=before)))

    @request_operation
    def explain(self, query, top_k: int = 20, allowed_ids: set[str] | None = None) -> list[str]:
        """The server's plan (``EXPLAIN``, no execution) for the statement ``search`` sends with its ids (an index lookup of
        the filter, or the HNSW index for no filter): what an operator checks to see how a search is served."""
        statement, params, before, _allowed = self._plan(query, top_k, allowed_ids)

        def operation(connection):
            statements = list(before) + [("EXPLAIN " + statement, params)]
            with self._driver.pipeline(connection):
                armed = connection.execute(self.ARM, self._arm_parameters(None, False))
                for sql, values in statements[:-1]:
                    connection.execute(sql, values)
                cursor = connection.execute(*statements[-1])
            self._noted(armed.fetchone())
            return [_REDACT.sub("'...'", row[0]) for row in cursor.fetchall()]
        return self._run(operation)

    @request_operation
    def count(self) -> int:
        def operation(connection):
            return int(self._read(connection, f"SELECT count(*) FROM {self._table}")[0][0])
        return self._run(operation)

    # -- status ----------------------------------------------------------------------------------------------------------

    def describe(self) -> dict:
        """What ``audit_status`` shows: no user, no password, no connection options. It reads what was noted when the
        connection was made, never the connection itself, which another thread may be using."""
        try:
            parsed = self._parsed()
        except PgVectorConfigError:
            parsed = {}
        env, peer = os.environ, self._peer
        host = peer.get("host") or parsed.get("host") or env.get("PGHOST")
        port = peer.get("port") or parsed.get("port") or env.get("PGPORT") or 5432
        database = peer.get("dbname") or parsed.get("dbname") or env.get("PGDATABASE")
        try:
            port = int(port)
        except (TypeError, ValueError):
            port = str(port)                    # several hosts: "5432,5433"
        return {"backend": "pgvector", "host": host, "port": port, "database": database, "schema": self.schema,
                "extension_version": self.extension_version, "server_version": self.server_version,
                "dimension": self.dim, "policy": self.policy, "hnsw": self.hnsw, "tls": self._tls,
                "loopback": _is_loopback(host), "connected": self._conn is not None}
