"""A throwaway pgvector server for the tests and the benchmark (synthetic data only).

``require_server()`` returns a ``PgServer``: either the server named by ``TSUKKOMI_PGVECTOR_TEST_DSN_FILE`` (a private
file with one libpq connection string; what the CI service container uses), or a container this module starts from
the image that is already on the machine (it never pulls) and always removes.

The container is the only one this module starts: named ``tsk-pgv-<random>``, published on 127.0.0.1 only, a random
password that exists in this process's memory and the ``docker run`` process environment (never on a command line),
a tmpfs data directory, no capabilities, ``no-new-privileges``. Containers with other names are never inspected,
stopped or removed.

``python tests/pgvector_support.py serve STATE_FILE`` keeps such a container running until interrupted and writes
its private connection-string file next to ``STATE_FILE``: for manual experiments and the benchmark.
"""
from __future__ import annotations

import atexit
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

IMAGE = os.environ.get("TSUKKOMI_PGVECTOR_TEST_IMAGE", "pgvector/pgvector:pg17")
NAME_PREFIX = "tsk-pgv-"
DATABASE = "tsukkomi_test"
USER = "tsukkomi"
POSTGRES_UID = 999          # the postgres user of the official image: no capability is needed to chown the data directory
_RUNNING: list["PgServer"] = []


class ServerUnavailable(RuntimeError):
    """No test server can be had (no Docker, no image, no psycopg); the message is the skip reason."""


def _docker(*args: str, env: dict | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout,
                          env=None if env is None else {**os.environ, **env})


class PgServer:
    """A PostgreSQL server with the ``vector`` extension created in ``dbname``."""

    def __init__(self, *, host: str, port: int, user: str, password: str, dbname: str, container: str | None = None,
                 extra: str = "", options: dict | None = None) -> None:
        self.host, self.port, self.user, self.password, self.dbname = host, port, user, password, dbname
        self.container = container          # None for an external server
        self.extra = extra                  # further libpq keywords of an external server (sslmode ...)
        self.options = options or {}        # what start_container was called with, for restart()
        self.stopped = False

    @property
    def managed(self) -> bool:
        """Whether this module started the server (so it can stop and restart it)."""
        return bool(self.options) or self.container is not None

    def restart(self) -> None:
        """Start a fresh container after ``stop()`` (the data directory is a tmpfs: the database is new, with the
        extension created). Host, port and password change; the object is updated in place."""
        if not self.options and self.container is None:
            raise ServerUnavailable("restart needs the container this module started")
        self.stop()
        fresh = start_container(**self.options)
        _RUNNING.remove(fresh)
        self.host, self.port, self.user, self.password, self.dbname = (fresh.host, fresh.port, fresh.user, fresh.password,
                                                                       fresh.dbname)
        self.container = fresh.container
        _RUNNING.append(self)

    def conninfo(self, dbname: str | None = None, **keywords) -> str:
        parts = {"host": self.host, "port": self.port, "dbname": dbname or self.dbname, "user": self.user,
                 "password": self.password, **keywords}
        text = " ".join(f"{key}={value}" for key, value in parts.items())
        return f"{text} {self.extra}".strip()

    def write_dsn_file(self, directory: Path, name: str = "dsn.txt", **keywords) -> Path:
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(self.conninfo(**keywords) + "\n")
        os.chmod(path, 0o600)
        return path

    def connect(self, dbname: str | None = None, **keywords):
        """An autocommit administrator connection (psycopg 3)."""
        import psycopg
        return psycopg.connect(self.conninfo(dbname, **keywords), autocommit=True)

    def execute(self, sql: str, params=None, dbname: str | None = None):
        with self.connect(dbname) as connection:
            cursor = connection.execute(sql, params)
            return cursor.fetchall() if cursor.description else None

    def new_database(self, name: str) -> None:
        self.execute(f'DROP DATABASE IF EXISTS "{name}"')
        self.execute(f'CREATE DATABASE "{name}"')

    def drop_schema(self, schema: str) -> None:
        self.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')

    def stop(self) -> None:
        """Stop and remove the container (a no-op for an external server)."""
        if self in _RUNNING:
            _RUNNING.remove(self)
        if self.container is not None:
            _docker("rm", "-f", self.container, timeout=60)
            self.container = None
            self.stopped = True


class TcpProxy:
    """A loopback TCP forwarder in front of a server: ``stop()`` is, for a client, a server that went away (open connections
    reset, new ones refused); ``start()`` brings it back on the same port. Works for any server, container or not."""

    def __init__(self, target_host: str, target_port: int) -> None:
        import socket
        import threading

        self._socket_module, self._threading = socket, threading
        self.target = (target_host, target_port)
        self.delay = 0.0
        self.port = 0
        self._listener = None
        self._connections: list = []
        self._lock = threading.Lock()
        self.start()

    def start(self) -> None:
        socket = self._socket_module
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.port))
        listener.listen(16)
        self.port = listener.getsockname()[1]
        self._listener = listener
        self._threading.Thread(target=self._accept, args=(listener,), daemon=True).start()

    def _accept(self, listener) -> None:
        while True:
            try:
                client, _ = listener.accept()
            except OSError:
                return
            try:
                upstream = self._socket_module.create_connection(self.target, timeout=10)
            except OSError:
                client.close()
                continue
            with self._lock:
                self._connections.extend((client, upstream))
            for source, sink in ((client, upstream), (upstream, client)):
                self._threading.Thread(target=self._pump, args=(source, sink), daemon=True).start()

    def _pump(self, source, sink) -> None:
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                if self.delay:
                    time.sleep(self.delay)          # an emulated one-way network delay (latency only, bandwidth is not limited)
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for end in (source, sink):
                try:
                    end.close()
                except OSError:
                    pass

    def drop_connections(self) -> None:
        with self._lock:
            connections, self._connections = self._connections, []
        for connection in connections:
            try:
                connection.shutdown(self._socket_module.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.shutdown(self._socket_module.SHUT_RDWR)      # wakes the accept() blocked in another thread
            except OSError:
                pass
            self._listener.close()
            self._listener = None
        self.drop_connections()


def _stop_all() -> None:
    for server in list(_RUNNING):
        server.stop()


atexit.register(_stop_all)


def _import_psycopg():
    try:
        import psycopg
    except ImportError:
        raise ServerUnavailable("psycopg 3 is not installed (pip install 'psycopg[binary]>=3.2')") from None
    return psycopg


def external_server() -> PgServer | None:
    """The server of ``TSUKKOMI_PGVECTOR_TEST_DSN_FILE``: a throwaway database that may be written freely."""
    name = os.environ.get("TSUKKOMI_PGVECTOR_TEST_DSN_FILE")
    if not name:
        return None
    psycopg = _import_psycopg()
    from psycopg.conninfo import conninfo_to_dict

    parsed = conninfo_to_dict(Path(name).read_text(encoding="utf-8").strip())
    extra = " ".join(f"{key}={value}" for key, value in parsed.items()
                     if key not in ("host", "port", "dbname", "user", "password"))
    server = PgServer(host=parsed.get("host", "127.0.0.1"), port=int(parsed.get("port", 5432)),
                      user=parsed.get("user", "postgres"), password=parsed.get("password", ""),
                      dbname=parsed.get("dbname", "postgres"), extra=extra)
    try:
        server.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.Error as exc:
        raise ServerUnavailable(f"the server of TSUKKOMI_PGVECTOR_TEST_DSN_FILE is not usable: {type(exc).__name__}") from None
    return server


def start_container(*, shared_buffers: str = "512MB", tmpfs_size: str = "3g", shm_size: str = "256m",
                    settings: tuple[str, ...] = (), wait_sec: float = 90) -> PgServer:
    """Start one throwaway container and return once ``CREATE EXTENSION vector`` has succeeded in its database."""
    psycopg = _import_psycopg()
    if shutil.which("docker") is None:
        raise ServerUnavailable("Docker is not installed")
    try:
        if _docker("image", "inspect", IMAGE, timeout=30).returncode != 0:
            raise ServerUnavailable(f"the image {IMAGE} is not present locally (this suite never pulls)")
    except (OSError, subprocess.TimeoutExpired):
        raise ServerUnavailable("the Docker daemon does not answer") from None
    name = NAME_PREFIX + secrets.token_hex(4)
    password = secrets.token_urlsafe(18)
    run = _docker(
        "run", "-d", "--rm", "--name", name, "--user", f"{POSTGRES_UID}:{POSTGRES_UID}", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--shm-size", shm_size,
        "--tmpfs", f"/var/lib/postgresql/data:size={tmpfs_size},uid={POSTGRES_UID},gid={POSTGRES_UID},mode=0700",
        "-p", "127.0.0.1::5432", "-e", "POSTGRES_PASSWORD", "-e", f"POSTGRES_USER={USER}", "-e", f"POSTGRES_DB={DATABASE}",
        IMAGE, "-c", f"shared_buffers={shared_buffers}", *[arg for item in settings for arg in ("-c", item)],
        env={"POSTGRES_PASSWORD": password})
    if run.returncode != 0:
        _docker("rm", "-f", name, timeout=60)
        raise ServerUnavailable("docker run failed: " + " ".join(run.stderr.split())[:200].replace(password, "***"))
    server = PgServer(host="127.0.0.1", port=0, user=USER, password=password, dbname=DATABASE, container=name,
                      options={"shared_buffers": shared_buffers, "tmpfs_size": tmpfs_size, "shm_size": shm_size,
                               "settings": settings, "wait_sec": wait_sec})
    _RUNNING.append(server)
    try:
        mapped = _docker("port", name, "5432/tcp").stdout.split()
        server.port = int(mapped[0].rsplit(":", 1)[1])
        deadline = time.monotonic() + wait_sec
        while True:
            ready = _docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", USER, "-d", DATABASE)
            if ready.returncode == 0:
                try:
                    server.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    return server
                except psycopg.Error:
                    pass
            if time.monotonic() > deadline:
                raise ServerUnavailable("the container did not become ready in time")
            time.sleep(0.5)
    except BaseException:
        server.stop()
        raise


def require_server(**options) -> PgServer:
    server = external_server()
    return server if server is not None else start_container(**options)


def serve(state_file: Path) -> None:
    """Start a container, write its private DSN file and keep it until interrupted."""
    state = Path(state_file)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))     # the cleanup below runs
    server = start_container(tmpfs_size=os.environ.get("TSK_PGV_TMPFS", "3g"), shm_size=os.environ.get("TSK_PGV_SHM", "1g"),
                             settings=tuple(filter(None, os.environ.get("TSK_PGV_SETTINGS", "").split(";"))))
    dsn = server.write_dsn_file(state.parent, state.name)
    print(f"{server.container} listening on 127.0.0.1:{server.port}; dsn file {dsn}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        dsn.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "serve":
        serve(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
