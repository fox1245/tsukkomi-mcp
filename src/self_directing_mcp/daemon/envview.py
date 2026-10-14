"""A per-request view of the process environment, for the daemon.

A hook or MCP server that runs as its own process reads its configuration from ``os.environ``. The daemon serves
many clients, each with its own environment, and the code it runs (settings readers, ``workflow``, the hook state
fallbacks) reads ``os.environ`` too. ``install()`` replaces ``os.environ`` with an ``OverlayEnviron``: inside
``request_environment(env)`` the *managed* names (``protocol.is_managed``: ``SELF_DIRECT_*``, ``TSUKKOMI_*``,
``HOME``, ``CODEX_SESSIONS_DIR``, ``CLAUDE_CONFIG_DIR``, ``OPENROUTER_API_KEY``) come from the request alone, never
from the daemon's own environment, and every other name (``PATH``, locale, ...) comes from the daemon. The view
is a ``ContextVar``: it follows ``asyncio.to_thread`` and ``contextvars.copy_context().run`` and is invisible to
every other thread, so concurrent requests cannot see each other's values.

Not forwarded, therefore absent inside a request: ``OPENROUTER_API_KEY`` and every other credential. The daemon
reads a key file named by ``SELF_DIRECT_OPENROUTER_API_KEY_FILE`` itself.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
import os

from self_directing_mcp.daemon.protocol import is_managed

_overlay: ContextVar[dict | None] = ContextVar("tsukkomi_request_environment", default=None)


class OverlayEnviron(MutableMapping):
    def __init__(self, base) -> None:
        self._base = base

    def __getitem__(self, key):
        overlay = _overlay.get()
        if overlay is not None and is_managed(key):
            return overlay[key]
        return self._base[key]

    def __setitem__(self, key, value) -> None:
        overlay = _overlay.get()
        if overlay is not None and is_managed(key):
            if not isinstance(value, str):
                raise TypeError("str expected, not " + type(value).__name__)
            overlay[key] = value
        else:
            self._base[key] = value

    def __delitem__(self, key) -> None:
        overlay = _overlay.get()
        if overlay is not None and is_managed(key):
            del overlay[key]
        else:
            del self._base[key]

    def __iter__(self):
        overlay = _overlay.get()
        if overlay is None:
            yield from self._base
            return
        for key in self._base:
            if not is_managed(key):
                yield key
        yield from overlay

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def copy(self) -> dict:
        return dict(self)

    def __repr__(self) -> str:
        return f"OverlayEnviron(<{len(self)} names>)"


def install() -> OverlayEnviron:
    """Replace ``os.environ`` once. Call before importing code that might hold ``os.environ`` itself."""
    if isinstance(os.environ, OverlayEnviron):
        return os.environ
    overlay = OverlayEnviron(os.environ)
    os.environ = overlay            # type: ignore[assignment]
    return overlay


def uninstall() -> None:
    """Undo ``install`` (tests)."""
    if isinstance(os.environ, OverlayEnviron):
        os.environ = os.environ._base       # type: ignore[assignment]


@contextmanager
def request_environment(env):
    token = _overlay.set(dict(env))
    try:
        yield
    finally:
        _overlay.reset(token)


def active() -> bool:
    return _overlay.get() is not None
