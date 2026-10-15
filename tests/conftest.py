

import hashlib
import os

import pytest

WINDOWS_PRIVILEGE_NOT_HELD = 1314


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Report symlink tests as skipped, not failed, where Windows denies symlink creation."""
    outcome = yield
    error = outcome.excinfo[1] if outcome.excinfo else None
    if isinstance(error, OSError) and getattr(error, "winerror", None) == WINDOWS_PRIVILEGE_NOT_HELD:
        outcome.force_exception(pytest.skip.Exception(
            "creating symlinks needs Windows Developer Mode or administrator rights", _use_item_location=True))


@pytest.fixture(autouse=True)
def explicit_test_backend(tmp_path, monkeypatch):
    # Offline default: NumPy. The sanitizer CI sets TSUKKOMI_TEST_DENSE_BACKEND=sqlite-vector together with
    # SQLITE_VECTOR_TEST_PATH (a sanitizer build of the official extension), so every engine the suite creates
    # without an explicit backend loads the native library.
    if os.environ.get("TSUKKOMI_TEST_DENSE_BACKEND") == "sqlite-vector":
        monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "sqlite-vector")
        monkeypatch.setenv("SELF_DIRECT_SQLITE_VECTOR_PATH", os.environ["SQLITE_VECTOR_TEST_PATH"])
    else:
        monkeypatch.setenv("SELF_DIRECT_DENSE_BACKEND", "numpy")
    monkeypatch.delenv("SELF_DIRECT_OPENROUTER_API_KEY_FILE", raising=False)
    # Every audit in the suite also runs the full evaluator beside the incremental one and compares them; a
    # difference returns the full result, so an incremental defect shows up as the mismatch assertions of
    # tests/test_incremental_*.py or as a changed finding, never as a silent wrong answer.
    monkeypatch.setenv("SELF_DIRECT_AUDIT_SHADOW_EVERY", "1")
    # The suite's sessions are small; keep monitor state for them anyway so every audit exercises the incremental path.
    monkeypatch.setenv("SELF_DIRECT_AUDIT_INCREMENTAL_MIN_EVENTS", "0")
    # Never load a developer's per-user TOML while running isolated regressions.
    config_file = tmp_path / "isolated-config.toml"
    config_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("SELF_DIRECT_CONFIG_FILE", str(config_file))


@pytest.fixture(autouse=True)
def defer_derived_everywhere(monkeypatch):
    """``TSUKKOMI_TEST_DEFER_DERIVED=1`` runs the suite with every engine holding its derived writes (the daemon's mode).

    Nothing but the engine's default changes, so every audit, differential and shadow test then runs against monitor
    states, verification stamps and AGY markers that are held in memory and written with a later commit. Tests that
    look at those rows in the database, or at the ``"saved"`` receipt, depend on the writes being synchronous and are
    marked ``synchronous_derived_writes``."""
    if os.environ.get("TSUKKOMI_TEST_DEFER_DERIVED") != "1":
        return
    from self_directing_mcp import engine as engine_module
    original = engine_module.SelfDirectEngine.__init__

    def deferring(self, settings=None, **kwargs):
        kwargs.setdefault("defer_derived", True)
        original(self, settings, **kwargs)

    monkeypatch.setattr(engine_module.SelfDirectEngine, "__init__", deferring)


@pytest.fixture
def no_forced_comparisons(monkeypatch):
    """Tests that measure the delta path alone (rows read, bytes hashed, memory) with ``audit_shadow_every=0`` also have to
    switch off the warm-up comparisons that follow a rebuild or a change of premise (``audit.shadow.WARMUP_REQUESTS``);
    without that, the first requests of every session also run the full evaluator. Violations, flips and high-risk contracts
    are still compared; those tests have none."""
    from self_directing_mcp.audit import shadow
    monkeypatch.setattr(shadow, "WARMUP_REQUESTS", 0)


class HashMeter:
    """Counts the bytes fed to every ``hashlib.sha256`` object created while installed (copies included).

    Used to prove how many times a source prefix is hashed: ``meter.bytes / file_size`` is the number of passes.
    Replacing the factory (not one call site) means a hash site that bypasses the helpers cannot hide.
    """

    def __init__(self, monkeypatch):
        self.bytes = 0
        real, meter = hashlib.sha256, self

        class Counted:
            __slots__ = ("_inner",)

            def __init__(self, inner):
                self._inner = inner

            def update(self, data):
                meter.bytes += len(data)
                self._inner.update(data)

            def copy(self):
                return Counted(self._inner.copy())

            def digest(self):
                return self._inner.digest()

            def hexdigest(self):
                return self._inner.hexdigest()

        def counting(data=b"", **kwargs):
            counted = Counted(real(**kwargs))
            if data:
                counted.update(data)
            return counted

        monkeypatch.setattr(hashlib, "sha256", counting)

    def lap(self) -> int:
        """Bytes hashed since the previous lap (or installation)."""
        taken, self.bytes = self.bytes, 0
        return taken


@pytest.fixture
def hash_meter(monkeypatch):
    return HashMeter(monkeypatch)


@pytest.fixture(autouse=True)
def strict_incremental(monkeypatch, request):
    """A shadow comparison that finds the incremental result different from the full one fails the test.

    In production a difference returns the full result and disables the incremental path for that contract; here it
    must be loud, or the whole suite would pass over an incremental defect. Tests that provoke a mismatch on purpose
    carry the ``allow_incremental_mismatch`` marker.
    """
    from self_directing_mcp.audit import incremental
    real = incremental.compare

    def strict(result, reference, *args, **kwargs):
        bad = real(result, reference, *args, **kwargs)
        if bad and request.node.get_closest_marker("allow_incremental_mismatch") is None:
            raise AssertionError(f"incremental audit differs from the full evaluator for {bad}: "
                                 f"{[f.model_dump() for f in result.findings]} != "
                                 f"{[f.model_dump() for f in reference.findings]}")
        return bad

    monkeypatch.setattr(incremental, "compare", strict)


def pytest_configure(config):
    config.addinivalue_line("markers", "allow_incremental_mismatch: the test provokes an incremental/full mismatch")


def pytest_sessionfinish(session, exitstatus):
    """Under the ASan profile of scripts/verify_memory.py, ask LeakSanitizer for unreachable blocks while the
    interpreter is still alive. Live Python objects are reachable then, so only genuine leaks are reported (the
    exit-time check would also list everything CPython does not free at shutdown)."""
    report = os.environ.get("TSUKKOMI_LSAN_RESULT")
    if not report:
        return
    import ctypes
    import gc
    gc.collect()
    leaked = ctypes.CDLL(None).__lsan_do_recoverable_leak_check()
    with open(report, "w", encoding="utf-8") as stream:
        stream.write(str(leaked))
