#!/usr/bin/env python3
"""Sanitizer verification for tsukkomi, after NeoGraph's ``verify_ci.py`` sanitizer profiles.

tsukkomi itself is Python; the native code it runs is neograph-engine and the sqlite-vector extension. The
``asan`` (AddressSanitizer + UndefinedBehaviorSanitizer + LeakSanitizer) and ``tsan`` (ThreadSanitizer) profiles
rebuild both from pinned, digest-verified sources, run the test suite against them with the sanitizer runtime
preloaded into CPython, and fail on any sanitizer report. Canaries first prove that each sanitizer is live, so
"no reports" means something.

    python scripts/verify_memory.py native --profile asan --out build/asan   # build both libraries (cached)
    python -m pip install build/asan/wheels/*.whl && python -m pip install -e ".[dev]"
    python scripts/verify_memory.py canary --profile asan --out build/asan   # detection works
    python scripts/verify_memory.py test   --profile asan --out build/asan   # suite under the sanitizer

LeakSanitizer and CPython: the interpreter does not free every object at exit, so the usual exit-time leak check
reports thousands of interpreter-owned blocks. The ``asan`` profile disables it and instead runs LSan's
recoverable check from ``tests/conftest.py`` when the pytest session finishes, while live Python objects are still
reachable; only unreachable blocks are reported then. Each *direct* leak is classified by the first frame above
the allocator: neograph or sqlite-vector code fails the run; CPython (including Python objects created through
pybind11) and third-party libraries (OpenSSL, libcurl, ...) are counted and shown. Indirect leaks hang off another
leaked block and are never the root cause.

Linux with GCC only. Standard library only, so it runs before the project is installed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# neograph-engine sdists, by the version pinned in pyproject.toml; digests as published on PyPI.
NEOGRAPH_SDISTS = {"0.14.1": "751cc3de3174840caf497549e2f42dab49a3720e79830ff7a0c3ca4f3431a244"}
# sqliteai/sqlite-vector release tag and the commit it must resolve to (the official 1.1.0 release).
SQLITE_VECTOR = ("1.1.0", "235a7c51b18858e1fea9fb2895e115e4cbbd9f7d")
SQLITE_VECTOR_REPO = "https://github.com/sqliteai/sqlite-vector.git"
PROFILES = {
    "asan": {"flags": "-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined -O1 -g",
             "link": "-fsanitize=address,undefined", "runtime": ("libasan.so", "libstdc++.so"), "needed": "libasan"},
    "tsan": {"flags": "-fsanitize=thread -fno-omit-frame-pointer -O1 -g",
             "link": "-fsanitize=thread", "runtime": ("libtsan.so",), "needed": "libtsan"},
}
# Both profiles run the whole suite. Under TSan, /bin/sh (dash) itself crashes with the runtime preloaded (SIGSEGV
# before it runs anything), so the tests that start hook commands through a shell are deselected there; Python
# child processes are unaffected.
TSAN_DESELECT = ("tests/test_claude_install.py::test_printed_commands_run_the_real_hook_and_fail_open_on_bad_input",
                 "tests/test_claude_install.py::test_index_dir_is_baked_into_every_command_and_must_be_absolute")
TSAN_EXIT = 66
CANARY = r"""
/* Sanitizer canaries: each function holds exactly one deliberate defect. */
#include <limits.h>
#include <stdlib.h>
#include <string.h>
static volatile long shared_counter;
/* Two Python threads call this through ctypes, which releases the GIL: the loops overlap with no ordering. */
void race_increment(long n) { for (long i = 0; i < n; ++i) shared_counter = shared_counter + 1; }
/* Writes one byte past the block through memset, which only ASan (not UBSan's object-size check) sees. */
int heap_overflow(int n) { char *p = malloc(n); memset(p, 1, (size_t)n + 1); int v = p[0]; free(p); return v; }
int signed_overflow(int n) { volatile int x = INT_MAX; return x + n; }
void *leak(int n) { void *volatile p = malloc(n); p = 0; return p; }
"""
CANARY_DRIVER = r"""
import ctypes, sys, threading
lib = ctypes.CDLL(sys.argv[2])
if sys.argv[1] == "race":
    threads = [threading.Thread(target=lib.race_increment, args=(ctypes.c_long(20_000_000),)) for _ in range(2)]
    [t.start() for t in threads]; [t.join() for t in threads]
elif sys.argv[1] == "overflow":
    lib.heap_overflow(16)
elif sys.argv[1] == "ubsan":
    lib.signed_overflow(1)
else:
    lib.leak(4096)
    print("leak check:", ctypes.CDLL(None).__lsan_do_recoverable_leak_check())
"""


def run(command, **kwargs):
    print("+", " ".join(str(part) for part in command), file=sys.stderr, flush=True)
    return subprocess.run([str(part) for part in command], check=kwargs.pop("check", True), **kwargs)


def gcc_file(name: str) -> str:
    location = run(["gcc", f"-print-file-name={name}"], capture_output=True, text=True).stdout.strip()
    if not Path(location).is_file():
        raise SystemExit(f"gcc did not resolve the sanitizer runtime {name}")
    return location


def neograph_version() -> str:
    match = re.search(r'"neograph-engine==([0-9][^"]*)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("pyproject.toml does not pin neograph-engine==<version>")
    return match.group(1)


def manifest(profile: str) -> dict:
    version = neograph_version()
    if version not in NEOGRAPH_SDISTS:
        raise SystemExit(f"neograph-engine {version} has no pinned sdist digest in {Path(__file__).name}")
    compiler = run(["gcc", "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    return {"profile": profile, "flags": PROFILES[profile]["flags"], "gcc": compiler,
            "python": "%d.%d" % sys.version_info[:2], "neograph_version": version,
            "neograph_sdist_sha256": NEOGRAPH_SDISTS[version], "sqlite_vector": list(SQLITE_VECTOR)}


# --- native builds ------------------------------------------------------------------------------------------

def fetch_neograph_sdist(version: str, directory: Path) -> Path:
    with urllib.request.urlopen(f"https://pypi.org/pypi/neograph-engine/{version}/json", timeout=60) as response:
        release = json.load(response)
    [sdist] = [item for item in release["urls"] if item["packagetype"] == "sdist"]
    if sdist["digests"]["sha256"] != NEOGRAPH_SDISTS[version]:
        raise SystemExit("PyPI's neograph-engine sdist digest differs from the pinned digest")
    target = directory / sdist["filename"]
    with urllib.request.urlopen(sdist["url"], timeout=300) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != NEOGRAPH_SDISTS[version]:
        raise SystemExit("downloaded neograph-engine sdist does not match the pinned digest")
    target.write_bytes(data)
    return target


def build_neograph(profile: str, out: Path, ccache: bool) -> None:
    spec = PROFILES[profile]
    with tempfile.TemporaryDirectory() as directory:
        sdist = fetch_neograph_sdist(neograph_version(), Path(directory))
        settings = {"cmake.build-type": "RelWithDebInfo", "build-dir": str(out / "neograph-build"),
                    "cmake.define.CMAKE_C_FLAGS": spec["flags"], "cmake.define.CMAKE_CXX_FLAGS": spec["flags"],
                    **{f"cmake.define.CMAKE_{kind}_LINKER_FLAGS": spec["link"] for kind in ("SHARED", "MODULE", "EXE")}}
        if ccache:
            settings.update({"cmake.define.CMAKE_C_COMPILER_LAUNCHER": "ccache",
                             "cmake.define.CMAKE_CXX_COMPILER_LAUNCHER": "ccache"})
        shutil.rmtree(out / "wheels", ignore_errors=True)
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", out / "wheels", sdist,
             *[f"-C{key}={value}" for key, value in settings.items()]])


def build_sqlite_vector(profile: str, out: Path) -> None:
    spec, (tag, commit) = PROFILES[profile], SQLITE_VECTOR
    source = out / "sqlite-vector"
    shutil.rmtree(source, ignore_errors=True)
    run(["git", "clone", "--quiet", "--depth", "1", "--branch", tag, SQLITE_VECTOR_REPO, source])
    head = run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"sqlite-vector {tag} resolved to {head}, expected {commit}")
    run(["make", "-C", source, "extension", "STRIP=true",
         f"CFLAGS=-Wall -Wno-unused-parameter -Isrc -Ilibs {spec['flags']}", f"LDFLAGS=-shared {spec['link']}"],
        stdout=subprocess.DEVNULL)
    shutil.copy2(source / "dist" / "vector.so", out / "vector.so")


def command_native(args) -> int:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = manifest(args.profile)
    stamp = out / "manifest.json"
    if (stamp.is_file() and json.loads(stamp.read_text()) == wanted and (out / "vector.so").is_file()
            and list((out / "wheels").glob("neograph_engine-*.whl"))):
        print(f"{args.profile}: native build is current ({stamp})")
        return 0
    build_sqlite_vector(args.profile, out)
    build_neograph(args.profile, out, args.ccache)
    stamp.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    print(f"{args.profile}: built {out / 'vector.so'} and {next((out / 'wheels').glob('*.whl')).name}")
    return 0


# --- running under a sanitizer ------------------------------------------------------------------------------

def environment(profile: str, out: Path, logs: Path) -> tuple[dict, list]:
    env = os.environ.copy()
    preload = ":".join(gcc_file(name) for name in PROFILES[profile]["runtime"])
    if profile == "asan":
        env["LD_PRELOAD"] = preload
        # Every Python object through malloc, so ASan sees native writes into interpreter memory too.
        env["PYTHONMALLOC"] = "malloc"
        # No exit-time leak check (see the module docstring). Never set LSAN_OPTIONS=exitcode=...: exitcode is a
        # flag shared with ASan, so it would also turn ASan errors into a zero exit status.
        env["ASAN_OPTIONS"] = (f"detect_leaks=1:leak_check_at_exit=0:halt_on_error=1:abort_on_error=0:"
                               f"log_path={logs / 'asan'}")
        env.pop("LSAN_OPTIONS", None)
        env["UBSAN_OPTIONS"] = f"print_stacktrace=1:halt_on_error=1:log_path={logs / 'ubsan'}"
        prefix = []
    else:
        env["TSAN_OPTIONS"] = (f"halt_on_error=0:second_deadlock_stack=1:exitcode={TSAN_EXIT}:"
                               f"log_path={logs / 'tsan'}")
        # TSan's shadow memory needs a fixed layout: setarch disables randomization, then env preloads the
        # runtime into the interpreter only (preloading into setarch itself would initialize TSan too early).
        prefix = ["setarch", platform.machine(), "-R", "env", f"LD_PRELOAD={preload}"]
    return env, prefix


_RUNTIME = re.compile(r"libsanitizer|/libasan|/libtsan|/libubsan|asan_malloc|asan_new_delete|libstdc\+\+|libc\.so"
                      r"|libgcc|/bits/|new_allocator\.h|alloc_traits\.h")
_PYTHON = re.compile(r"/python3(\.\d+)?\b|libpython|\(/usr/bin/python|\bpython3\.\d+\+0x|\b_?Py[A-Z_]")


def classify_leaks(logs: Path, project: re.Pattern) -> collections.Counter:
    """Count direct leaks by owner: ``project``, ``cpython``, ``third_party``; print project leak stacks."""
    owners = collections.Counter()
    for log in sorted(logs.glob("asan*")):
        for block in re.split(r"\n(?=(?:Direct|Indirect) leak of)", log.read_text(errors="replace")):
            if not block.startswith("Direct leak of"):
                continue
            frames = [line.strip() for line in block.splitlines()[1:] if line.strip().startswith("#")]
            owner = "third_party"
            for frame in frames[1:]:
                if _RUNTIME.search(frame):
                    continue
                owner = "project" if project.search(frame) else "cpython" if _PYTHON.search(frame) else "third_party"
                break
            owners[owner] += 1
            if owner == "project":
                print("\n".join(block.splitlines()[:16]), "\n")
    return owners


def scan_reports(logs: Path) -> collections.Counter:
    counts = collections.Counter()
    for log in logs.glob("*"):
        text = log.read_text(errors="replace")
        counts["asan_errors"] += len(re.findall(r"ERROR: AddressSanitizer", text))
        counts["ubsan_errors"] += len(re.findall(r"runtime error:", text))
        counts["tsan_warnings"] += len(re.findall(r"WARNING: ThreadSanitizer", text))
    return counts


def dump_logs(logs: Path, limit: int = 120) -> None:
    """Print the head of every sanitizer log, so a CI failure is diagnosable from the job output alone."""
    for log in sorted(logs.glob("*")):
        lines = log.read_text(errors="replace").splitlines()
        print(f"--- {log.name} ({len(lines)} lines)")
        print("\n".join(lines[:limit]))


def require_instrumented(profile: str) -> None:
    """The installed neograph-engine must be the sanitizer build, not the release wheel."""
    import importlib.util
    spec = importlib.util.find_spec("neograph_engine")
    if spec is None or spec.origin is None:
        raise SystemExit("neograph_engine is not installed")
    module = next(Path(spec.origin).parent.glob("_neograph*.so"))
    needed = run(["readelf", "-d", module], capture_output=True, text=True).stdout
    if PROFILES[profile]["needed"] not in needed:
        raise SystemExit(f"{module.name} is not a {profile} build; install the wheel from 'native' first")


def command_key(args) -> int:
    """Cache key of a native build: changes only with the pins, flags, compiler and Python minor version."""
    print(hashlib.sha256(json.dumps(manifest(args.profile), sort_keys=True).encode()).hexdigest()[:32])
    return 0


def command_canary(args) -> int:
    out, spec = args.out.resolve(), PROFILES[args.profile]
    work = out / "canary"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "canary.c").write_text(CANARY, encoding="utf-8")
    (work / "driver.py").write_text(CANARY_DRIVER, encoding="utf-8")
    library = work / "canary.so"
    run(["gcc", "-shared", "-fPIC", *spec["flags"].split(), "-o", library, work / "canary.c"])
    failures = []
    # The ASan canary is built with ASan alone: UBSan's object-size check would otherwise report it first.
    asan_only = work / "canary-asan-only.so"
    if args.profile == "asan":
        run(["gcc", "-shared", "-fPIC", "-fsanitize=address", "-fno-omit-frame-pointer", "-g", "-o", asan_only,
             work / "canary.c"])
    cases = [("race", library, "WARNING: ThreadSanitizer: data race")] if args.profile == "tsan" else \
        [("overflow", asan_only, "ERROR: AddressSanitizer"), ("ubsan", library, "runtime error:"),
         ("leak", library, None)]
    for mode, target, marker in cases:
        logs = work / f"logs-{mode}"
        logs.mkdir()
        env, prefix = environment(args.profile, out, logs)
        result = run([*prefix, sys.executable, work / "driver.py", mode, target], env=env, check=False,
                     capture_output=True, text=True)
        text = result.stdout + result.stderr + "".join(path.read_text(errors="replace") for path in logs.glob("*"))
        if mode == "leak":
            owners = classify_leaks(logs, re.compile(r"canary"))
            if owners["project"] != 1:
                failures.append(f"leak canary: expected 1 project-owned direct leak, classified {dict(owners)}")
        elif marker not in text or result.returncode == 0:
            failures.append(f"{mode} canary: no '{marker}' report (exit {result.returncode})")
    for failure in failures:
        print("FAIL", failure)
    print(f"{args.profile} canaries:", "FAILED" if failures else "detected")
    return 1 if failures else 0


def command_test(args) -> int:
    out = args.out.resolve()
    require_instrumented(args.profile)
    vector = out / "vector.so"
    if not vector.is_file():
        raise SystemExit(f"{vector} is missing; run 'native' first")
    logs = out / "logs"
    shutil.rmtree(logs, ignore_errors=True)
    logs.mkdir(parents=True)
    env, prefix = environment(args.profile, out, logs)
    env.update(TSUKKOMI_TEST_DENSE_BACKEND="sqlite-vector", SQLITE_VECTOR_TEST_PATH=str(vector))
    leak_result = logs / "lsan-result.txt"
    if args.profile == "asan":
        env["TSUKKOMI_LSAN_RESULT"] = str(leak_result)
    # A sanitizer stops the process with _exit(), which drops block-buffered output: keep it unbuffered.
    env["PYTHONUNBUFFERED"] = "1"
    preflight = run([*prefix, sys.executable, "-c", "import neograph_engine, self_directing_mcp.engine; print('preflight ok')"],
                    env=env, check=False, cwd=ROOT, capture_output=True, text=True)
    print(preflight.stdout + preflight.stderr, end="")
    if preflight.returncode:
        dump_logs(logs)
        raise SystemExit(f"importing the sanitizer build failed (exit {preflight.returncode})")
    tests = list(args.pytest_args)
    if args.profile == "tsan":
        tests += [f"--deselect={test}" for test in TSAN_DESELECT]
    # Unbuffered output (below) keeps the progress written before a native exit, so a run that dies early still
    # shows how far it got. pytest keeps its default fd capture: the MCP stdio tests hand sys.stderr to a child.
    command = [*prefix, sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    print("+", " ".join(command), flush=True)
    # UBSan writes to stderr even when ASan logs to files, so the run's output is kept and scanned too.
    env = {**env, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace") as process, open(logs / "pytest.out", "w") as copy:
        for line in process.stdout:
            sys.stdout.write(line)
            copy.write(line)
    result = process
    counts = scan_reports(logs)
    problems = []
    if result.returncode:
        problems.append(f"pytest exited {result.returncode}")
    for key in ("asan_errors", "ubsan_errors", "tsan_warnings"):
        if counts[key]:
            problems.append(f"{counts[key]} {key.replace('_', ' ')} (logs in {logs})")
    if args.profile == "asan":
        if not leak_result.is_file():
            problems.append("the session-end leak check did not run")
        owners = classify_leaks(logs, re.compile(r"neograph|/sqlite-vector/|vector\.so"))
        print("direct leaks by owner:", dict(owners))
        if owners["project"]:
            problems.append(f"{owners['project']} direct leaks allocated by neograph or sqlite-vector code")
    for problem in problems:
        print("FAIL", problem)
    if problems:
        dump_logs(logs)
    print(f"{args.profile}:", "FAILED" if problems else "no sanitizer reports")
    return 1 if problems else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("key", "native", "canary", "test"):
        command = sub.add_parser(name)
        command.add_argument("--profile", choices=sorted(PROFILES), required=True)
        if name != "key":
            command.add_argument("--out", type=Path, required=True, help="build and log directory for this profile")
        if name == "native":
            command.add_argument("--ccache", action="store_true")
        if name == "test":
            command.add_argument("pytest_args", nargs="*", help="tests or pytest options after --")
    args = parser.parse_args(argv)
    if platform.system() != "Linux":
        raise SystemExit("sanitizer profiles need Linux and GCC")
    return {"key": command_key, "native": command_native, "canary": command_canary,
            "test": command_test}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
