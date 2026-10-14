"""NumPy loads with one BLAS thread (a default OpenBLAS starts one spinning thread per core at import)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import os, sys
import self_directing_mcp.engine
from self_directing_mcp._runtime import np
np.zeros(1)    # NumPy loads on first use (the engine import alone does not load it)
print(len(os.listdir('/proc/self/task')), os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('OMP_NUM_THREADS'))
"""

pytestmark = pytest.mark.skipif(not Path("/proc/self/task").is_dir() or (os.cpu_count() or 1) < 4,
                                reason="counts native threads through /proc on a multi-core host")


def _probe(**extra):
    env = {key: value for key, value in os.environ.items()
           if key not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env.update(PYTHONPATH=str(ROOT / "src"), **extra)
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=120, check=True)
    threads, openblas, omp = out.stdout.split()
    return int(threads), openblas, omp


def test_engine_import_starts_no_blas_thread_pool_and_restores_the_environment():
    threads, openblas, omp = _probe()
    assert threads <= 3, threads                       # the main thread, not one per core
    assert (openblas, omp) == ("None", "None")         # children (workflow checks) inherit the caller's environment


def test_an_explicit_user_setting_is_respected():
    threads, openblas, _omp = _probe(OPENBLAS_NUM_THREADS="4")
    assert openblas == "4" and threads >= 4
