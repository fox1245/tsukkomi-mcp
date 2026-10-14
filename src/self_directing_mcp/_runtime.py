"""NumPy, loaded with one BLAS thread unless the user chose otherwise.

tsukkomi does no large linear algebra, but a default OpenBLAS build starts one spinning thread per core when NumPy
loads: about a CPU second per hook process and idle threads in every server. The thread variables are read only while
the library loads, so they are restored right afterwards and child processes (workflow checks, user tests) keep the
caller's environment. Modules import ``np`` from here instead of importing NumPy directly.
"""
from __future__ import annotations

import os
import sys

_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS")


def _load_numpy():
    if "numpy" in sys.modules:          # already loaded by the host process; its thread setup stands
        import numpy
        return numpy
    saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    for name in _THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    try:
        import numpy
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return numpy


np = _load_numpy()
