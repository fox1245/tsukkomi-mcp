"""NumPy, loaded on first use and with one BLAS thread unless the user chose otherwise.

tsukkomi does no large linear algebra, but a default OpenBLAS build starts one spinning thread per core when NumPy
loads: about a CPU second per hook process and idle threads in every server. The thread variables are read only while
the library loads, so they are restored right afterwards and child processes (workflow checks, user tests) keep the
caller's environment. Modules import ``np`` from here instead of importing NumPy directly; this is the only place that
imports it.

``np`` is a lazy proxy: importing this module costs nothing, and NumPy loads at the first attribute access
(``np.asarray``, ``np.float32``, ...). Code that never touches a dense vector, such as the local-only command hooks,
therefore never loads it (about 40 ms of import time and 25 MB of memory). Annotations such as ``np.ndarray`` are
strings under ``from __future__ import annotations`` and do not load it either. A long-lived server that will use
dense vectors can call ``load_numpy()`` at startup so the environment juggling happens while it is single-threaded.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import TYPE_CHECKING

_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS")
_load_lock = threading.Lock()


def _import_numpy():
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


class _LazyNumpy:
    """Stands in for the ``numpy`` module; the first attribute access imports it (once, under a lock)."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):    # copy/pickle/introspection probes must not import it
            raise AttributeError(name)
        module = load_numpy()
        value = getattr(module, name)
        object.__setattr__(self, name, value)               # later accesses skip __getattr__ entirely
        return value

    def __dir__(self):
        return dir(load_numpy())

    def __repr__(self) -> str:
        return "<lazy numpy: loaded>" if "numpy" in sys.modules else "<lazy numpy: not loaded yet>"


def load_numpy():
    """The real ``numpy`` module, imported now if it was not already (thread-safe, one BLAS thread by default)."""
    with _load_lock:
        return _import_numpy()


def numpy_loaded() -> bool:
    return "numpy" in sys.modules


if TYPE_CHECKING:
    import numpy as np
else:
    np = _LazyNumpy()
