"""Resident daemon (experimental, Linux only, off by default): one process per user and OS environment.

Command hooks and MCP entry points become thin clients of that process when ``SELF_DIRECT_DAEMON=on``. The
daemon calls the same ``run_hook`` functions and MCP tool code as the standalone processes: there is one judgement
code path. See README, "Resident daemon".

This package's ``protocol``, ``rundir``, ``client`` and ``relay`` modules import only the standard library; the
client side must never load the engine, pydantic, NumPy, NeoGraph or the MCP SDK. ``core`` and ``envview`` are the
daemon's own side and load the full runtime.
"""
