"""Maintenance commands that run outside a request.

    python -m self_directing_mcp.maintenance sparse-rebuild [--budget-sec N] [--batch N] [--index-dir DIR]

``sparse-rebuild`` materializes the sparse (FTS5) search index that is missing: it converts a whole index to the compact
format (``SELF_DIRECT_SPARSE_FORMAT``, README "Sparse index formats") without waiting for searches and syncs to do it
session by session, and it repairs a compact index that lost documents. The canonical store (``meta.sqlite``) is the only
source. The legacy ``sparse.sqlite`` is neither read nor written. The command is bounded (``--budget-sec``), resumable and
safe to interrupt: every batch is its own transaction and only missing documents are written, so running it again
continues where it stopped, and on a finished index it changes nothing. Progress goes to stderr; stdout gets one JSON line.

Exit status: 0 complete, 3 partial (time budget or busy index: run it again), 2 nothing to do or failed, 130 interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

EXIT_COMPLETE, EXIT_FAILED, EXIT_PARTIAL, EXIT_INTERRUPTED = 0, 2, 3, 130


def _progress_printer(stream, every: float = 1.0):
    started = time.monotonic()
    last = [0.0]

    def show(done: int, total: int, written: int, *, final: bool = False) -> None:
        now = time.monotonic()
        if not final and now - last[0] < every:
            return
        last[0] = now
        rate = written / max(now - started, 1e-9)
        percent = 100.0 * done / total if total else 100.0
        print(f"sparse-rebuild: {done}/{total} events ({percent:.1f} %), {written} written this run, {rate:.0f} events/s",
              file=stream, flush=True)

    return show


def sparse_rebuild(args, stderr=None, stdout=None) -> int:
    stderr = stderr if stderr is not None else sys.stderr
    stdout = stdout if stdout is not None else sys.stdout
    from self_directing_mcp.config import get_settings
    from self_directing_mcp.engine import SelfDirectEngine
    from self_directing_mcp.request_control import IndexBusy
    from self_directing_mcp.index.sparse import SparseFormatError, compact_support

    overrides = {"local_only": True}      # no embedder and no vector index are needed (or created) for a text index
    if args.index_dir:
        overrides["index_dir"] = args.index_dir
    settings = get_settings(**overrides)
    if args.format:
        settings.sparse_format = args.format
    # Refuse before an engine opens (and so creates) a legacy file that nothing here would use.
    if settings.sparse_format == "legacy":
        print("sparse-rebuild: SELF_DIRECT_SPARSE_FORMAT is legacy; there is nothing to convert (use compact or auto)", file=stderr)
        return EXIT_FAILED
    reason = compact_support()
    if reason is not None:
        print(f"sparse-rebuild: {reason}", file=stderr)
        return EXIT_FAILED
    engine = SelfDirectEngine(settings)
    show = _progress_printer(stderr)
    try:
        try:
            info = engine.rebuild_sparse(budget_sec=args.budget_sec, batch=args.batch, progress=show)
        except SparseFormatError as exc:
            print(f"sparse-rebuild: {exc}", file=stderr)
            return EXIT_FAILED
        except IndexBusy:
            print("sparse-rebuild: another process holds the index; run the command again", file=stderr)
            return EXIT_PARTIAL
        except KeyboardInterrupt:
            print("sparse-rebuild: interrupted; every finished batch is kept, run the command again to continue", file=stderr)
            return EXIT_INTERRUPTED
    finally:
        engine.close()
    if info.get("canonical") is not None:
        show(info["canonical"] - info["remaining"], info["canonical"], info["written"], final=True)
    print(json.dumps(info, sort_keys=True), file=stdout)
    if not info["ok"]:
        print(f"sparse-rebuild: {info.get('reason') or info.get('error')}", file=stderr)
        return EXIT_FAILED
    if info["state"] != "complete":
        print(f"sparse-rebuild: stopped by {info['stopped_by']} with {info['remaining']} events left; run the command again to continue",
              file=stderr)
        return EXIT_PARTIAL
    if info.get("statistics") and not info["statistics"]["ok"]:
        print(f"sparse-rebuild: the FTS5 statistics do not match the documents: {info['statistics']}", file=stderr)
        return EXIT_FAILED
    return EXIT_COMPLETE


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m self_directing_mcp.maintenance", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rebuild = commands.add_parser("sparse-rebuild", help="materialize the missing documents of the sparse search index")
    rebuild.add_argument("--budget-sec", type=float, default=None, help="stop after this many seconds (default: run to the end)")
    rebuild.add_argument("--batch", type=int, default=None, help="documents per transaction (default 500)")
    rebuild.add_argument("--index-dir", default=None, help="the index directory (default: SELF_DIRECT_INDEX_DIR or the user's default)")
    rebuild.add_argument("--format", choices=("auto", "compact", "legacy"), default=None,
                         help="override SELF_DIRECT_SPARSE_FORMAT for this run")
    args = parser.parse_args(argv)
    if args.command == "sparse-rebuild":
        if args.budget_sec is not None and args.budget_sec < 0:
            parser.error("--budget-sec must not be negative")
        if args.batch is not None and args.batch < 1:
            parser.error("--batch must be at least 1")
        return sparse_rebuild(args)
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
