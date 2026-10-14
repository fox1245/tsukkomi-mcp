"""Owner approval and local execution CLI for evidence-bound workflows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="tsukkomi-workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    approve = commands.add_parser("approve", help="Owner-only: freeze a reviewed manifest and original sources")
    approve.add_argument("--root", type=Path, required=True)
    approve.add_argument("--manifest", type=Path, required=True)
    approve.add_argument("--output", type=Path, required=True)
    for name in ("status", "wiki", "run-checks", "classify", "transition", "authorize", "execute"):
        sub = commands.add_parser(name)
        sub.add_argument("--approval", type=Path, default=os.environ.get("TSUKKOMI_WORKFLOW_APPROVAL"))
        if name == "wiki":
            sub.add_argument("--output", type=Path, required=True)
            continue
        sub.add_argument("--state", type=Path, default=os.environ.get("TSUKKOMI_WORKFLOW_STATE"))
        sub.add_argument("--session", required=True)
        sub.add_argument("--provider", choices=("agy", "codex", "omp", "grokbot"), default="agy")
        sub.add_argument("--lean", default=os.environ.get("TSUKKOMI_LEAN"))
        if name == "run-checks":
            sub.add_argument("--kind", choices=("all", "proof", "test"), default="all")
        if name in ("transition", "authorize"):
            sub.add_argument("--classification", required=True)
        if name in ("authorize", "execute"):
            sub.add_argument("--action", type=Path, required=True, help="JSON workflow_command with argv and cwd='.'")
        if name == "execute":
            sub.add_argument("--grant", required=True, help="Single-use grant from authorize")
    args = parser.parse_args()
    try:
        from self_directing_mcp.workflow_requirements import ApprovedProject, approve_project
        if args.command == "approve":
            project = approve_project(args.root, args.manifest, args.output)
            result = {"ok": True, "approval_digest": project.approval_digest}
        elif not args.approval:
            parser.error("--approval or TSUKKOMI_WORKFLOW_APPROVAL is required")
        elif args.command == "wiki":
            pages = ApprovedProject.load(args.approval).render_wiki(args.output)
            result = {"ok": True, "pages": [p.name for p in pages], "derived": True}
        else:
            if not args.state:
                parser.error("--state or TSUKKOMI_WORKFLOW_STATE is required")
            from self_directing_mcp.workflow import WorkflowController
            controller = WorkflowController(args.approval, args.state, lean=args.lean)
            common = {"session_id": args.session, "provider": args.provider}
            if args.command == "run-checks":
                result = controller.run_checks(**common, kind=args.kind)
            elif args.command in ("status", "classify"):
                result = getattr(controller, args.command)(**common)
            elif args.command == "transition":
                result = controller.transition(**common, classification_id=args.classification)
            else:
                action = json.loads(args.action.read_text(encoding="utf-8"))
                if args.command == "authorize":
                    result = controller.authorize(**common, classification_id=args.classification, action=action)
                else:
                    result = controller.execute(**common, grant=args.grant, action=action)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        if not result.get("ok"):
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        from self_directing_mcp.security.mask import mask_secrets
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": mask_secrets(str(exc))}), file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": "cancelled"}), file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
