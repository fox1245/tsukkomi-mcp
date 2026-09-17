import os
from pathlib import Path, PureWindowsPath

import pytest

from self_directing_mcp.codex.discover import ensure_under_root, _windows_comparison_path, PathTraversalError


@pytest.mark.parametrize("value", [r"C:\sessions\day\file.jsonl", r"\\?\C:\sessions\day\file.jsonl"])
def test_dos_spellings_share_comparison_namespace(value):
    assert _windows_comparison_path(PureWindowsPath(value)) == PureWindowsPath(r"\\?\C:\sessions\day\file.jsonl")


@pytest.mark.parametrize("value", [r"\\server\share\sessions", r"\\?\UNC\server\share\sessions"])
def test_unc_spellings_share_comparison_namespace_without_network(value):
    assert _windows_comparison_path(PureWindowsPath(value)) == PureWindowsPath(r"\\?\UNC\server\share\sessions")


@pytest.mark.parametrize("value", [r"\\.\PhysicalDrive0", r"\\?\GLOBALROOT\Device\HarddiskVolume1", r"C:relative"])
def test_device_or_drive_relative_paths_are_not_session_locations(value):
    with pytest.raises(PathTraversalError):
        _windows_comparison_path(PureWindowsPath(value))


@pytest.mark.skipif(os.name != "nt", reason="native Windows extended paths")
def test_real_file_accepts_all_root_and_file_spellings(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "rollout.jsonl"
    path.write_text('{}\n')
    extended = lambda p: Path('\\\\?\\' + str(p.resolve()))
    for r in (root, extended(root)):
        for p in (path, extended(path)):
            resolved = ensure_under_root(p, r)
            assert resolved.samefile(path)
    outside = tmp_path / "sessions-other" / "rollout.jsonl"
    outside.parent.mkdir()
    outside.write_text('{}\n')
    for p in (outside, extended(outside), root / '..' / 'sessions-other' / 'rollout.jsonl'):
        with pytest.raises(PathTraversalError):
            ensure_under_root(p, root)


def test_hook_exception_with_non_requirement_prompt_stays_advisory():
    from self_directing_mcp.codex_hooks import handle_hook
    class Broken:
        def hook_obligations(self, *args, **kwargs):
            raise OSError('unavailable')
    result = handle_hook(Broken(), 'UserPromptSubmit', 'session', tool_input='hello')
    assert 'unknown (OSError)' in result['hookSpecificOutput']['additionalContext']
