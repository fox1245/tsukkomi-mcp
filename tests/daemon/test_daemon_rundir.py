"""The runtime directory: location, creation, and refusal of anything unsafe."""
from __future__ import annotations

import os
import stat

import pytest

from self_directing_mcp.daemon import rundir


def mode_of(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_xdg_runtime_dir_is_preferred_and_created_with_mode_0700(tmp_path):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    with rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg), "HOME": str(tmp_path / "home")}) as runtime:
        assert runtime.path == str(xdg / "tsukkomi")
        assert mode_of(runtime.path) == 0o700
    assert not (tmp_path / "home").exists()                                  # nothing under HOME was needed


def test_a_restrictive_or_permissive_umask_still_yields_exactly_0700(tmp_path):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    old = os.umask(0o000)
    try:
        with rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)}) as runtime:
            assert mode_of(runtime.path) == 0o700
    finally:
        os.umask(old)


def test_without_xdg_the_directory_is_under_the_user_cache(tmp_path):
    environ = {"HOME": str(tmp_path / "home")}
    with rundir.open_runtime_dir(environ) as runtime:
        assert runtime.path == str(tmp_path / "home" / ".cache" / "tsukkomi" / "run")
        assert mode_of(runtime.path) == 0o700


@pytest.mark.parametrize("kind", ["missing", "relative", "file", "world_writable", "foreign_owner"])
def test_an_unusable_xdg_runtime_dir_falls_back_to_the_cache_directory(tmp_path, monkeypatch, kind):
    xdg = tmp_path / "xdg"
    value = str(xdg)
    if kind == "relative":
        value = "relative/xdg"
    elif kind == "file":
        xdg.write_text("not a directory")
    elif kind == "world_writable":
        xdg.mkdir()
        os.chmod(xdg, 0o777)
    elif kind == "foreign_owner":
        xdg.mkdir(mode=0o700)
        os.chmod(xdg, 0o700)
        monkeypatch.setattr(rundir, "current_uid", lambda: os.geteuid() + 1)
    path = rundir.runtime_dir_path({"XDG_RUNTIME_DIR": value, "HOME": str(tmp_path / "h")})
    assert path == str(tmp_path / "h" / ".cache" / "tsukkomi" / "run")


def test_a_directory_with_the_wrong_mode_is_refused_with_a_clear_error(tmp_path):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    bad = xdg / "tsukkomi"
    bad.mkdir()
    for mode in (0o755, 0o750, 0o770, 0o777, 0o500, 0o701, 0o600):
        os.chmod(bad, mode)
        with pytest.raises(rundir.RuntimeDirError, match="must be exactly 0700"):
            rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)})
    assert mode_of(bad) == 0o600                                          # refused, not "repaired"


def test_a_symbolic_link_is_refused_even_when_it_points_at_a_perfect_directory(tmp_path):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    os.chmod(target, 0o700)
    os.symlink(target, xdg / "tsukkomi")
    with pytest.raises(rundir.RuntimeDirError, match="symbolic link"):
        rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)})
    assert list(target.iterdir()) == []                                  # nothing was created through the link


def test_a_dangling_symbolic_link_is_refused_and_not_followed(tmp_path):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    os.symlink(tmp_path / "does-not-exist", xdg / "tsukkomi")
    with pytest.raises(rundir.RuntimeDirError, match="symbolic link"):
        rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)})
    assert not (tmp_path / "does-not-exist").exists()


def test_a_directory_owned_by_someone_else_is_refused(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    with rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)}):
        pass
    # The owner check, simulated: the directory belongs to uid X while the process claims to be uid X + 1.
    monkeypatch.setattr(rundir, "current_uid", lambda: os.geteuid() + 1)
    # With the foreign uid the XDG directory itself is no longer "usable", so the fallback location is chosen;
    # pre-create that one to make the owner check the thing under test.
    home = tmp_path / "home"
    target = home / ".cache" / "tsukkomi" / "run"
    monkeypatch.setattr(rundir, "current_uid", os.geteuid)
    with rundir.open_runtime_dir({"HOME": str(home)}):
        pass
    monkeypatch.setattr(rundir, "current_uid", lambda: os.geteuid() + 1)
    with pytest.raises(rundir.RuntimeDirError, match="not owned by the current user"):
        rundir.open_runtime_dir({"HOME": str(home)})
    assert target.is_dir()


def test_a_path_that_is_a_regular_file_is_refused(tmp_path):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    (xdg / "tsukkomi").write_text("x")
    with pytest.raises(rundir.RuntimeDirError):
        rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)})


def test_create_false_never_creates(tmp_path):
    with pytest.raises(rundir.RuntimeDirError, match="does not exist"):
        rundir.open_runtime_dir({"HOME": str(tmp_path / "h")}, create=False)
    assert not (tmp_path / "h").exists()


def test_a_long_runtime_path_still_yields_a_usable_socket_address(tmp_path):
    import socket
    deep = tmp_path
    for part in ("a" * 40, "b" * 40, "c" * 40):
        deep = deep / part
    deep.mkdir(parents=True, mode=0o700)
    os.chmod(deep, 0o700)
    environ = {"XDG_RUNTIME_DIR": str(deep)}
    with rundir.open_runtime_dir(environ) as runtime:
        assert len(os.path.join(runtime.path, rundir.SOCKET_NAME).encode()) > 108
        address = runtime.address()
        assert address.startswith("/proc/self/fd/") and len(address.encode()) < 100
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(address)
        server.listen(1)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(address)
        conn, _ = server.accept()
        client.sendall(b"x")
        assert conn.recv(1) == b"x"
        for s in (client, conn, server):
            s.close()
        runtime.unlink(rundir.SOCKET_NAME)
        assert not runtime.exists(rundir.SOCKET_NAME)


def test_files_are_opened_relative_to_the_verified_directory_and_never_through_a_link(tmp_path):
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    with rundir.open_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)}) as runtime:
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        os.symlink(outside, os.path.join(runtime.path, "planted"))
        with pytest.raises(OSError):
            runtime.open_file("planted", os.O_RDONLY)
        fd = runtime.open_file("fresh", os.O_WRONLY | os.O_CREAT)
        os.close(fd)
        assert mode_of(os.path.join(runtime.path, "fresh")) == 0o600
