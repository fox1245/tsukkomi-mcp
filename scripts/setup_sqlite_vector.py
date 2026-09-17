"""Install the official sqliteai/sqlite-vector release library; no wheel or compiler."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sqlite3
import struct
import tempfile
import uuid
import zipfile

import httpx

VERSION = "1.1.0"
# SHA-256 digests published on the official GitHub release assets.
ASSETS = {
    ("Windows", "x86_64"): ("windows-x86_64", "dll", "ff648d50272df30b92b8ff67637c0d48197a4a11e85676091dd7664ebf9d406d"),
    ("Linux", "x86_64"): ("linux-x86_64", "so", "cc3c4ee21dfd90a218184c76bc355bd93d4d89e1a58ebc6c477996fa22ec68e0"),
    ("Linux", "arm64"): ("linux-arm64", "so", "f5948dc1131e643ec6cb6224ba7f4a39d297686d908bc7e1f3f050e846977275"),
    ("Darwin", "x86_64"): ("macos-x86_64", "dylib", "37acc65e770cecebbd6975d4af17375e785c9a62d1b0d280232af151cac94b7b"),
    ("Darwin", "arm64"): ("macos-arm64", "dylib", "a5342b225487ea14d4dd66fb78b4fd583c733501d56da669cc6985f71e89758f"),
}


def verify(path):
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(Path(path).resolve()))
        finally:
            conn.enable_load_extension(False)
        version, backend = conn.execute("SELECT vector_version(), vector_backend()").fetchone()
        if version != VERSION:
            raise RuntimeError(f"Expected sqlite-vector {VERSION}, found {version}")
        conn.execute("CREATE TABLE smoke (embedding BLOB)")
        conn.execute("INSERT INTO smoke VALUES (?)", (struct.pack("ff", 1, 0),))
        conn.execute("SELECT vector_init('smoke', 'embedding', 'type=FLOAT32,dimension=2,distance=COSINE')")
        row = conn.execute("SELECT rowid,distance FROM vector_full_scan('smoke','embedding',?)",
                           (struct.pack("ff", 1, 0),)).fetchone()
        if row is None or row[0] != 1 or abs(row[1]) > 1e-6:
            raise RuntimeError("Native cosine search smoke test failed")
        return {"version": version, "compute_backend": backend}


def install(output_dir, archive=None):
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    target = ASSETS.get((platform.system(), machine))
    if target is None or struct.calcsize("P") != 8:
        raise RuntimeError("No pinned release for this Python architecture; select a matching official native library")
    if platform.system() == "Linux" and platform.libc_ver()[0] != "glibc":
        raise RuntimeError("This installer pins glibc Linux assets; choose the matching official musl library manually")
    name, suffix, expected = target
    url = f"https://github.com/sqliteai/sqlite-vector/releases/download/{VERSION}/vector-{name}-{VERSION}.zip"
    if archive is None:
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        data = response.content
    else:
        data = Path(archive).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("Official sqlite-vector archive checksum mismatch")
    library_name = f"vector.{suffix}"
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        entries = [entry for entry in zipped.infolist()
                   if not entry.is_dir() and Path(entry.filename).name == library_name]
        if len(entries) != 1:
            raise RuntimeError("Release must contain exactly one matching vector library")
        library = zipped.read(entries[0])  # Never extract arbitrary archive paths.
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / library_name
    digest = hashlib.sha256(library).hexdigest()
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
        result = verify(destination)
    else:
        # Validate before replacing an existing runtime. The library must retain
        # its vector basename so SQLite resolves sqlite3_vector_init correctly.
        with tempfile.TemporaryDirectory(prefix=".sqlite-vector-", dir=output_dir) as temporary:
            candidate = Path(temporary) / library_name
            candidate.write_bytes(library)
            result = verify(candidate)
            # A TemporaryDirectory can grant access only to its creator on
            # Windows. Renaming its DLL would preserve that restrictive ACL.
            # Create the final staging file in the destination directory so
            # it inherits permissions for the account that runs the MCP.
            staging = output_dir / ("." + library_name + "." + uuid.uuid4().hex + ".tmp")
            try:
                with staging.open("xb") as stream:
                    stream.write(library)
                os.replace(staging, destination)
            finally:
                staging.unlink(missing_ok=True)
    result.update(path=str(destination), archive_sha256=expected, library_sha256=digest, url=url)
    (output_dir / "sqlite-vector-installation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".native"))
    parser.add_argument("--archive", type=Path, help="Verify/install a previously downloaded official ZIP without network access")
    args = parser.parse_args()
    print(json.dumps(install(args.output_dir, args.archive), indent=2))


if __name__ == "__main__":
    main()
