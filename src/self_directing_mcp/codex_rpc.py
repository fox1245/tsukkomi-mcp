"""Minimal local Codex app-server client. No model turns are started."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


class CodexRPC:
    def __init__(self, executable: str, codex_home: Path, cwd: Path):
        self.executable, self.home, self.cwd = executable, Path(codex_home), Path(cwd)
        self._messages = queue.Queue()
        self._next_id = 0

    def __enter__(self):
        env = dict(os.environ, CODEX_HOME=str(self.home))
        self._process = subprocess.Popen(
            [self.executable, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env, cwd=self.cwd,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        def reader():
            for line in self._process.stdout:
                try:
                    self._messages.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
            self._messages.put(None)
        threading.Thread(target=reader, daemon=True).start()
        try:
            self.request("initialize", {
                "clientInfo": {"name": "self-directing-installer", "version": "0.3.0"},
                "capabilities": {"experimentalApi": True}})
            self._send({"method": "initialized", "params": {}})
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _send(self, value):
        self._process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def request(self, method, params, timeout=45):
        self._next_id += 1
        call_id = self._next_id
        self._send({"id": call_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self._messages.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError(f"Codex {method} timed out") from None
            if message is None:
                raise RuntimeError(f"Codex app-server ended while calling {method}")
            if message.get("id") == call_id and ("result" in message or "error" in message):
                if "error" in message:
                    # Error messages may contain config values; do not echo them.
                    raise RuntimeError(f"Codex {method} failed (code {message['error'].get('code')})")
                return message["result"]
            if "id" in message and "method" in message:
                self._send({"id": message["id"], "error": {"code": -32601, "message": "Installer does not grant runtime approvals"}})
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Codex {method} timed out")

    def __exit__(self, *args):
        process = getattr(self, "_process", None)
        if not process:
            return
        try:
            process.stdin.close()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            process.stdout.close()
