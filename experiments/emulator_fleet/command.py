"""Safe command execution for vendor emulator CLIs.

Commands are always passed as argv arrays with ``shell=False``.  Mutation
planning is kept separate from execution, and every process call yields a
receipt even when it times out or the executable is missing.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from experiments.emulator_fleet.schema import json_bytes, sha256_json


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    argv: tuple[str, ...]
    returncode: Optional[int]
    stdout: bytes
    stderr: bytes
    started_ms: float
    completed_ms: float
    timed_out: bool = False
    missing_executable: bool = False
    planned_only: bool = False
    cwd: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.completed_ms - self.started_ms)

    @property
    def ok(self) -> bool:
        return bool(
            not self.planned_only
            and not self.timed_out
            and not self.missing_executable
            and self.returncode == 0
        )

    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout_sha256": __import__("hashlib").sha256(self.stdout).hexdigest(),
            "stderr_sha256": __import__("hashlib").sha256(self.stderr).hexdigest(),
            "stdout_text": self.stdout_text()[:2000],
            "stderr_text": self.stderr_text()[:2000],
            "started_ms": self.started_ms,
            "completed_ms": self.completed_ms,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "missing_executable": self.missing_executable,
            "planned_only": self.planned_only,
            "cwd": self.cwd,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float = 30.0,
        cwd: Optional[str | Path] = None,
        env: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """Run one attached process and kill its process tree on timeout."""

    def __init__(self, *, now_ms=None) -> None:
        self.now_ms = now_ms or (lambda: time.monotonic() * 1000.0)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float = 30.0,
        cwd: Optional[str | Path] = None,
        env: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CommandResult:
        args = tuple(str(v) for v in argv)
        if not args:
            raise ValueError("argv cannot be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        started = self.now_ms()
        command_id = "command_" + sha256_json(
            {"argv": list(args), "cwd": str(cwd) if cwd is not None else None, "started_ms": round(started, 3)}
        )
        creationflags = 0
        start_new_session = False
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            start_new_session = True
        try:
            proc = subprocess.Popen(
                args,
                cwd=str(cwd) if cwd is not None else None,
                env=(dict(os.environ) | dict(env or {})),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
        except FileNotFoundError as exc:
            completed = self.now_ms()
            return CommandResult(
                command_id=command_id,
                argv=args,
                returncode=None,
                stdout=b"",
                stderr=str(exc).encode("utf-8", errors="replace"),
                started_ms=started,
                completed_ms=completed,
                missing_executable=True,
                cwd=str(cwd) if cwd is not None else None,
                metadata=dict(metadata or {}),
            )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            completed = self.now_ms()
            return CommandResult(
                command_id=command_id,
                argv=args,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                started_ms=started,
                completed_ms=completed,
                cwd=str(cwd) if cwd is not None else None,
                metadata=dict(metadata or {}),
            )
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            stdout, stderr = proc.communicate()
            completed = self.now_ms()
            return CommandResult(
                command_id=command_id,
                argv=args,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                started_ms=started,
                completed_ms=completed,
                timed_out=True,
                cwd=str(cwd) if cwd is not None else None,
                metadata=dict(metadata or {}),
            )

    @staticmethod
    def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()


class RecordingCommandRunner:
    """Deterministic runner for command-building and parser tests."""

    def __init__(self, responses: Optional[Mapping[tuple[str, ...], tuple[int, bytes, bytes]]] = None) -> None:
        self.responses = dict(responses or {})
        self.calls: list[tuple[str, ...]] = []
        self.clock_ms = 0.0

    def add_response(
        self,
        argv: Sequence[str],
        *,
        returncode: int = 0,
        stdout: bytes | str = b"",
        stderr: bytes | str = b"",
    ) -> None:
        self.responses[tuple(str(v) for v in argv)] = (
            int(returncode),
            stdout.encode() if isinstance(stdout, str) else bytes(stdout),
            stderr.encode() if isinstance(stderr, str) else bytes(stderr),
        )

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float = 30.0,
        cwd: Optional[str | Path] = None,
        env: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CommandResult:
        del timeout_s, env
        args = tuple(str(v) for v in argv)
        self.calls.append(args)
        started = self.clock_ms
        self.clock_ms += 5.0
        returncode, stdout, stderr = self.responses.get(args, (0, b"", b""))
        return CommandResult(
            command_id="command_" + sha256_json({"argv": list(args), "call": len(self.calls)}),
            argv=args,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            started_ms=started,
            completed_ms=self.clock_ms,
            cwd=str(cwd) if cwd is not None else None,
            metadata=dict(metadata or {}),
        )


def planned_result(argv: Sequence[str], *, metadata: Optional[Mapping[str, Any]] = None) -> CommandResult:
    args = tuple(str(v) for v in argv)
    return CommandResult(
        command_id="planned_" + sha256_json({"argv": list(args), "metadata": dict(metadata or {})}),
        argv=args,
        returncode=None,
        stdout=b"",
        stderr=b"",
        started_ms=0.0,
        completed_ms=0.0,
        planned_only=True,
        metadata=dict(metadata or {}),
    )
