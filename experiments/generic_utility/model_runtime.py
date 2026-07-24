"""Run-scoped local-model execution with hard deadlines and receipts.

The experiment harness never requires a persistent model daemon.  A provider may
be a deterministic emulator, a subprocess that reads/writes JSON, or a future
in-process grounder.  Timeouts terminate the process tree and return an explicit
receipt before any motor call is permitted.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

import psutil

from experiments.generic_utility.schema import MetricKind, json_bytes, sha256_json


class ModelRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    kind: str
    payload: Mapping[str, Any]
    timeout_ms: float = 12000.0

    @classmethod
    def create(
        cls,
        kind: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        timeout_ms: float = 12000.0,
    ) -> "ModelRequest":
        payload = dict(payload or {})
        return cls(
            request_id="model_request_" + sha256_json(
                {"kind": kind, "payload": payload, "timeout_ms": timeout_ms}
            ),
            kind=str(kind),
            payload=payload,
            timeout_ms=float(timeout_ms),
        )


@dataclass(frozen=True)
class ModelReceipt:
    request_id: str
    provider: str
    kind: str
    status: str
    started_ms: float
    completed_ms: float
    load_ms: float
    inference_ms: float
    peak_vram_mb: float
    quantization: Optional[str]
    input_resolution: Optional[str]
    metric_kind: str
    result: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None
    process_terminated: bool = False
    motor_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["result"] = dict(self.result) if self.result is not None else None
        return value


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def run(self, request: ModelRequest) -> ModelReceipt: ...


class EmulatedModelProvider:
    """Deterministic provider for campaign plumbing and cost ablations.

    The receipt is explicitly ``metric_kind=simulated``.  It may act as an oracle
    for cold-path planning in the emulator, but cannot support a real-model claim.
    """

    def __init__(
        self,
        name: str,
        *,
        load_ms: float,
        inference_ms: float,
        peak_vram_mb: float,
        responder: Optional[Callable[[ModelRequest], Mapping[str, Any]]] = None,
        fail: bool = False,
        timeout: bool = False,
    ) -> None:
        self.name = name
        self.load_ms = float(load_ms)
        self.inference_ms = float(inference_ms)
        self.peak_vram_mb = float(peak_vram_mb)
        self.responder = responder or (lambda request: {"ok": True, "kind": request.kind})
        self.fail = bool(fail)
        self.timeout = bool(timeout)
        self.calls = 0
        self.loaded_kinds: set[str] = set()

    def run(self, request: ModelRequest) -> ModelReceipt:
        self.calls += 1
        started = time.monotonic() * 1000.0
        load = 0.0 if request.kind in self.loaded_kinds else self.load_ms
        if not self.timeout:
            self.loaded_kinds.add(request.kind)
        if self.timeout:
            completed = started + request.timeout_ms
            return ModelReceipt(
                request.request_id,
                self.name,
                request.kind,
                "timeout",
                started,
                completed,
                load,
                request.timeout_ms,
                self.peak_vram_mb,
                None,
                request.payload.get("input_resolution"),
                MetricKind.SIMULATED.value,
                error="emulated hard timeout",
                process_terminated=True,
            )
        if self.fail:
            completed = started + load + self.inference_ms
            return ModelReceipt(
                request.request_id,
                self.name,
                request.kind,
                "error",
                started,
                completed,
                load,
                self.inference_ms,
                self.peak_vram_mb,
                None,
                request.payload.get("input_resolution"),
                MetricKind.SIMULATED.value,
                error="emulated provider failure",
            )
        result = dict(self.responder(request))
        completed = started + load + self.inference_ms
        return ModelReceipt(
            request.request_id,
            self.name,
            request.kind,
            "done",
            started,
            completed,
            load,
            self.inference_ms,
            self.peak_vram_mb,
            request.payload.get("quantization"),
            request.payload.get("input_resolution"),
            MetricKind.SIMULATED.value,
            result=result,
        )


class _NvidiaSampler:
    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_mb

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                values = [float(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()]
                if values:
                    self.peak_mb = max(self.peak_mb, sum(values))
            except Exception:
                return


def _terminate_tree(process: subprocess.Popen[Any]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            parent.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs([*children, parent], timeout=1.5)
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        try:
            process.kill()
        except Exception:
            pass


class JsonSubprocessModelProvider:
    """Execute a one-shot JSON model command under a hard deadline.

    The command receives ``--input <json> --output <json>`` by default.  Use
    ``{input}`` and ``{output}`` placeholders in the configured command to place
    the paths elsewhere.  No listener or background service is started.
    """

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        quantization: Optional[str] = None,
        cwd: Optional[str | Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not command:
            raise ValueError("model command cannot be empty")
        self.name = name
        self.command = tuple(str(v) for v in command)
        self.quantization = quantization
        self.cwd = str(cwd) if cwd is not None else None
        self.env = dict(env or {})

    def run(self, request: ModelRequest) -> ModelReceipt:
        started = time.monotonic() * 1000.0
        with tempfile.TemporaryDirectory(prefix="screenghost-model-") as temp:
            input_path = Path(temp) / "request.json"
            output_path = Path(temp) / "response.json"
            input_path.write_bytes(
                json_bytes(
                    {
                        "request_id": request.request_id,
                        "kind": request.kind,
                        "payload": dict(request.payload),
                    }
                )
            )
            has_placeholders = any("{input}" in arg or "{output}" in arg for arg in self.command)
            command = [
                arg.replace("{input}", str(input_path)).replace("{output}", str(output_path))
                for arg in self.command
            ]
            if not has_placeholders:
                command.extend(["--input", str(input_path), "--output", str(output_path)])
            sampler = _NvidiaSampler()
            sampler.start()
            process = subprocess.Popen(
                command,
                cwd=self.cwd,
                env={**os.environ, **self.env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name != "nt"),
            )
            terminated = False
            try:
                stdout, stderr = process.communicate(timeout=request.timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                terminated = True
                _terminate_tree(process)
                stdout, stderr = process.communicate(timeout=2)
                completed = time.monotonic() * 1000.0
                peak = sampler.stop()
                return ModelReceipt(
                    request.request_id,
                    self.name,
                    request.kind,
                    "timeout",
                    started,
                    completed,
                    0.0,
                    completed - started,
                    peak,
                    self.quantization,
                    request.payload.get("input_resolution"),
                    MetricKind.MEASURED.value,
                    error=f"model deadline exceeded; stderr={stderr[-500:]!r}",
                    process_terminated=True,
                )
            completed = time.monotonic() * 1000.0
            peak = sampler.stop()
            if process.returncode != 0:
                return ModelReceipt(
                    request.request_id,
                    self.name,
                    request.kind,
                    "error",
                    started,
                    completed,
                    0.0,
                    completed - started,
                    peak,
                    self.quantization,
                    request.payload.get("input_resolution"),
                    MetricKind.MEASURED.value,
                    error=f"exit={process.returncode}; stderr={stderr[-1000:]!r}; stdout={stdout[-500:]!r}",
                )
            if not output_path.exists():
                return ModelReceipt(
                    request.request_id,
                    self.name,
                    request.kind,
                    "error",
                    started,
                    completed,
                    0.0,
                    completed - started,
                    peak,
                    self.quantization,
                    request.payload.get("input_resolution"),
                    MetricKind.MEASURED.value,
                    error="model process exited successfully but wrote no response file",
                )
            try:
                result = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return ModelReceipt(
                    request.request_id,
                    self.name,
                    request.kind,
                    "error",
                    started,
                    completed,
                    0.0,
                    completed - started,
                    peak,
                    self.quantization,
                    request.payload.get("input_resolution"),
                    MetricKind.MEASURED.value,
                    error=f"invalid JSON response: {exc}",
                )
            return ModelReceipt(
                request.request_id,
                self.name,
                request.kind,
                "done",
                started,
                completed,
                float(result.get("load_ms", 0.0)),
                float(result.get("inference_ms", completed - started)),
                peak,
                self.quantization,
                request.payload.get("input_resolution"),
                MetricKind.MEASURED.value,
                result=result,
                process_terminated=terminated,
            )

class AttachedJsonModelProvider:
    """Run-scoped JSONL worker with one model resident and no listener.

    The child is attached through stdin/stdout, emits one ``ready`` record after
    loading, processes request records serially, and is terminated when the
    provider closes or a deadline expires.  It never binds a port and cannot
    outlive a well-formed campaign.
    """

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        startup_timeout_ms: float = 120000.0,
        quantization: Optional[str] = None,
        cwd: Optional[str | Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        import queue

        if not command:
            raise ValueError("attached model command cannot be empty")
        self.name = str(name)
        self.command = tuple(str(v) for v in command)
        self.startup_timeout_ms = float(startup_timeout_ms)
        self.quantization = quantization
        self.cwd = str(cwd) if cwd is not None else None
        self.env = dict(env or {})
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stderr_tail: list[str] = []
        self._process: Optional[subprocess.Popen[str]] = None
        self._reader: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None
        self._ready: Optional[dict[str, Any]] = None
        self._started_ms: Optional[float] = None
        self._closed = False
        self._start()

    @property
    def process_id(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def _start(self) -> None:
        import queue

        self._started_ms = time.monotonic() * 1000.0
        self._process = subprocess.Popen(
            list(self.command),
            cwd=self.cwd,
            env={**os.environ, **self.env},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=(os.name != "nt"),
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None

        def read_stdout() -> None:
            assert self._process is not None and self._process.stdout is not None
            for line in self._process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._queue.put(json.loads(line))
                except json.JSONDecodeError:
                    self._queue.put({"type": "protocol_error", "raw": line[:2000]})

        def read_stderr() -> None:
            assert self._process is not None and self._process.stderr is not None
            for line in self._process.stderr:
                self._stderr_tail.append(line.rstrip())
                del self._stderr_tail[:-80]

        self._reader = threading.Thread(target=read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()
        deadline = time.monotonic() + self.startup_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise ModelRuntimeError(
                    f"attached model exited during startup: {self._process.returncode}; "
                    f"stderr={self._stderr_text()!r}"
                )
            try:
                message = self._queue.get(timeout=min(0.1, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if message.get("type") == "ready":
                self._ready = message
                return
            if message.get("type") == "error":
                raise ModelRuntimeError(f"attached model startup failed: {message.get('error')}")
        self.close(force=True)
        raise ModelRuntimeError("attached model did not become ready before startup deadline")

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail)[-4000:]

    def run(self, request: ModelRequest) -> ModelReceipt:
        import queue

        if self._closed or self._process is None or self._process.poll() is not None:
            raise ModelRuntimeError("attached model provider is closed")
        assert self._process.stdin is not None
        started = time.monotonic() * 1000.0
        sampler = _NvidiaSampler()
        sampler.start()
        payload = {
            "type": "request",
            "request_id": request.request_id,
            "kind": request.kind,
            "payload": dict(request.payload),
        }
        try:
            self._process.stdin.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            peak = sampler.stop()
            completed = time.monotonic() * 1000.0
            return ModelReceipt(
                request.request_id,
                self.name,
                request.kind,
                "error",
                started,
                completed,
                0.0,
                completed - started,
                peak,
                self.quantization,
                request.payload.get("input_resolution"),
                MetricKind.MEASURED.value,
                error=f"worker pipe failed: {exc}; stderr={self._stderr_text()!r}",
            )
        deadline = time.monotonic() + request.timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                peak = sampler.stop()
                completed = time.monotonic() * 1000.0
                return ModelReceipt(
                    request.request_id,
                    self.name,
                    request.kind,
                    "error",
                    started,
                    completed,
                    0.0,
                    completed - started,
                    peak,
                    self.quantization,
                    request.payload.get("input_resolution"),
                    MetricKind.MEASURED.value,
                    error=f"worker exited={self._process.returncode}; stderr={self._stderr_text()!r}",
                )
            try:
                message = self._queue.get(timeout=min(0.1, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if message.get("request_id") != request.request_id:
                # The provider is intentionally serial.  An unrelated message is
                # a protocol failure rather than something to reorder silently.
                if message.get("type") == "log":
                    continue
                peak = sampler.stop()
                completed = time.monotonic() * 1000.0
                return ModelReceipt(
                    request.request_id,
                    self.name,
                    request.kind,
                    "error",
                    started,
                    completed,
                    0.0,
                    completed - started,
                    peak,
                    self.quantization,
                    request.payload.get("input_resolution"),
                    MetricKind.MEASURED.value,
                    error=f"worker protocol mismatch: {message!r}",
                )
            peak = sampler.stop()
            completed = time.monotonic() * 1000.0
            if message.get("type") == "response":
                result = message.get("result")
                if not isinstance(result, Mapping):
                    result = {"value": result}
                return ModelReceipt(
                    request.request_id,
                    self.name,
                    request.kind,
                    "done",
                    started,
                    completed,
                    float(self._ready.get("load_ms", 0.0) if self._ready else 0.0),
                    float(message.get("inference_ms", completed - started)),
                    peak,
                    self.quantization,
                    request.payload.get("input_resolution"),
                    MetricKind.MEASURED.value,
                    result=dict(result),
                )
            return ModelReceipt(
                request.request_id,
                self.name,
                request.kind,
                "error",
                started,
                completed,
                0.0,
                completed - started,
                peak,
                self.quantization,
                request.payload.get("input_resolution"),
                MetricKind.MEASURED.value,
                error=str(message.get("error") or message),
            )
        peak = sampler.stop()
        self.close(force=True)
        completed = time.monotonic() * 1000.0
        return ModelReceipt(
            request.request_id,
            self.name,
            request.kind,
            "timeout",
            started,
            completed,
            0.0,
            completed - started,
            peak,
            self.quantization,
            request.payload.get("input_resolution"),
            MetricKind.MEASURED.value,
            error="attached model deadline exceeded; process tree terminated",
            process_terminated=True,
        )

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        if not force and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"type":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=2.0)
            except Exception:
                force = True
        if force and process.poll() is None:
            _terminate_tree(process)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _terminate_tree(process)

    def __enter__(self) -> "AttachedJsonModelProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(force=exc is not None)
