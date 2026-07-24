from __future__ import annotations

import json
import sys
from pathlib import Path

import psutil

from experiments.generic_utility.model_runtime import (
    AttachedJsonModelProvider,
    EmulatedModelProvider,
    JsonSubprocessModelProvider,
    ModelRequest,
)


WORKER = Path(__file__).resolve().parents[2] / "experiments" / "generic_utility" / "model_workers" / "fake_grounder_worker.py"


def test_emulated_model_receipt_is_explicitly_simulated():
    provider = EmulatedModelProvider("x", load_ms=10, inference_ms=20, peak_vram_mb=30)
    receipt = provider.run(ModelRequest.create("ground", {}))
    assert receipt.status == "done" and receipt.metric_kind == "simulated"
    second = provider.run(ModelRequest.create("ground", {"n": 2}))
    assert second.load_ms == 0


def test_emulated_timeout_marks_process_terminated():
    provider = EmulatedModelProvider("hung", load_ms=1, inference_ms=0, peak_vram_mb=10, timeout=True)
    receipt = provider.run(ModelRequest.create("ground", {}, timeout_ms=50))
    assert receipt.status == "timeout" and receipt.process_terminated


def test_attached_worker_loads_once_and_serves_multiple_requests():
    with AttachedJsonModelProvider(
        "fixture",
        [sys.executable, str(WORKER), "--mode", "echo"],
        startup_timeout_ms=5000,
    ) as provider:
        pid = provider.process_id
        one = provider.run(ModelRequest.create("ground", {"point": [0.2, 0.3]}, timeout_ms=1000))
        two = provider.run(ModelRequest.create("ground", {"point": [0.4, 0.5]}, timeout_ms=1000))
        assert one.status == two.status == "done"
        assert one.result["point"] == [0.2, 0.3]
        assert two.result["point"] == [0.4, 0.5]
        assert provider.process_id == pid
    assert not psutil.pid_exists(pid)


def test_attached_worker_timeout_kills_process_tree():
    provider = AttachedJsonModelProvider(
        "hung",
        [sys.executable, str(WORKER), "--mode", "hang"],
        startup_timeout_ms=5000,
    )
    pid = provider.process_id
    receipt = provider.run(ModelRequest.create("ground", {}, timeout_ms=100))
    assert receipt.status == "timeout" and receipt.process_terminated
    assert not psutil.pid_exists(pid)


def test_one_shot_json_provider_reads_response(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
        """import argparse,json\np=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args();json.dump({'point':[0.1,0.2]},open(a.output,'w'))\n"""
    )
    provider = JsonSubprocessModelProvider("one", [sys.executable, str(script)])
    receipt = provider.run(ModelRequest.create("ground", {}, timeout_ms=1000))
    assert receipt.status == "done" and receipt.result["point"] == [0.1, 0.2]


def test_one_shot_timeout_is_fail_closed(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text("import time; time.sleep(30)\n")
    provider = JsonSubprocessModelProvider("hang", [sys.executable, str(script)])
    receipt = provider.run(ModelRequest.create("ground", {}, timeout_ms=100))
    assert receipt.status == "timeout" and receipt.process_terminated
