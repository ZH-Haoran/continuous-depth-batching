import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from looped_cdb.benchmarks import wandb_serving
from looped_cdb.benchmarks.latency_events import write_latency_events


def test_serving_wandb_upload_uses_metrics_and_raw_file(tmp_path: Path, monkeypatch) -> None:
    summary = {
        "run_id": "cb-test",
        "config": {
            "backend": "cb",
            "refill": False,
            "workload_name": "sharegpt",
            "request_rate_rps": 3.0,
            "exit_threshold": 0.2,
            "measured_repeat": 0,
        },
        "completed_requests": 1,
        "generated_tokens": 2,
        "completed_requests_per_second": 1.0,
        "generated_tokens_per_second": 2.0,
        "device_name": "test GPU",
        "kv_cache": {"num_blocks": 8},
    }
    output = SimpleNamespace(
        request_id="r",
        created_time=10.0,
        lifespan=(10.1, 11.0),
        first_token_time=10.2,
        token_ready_times=[10.2, 10.8],
        prompt_ids=[1],
        generated_tokens=[2, 3],
    )
    path = write_latency_events(
        tmp_path,
        summary=summary,
        outputs=[output],
        queue_samples=[(10.1, 1)],
        start_time=10,
        kv_usage_samples=[(10.1, 2), (10.3, 4)],
    )
    logged = []
    artifacts = []

    class FakeRun:
        id = "test-id"
        url = "https://wandb.ai/test/run"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def log(self, values):
            logged.append(values)

        def log_artifact(self, artifact):
            artifacts.append(artifact)

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, file):
            self.files.append(file)

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: FakeRun(),
        Table=lambda **kwargs: kwargs,
        plot=SimpleNamespace(line=lambda table, x, y, title: {"table": table, "x": x, "y": y, "title": title}),
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(wandb_serving, "_git_commit", lambda: "abc123")

    url = wandb_serving.log_serving_run(path, project="test-project")

    assert url == "https://wandb.ai/test/run"
    assert logged[0]["latency/itl_p50_ms"] == pytest.approx(600.0)
    assert logged[0]["latency/tbt_p50_ms"] == pytest.approx(600.0)
    assert logged[0]["queue/peak_waiting_requests"] == 1
    assert len(logged[1]["queue/waiting_requests"]["data"]) == 1
    kv_chart = logged[2]["kv/occupancy_pct"]
    assert kv_chart["table"]["data"] == [[pytest.approx(0.1), 25.0], [pytest.approx(0.3), 50.0]]
    assert kv_chart["x"] == "elapsed_s"
    assert kv_chart["y"] == "used_pct"
    assert artifacts[0].files == [str(path)]
