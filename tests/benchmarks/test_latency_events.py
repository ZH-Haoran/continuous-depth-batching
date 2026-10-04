from types import SimpleNamespace

import pytest

from looped_cdb.benchmarks.latency_events import comparison_row, write_latency_events


def test_event_export_recomputes_token_gap_distribution(tmp_path) -> None:
    summary = {
        "run_id": "cdb-refill-sharegpt-r3",
        "config": {
            "backend": "cdb",
            "refill": True,
            "workload_name": "sharegpt",
            "request_rate_rps": 3.0,
            "exit_threshold": 0.2,
            "measured_repeat": 0,
        },
        "completed_requests": 2,
        "generated_tokens": 5,
        "completed_requests_per_second": 1.0,
        "generated_tokens_per_second": 2.5,
    }
    outputs = [
        SimpleNamespace(
            request_id="request-0",
            created_time=10.0,
            lifespan=(10.1, 11.4),
            first_token_time=10.2,
            token_ready_times=[10.2, 10.3, 11.3],
            prompt_ids=[1, 2],
            generated_tokens=[3, 4, 5],
        ),
        SimpleNamespace(
            request_id="request-1",
            created_time=10.5,
            lifespan=(10.6, 11.0),
            first_token_time=10.7,
            token_ready_times=[10.7, 10.9],
            prompt_ids=[1],
            generated_tokens=[6, 7],
        ),
    ]
    path = write_latency_events(
        tmp_path, summary=summary, outputs=outputs, queue_samples=[(10.1, 2), (11.4, 0)], start_time=10.0
    )

    row = comparison_row(path)

    assert row["mode"] == "cdb-refill"
    assert row["requests"] == 2
    assert row["queue_peak_waiting"] == 2
    assert row["ttft_p50_ms"] == pytest.approx(200.0)
    assert row["itl_p50_ms"] == pytest.approx(200.0)
    assert row["request_max_itl_p50_ms"] == pytest.approx(600.0)
    assert row["normalized_mean_ms_per_token"] == pytest.approx(((1400 / 3) + (500 / 2)) / 2)
