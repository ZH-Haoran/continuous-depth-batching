import sqlite3
from pathlib import Path

import analyze_paper_nsys
from analyze_paper_nsys import Interval


def test_merge_intervals_and_busy_gap_split() -> None:
    busy = analyze_paper_nsys.merge_intervals(
        [
            Interval(10, 20),
            Interval(15, 25),
            Interval(40, 50),
        ]
    )

    assert busy == [Interval(10, 25), Interval(40, 50)]
    busy_ns, gaps = analyze_paper_nsys.busy_duration_and_gaps([Interval(0, 60)], busy)
    assert busy_ns == 25
    assert gaps == [
        Interval(0, 10),
        Interval(25, 40),
        Interval(50, 60),
    ]


def test_summarize_sqlite_reports_paper_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, name TEXT, correlationId INTEGER)"
        )
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind TEXT, bytes INTEGER)"
        )
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start INTEGER, end INTEGER, name INTEGER, globalTid INTEGER, correlationId INTEGER)"
        )
        conn.execute("INSERT INTO StringIds VALUES (7, 'cudaLaunchKernel')")
        conn.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (0, 1000, "benchmark.generate", 1),
                (700, 800, "cdb.exit_kv_copy", 1),
                (100, 700, "cdb.recurrent", 1),
            ],
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            [
                (100, 300, "kernel_a", 9),
                (500, 800, "kernel_b", 11),
            ],
        )
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (820, 900, 'DtoH', 128)")
        # Both kernels are launched inside cdb.recurrent, but kernel_b executes into the
        # cdb.exit_kv_copy window, so overlap and launch-site attribution disagree on it.
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            [
                (120, 130, 7, 1, 9),
                (450, 460, 7, 1, 11),
            ],
        )

    # stage_labels=None discovers every NVTX label in the trace besides the measured
    # window, emitted in pipeline order rather than alphabetically.
    summary = analyze_paper_nsys.summarize_sqlite(db_path, label="toy", stage_labels=None)

    assert summary["label"] == "toy"
    assert summary["measured_label"] == "benchmark.generate"
    assert list(summary["stages"]) == ["cdb.recurrent", "cdb.exit_kv_copy"]
    assert summary["measured"]["100us"]["gpu_busy_s"] == 580 / 1e9
    assert summary["measured"]["100us"]["gpu_idle_s"] == 420 / 1e9
    assert summary["stages"]["cdb.recurrent"]["100us"]["gpu_busy_s"] == 400 / 1e9
    # Launch-site attribution credits kernel_b (300 ns) to the stage that launched it,
    # even though it executes into the cdb.exit_kv_copy window.
    assert summary["stages"]["cdb.recurrent"]["100us"]["gpu_launched_s"] == 500 / 1e9
    assert summary["stages"]["cdb.exit_kv_copy"]["100us"]["gpu_launched_s"] == 0.0
    assert summary["measured"]["100us"]["gpu_launched_s"] == 500 / 1e9
    assert summary["kernels"]["kernel_launch_count"] == 2
    assert summary["memcpy"]["memcpy_count"] == 1
    assert summary["cuda_api"]["cuda_api_call_count"] == 2
    assert summary["cuda_api"]["top_cuda_api"][0]["name"] == "cudaLaunchKernel"


def test_summarize_sqlite_clips_to_the_steady_window_when_the_trace_holds_one(tmp_path: Path) -> None:
    # The steady window lies inside the measured run; everything is clipped to it, and neither
    # window label is reported as a stage.
    db_path = tmp_path / "steady.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, name TEXT, correlationId INTEGER)"
        )
        conn.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (0, 1000, "benchmark.generate", 1),
                (200, 900, "benchmark.steady", 1),
                (100, 700, "cdb.recurrent", 1),
            ],
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            [(100, 300, "kernel_a", 9), (500, 800, "kernel_b", 11)],
        )

    summary = analyze_paper_nsys.summarize_sqlite(db_path, label="toy", stage_labels=None)

    assert summary["measured_label"] == "benchmark.steady"
    assert list(summary["stages"]) == ["cdb.recurrent"]
    assert summary["measured"]["100us"]["gpu_busy_s"] == 400 / 1e9
    assert summary["measured"]["100us"]["gpu_idle_s"] == 300 / 1e9
    assert summary["stages"]["cdb.recurrent"]["100us"]["gpu_busy_s"] == 300 / 1e9


def test_summarize_sqlite_can_select_the_full_generation_window(tmp_path: Path) -> None:
    db_path = tmp_path / "full.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER)")
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, name TEXT, correlationId INTEGER)"
        )
        conn.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (0, 1000, "benchmark.generate", 1),
                (200, 900, "benchmark.steady", 1),
                (100, 700, "cdb.recurrent", 1),
            ],
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
            [(100, 300, "kernel_a", 9), (500, 800, "kernel_b", 11)],
        )

    summary = analyze_paper_nsys.summarize_sqlite(
        db_path,
        label="toy",
        stage_labels=None,
        measured_label="benchmark.generate",
    )

    assert summary["measured_label"] == "benchmark.generate"
    assert list(summary["stages"]) == ["cdb.recurrent"]
    assert summary["measured"]["100us"]["gpu_busy_s"] == 500 / 1e9
    assert summary["measured"]["100us"]["gpu_idle_s"] == 500 / 1e9
    assert summary["stages"]["cdb.recurrent"]["100us"]["gpu_busy_s"] == 400 / 1e9


def test_summarize_sqlite_warns_for_graph_launches_without_node_trace(tmp_path: Path) -> None:
    db_path = tmp_path / "graph.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        conn.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT)")
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, name TEXT)")
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, name INTEGER)")
        conn.execute("INSERT INTO StringIds VALUES (1, 'cudaGraphLaunch_v10000')")
        conn.execute("INSERT INTO NVTX_EVENTS VALUES (0, 1000, 'benchmark.generate')")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (200, 300, 'visible_kernel')")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 180, 1)")

    summary = analyze_paper_nsys.summarize_sqlite(db_path, label="toy", stage_labels=())

    assert summary["stages"] == {}
    assert summary["diagnostics"]["cuda_graph_launch_count"] == 1
    assert summary["diagnostics"]["has_graph_node_kernel_activity"] is False
    assert summary["diagnostics"]["warnings"]
