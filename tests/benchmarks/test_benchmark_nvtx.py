import torch

from looped_cdb.benchmarks import nvtx
from looped_cdb.continuous_depth_batching.requests import PendingGateResult


class FakeNvtx:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []

    def range_push(self, name: str) -> None:
        self.events.append(("push", name))

    def registered_range_push(self, name: str) -> None:
        self.events.append(("registered_push", name))

    def range_pop(self) -> None:
        self.events.append(("pop", None))

    def range_start(self, name: str) -> int:
        self.events.append(("start", name))
        return 7

    def registered_range_start(self, name: str) -> int:
        self.events.append(("registered_start", name))
        return 7

    def range_end(self, handle: int) -> None:
        self.events.append(("end", str(handle)))


def test_nvtx_range_is_noop_when_disabled(monkeypatch) -> None:
    nvtx.set_enabled(False)
    monkeypatch.setattr(nvtx, "_cuda_nvtx", lambda: (_ for _ in ()).throw(AssertionError("should not import torch")))

    with nvtx.range("benchmark.generate"):
        pass

    assert not nvtx.is_enabled()


def test_nvtx_range_pushes_and_pops_when_enabled(monkeypatch) -> None:
    fake = FakeNvtx()
    nvtx.set_enabled(True)
    monkeypatch.setattr(nvtx, "_cuda_nvtx", lambda: fake)

    with nvtx.range("benchmark.generate"):
        fake.events.append(("inside", None))

    nvtx.set_enabled(False)
    assert fake.events == [
        ("push", "benchmark.generate"),
        ("inside", None),
        ("pop", None),
    ]


def test_gate_wait_range_marks_only_blocking_waits(monkeypatch) -> None:
    class FakeEvent:
        def __init__(self, ready: bool) -> None:
            self.ready = ready
            self.sync_count = 0

        def query(self) -> bool:
            return self.ready

        def synchronize(self) -> None:
            self.sync_count += 1

    def pending(event: FakeEvent) -> PendingGateResult:
        return PendingGateResult(host_logits=torch.tensor([1.0]), batch_index=0, recurrent_step=0, ready_event=event)

    fake = FakeNvtx()
    nvtx.set_enabled(True)
    monkeypatch.setattr(nvtx, "_cuda_nvtx", lambda: fake)

    blocking = FakeEvent(ready=False)
    pending(blocking).signal()
    complete = FakeEvent(ready=True)
    pending(complete).signal()

    nvtx.set_enabled(False)
    # Both waits synchronize, but only the one that actually blocked appears in the trace.
    assert blocking.sync_count == 1
    assert complete.sync_count == 1
    assert fake.events == [("push", "cdb.gate_wait"), ("pop", None)]

    # Disabled path: no CUDA query, no range, still synchronizes.
    class QueryForbiddenEvent(FakeEvent):
        def query(self) -> bool:
            raise AssertionError("the unprofiled path must not issue per-item event queries")

    unprofiled = QueryForbiddenEvent(ready=False)
    pending(unprofiled).signal()
    assert unprofiled.sync_count == 1


def test_nvtx_start_end_range_spans_a_push_pop_range_and_is_noop_when_disabled(monkeypatch) -> None:
    fake = FakeNvtx()
    nvtx.set_enabled(False)
    assert nvtx.range_start("benchmark.steady") is None

    nvtx.set_enabled(True)
    monkeypatch.setattr(nvtx, "_cuda_nvtx", lambda: fake)
    with nvtx.range("cdb.schedule"):
        handle = nvtx.range_start("benchmark.steady", registered=True)
    with nvtx.range("cdb.recurrent"):
        pass
    assert handle is not None
    nvtx.range_end(handle)

    nvtx.set_enabled(False)
    assert fake.events == [
        ("push", "cdb.schedule"),
        ("registered_start", "benchmark.steady"),
        ("pop", None),
        ("push", "cdb.recurrent"),
        ("pop", None),
        ("end", "7"),
    ]


def test_nvtx_range_can_use_registered_message_when_enabled(monkeypatch) -> None:
    fake = FakeNvtx()
    nvtx.set_enabled(True)
    monkeypatch.setattr(nvtx, "_cuda_nvtx", lambda: fake)

    with nvtx.range("benchmark.generate", registered=True):
        fake.events.append(("inside", None))

    nvtx.set_enabled(False)
    assert fake.events == [
        ("registered_push", "benchmark.generate"),
        ("inside", None),
        ("pop", None),
    ]
