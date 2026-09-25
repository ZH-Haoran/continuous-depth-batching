"""Record CDB stage launches and scheduler queues."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

TraceStage = Literal["prefill", "prelude", "recurrent", "coda"]
TRACE_STAGES: tuple[TraceStage, ...] = get_args(TraceStage)


@dataclass(frozen=True)
class QueueSnapshot:
    """Queue sizes after a stage launch.

    ``decode`` excludes the running no-refill cohort.
    ``coda_in_flight`` excludes its carried boundary coda.
    """

    waiting: int
    recurrent: int
    coda: int
    decode: int
    coda_in_flight: int
    active: int


@dataclass(frozen=True)
class TraceEvent:
    """One stage launch, with zero-based recurrent depths."""

    launch: int
    tick: int
    stage: TraceStage
    size: int
    queues: QueueSnapshot
    steady: bool = False
    depths: dict[int, int] | None = None
    exited: int | None = None
    site: str | None = None
    tokens: int | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if self.depths is not None:
            record["depths"] = {str(step): count for step, count in sorted(self.depths.items())}
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TraceEvent:
        depths = record.get("depths")
        return cls(
            launch=int(record["launch"]),
            tick=int(record["tick"]),
            stage=record["stage"],
            size=int(record["size"]),
            queues=QueueSnapshot(**record["queues"]),
            steady=bool(record.get("steady", False)),
            depths=None if depths is None else {int(step): int(count) for step, count in depths.items()},
            exited=record.get("exited"),
            site=record.get("site"),
            tokens=record.get("tokens"),
        )


@dataclass
class ScheduleTrace:
    """Collect stage launches for one generation run."""

    events: list[TraceEvent] = field(default_factory=list)
    tick: int = 0
    running_cohort: int = 0

    def reset(self) -> None:
        self.events.clear()
        self.tick = 0
        self.running_cohort = 0

    def record(self, stage: TraceStage, size: int, queues: QueueSnapshot, **detail: Any) -> TraceEvent:
        event = TraceEvent(launch=len(self.events) + 1, tick=self.tick, stage=stage, size=size, queues=queues, **detail)
        self.events.append(event)
        return event

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_record()) + "\n")

    @staticmethod
    def load_jsonl(path: Path) -> list[TraceEvent]:
        with path.open() as handle:
            return [TraceEvent.from_record(json.loads(line)) for line in handle if line.strip()]
