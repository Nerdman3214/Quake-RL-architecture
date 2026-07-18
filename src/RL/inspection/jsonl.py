"""Strict JSONL storage for inspectable agent episodes."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from RL.inspection.contracts import AgentTransition


RECORD_TYPES = {
    "episode_started",
    "agent_transition",
    "episode_ended",
}


def _is_nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_nonempty_string(
    value: Any,
    field_name: str,
    line_number: int,
) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Invalid {field_name} on line {line_number}"
        )


@dataclass(frozen=True)
class EpisodeStarted:
    """Metadata written before the first transition."""

    episode_id: str
    started_at: str
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError(
                "episode id must not be empty"
            )

        if not self.started_at:
            raise ValueError(
                "started_at must not be empty"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "type": "episode_started",
            "data": {
                "episode_id": self.episode_id,
                "started_at": self.started_at,
                "metadata": dict(self.metadata),
            },
        }


@dataclass(frozen=True)
class EpisodeEnded:
    """Metadata written after the final transition."""

    episode_id: str
    ended_at: str
    steps: int
    terminated: bool
    truncated: bool
    outcome: str | None = None
    summary: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError(
                "episode id must not be empty"
            )

        if not self.ended_at:
            raise ValueError(
                "ended_at must not be empty"
            )

        if self.steps < 0:
            raise ValueError(
                "episode steps must not be negative"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "type": "episode_ended",
            "data": {
                "episode_id": self.episode_id,
                "ended_at": self.ended_at,
                "steps": self.steps,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "outcome": self.outcome,
                "summary": dict(self.summary),
            },
        }


@dataclass(frozen=True)
class InspectionRecord:
    """One validated inspection JSONL record."""

    type: str
    data: dict[str, Any]


class InspectionJSONLWriter:
    """Write inspection records with immediate flushing."""

    def __init__(
        self,
        filepath: str | Path,
        *,
        append: bool = False,
    ) -> None:
        self.path = Path(filepath)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mode = "a" if append else "w"

        self._file = self.path.open(
            mode,
            encoding="utf-8",
        )
        self._lock = threading.Lock()
        self._closed = False

    def _write_record(
        self,
        record: Mapping[str, Any],
    ) -> None:
        if self._closed:
            raise ValueError(
                "inspection writer is closed"
            )

        line = json.dumps(
            dict(record),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def write_episode_started(
        self,
        episode: EpisodeStarted,
    ) -> None:
        if not isinstance(episode, EpisodeStarted):
            raise TypeError(
                "episode must be an EpisodeStarted instance"
            )

        self._write_record(episode.to_record())

    def write_transition(
        self,
        transition: AgentTransition,
    ) -> None:
        if not isinstance(transition, AgentTransition):
            raise TypeError(
                "transition must be an AgentTransition instance"
            )

        self._write_record(
            {
                "type": "agent_transition",
                "data": transition.to_record(),
            }
        )

    def write_episode_ended(
        self,
        episode: EpisodeEnded,
    ) -> None:
        if not isinstance(episode, EpisodeEnded):
            raise TypeError(
                "episode must be an EpisodeEnded instance"
            )

        self._write_record(episode.to_record())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            self._file.close()
            self._closed = True

    def __enter__(self) -> "InspectionJSONLWriter":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


class InspectionJSONLReader:
    """Read and strictly validate inspection JSONL."""

    def __init__(
        self,
        filepath: str | Path,
    ) -> None:
        self.path = Path(filepath)

        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

    @staticmethod
    def _validate_record(
        record_type: str,
        data: dict[str, Any],
        line_number: int,
    ) -> None:
        _require_nonempty_string(
            data.get("episode_id"),
            "episode id",
            line_number,
        )

        if record_type == "episode_started":
            _require_nonempty_string(
                data.get("started_at"),
                "started_at",
                line_number,
            )

            if not isinstance(
                data.get("metadata"),
                dict,
            ):
                raise ValueError(
                    "Invalid episode metadata on "
                    f"line {line_number}"
                )

            return

        if record_type == "episode_ended":
            _require_nonempty_string(
                data.get("ended_at"),
                "ended_at",
                line_number,
            )

            if not _is_nonnegative_integer(
                data.get("steps")
            ):
                raise ValueError(
                    "Invalid episode steps on "
                    f"line {line_number}"
                )

            if not isinstance(
                data.get("terminated"),
                bool,
            ):
                raise ValueError(
                    "Invalid terminated flag on "
                    f"line {line_number}"
                )

            if not isinstance(
                data.get("truncated"),
                bool,
            ):
                raise ValueError(
                    "Invalid truncated flag on "
                    f"line {line_number}"
                )

            outcome = data.get("outcome")

            if outcome is not None and not isinstance(
                outcome,
                str,
            ):
                raise ValueError(
                    "Invalid episode outcome on "
                    f"line {line_number}"
                )

            if not isinstance(
                data.get("summary"),
                dict,
            ):
                raise ValueError(
                    "Invalid episode summary on "
                    f"line {line_number}"
                )

            return

        if not _is_nonnegative_integer(
            data.get("step_index")
        ):
            raise ValueError(
                "Invalid transition step index on "
                f"line {line_number}"
            )

        if not isinstance(
            data.get("observation"),
            dict,
        ):
            raise ValueError(
                "Invalid transition observation on "
                f"line {line_number}"
            )

        if not isinstance(
            data.get("decision"),
            dict,
        ):
            raise ValueError(
                "Invalid transition decision on "
                f"line {line_number}"
            )

        if not _is_finite_number(
            data.get("reward")
        ):
            raise ValueError(
                "Invalid transition reward on "
                f"line {line_number}"
            )

        next_observation = data.get(
            "next_observation"
        )

        if (
            next_observation is not None
            and not isinstance(
                next_observation,
                dict,
            )
        ):
            raise ValueError(
                "Invalid next observation on "
                f"line {line_number}"
            )

        if not isinstance(
            data.get("terminated"),
            bool,
        ):
            raise ValueError(
                "Invalid terminated flag on "
                f"line {line_number}"
            )

        if not isinstance(
            data.get("truncated"),
            bool,
        ):
            raise ValueError(
                "Invalid truncated flag on "
                f"line {line_number}"
            )

        goal = data.get("goal")

        if goal is not None and not isinstance(
            goal,
            dict,
        ):
            raise ValueError(
                "Invalid transition goal on "
                f"line {line_number}"
            )

        if not isinstance(
            data.get("reward_components"),
            dict,
        ):
            raise ValueError(
                "Invalid reward components on "
                f"line {line_number}"
            )

        if not isinstance(
            data.get("info"),
            dict,
        ):
            raise ValueError(
                "Invalid transition info on "
                f"line {line_number}"
            )

    def read_records(
        self,
    ) -> Iterator[InspectionRecord]:
        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, raw_line in enumerate(
                file,
                1,
            ):
                line = raw_line.strip()

                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid JSON on line "
                        f"{line_number}"
                    ) from error

                if not isinstance(payload, dict):
                    raise ValueError(
                        "Invalid inspection payload on "
                        f"line {line_number}"
                    )

                record_type = payload.get("type")
                data = payload.get("data")

                if record_type not in RECORD_TYPES:
                    raise ValueError(
                        "Invalid inspection record type "
                        f"on line {line_number}"
                    )

                if not isinstance(data, dict):
                    raise ValueError(
                        "Invalid inspection record data "
                        f"on line {line_number}"
                    )

                self._validate_record(
                    record_type,
                    data,
                    line_number,
                )

                yield InspectionRecord(
                    type=record_type,
                    data=data,
                )

    def read_episode(
        self,
    ) -> list[InspectionRecord]:
        records = list(self.read_records())

        if not records:
            raise ValueError(
                "Inspection JSONL is empty"
            )

        if records[0].type != "episode_started":
            raise ValueError(
                "Episode must begin with episode_started"
            )

        if records[-1].type != "episode_ended":
            raise ValueError(
                "Episode must end with episode_ended"
            )

        if any(
            record.type == "episode_started"
            for record in records[1:]
        ):
            raise ValueError(
                "Episode contains multiple start records"
            )

        if any(
            record.type == "episode_ended"
            for record in records[:-1]
        ):
            raise ValueError(
                "Episode ended before the final record"
            )

        episode_id = records[0].data["episode_id"]

        for record in records:
            if record.data["episode_id"] != episode_id:
                raise ValueError(
                    "Episode contains mismatched "
                    "episode identifiers"
                )

        transitions = [
            record
            for record in records
            if record.type == "agent_transition"
        ]

        for expected_step, transition in enumerate(
            transitions
        ):
            if (
                transition.data["step_index"]
                != expected_step
            ):
                raise ValueError(
                    "Transition step indexes must be "
                    "contiguous and begin at zero"
                )

        ended_steps = records[-1].data["steps"]

        if ended_steps != len(transitions):
            raise ValueError(
                "Episode-ended step count does not "
                "match transition count"
            )

        return records
