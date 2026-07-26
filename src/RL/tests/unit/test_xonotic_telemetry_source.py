from __future__ import annotations

import json
import os
from pathlib import Path

from RL.recording.xonotic_telemetry_source import (
    XonoticTelemetrySource,
)


class FakeClock:
    def __init__(
        self,
        *,
        monotonic_value: float = 50.0,
        wall_value: float = 1000.0,
    ) -> None:
        self.monotonic_value = (
            monotonic_value
        )
        self.wall_value = wall_value

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall_time(self) -> float:
        return self.wall_value


def telemetry_record(
    *,
    health: int = 100,
    alive: int = 1,
    score: int = 0,
    sample_time: float = 4.5,
) -> dict[str, object]:
    return {
        "sample_time": sample_time,
        "match_time_seconds": 2.5,
        "health": health,
        "armor": 25,
        "ammo": 14,
        "ammo_stat": 6,
        "weapon": "shotgun",
        "weapon_id": 2,
        "alive": alive,
        "score": score,
        "spectatee_status": 0,
        "current_player": 0,
    }


def test_reads_latest_complete_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    clock = FakeClock()

    path.write_text(
        json.dumps(
            telemetry_record()
        )
        + "\n",
        encoding="utf-8",
    )

    os.utime(
        path,
        (
            clock.wall_value,
            clock.wall_value,
        ),
    )

    source = XonoticTelemetrySource(
        path,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
    )

    snapshot = source.read_latest()

    assert snapshot is not None
    assert snapshot.telemetry.health == 100
    assert snapshot.telemetry.alive
    assert snapshot.telemetry.weapon == (
        "shotgun"
    )
    assert snapshot.weapon_id == 2
    assert snapshot.ammo_stat == 6
    assert snapshot.fresh
    assert source.valid_record_count == 1
    assert source.invalid_record_count == 0


def test_waits_for_partial_line_completion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    record_text = json.dumps(
        telemetry_record()
    )

    path.write_text(
        record_text,
        encoding="utf-8",
    )

    source = XonoticTelemetrySource(path)

    assert source.read_latest() is None

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write("\n")

    snapshot = source.read_latest()

    assert snapshot is not None
    assert snapshot.telemetry.health == 100
    assert source.valid_record_count == 1


def test_skips_invalid_records_and_handles_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"

    path.write_text(
        "not-json\n"
        + json.dumps(
            telemetry_record(
                health=75,
                score=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    source = XonoticTelemetrySource(path)

    first = source.read_latest()

    assert first is not None
    assert first.telemetry.health == 75
    assert first.telemetry.score == 1
    assert source.invalid_record_count == 1

    replacement = tmp_path / "replacement.jsonl"

    replacement.write_text(
        json.dumps(
            telemetry_record(
                health=-20,
                alive=0,
                score=2,
                sample_time=9.0,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    replacement.replace(path)

    second = source.read_latest()

    assert second is not None
    assert second.telemetry.health == -20
    assert not second.telemetry.alive
    assert second.telemetry.score == 2
    assert second.sample_time == 9.0
    assert source.file_reset_count >= 2


def test_marks_old_initial_file_stale(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    clock = FakeClock(
        monotonic_value=50.0,
        wall_value=1000.0,
    )

    path.write_text(
        json.dumps(
            telemetry_record()
        )
        + "\n",
        encoding="utf-8",
    )

    os.utime(
        path,
        (990.0, 990.0),
    )

    source = XonoticTelemetrySource(
        path,
        max_age_seconds=0.25,
        monotonic=clock.monotonic,
        wall_time=clock.wall_time,
    )

    snapshot = source.read_latest()

    assert snapshot is not None
    assert snapshot.age_seconds == 10.0
    assert not snapshot.fresh
