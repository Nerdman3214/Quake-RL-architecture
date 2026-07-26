"""Incremental reader for Xonotic CSQC telemetry JSONL."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from RL.observations.contracts import (
    PlayerTelemetry,
)


@dataclass(frozen=True)
class XonoticTelemetrySnapshot:
    """Latest Xonotic telemetry aligned for one recorder frame."""

    telemetry: PlayerTelemetry
    sample_time: float
    age_seconds: float
    fresh: bool
    weapon_id: int
    ammo_stat: int
    spectatee_status: int
    current_player: int
    pre_spawn: bool
    source_path: Path

    def to_sync_record(
        self,
    ) -> dict[str, object]:
        """Return JSON-serializable synchronization metadata."""

        return {
            "source": "xonotic_csqc_jsonl",
            "source_path": str(self.source_path),
            "sample_time": self.sample_time,
            "age_seconds": self.age_seconds,
            "fresh": self.fresh,
            "weapon_id": self.weapon_id,
            "ammo_stat": self.ammo_stat,
            "spectatee_status": (
                self.spectatee_status
            ),
            "current_player": self.current_player,
            "pre_spawn": self.pre_spawn,
        }


@dataclass(frozen=True)
class _ParsedTelemetryRecord:
    telemetry: PlayerTelemetry
    sample_time: float
    weapon_id: int
    ammo_stat: int
    spectatee_status: int
    current_player: int
    pre_spawn: bool


def _integer_value(
    record: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    if key not in record:
        if default is not None:
            return default

        raise ValueError(
            f"missing telemetry field: {key}"
        )

    value = record[key]

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if (
        isinstance(value, float)
        and value.is_integer()
    ):
        return int(value)

    raise ValueError(
        f"telemetry field {key} must be an integer"
    )


def _float_value(
    record: Mapping[str, Any],
    key: str,
) -> float:
    if key not in record:
        raise ValueError(
            f"missing telemetry field: {key}"
        )

    value = record[key]

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"telemetry field {key} must be numeric"
        )

    return float(value)


def _parse_record(
    record: object,
) -> _ParsedTelemetryRecord:
    if not isinstance(record, Mapping):
        raise ValueError(
            "telemetry record must be a mapping"
        )

    health = _integer_value(
        record,
        "health",
    )
    armor = _integer_value(
        record,
        "armor",
    )
    ammo = _integer_value(
        record,
        "ammo",
    )
    score = _integer_value(
        record,
        "score",
    )

    alive_integer = _integer_value(
        record,
        "alive",
    )

    if alive_integer not in (0, 1):
        raise ValueError(
            "alive must be encoded as 0 or 1"
        )

    weapon = record.get("weapon")

    if not isinstance(weapon, str):
        raise ValueError(
            "weapon must be a string"
        )

    sample_time = _float_value(
        record,
        "sample_time",
    )

    match_time_seconds = _float_value(
        record,
        "match_time_seconds",
    )

    weapon_id = _integer_value(
        record,
        "weapon_id",
        default=-1,
    )
    ammo_stat = _integer_value(
        record,
        "ammo_stat",
        default=-1,
    )
    spectatee_status = _integer_value(
        record,
        "spectatee_status",
        default=-1,
    )
    current_player = _integer_value(
        record,
        "current_player",
        default=-1,
    )

    pre_spawn = (
        health <= -600
        or spectatee_status < 0
        or current_player < 0
    )

    return _ParsedTelemetryRecord(
        telemetry=PlayerTelemetry(
            health=health,
            armor=armor,
            ammo=ammo,
            weapon=weapon,
            alive=bool(alive_integer),
            score=score,
            match_time_seconds=(
                match_time_seconds
            ),
        ),
        sample_time=sample_time,
        weapon_id=weapon_id,
        ammo_stat=ammo_stat,
        spectatee_status=spectatee_status,
        current_player=current_player,
        pre_spawn=pre_spawn,
    )


class XonoticTelemetrySource:
    """Tail an append-only Xonotic telemetry file.

    The reader keeps only the latest valid complete JSON record.
    It tolerates partial lines, invalid records, truncation, and
    replacement of the telemetry file.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_age_seconds: float = 0.25,
        initial_tail_bytes: int = 65536,
        monotonic: Callable[[], float] = (
            time.monotonic
        ),
        wall_time: Callable[[], float] = (
            time.time
        ),
    ) -> None:
        if max_age_seconds <= 0.0:
            raise ValueError(
                "max_age_seconds must be positive"
            )

        if initial_tail_bytes <= 0:
            raise ValueError(
                "initial_tail_bytes must be positive"
            )

        if not callable(monotonic):
            raise TypeError(
                "monotonic must be callable"
            )

        if not callable(wall_time):
            raise TypeError(
                "wall_time must be callable"
            )

        self.path = Path(path).expanduser()

        self.max_age_seconds = float(
            max_age_seconds
        )
        self.initial_tail_bytes = int(
            initial_tail_bytes
        )

        self._monotonic = monotonic
        self._wall_time = wall_time

        self._identity: (
            tuple[int, int] | None
        ) = None

        self._offset = 0
        self._pending = b""

        self._latest: (
            _ParsedTelemetryRecord | None
        ) = None

        self._latest_observed_at: (
            float | None
        ) = None

        self.valid_record_count = 0
        self.invalid_record_count = 0
        self.file_reset_count = 0

    def _clear_file_state(
        self,
    ) -> None:
        self._identity = None
        self._offset = 0
        self._pending = b""
        self._latest = None
        self._latest_observed_at = None

    def _consume(
        self,
        data: bytes,
        *,
        observed_at: float,
    ) -> int:
        combined = self._pending + data
        lines = combined.split(b"\n")

        self._pending = lines.pop()

        accepted = 0

        for raw_line in lines:
            if not raw_line.strip():
                continue

            try:
                decoded = raw_line.decode(
                    "utf-8"
                )
                record = json.loads(decoded)
                parsed = _parse_record(record)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ):
                self.invalid_record_count += 1
                continue

            self._latest = parsed
            self._latest_observed_at = (
                observed_at
            )
            self.valid_record_count += 1
            accepted += 1

        return accepted

    def _open_current_file(
        self,
        stat_result: object,
    ) -> int:
        size = int(
            getattr(
                stat_result,
                "st_size",
            )
        )

        self._identity = (
            int(
                getattr(
                    stat_result,
                    "st_dev",
                )
            ),
            int(
                getattr(
                    stat_result,
                    "st_ino",
                )
            ),
        )

        self._offset = 0
        self._pending = b""
        self._latest = None
        self._latest_observed_at = None
        self.file_reset_count += 1

        start = max(
            0,
            size - self.initial_tail_bytes,
        )

        with self.path.open("rb") as file:
            file.seek(start)
            data = file.read()

        self._offset = start + len(data)

        if start > 0:
            newline_index = data.find(b"\n")

            if newline_index < 0:
                return 0

            data = data[newline_index + 1 :]

        file_age_seconds = max(
            0.0,
            self._wall_time()
            - float(
                getattr(
                    stat_result,
                    "st_mtime",
                )
            ),
        )

        observed_at = (
            self._monotonic()
            - file_age_seconds
        )

        return self._consume(
            data,
            observed_at=observed_at,
        )

    def poll(
        self,
    ) -> int:
        """Read all newly completed records."""

        try:
            stat_result = self.path.stat()
        except FileNotFoundError:
            if self._identity is not None:
                self._clear_file_state()

            return 0

        identity = (
            int(stat_result.st_dev),
            int(stat_result.st_ino),
        )

        if (
            self._identity != identity
            or stat_result.st_size
            < self._offset
        ):
            return self._open_current_file(
                stat_result
            )

        if stat_result.st_size == self._offset:
            return 0

        with self.path.open("rb") as file:
            file.seek(self._offset)
            data = file.read()

        self._offset += len(data)

        return self._consume(
            data,
            observed_at=self._monotonic(),
        )

    def read_latest(
        self,
    ) -> XonoticTelemetrySnapshot | None:
        """Poll the file and return the latest cached sample."""

        self.poll()

        latest = self._latest
        observed_at = self._latest_observed_at

        if latest is None or observed_at is None:
            return None

        age_seconds = max(
            0.0,
            self._monotonic() - observed_at,
        )

        return XonoticTelemetrySnapshot(
            telemetry=latest.telemetry,
            sample_time=latest.sample_time,
            age_seconds=age_seconds,
            fresh=(
                age_seconds
                <= self.max_age_seconds
            ),
            weapon_id=latest.weapon_id,
            ammo_stat=latest.ammo_stat,
            spectatee_status=(
                latest.spectatee_status
            ),
            current_player=latest.current_player,
            pre_spawn=latest.pre_spawn,
            source_path=self.path,
        )
