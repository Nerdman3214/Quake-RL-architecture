"""Tests for bounded terminal event-tail collection."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from RL.events import (
    Event,
    TerminalEventBatchCollector,
)


def event(
    event_type: str,
    **data: object,
) -> Event:
    return Event(
        type=event_type,
        data=dict(data),
    )


class FakeClock:
    """Deterministic monotonic clock and sleep recorder."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TimedEventSource:
    """Release scheduled events according to a fake clock."""

    def __init__(
        self,
        clock: FakeClock,
        scheduled: Iterable[
            tuple[float, Event]
        ],
    ) -> None:
        self.clock = clock
        self.scheduled = list(scheduled)
        self.index = 0
        self.calls = 0

    def read_new_events(
        self,
    ) -> tuple[Event, ...]:
        self.calls += 1

        available: list[Event] = []

        while self.index < len(self.scheduled):
            available_at, record = (
                self.scheduled[self.index]
            )

            if available_at > self.clock.monotonic():
                break

            available.append(record)
            self.index += 1

        return tuple(available)


def make_collector(
    source: TimedEventSource,
    clock: FakeClock,
    *,
    quiet: float = 0.05,
    maximum: float = 0.25,
    poll: float = 0.005,
) -> TerminalEventBatchCollector:
    return TerminalEventBatchCollector(
        source,
        quiet_period_seconds=quiet,
        max_wait_seconds=maximum,
        poll_interval_seconds=poll,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def test_empty_batch_returns_without_waiting() -> None:
    clock = FakeClock()
    source = TimedEventSource(clock, [])

    collector = make_collector(
        source,
        clock,
    )

    assert collector.read_batch() == ()
    assert source.calls == 1
    assert clock.sleeps == []


def test_nonterminal_batch_returns_without_waiting() -> None:
    clock = FakeClock()

    source = TimedEventSource(
        clock,
        [
            (
                0.0,
                event(
                    "player_kill",
                    killer="Noobnog",
                ),
            ),
            (
                0.001,
                event("unrelated_later_event"),
            ),
        ],
    )

    collector = make_collector(
        source,
        clock,
    )

    records = collector.read_batch()

    assert [
        item.type
        for item in records
    ] == ["player_kill"]

    assert source.calls == 1
    assert clock.sleeps == []


def test_session_end_does_not_trigger_tail_wait() -> None:
    clock = FakeClock()

    source = TimedEventSource(
        clock,
        [
            (
                0.0,
                event("session_ended"),
            ),
            (
                0.001,
                event("later_event"),
            ),
        ],
    )

    collector = make_collector(
        source,
        clock,
    )

    records = collector.read_batch()

    assert [
        item.type
        for item in records
    ] == ["session_ended"]

    assert clock.sleeps == []


def test_collects_immediate_winner_tail() -> None:
    clock = FakeClock()

    source = TimedEventSource(
        clock,
        [
            (
                0.0,
                event("match_ended"),
            ),
            (
                0.001,
                event(
                    "team_match_win",
                    team="YELLOW",
                ),
            ),
            (
                0.002,
                event(
                    "player_match_win",
                    player_name="Noobnog",
                ),
            ),
        ],
    )

    collector = make_collector(
        source,
        clock,
    )

    records = collector.read_batch()

    assert [
        item.type
        for item in records
    ] == [
        "match_ended",
        "team_match_win",
        "player_match_win",
    ]

    assert clock.now >= 0.05
    assert clock.now <= 0.06


def test_new_tail_events_reset_quiet_window() -> None:
    clock = FakeClock()

    source = TimedEventSource(
        clock,
        [
            (
                0.0,
                event("match_ended"),
            ),
            (
                0.04,
                event(
                    "team_match_win",
                    team="RED",
                ),
            ),
            (
                0.08,
                event(
                    "player_match_win",
                    player_name="Noobnog",
                ),
            ),
        ],
    )

    collector = make_collector(
        source,
        clock,
        quiet=0.05,
        maximum=0.20,
        poll=0.01,
    )

    records = collector.read_batch()

    assert [
        item.type
        for item in records
    ] == [
        "match_ended",
        "team_match_win",
        "player_match_win",
    ]

    assert clock.now >= 0.13
    assert clock.now <= 0.14


def test_late_cleanup_events_are_not_collected() -> None:
    clock = FakeClock()

    source = TimedEventSource(
        clock,
        [
            (
                0.0,
                event("match_ended"),
            ),
            (
                0.001,
                event(
                    "player_match_win",
                    player_name="Noobnog",
                ),
            ),
            (
                0.70,
                event("console_line"),
            ),
            (
                5.0,
                event(
                    "player_disconnected",
                    player_name="Noobnog",
                ),
            ),
        ],
    )

    collector = make_collector(
        source,
        clock,
    )

    records = collector.read_batch()

    assert [
        item.type
        for item in records
    ] == [
        "match_ended",
        "player_match_win",
    ]

    assert clock.now < 0.25
    assert source.index == 2


def test_maximum_wait_bounds_continuous_tail() -> None:
    clock = FakeClock()

    scheduled = [
        (
            0.0,
            event("match_ended"),
        )
    ]

    for index in range(1, 20):
        scheduled.append(
            (
                index * 0.01,
                event(
                    "tail_event",
                    sequence=index,
                ),
            )
        )

    source = TimedEventSource(
        clock,
        scheduled,
    )

    collector = make_collector(
        source,
        clock,
        quiet=0.03,
        maximum=0.05,
        poll=0.01,
    )

    records = collector.read_batch()

    assert records[0].type == "match_ended"
    assert clock.now == pytest.approx(0.05)

    assert all(
        float(item.data.get("sequence", 0))
        <= 5.0
        for item in records[1:]
    )


def test_invalid_source_event_is_rejected() -> None:
    class InvalidSource:
        def read_new_events(
            self,
        ) -> tuple[object, ...]:
            return (
                "not-an-event",
            )

    collector = TerminalEventBatchCollector(
        InvalidSource()
    )

    with pytest.raises(
        TypeError,
        match="only Event instances",
    ):
        collector.read_batch()


def test_source_errors_are_not_suppressed() -> None:
    class FailingSource:
        def read_new_events(
            self,
        ) -> tuple[Event, ...]:
            raise RuntimeError(
                "cursor failed"
            )

    collector = TerminalEventBatchCollector(
        FailingSource()
    )

    with pytest.raises(
        RuntimeError,
        match="cursor failed",
    ):
        collector.read_batch()


def test_constructor_rejects_invalid_settings() -> None:
    clock = FakeClock()
    source = TimedEventSource(clock, [])

    invalid_arguments = [
        {
            "quiet_period_seconds": 0.0,
        },
        {
            "quiet_period_seconds": 0.10,
            "max_wait_seconds": 0.05,
        },
        {
            "poll_interval_seconds": 0.0,
        },
    ]

    for arguments in invalid_arguments:
        with pytest.raises(ValueError):
            TerminalEventBatchCollector(
                source,
                **arguments,
            )
