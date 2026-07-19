"""Bounded collection of events following match completion."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from RL.events.contracts import Event


class IncrementalEventSource(Protocol):
    """Incremental event-source operations used by the collector."""

    def read_new_events(self) -> tuple[Event, ...]:
        """Return events appended since the previous read."""


class TerminalEventBatchCollector:
    """Collect an immediately following match-completion tail.

    Normal batches return after one source read. A batch containing
    ``match_ended`` receives a short bounded drain so winner
    announcements written immediately afterward remain in the same
    lifecycle batch.

    The quiet window is reset whenever another event arrives. Total
    waiting remains capped by ``max_wait_seconds``.
    """

    _TAIL_TRIGGER_TYPES = frozenset(
        {
            "match_ended",
        }
    )

    def __init__(
        self,
        source: IncrementalEventSource,
        *,
        quiet_period_seconds: float = 0.05,
        max_wait_seconds: float = 0.25,
        poll_interval_seconds: float = 0.005,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        reader = getattr(
            source,
            "read_new_events",
            None,
        )

        if not callable(reader):
            raise TypeError(
                "source must provide read_new_events"
            )

        if quiet_period_seconds <= 0.0:
            raise ValueError(
                "quiet_period_seconds must be positive"
            )

        if max_wait_seconds < quiet_period_seconds:
            raise ValueError(
                "max_wait_seconds must be greater than "
                "or equal to quiet_period_seconds"
            )

        if poll_interval_seconds <= 0.0:
            raise ValueError(
                "poll_interval_seconds must be positive"
            )

        if not callable(monotonic):
            raise TypeError(
                "monotonic must be callable"
            )

        if not callable(sleep):
            raise TypeError(
                "sleep must be callable"
            )

        self._source = source

        self.quiet_period_seconds = float(
            quiet_period_seconds
        )

        self.max_wait_seconds = float(
            max_wait_seconds
        )

        self.poll_interval_seconds = float(
            poll_interval_seconds
        )

        self._monotonic = monotonic
        self._sleep = sleep

    @staticmethod
    def _validate_events(
        events: object,
    ) -> tuple[Event, ...]:
        try:
            records = tuple(events)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError(
                "event source must return an iterable"
            ) from error

        for event in records:
            if not isinstance(event, Event):
                raise TypeError(
                    "event source must return only "
                    "Event instances"
                )

        return records

    def _read_once(self) -> tuple[Event, ...]:
        return self._validate_events(
            self._source.read_new_events()
        )

    @classmethod
    def _requires_tail(
        cls,
        events: tuple[Event, ...],
    ) -> bool:
        return any(
            event.type in cls._TAIL_TRIGGER_TYPES
            for event in events
        )

    def read_batch(self) -> tuple[Event, ...]:
        """Read one event batch with bounded terminal-tail draining."""

        initial = self._read_once()

        if not self._requires_tail(initial):
            return initial

        records = list(initial)

        started_at = self._monotonic()
        last_event_at = started_at

        while True:
            now = self._monotonic()

            max_remaining = (
                self.max_wait_seconds
                - (now - started_at)
            )

            quiet_remaining = (
                self.quiet_period_seconds
                - (now - last_event_at)
            )

            if (
                max_remaining <= 0.0
                or quiet_remaining <= 0.0
            ):
                break

            delay = min(
                self.poll_interval_seconds,
                max_remaining,
                quiet_remaining,
            )

            self._sleep(delay)

            appended = self._read_once()
            now = self._monotonic()

            if appended:
                records.extend(appended)
                last_event_at = now

        return tuple(records)
