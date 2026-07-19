"""Tests for the authoritative JSONL event step processor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from RL.env.core.event_pipeline import (
    JSONLEventStepProcessor,
)
from RL.events import Event


def event(
    event_type: str,
    **data: object,
) -> Event:
    return Event(
        type=event_type,
        data=dict(data),
    )


def payload(record: Event) -> str:
    return json.dumps(
        {
            "type": record.type,
            "data": record.data,
        },
        separators=(",", ":"),
    )


def write_events(
    path: Path,
    *records: Event,
) -> None:
    text = "".join(
        payload(record) + "\n"
        for record in records
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def append_events(
    path: Path,
    *records: Event,
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(payload(record) + "\n")

        file.flush()


def active_history(path: Path) -> None:
    write_events(
        path,
        event("session_started"),
        event(
            "match_started",
            game_mode="dm",
        ),
        event(
            "player_now_playing",
            player_name="Noobnog",
            team="RED",
        ),
    )


def test_requires_reset_before_step(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    write_events(path)

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
    )

    with pytest.raises(
        RuntimeError,
        match="reset before step",
    ):
        processor.process_step()


def test_reset_primes_active_state_without_reward(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    write_events(
        path,
        event("session_started"),
        event(
            "match_started",
            game_mode="dm",
        ),
        event(
            "player_team_changed",
            player_name="Noobnog",
            team="RED",
        ),
        event(
            "player_kill",
            killer="Noobnog",
            victim="[BOT]Dominator",
        ),
    )

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
    )

    processor.reset_episode()
    outcome = processor.process_step()

    assert processor.initialized
    assert processor.priming_reason == "active_match"
    assert processor.primed_event_count == 3
    assert processor.history_event_count == 4
    assert processor.match_active
    assert processor.current_mode == "dm"
    assert processor.controlled_team == "RED"

    assert outcome.events == ()
    assert outcome.reward == 0.0
    assert not outcome.terminated
    assert not outcome.truncated


def test_appended_reward_is_consumed_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    active_history(path)

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
    )

    processor.reset_episode()

    append_events(
        path,
        event(
            "player_kill",
            killer="Noobnog",
            victim="[BOT]Dominator",
        ),
    )

    first = processor.process_step()
    second = processor.process_step()

    assert first.reward == 1.0
    assert first.reward_components["frag"] == 1.0
    assert second.reward == 0.0
    assert second.events == ()


def test_inactive_orphan_win_is_ignored(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    write_events(
        path,
        event("session_started"),
    )

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
    )

    processor.reset_episode()

    append_events(
        path,
        event(
            "player_match_win",
            player_name="Noobnog",
        ),
    )

    outcome = processor.process_step()

    assert [
        item.type
        for item in outcome.events
    ] == ["player_match_win"]

    assert outcome.reward == 0.0
    assert not outcome.terminated
    assert not outcome.truncated
    assert not processor.match_active


def test_new_match_start_arms_inactive_processor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    write_events(
        path,
        event("session_started"),
    )

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
    )

    processor.reset_episode()

    append_events(
        path,
        event(
            "player_match_win",
            player_name="Noobnog",
        ),
        event(
            "match_started",
            game_mode="dm",
        ),
        event(
            "player_kill",
            killer="Noobnog",
            victim="[BOT]Dominator",
        ),
    )

    outcome = processor.process_step()

    assert [
        item.type
        for item in outcome.events
    ] == [
        "match_started",
        "player_kill",
    ]

    assert outcome.reward == 1.0
    assert outcome.match_active
    assert processor.match_active
    assert processor.current_mode == "dm"


class AppendingClock:
    """Fake clock that appends one winner during tail waiting."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path
        self.now = 0.0
        self.appended = False

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

        if not self.appended:
            append_events(
                self.path,
                event(
                    "player_match_win",
                    player_name="Noobnog",
                ),
            )

            self.appended = True


def test_terminal_tail_keeps_winner_reward(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    active_history(path)

    clock = AppendingClock(path)

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
        quiet_period_seconds=0.05,
        max_wait_seconds=0.25,
        poll_interval_seconds=0.005,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    processor.reset_episode()

    append_events(
        path,
        event("match_ended"),
    )

    outcome = processor.process_step()

    assert [
        item.type
        for item in outcome.events
    ] == [
        "match_ended",
        "player_match_win",
    ]

    assert outcome.reward == 5.0
    assert outcome.reward_components["win"] == 5.0
    assert outcome.terminated
    assert not outcome.truncated
    assert outcome.termination_reason == "match_ended"
    assert processor.episode_done


def test_terminal_outcome_requires_reset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    active_history(path)

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
    )

    processor.reset_episode()

    append_events(
        path,
        event(
            "player_disconnected",
            player_name="Noobnog",
        ),
    )

    outcome = processor.process_step()

    assert outcome.reward == -2.0
    assert outcome.truncated
    assert processor.episode_done

    with pytest.raises(
        RuntimeError,
        match="reset after",
    ):
        processor.process_step()


def test_reset_rebuilds_context_for_new_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    active_history(path)

    processor = JSONLEventStepProcessor(
        path,
        "Noobnog",
    )

    processor.reset_episode()

    append_events(
        path,
        event(
            "player_disconnected",
            player_name="Noobnog",
        ),
    )

    assert processor.process_step().truncated

    append_events(
        path,
        event(
            "match_started",
            game_mode="ctf",
        ),
        event(
            "player_team_changed",
            player_name="Noobnog",
            team="YELLOW",
        ),
    )

    processor.reset_episode()
    outcome = processor.process_step()

    assert processor.priming_reason == "active_match"
    assert processor.match_active
    assert processor.current_mode == "ctf"
    assert processor.controlled_team == "YELLOW"
    assert not processor.episode_done

    assert outcome.events == ()
    assert outcome.reward == 0.0
