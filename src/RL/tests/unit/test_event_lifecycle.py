"""Tests for ordered reward and lifecycle event processing."""

from __future__ import annotations

import pytest

from RL.events import Event
from RL.rewards import (
    EventLifecycleGate,
    RewardMapper,
)


def event(
    event_type: str,
    **data: object,
) -> Event:
    return Event(
        type=event_type,
        data=dict(data),
    )


def test_empty_batch_has_zero_reward() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process([])

    assert outcome.events == ()
    assert outcome.reward == 0.0
    assert outcome.reward_ledger.is_zero
    assert not outcome.terminated
    assert not outcome.truncated
    assert outcome.termination_reason is None
    assert not outcome.match_active
    assert not gate.episode_done


def test_match_restarted_marks_match_active() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event("match_restarted"),
        ]
    )

    assert outcome.match_active
    assert gate.match_active
    assert not outcome.terminated
    assert not outcome.truncated


def test_accumulates_ordered_rewards() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event(
                "player_kill",
                killer="Noobnog",
                victim="[BOT]Dominator",
            ),
            event(
                "player_kill",
                killer="[BOT]Dominator",
                victim="Noobnog",
            ),
        ]
    )

    assert outcome.reward_components["frag"] == 1.0
    assert outcome.reward_components["death"] == -1.0
    assert outcome.reward == 0.0
    assert outcome.match_active
    assert not outcome.terminated
    assert not outcome.truncated


def test_match_end_before_win_keeps_win_reward() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event("match_ended"),
            event(
                "player_match_win",
                player_name="Noobnog",
            ),
        ]
    )

    assert outcome.reward == 5.0
    assert outcome.reward_components["win"] == 5.0
    assert outcome.terminated
    assert not outcome.truncated
    assert outcome.termination_reason == "match_ended"
    assert not outcome.match_active
    assert gate.episode_done


def test_post_match_disconnect_is_cleanup() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event("match_ended"),
            event(
                "player_disconnected",
                player_name="Noobnog",
            ),
        ]
    )

    assert outcome.reward_components[
        "disconnect"
    ] == 0.0

    assert outcome.terminated
    assert not outcome.truncated
    assert outcome.termination_reason == "match_ended"


def test_active_controlled_disconnect_truncates() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event(
                "player_disconnected",
                player_name="Noobnog",
            ),
        ]
    )

    assert outcome.reward == -2.0
    assert outcome.reward_components[
        "disconnect"
    ] == -2.0

    assert not outcome.terminated
    assert outcome.truncated
    assert outcome.termination_reason == (
        "controlled_player_disconnected"
    )

    assert not outcome.match_active
    assert gate.episode_done


def test_bot_disconnect_does_not_truncate() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event(
                "player_disconnected",
                player_name="[BOT]Dominator",
            ),
        ]
    )

    assert outcome.reward == 0.0
    assert not outcome.terminated
    assert not outcome.truncated
    assert outcome.match_active
    assert not gate.episode_done


def test_session_end_during_match_truncates() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event("session_ended"),
        ]
    )

    assert not outcome.terminated
    assert outcome.truncated
    assert outcome.termination_reason == "session_ended"
    assert not outcome.match_active


def test_session_end_after_match_end_keeps_termination() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event("match_ended"),
            event("session_ended"),
        ]
    )

    assert outcome.terminated
    assert not outcome.truncated
    assert outcome.termination_reason == "match_ended"


def test_first_terminal_signal_wins() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event(
                "player_disconnected",
                player_name="Noobnog",
            ),
            event("match_ended"),
        ]
    )

    assert not outcome.terminated
    assert outcome.truncated
    assert outcome.termination_reason == (
        "controlled_player_disconnected"
    )

    assert outcome.reward == -2.0


def test_processing_after_terminal_is_rejected() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event("match_ended"),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="must be reset",
    ):
        gate.process([])


def test_reset_allows_new_episode() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event("match_ended"),
        ]
    )

    gate.reset_episode()

    outcome = gate.process(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event(
                "player_kill",
                killer="Noobnog",
                victim="[BOT]Dominator",
            ),
        ]
    )

    assert outcome.reward == 1.0
    assert outcome.match_active
    assert not outcome.terminated
    assert not outcome.truncated
    assert not gate.episode_done


def test_constructor_supports_primed_mapper_state() -> None:
    mapper = RewardMapper("Noobnog")

    # Historical priming ledgers are deliberately discarded.
    mapper.map_event(
        event(
            "match_started",
            game_mode="kh",
        )
    )

    mapper.map_event(
        event(
            "player_team_changed",
            player_name="Noobnog",
            team="YELLOW",
        )
    )

    gate = EventLifecycleGate(
        mapper,
        match_active=True,
    )

    outcome = gate.process(
        [
            event("match_ended"),
            event(
                "team_match_win",
                team="YELLOW",
            ),
        ]
    )

    assert outcome.reward == 5.0
    assert outcome.terminated
    assert outcome.termination_reason == "match_ended"


def test_invalid_items_are_rejected_before_mapping() -> None:
    gate = EventLifecycleGate(
        RewardMapper("Noobnog")
    )

    with pytest.raises(
        TypeError,
        match="only Event instances",
    ):
        gate.process(
            [
                event(
                    "match_started",
                    game_mode="dm",
                ),
                "not-an-event",
            ]
        )

    # Validation occurred before the match_started event was mapped.
    outcome = gate.process([])

    assert not outcome.match_active
    assert outcome.reward == 0.0
