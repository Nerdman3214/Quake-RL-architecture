"""Tests for historical reward-state priming selection."""

from __future__ import annotations

import pytest

from RL.events import Event
from RL.rewards import (
    RewardMapper,
    build_reward_priming_plan,
)


def event(
    event_type: str,
    **data: object,
) -> Event:
    return Event(
        type=event_type,
        data=dict(data),
    )


def event_types(
    events: tuple[Event, ...],
) -> list[str]:
    return [
        item.type
        for item in events
    ]


def test_selects_latest_active_match_segment() -> None:
    records = [
        event("session_started"),
        event(
            "match_started",
            game_mode="dm",
        ),
        event("match_ended"),
        event(
            "match_started",
            game_mode="kh",
        ),
        event(
            "player_team_changed",
            player_name="Noobnog",
            team="YELLOW",
        ),
        event(
            "player_kill",
            killer="Noobnog",
            victim="[BOT]Dominator",
        ),
    ]

    plan = build_reward_priming_plan(records)

    assert plan.reason == "active_match"
    assert plan.session_open
    assert plan.match_active
    assert plan.boundary_index == 3
    assert plan.selected_count == 3

    assert event_types(plan.events) == [
        "match_started",
        "player_team_changed",
        "player_kill",
    ]


def test_completed_match_has_no_priming_segment() -> None:
    plan = build_reward_priming_plan(
        [
            event("session_started"),
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

    assert plan.events == ()
    assert plan.reason == "match_inactive"
    assert plan.session_open
    assert not plan.match_active
    assert plan.boundary_index is None


def test_session_end_invalidates_shutdown_autostart() -> None:
    plan = build_reward_priming_plan(
        [
            event("session_started"),
            event(
                "match_started",
                game_mode="kh",
            ),
            event("match_ended"),
            event(
                "match_started",
                game_mode="kh",
            ),
            event("match_mutators"),
            event("match_info_complete"),
            event("session_ended"),
        ]
    )

    assert plan.events == ()
    assert plan.reason == "session_closed"
    assert not plan.session_open
    assert not plan.match_active


def test_restart_keeps_original_match_boundary() -> None:
    records = [
        event("session_started"),
        event(
            "match_started",
            game_mode="ctf",
        ),
        event(
            "player_team_changed",
            player_name="Noobnog",
            team="RED",
        ),
        event("match_restarted"),
        event("match_start_delay_ended"),
    ]

    plan = build_reward_priming_plan(records)

    assert plan.reason == "active_match"
    assert plan.boundary_index == 1

    assert event_types(plan.events) == [
        "match_started",
        "player_team_changed",
        "match_restarted",
        "match_start_delay_ended",
    ]


def test_new_session_discards_previous_context() -> None:
    plan = build_reward_priming_plan(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event(
                "player_team_changed",
                player_name="Noobnog",
                team="RED",
            ),
            event("session_started"),
        ]
    )

    assert plan.events == ()
    assert plan.reason == "no_match_start"
    assert plan.session_open
    assert not plan.match_active


def test_stream_without_session_records_is_supported() -> None:
    plan = build_reward_priming_plan(
        [
            event(
                "match_started",
                game_mode="dm",
            ),
            event(
                "player_now_playing",
                player_name="Noobnog",
                team="RED",
            ),
        ]
    )

    assert plan.reason == "active_match"
    assert plan.boundary_index == 0
    assert plan.selected_count == 2


def test_no_match_start_returns_empty_plan() -> None:
    plan = build_reward_priming_plan(
        [
            event("session_started"),
            event(
                "player_team_changed",
                player_name="Noobnog",
                team="YELLOW",
            ),
        ]
    )

    assert plan.events == ()
    assert plan.reason == "no_match_start"
    assert plan.boundary_index is None


def test_rejects_non_event_items() -> None:
    with pytest.raises(
        TypeError,
        match="only Event instances",
    ):
        build_reward_priming_plan(
            [
                event("session_started"),
                "not-an-event",
            ]
        )


def test_selected_events_reconstruct_mapper_state() -> None:
    records = [
        event("session_started"),
        event(
            "match_started",
            game_mode="kh",
        ),
        event(
            "player_team_changed",
            player_name="Noobnog",
            team="SPECTATOR",
        ),
        event(
            "player_team_changed",
            player_name="Noobnog",
            team="YELLOW",
        ),
        event(
            "player_now_playing",
            player_name="Noobnog",
            team="YELLOW",
        ),
        event("match_restarted"),
    ]

    plan = build_reward_priming_plan(records)
    mapper = RewardMapper("Noobnog")

    # Priming deliberately discards every generated ledger.
    for historical_event in plan.events:
        mapper.map_event(historical_event)

    assert mapper.current_mode is not None
    assert mapper.current_mode.value == "kh"
    assert mapper.controlled_team == "YELLOW"

    disconnect = mapper.map_event(
        event(
            "player_disconnected",
            player_name="Noobnog",
        )
    )

    assert disconnect.disconnect == -2.0
