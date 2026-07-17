"""Tests for conservative adaptive matchmaking."""

from pathlib import Path

from RL.matchmaking import (
    AdaptiveState,
    MatchPerformance,
    load_state,
    save_state,
    update_state,
)


def make_performance(
    *,
    won: bool,
    score_margin: float,
    combat_score: float,
    objective_score: float,
    discipline_penalty: float = 0.0,
) -> MatchPerformance:
    return MatchPerformance(
        mode="kh",
        won=won,
        score_margin=score_margin,
        combat_score=combat_score,
        objective_score=objective_score,
        discipline_penalty=discipline_penalty,
        bot_skill=4,
        bot_count=7,
        session_id="test-session",
    )


def test_missing_state_uses_defaults(
    tmp_path: Path,
) -> None:
    state = load_state(
        tmp_path / "agent_rating.json"
    )

    assert state.rating == 4.0
    assert state.current_skill == 4
    assert state.matches == 0
    assert state.recent_matches == ()


def test_state_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_rating.json"

    state = update_state(
        AdaptiveState(),
        make_performance(
            won=True,
            score_margin=0.4,
            combat_score=0.2,
            objective_score=0.6,
        ),
    )

    save_state(state, path)
    loaded = load_state(path)

    assert loaded == state


def test_single_win_does_not_raise_skill() -> None:
    state = AdaptiveState(
        rating=7.0,
        current_skill=7,
    )

    updated = update_state(
        state,
        make_performance(
            won=True,
            score_margin=0.8,
            combat_score=0.8,
            objective_score=0.8,
        ),
    )

    assert updated.current_skill == 7
    assert updated.matches == 1


def test_sustained_balanced_success_raises_once() -> None:
    state = AdaptiveState()

    performance = make_performance(
        won=True,
        score_margin=0.8,
        combat_score=0.6,
        objective_score=0.8,
        discipline_penalty=0.1,
    )

    for _ in range(3):
        state = update_state(
            state,
            performance,
        )

    assert state.current_skill == 5
    assert state.matches == 3
    assert state.last_adjustment_match == 3


def test_carried_wins_do_not_raise_skill() -> None:
    state = AdaptiveState()

    carried_win = make_performance(
        won=True,
        score_margin=0.6,
        combat_score=-0.6,
        objective_score=0.0,
        discipline_penalty=0.3,
    )

    for _ in range(3):
        state = update_state(
            state,
            carried_win,
        )

    assert state.current_skill == 4


def test_sustained_losses_lower_skill() -> None:
    state = AdaptiveState()

    weak_loss = make_performance(
        won=False,
        score_margin=-0.8,
        combat_score=-0.6,
        objective_score=-0.4,
        discipline_penalty=0.2,
    )

    for _ in range(3):
        state = update_state(
            state,
            weak_loss,
        )

    assert state.current_skill == 3
    assert state.last_adjustment_match == 3
