"""Tests for readable agent inspection records."""

import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.inspection import (
    AgentTransition,
    FrameSnapshot,
    GoalSnapshot,
    ObservationSnapshot,
    PolicyDecision,
)
from RL.observations.contracts import PlayerTelemetry


def make_observation(
    tick: int,
) -> ObservationSnapshot:
    return ObservationSnapshot(
        tick=tick,
        telemetry=PlayerTelemetry(
            health=78,
            armor=25,
            ammo=16,
            weapon="hagar",
            alive=True,
            score=42,
            match_time_seconds=31.5,
        ),
        raw_frame=FrameSnapshot(
            path="raw/frame_000184.png",
            shape=(720, 1280, 3),
            dtype="uint8",
            encoding="png",
        ),
        policy_frame=FrameSnapshot(
            path="policy/frame_000184.npy",
            shape=(4, 84, 84),
            dtype="float32",
            encoding="npy",
            transform=(
                "resize=84x84; grayscale; "
                "normalize=0..1; stack=4"
            ),
        ),
    )


def test_transition_exposes_what_policy_saw() -> None:
    transition = AgentTransition(
        episode_id="episode_000012",
        step_index=184,
        observation=make_observation(18422),
        decision=PolicyDecision(
            action=ActionCommand(
                action=DiscreteAction.FORWARD,
                duration_ticks=2,
            ),
            policy_name="scripted-smoke-policy",
            policy_version="v1",
            action_scores={
                "FORWARD": 0.84,
                "FIRE": 0.68,
            },
            deterministic=True,
        ),
        reward=1.0,
        reward_components={
            "combat": 1.0,
            "objective": 0.0,
        },
        next_observation=make_observation(18424),
        terminated=False,
        truncated=False,
        goal=GoalSnapshot(
            goal_id="engage-opponent-001",
            category="combat",
            target="visible-opponent",
            trigger="opponent_visible",
            progress=0.61,
            status="active",
            steps_active=18,
        ),
        info={
            "match_index": 9,
            "mode": "kh",
        },
    )

    record = transition.to_record()

    assert record["episode_id"] == "episode_000012"
    assert record["step_index"] == 184

    observation = record["observation"]

    assert observation["raw_frame"]["path"] == (
        "raw/frame_000184.png"
    )
    assert observation["policy_frame"]["path"] == (
        "policy/frame_000184.npy"
    )
    assert observation["policy_frame"]["shape"] == [
        4,
        84,
        84,
    ]
    assert observation["telemetry"]["health"] == 78

    decision = record["decision"]

    assert decision["action"]["name"] == "FORWARD"
    assert decision["action"]["duration_ticks"] == 2
    assert decision["action_scores"]["FIRE"] == 0.68

    assert record["reward"] == 1.0
    assert record["reward_components"]["combat"] == 1.0

    goal = record["goal"]

    assert goal["goal_id"] == "engage-opponent-001"
    assert goal["category"] == "combat"
    assert goal["target"] == "visible-opponent"
    assert goal["trigger"] == "opponent_visible"
    assert goal["progress"] == 0.61
    assert goal["status"] == "active"
    assert goal["steps_active"] == 18
    assert record["next_observation"]["tick"] == 18424
    assert record["terminated"] is False


def test_frame_snapshot_rejects_invalid_shape() -> None:
    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        FrameSnapshot(
            path="frame.png",
            shape=(84, 0, 3),
            dtype="uint8",
            encoding="png",
        )


def test_transition_rejects_negative_step() -> None:
    with pytest.raises(
        ValueError,
        match="step index",
    ):
        AgentTransition(
            episode_id="episode_000001",
            step_index=-1,
            observation=make_observation(0),
            decision=PolicyDecision(
                action=ActionCommand(
                    action=DiscreteAction.NO_OP,
                ),
                policy_name="test-policy",
                policy_version="v1",
            ),
            reward=0.0,
            next_observation=None,
            terminated=True,
            truncated=False,
        )



def test_goal_snapshot_serializes_completion() -> None:
    goal = GoalSnapshot(
        goal_id="take-cover-004",
        category="survival",
        target="nearest-cover",
        trigger="health_below_threshold",
        progress=1.0,
        status="completed",
        steps_active=12,
        completion_reason="line_of_sight_broken",
    )

    record = goal.to_record()

    assert record == {
        "goal_id": "take-cover-004",
        "category": "survival",
        "target": "nearest-cover",
        "trigger": "health_below_threshold",
        "progress": 1.0,
        "status": "completed",
        "steps_active": 12,
        "completion_reason": "line_of_sight_broken",
        "failure_reason": None,
    }


def test_goal_snapshot_rejects_invalid_progress() -> None:
    with pytest.raises(
        ValueError,
        match="progress",
    ):
        GoalSnapshot(
            goal_id="find-opponent-001",
            category="search",
            target=None,
            trigger="opponent_not_visible",
            progress=1.25,
            status="active",
            steps_active=3,
        )


def test_goal_snapshot_rejects_unknown_status() -> None:
    with pytest.raises(
        ValueError,
        match="status",
    ):
        GoalSnapshot(
            goal_id="find-opponent-001",
            category="search",
            target=None,
            trigger="opponent_not_visible",
            progress=0.2,
            status="reward_farming",
            steps_active=3,
        )
