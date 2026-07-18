"""Tests for strict agent-inspection JSONL storage."""

import json
from pathlib import Path

import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.inspection import (
    AgentTransition,
    EpisodeEnded,
    EpisodeStarted,
    FrameSnapshot,
    GoalSnapshot,
    InspectionJSONLReader,
    InspectionJSONLWriter,
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
            health=100,
            armor=0,
            ammo=20,
            weapon="shotgun",
            alive=True,
            score=0,
            match_time_seconds=float(tick),
        ),
        raw_frame=FrameSnapshot(
            path=f"synthetic/raw/{tick:06d}.png",
            shape=(720, 1280, 3),
            dtype="uint8",
            encoding="png",
        ),
        policy_frame=FrameSnapshot(
            path=f"synthetic/policy/{tick:06d}.npy",
            shape=(4, 84, 84),
            dtype="float32",
            encoding="npy",
            transform="grayscale; resize=84x84; stack=4",
        ),
    )


def make_transition(
    episode_id: str,
    step_index: int,
) -> AgentTransition:
    return AgentTransition(
        episode_id=episode_id,
        step_index=step_index,
        observation=make_observation(step_index),
        decision=PolicyDecision(
            action=ActionCommand(
                action=DiscreteAction.TURN_RIGHT,
            ),
            policy_name="synthetic-policy",
            policy_version="v1",
            action_scores={
                "TURN_RIGHT": 0.8,
                "FORWARD": 0.2,
            },
        ),
        reward=0.0,
        next_observation=make_observation(
            step_index + 1
        ),
        terminated=False,
        truncated=False,
        goal=GoalSnapshot(
            goal_id=f"find-opponent-{step_index}",
            category="search",
            target=None,
            trigger="opponent_not_visible",
            progress=0.25,
            status="active",
            steps_active=step_index + 1,
        ),
        info={
            "synthetic": True,
        },
    )


def test_inspection_episode_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode.jsonl"
    episode_id = "synthetic-episode-001"

    with InspectionJSONLWriter(path) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id=episode_id,
                started_at="2026-07-18T14:30:00+00:00",
                metadata={
                    "curriculum_stage": "open_arena_1v1",
                    "bot_skill": 1,
                },
            )
        )
        writer.write_transition(
            make_transition(episode_id, 0)
        )
        writer.write_transition(
            make_transition(episode_id, 1)
        )
        writer.write_episode_ended(
            EpisodeEnded(
                episode_id=episode_id,
                ended_at="2026-07-18T14:31:00+00:00",
                steps=2,
                terminated=True,
                truncated=False,
                outcome="synthetic_complete",
                summary={
                    "reward_mode": "disabled",
                },
            )
        )

    records = InspectionJSONLReader(
        path
    ).read_episode()

    assert [
        record.type
        for record in records
    ] == [
        "episode_started",
        "agent_transition",
        "agent_transition",
        "episode_ended",
    ]

    assert records[1].data["goal"]["category"] == (
        "search"
    )
    assert records[1].data["reward"] == 0.0
    assert records[-1].data["steps"] == 2


def test_reader_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"type":"episode_started"',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        list(
            InspectionJSONLReader(
                path
            ).read_records()
        )


def test_reader_rejects_unknown_record_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "reward_farming",
                "data": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="record type",
    ):
        list(
            InspectionJSONLReader(
                path
            ).read_records()
        )


def test_episode_rejects_noncontiguous_steps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "steps.jsonl"
    episode_id = "synthetic-episode-002"

    with InspectionJSONLWriter(path) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id=episode_id,
                started_at="start",
            )
        )
        writer.write_transition(
            make_transition(episode_id, 0)
        )
        writer.write_transition(
            make_transition(episode_id, 2)
        )
        writer.write_episode_ended(
            EpisodeEnded(
                episode_id=episode_id,
                ended_at="end",
                steps=2,
                terminated=True,
                truncated=False,
            )
        )

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        InspectionJSONLReader(
            path
        ).read_episode()


def test_episode_rejects_mismatched_identifier(
    tmp_path: Path,
) -> None:
    path = tmp_path / "identifier.jsonl"

    with InspectionJSONLWriter(path) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id="episode-a",
                started_at="start",
            )
        )
        writer.write_transition(
            make_transition("episode-b", 0)
        )
        writer.write_episode_ended(
            EpisodeEnded(
                episode_id="episode-a",
                ended_at="end",
                steps=1,
                terminated=True,
                truncated=False,
            )
        )

    with pytest.raises(
        ValueError,
        match="mismatched",
    ):
        InspectionJSONLReader(
            path
        ).read_episode()


def test_episode_rejects_wrong_step_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "count.jsonl"
    episode_id = "synthetic-episode-003"

    with InspectionJSONLWriter(path) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id=episode_id,
                started_at="start",
            )
        )
        writer.write_transition(
            make_transition(episode_id, 0)
        )
        writer.write_episode_ended(
            EpisodeEnded(
                episode_id=episode_id,
                ended_at="end",
                steps=2,
                terminated=True,
                truncated=False,
            )
        )

    with pytest.raises(
        ValueError,
        match="step count",
    ):
        InspectionJSONLReader(
            path
        ).read_episode()
