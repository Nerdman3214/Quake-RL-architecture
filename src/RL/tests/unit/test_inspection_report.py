"""Tests for post-episode inspection reports."""

import json
from pathlib import Path

import pytest

from RL.inspection import InspectionRecord
from RL.inspection.report import (
    build_episode_report,
    render_episode_report_text,
    write_episode_report_files,
)


def make_records() -> list[InspectionRecord]:
    return [
        InspectionRecord(
            type="episode_started",
            data={
                "episode_id": "episode-001",
                "started_at": "start",
                "metadata": {
                    "curriculum_stage": (
                        "open_arena_1v1"
                    ),
                    "bot_skill": 1,
                    "learning_phase": (
                        "pre_training_inspection"
                    ),
                    "frame_references_are_real": (
                        False
                    ),
                },
            },
        ),
        InspectionRecord(
            type="agent_transition",
            data={
                "episode_id": "episode-001",
                "step_index": 0,
                "observation": {
                    "tick": 100,
                    "telemetry": {
                        "health": 100,
                        "armor": 25,
                        "ammo": 20,
                        "weapon": "shotgun",
                        "alive": True,
                        "score": 0,
                        "match_time_seconds": 10.0,
                    },
                    "raw_frame": {
                        "path": "raw/100.png",
                    },
                    "policy_frame": {
                        "path": "policy/100.npy",
                        "transform": "resize=84x84",
                    },
                },
                "decision": {
                    "action": {
                        "name": "TURN_RIGHT",
                    },
                },
                "reward": 0.0,
                "goal": {
                    "goal_id": "find-opponent-001",
                    "category": "search",
                    "target": None,
                    "trigger": "opponent_not_visible",
                    "progress": 0.5,
                    "status": "active",
                    "steps_active": 1,
                    "completion_reason": None,
                    "failure_reason": None,
                },
                "next_observation": {
                    "tick": 101,
                    "telemetry": {
                        "health": 90,
                        "armor": 25,
                        "ammo": 20,
                        "weapon": "shotgun",
                        "alive": True,
                        "score": 0,
                        "match_time_seconds": 10.1,
                    },
                },
                "terminated": False,
                "truncated": False,
                "reward_components": {},
                "info": {},
            },
        ),
        InspectionRecord(
            type="agent_transition",
            data={
                "episode_id": "episode-001",
                "step_index": 1,
                "observation": {
                    "tick": 101,
                    "telemetry": {
                        "health": 90,
                        "armor": 25,
                        "ammo": 20,
                        "weapon": "shotgun",
                        "alive": True,
                        "score": 0,
                        "match_time_seconds": 10.1,
                    },
                    "raw_frame": {
                        "path": "raw/101.png",
                    },
                    "policy_frame": {
                        "path": "policy/101.npy",
                        "transform": "resize=84x84",
                    },
                },
                "decision": {
                    "action": {
                        "name": "FIRE",
                    },
                },
                "reward": 0.0,
                "goal": {
                    "goal_id": "engage-001",
                    "category": "combat",
                    "target": "visible-opponent",
                    "trigger": "opponent_visible",
                    "progress": 1.0,
                    "status": "completed",
                    "steps_active": 2,
                    "completion_reason": (
                        "shot_attempted"
                    ),
                    "failure_reason": None,
                },
                "next_observation": None,
                "terminated": True,
                "truncated": False,
                "reward_components": {},
                "info": {},
            },
        ),
        InspectionRecord(
            type="episode_ended",
            data={
                "episode_id": "episode-001",
                "ended_at": "end",
                "steps": 2,
                "terminated": True,
                "truncated": False,
                "outcome": "complete",
                "summary": {
                    "reward_mode": "disabled",
                },
            },
        ),
    ]


def test_build_episode_report() -> None:
    report = build_episode_report(
        make_records()
    )

    assert report.episode_id == "episode-001"
    assert report.step_count == 2
    assert report.total_reward == 0.0
    assert report.action_counts == {
        "FIRE": 1,
        "TURN_RIGHT": 1,
    }
    assert report.goal_status_counts == {
        "active": 1,
        "completed": 1,
    }
    assert report.goal_category_counts == {
        "combat": 1,
        "search": 1,
    }
    assert report.telemetry_delta["health"] == (
        -10.0
    )
    assert report.telemetry_delta[
        "match_time_seconds"
    ] == 0.1
    assert report.frame_references[0][
        "raw_frame"
    ] == "raw/100.png"
    assert report.important_transitions[0][
        "reasons"
    ] == ["health_changed"]


def test_text_report_is_readable() -> None:
    text = render_episode_report_text(
        build_episode_report(make_records())
    )

    assert (
        "AGENT EPISODE INSPECTION REPORT"
        in text
    )
    assert "goal=find-opponent-001" in text
    assert "action=FIRE" in text
    assert "raw_frame=raw/100.png" in text
    assert (
        "frame_references_are_real=False"
        in text
    )


def test_write_report_files(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "report.txt"
    json_path = tmp_path / "report.json"

    write_episode_report_files(
        build_episode_report(make_records()),
        text_path=text_path,
        json_path=json_path,
    )

    assert text_path.exists()
    assert json_path.exists()

    payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["episode_id"] == (
        "episode-001"
    )
    assert payload["action_counts"]["FIRE"] == 1


def test_report_rejects_empty_records() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_episode_report([])


def test_report_requires_transition() -> None:
    records = [
        make_records()[0],
        make_records()[-1],
    ]

    with pytest.raises(
        ValueError,
        match="at least one transition",
    ):
        build_episode_report(records)
