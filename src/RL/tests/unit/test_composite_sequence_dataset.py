"""Tests for recurrent composite demonstration datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.inspection import (
    AgentTransition,
    EpisodeEnded,
    EpisodeStarted,
    FrameSnapshot,
    InspectionJSONLWriter,
    ObservationSnapshot,
    PolicyDecision,
)
from RL.training.imitation import (
    CompositeSequenceDataset,
    GAME_MODE_TO_INDEX,
    make_composite_sequence_dataloader,
)


FRAME_SHAPE = (4, 3, 90, 160)


def write_composite_episode(
    root: Path,
    *,
    step_count: int = 12,
    mode: str = "dm",
) -> Path:
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_path = root / "episode.jsonl"
    frames = root / "frames"
    frames.mkdir(exist_ok=True)

    frame_path = frames / "policy.npy"

    np.save(
        frame_path,
        np.full(
            FRAME_SHAPE,
            0.25,
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    observation = ObservationSnapshot(
        tick=0,
        telemetry=None,
        raw_frame=None,
        policy_frame=FrameSnapshot(
            path="frames/policy.npy",
            shape=FRAME_SHAPE,
            dtype="float32",
            encoding="npy",
            transform=(
                "RGB; resize=160x90; "
                "normalize=0..1; stack=4"
            ),
        ),
    )

    with InspectionJSONLWriter(
        episode_path
    ) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id="composite-episode",
                started_at=(
                    "2026-07-24T03:00:00+00:00"
                ),
                metadata={
                    "frame_shape": list(
                        FRAME_SHAPE
                    ),
                    "frame_encoding": "npy",
                    "frame_references_are_real": True,
                    "expert_demonstration": True,
                    "demonstration_source": "human",
                    "composite_action_labels_available": True,
                    "label_authority": (
                        "xinput2-raw-events"
                    ),
                    "game_mode": mode,
                },
            )
        )

        for step in range(step_count):
            forward_axis = (-1, 0, 1)[
                step % 3
            ]
            strafe_axis = (1, 0, -1)[
                step % 3
            ]
            fire = step % 4 == 0
            jump = step % 5 == 0
            weapon_delta = (-1, 0, 1)[
                step % 3
            ]

            action = (
                DiscreteAction.FIRE
                if fire
                else DiscreteAction.FORWARD
            )

            writer.write_transition(
                AgentTransition(
                    episode_id=(
                        "composite-episode"
                    ),
                    step_index=step,
                    observation=observation,
                    decision=PolicyDecision(
                        action=ActionCommand(
                            action=action,
                            duration_ticks=1,
                        ),
                        policy_name=(
                            "human-demonstration"
                        ),
                        policy_version="v1",
                        action_scores={},
                        deterministic=True,
                    ),
                    reward=0.0,
                    next_observation=None,
                    terminated=False,
                    truncated=False,
                    reward_components={},
                    info={
                        "human_input": {
                            "forward_axis": (
                                forward_axis
                            ),
                            "strafe_axis": (
                                strafe_axis
                            ),
                            "turn_delta_x": (
                                float(step)
                            ),
                            "look_delta_y": (
                                float(-step)
                            ),
                            "fire": fire,
                            "jump": jump,
                            "weapon_delta": (
                                weapon_delta
                            ),
                            "duration_ticks": 1,
                        },
                        "pressed_keycodes": [],
                        "pressed_buttons": [],
                        "timestamp_ns": (
                            1_000_000_000
                            + step * 200_000_000
                        ),
                        "label_authority": (
                            "xinput2-raw-events"
                        ),
                        "derived_discrete_action": (
                            action.name
                        ),
                        "derived_label_is_primary": (
                            False
                        ),
                    },
                )
            )

        writer.write_episode_ended(
            EpisodeEnded(
                episode_id="composite-episode",
                ended_at=(
                    "2026-07-24T03:01:00+00:00"
                ),
                steps=step_count,
                terminated=False,
                truncated=True,
                outcome="frame_limit_reached",
            )
        )

    return episode_path


def mutate_record(
    episode_path: Path,
    *,
    record_type: str,
    mutation: Callable[
        [dict[str, Any]],
        None,
    ],
    occurrence: int = 0,
) -> None:
    lines = episode_path.read_text(
        encoding="utf-8"
    ).splitlines()

    found = 0

    for index, line in enumerate(lines):
        payload = json.loads(line)

        if payload.get("type") != record_type:
            continue

        if found == occurrence:
            mutation(payload["data"])
            lines[index] = json.dumps(
                payload,
                separators=(",", ":"),
            )
            break

        found += 1
    else:
        raise AssertionError(
            f"record not found: {record_type}"
        )

    episode_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_dataset_builds_expected_sequence(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path,
        step_count=12,
    )

    dataset = CompositeSequenceDataset(
        episode,
        sequence_length=4,
        stride=2,
    )

    assert len(dataset) == 5

    sample = dataset[0]

    assert sample.frames.shape == (
        4,
        *FRAME_SHAPE,
    )
    assert sample.previous_action_features.shape == (
        4,
        8,
    )
    assert sample.forward_classes.tolist() == [
        0,
        1,
        2,
        0,
    ]
    assert sample.strafe_classes.tolist() == [
        2,
        1,
        0,
        2,
    ]
    assert sample.weapon_classes.tolist() == [
        0,
        1,
        2,
        0,
    ]
    assert sample.previous_action_features[
        0
    ].tolist() == [0.0] * 8
    assert sample.previous_action_features[
        1,
        0,
    ].item() == -1.0
    assert sample.mode_index.item() == (
        GAME_MODE_TO_INDEX["dm"]
    )
    assert sample.valid_mask.tolist() == [
        True,
        True,
        True,
        True,
    ]


def test_sixty_steps_produce_fourteen_windows(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path,
        step_count=60,
    )

    dataset = CompositeSequenceDataset(
        episode,
        sequence_length=8,
        stride=4,
    )

    assert len(dataset) == 14
    assert dataset[-1].start_step_index == 52


def test_dataloader_preserves_sequence_order(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path,
        step_count=12,
    )

    dataset = CompositeSequenceDataset(
        episode,
        sequence_length=4,
        stride=2,
    )

    loader = make_composite_sequence_dataloader(
        dataset,
        batch_size=2,
    )

    batch = next(iter(loader))

    assert batch.frames.shape == (
        2,
        4,
        *FRAME_SHAPE,
    )
    assert batch.previous_action_features.shape == (
        2,
        4,
        8,
    )
    assert batch.mode_indices.tolist() == [
        0,
        0,
    ]
    assert batch.start_step_indices.tolist() == [
        0,
        2,
    ]
    assert batch.valid_mask.dtype == torch.bool


def test_dataset_rejects_missing_composite_flag(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path
    )

    mutate_record(
        episode,
        record_type="episode_started",
        mutation=lambda data: data[
            "metadata"
        ].update(
            {
                "composite_action_labels_available": (
                    False
                ),
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="composite action labels",
    ):
        CompositeSequenceDataset(episode)


def test_dataset_rejects_invalid_axis(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path
    )

    mutate_record(
        episode,
        record_type="agent_transition",
        mutation=lambda data: data[
            "info"
        ]["human_input"].update(
            {
                "forward_axis": 2,
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="axis value",
    ):
        CompositeSequenceDataset(episode)


def test_dataset_rejects_nonmonotonic_timestamp(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path
    )

    mutate_record(
        episode,
        record_type="agent_transition",
        occurrence=2,
        mutation=lambda data: data[
            "info"
        ].update(
            {
                "timestamp_ns": 1,
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="timestamps must be monotonic",
    ):
        CompositeSequenceDataset(episode)


def test_dataset_rejects_noncontiguous_step(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path
    )

    mutate_record(
        episode,
        record_type="agent_transition",
        occurrence=2,
        mutation=lambda data: data.update(
            {
                "step_index": 8,
            }
        ),
    )

    with pytest.raises(ValueError):
        CompositeSequenceDataset(episode)


def test_dataset_rejects_unsupported_mode(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path
    )

    mutate_record(
        episode,
        record_type="episode_started",
        mutation=lambda data: data[
            "metadata"
        ].update(
            {
                "game_mode": "unknown-mode",
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="unsupported game mode",
    ):
        CompositeSequenceDataset(episode)


def test_dataset_rejects_primary_discrete_label(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path
    )

    mutate_record(
        episode,
        record_type="agent_transition",
        mutation=lambda data: data[
            "info"
        ].update(
            {
                "derived_label_is_primary": True,
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="must not be primary",
    ):
        CompositeSequenceDataset(episode)


def test_dataset_rejects_short_episode(
    tmp_path: Path,
) -> None:
    episode = write_composite_episode(
        tmp_path,
        step_count=3,
    )

    with pytest.raises(
        ValueError,
        match="no complete composite sequences",
    ):
        CompositeSequenceDataset(
            episode,
            sequence_length=8,
        )



def test_telemetry_wrapper_extracts_features_and_masks_prespawn(
    tmp_path: Path,
) -> None:
    import json

    from RL.training.imitation.telemetry_sequence_dataset import (
        TelemetryCompositeSequenceDataset,
        make_telemetry_composite_sequence_dataloader,
    )

    episode = write_composite_episode(
        tmp_path / "telemetry",
        step_count=8,
    )

    records = [
        json.loads(line)
        for line in episode.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    transition_index = 0

    for record in records:
        if record.get("type") != "agent_transition":
            continue

        data = record["data"]

        pre_spawn = transition_index == 1

        data["observation"]["telemetry"] = {
            "health": (
                -666 if pre_spawn else 100
            ),
            "armor": 25,
            "ammo": 15,
            "weapon": "shotgun",
            "alive": not pre_spawn,
            "score": 2,
            "match_time_seconds": 12.0,
        }
        data["info"]["telemetry_sync"] = {
            "fresh": True,
            "pre_spawn": pre_spawn,
            "weapon_id": 2,
        }

        transition_index += 1

    episode.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = TelemetryCompositeSequenceDataset(
        [episode],
        sequence_length=4,
        stride=4,
        telemetry_weapon_count=64,
    )

    sample = dataset[0]

    assert sample.telemetry_features.shape == (
        4,
        8,
    )
    assert (
        sample.telemetry_features[0, 7].item()
        == 1.0
    )
    assert (
        sample.telemetry_weapon_indices[0].item()
        == 3
    )
    assert sample.base_sample.valid_mask.tolist() == [
        True,
        True,
        True,
        True,
    ]
    assert sample.telemetry_trainable_mask.tolist() == [
        True,
        False,
        True,
        True,
    ]

    loader = (
        make_telemetry_composite_sequence_dataloader(
            dataset,
            batch_size=1,
        )
    )
    batch = next(iter(loader))

    assert batch.telemetry_features.shape == (
        1,
        4,
        8,
    )
    assert batch.telemetry_weapon_indices.shape == (
        1,
        4,
    )
    assert batch.valid_mask.tolist() == [
        [
            True,
            False,
            True,
            True,
        ]
    ]
