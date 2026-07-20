"""Tests for strict offline demonstration datasets."""

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
    DemonstrationDataset,
    make_demonstration_dataloader,
)


FRAME_SHAPE = (
    4,
    3,
    90,
    160,
)


def write_episode(
    root: Path,
    name: str,
    *,
    action: DiscreteAction = (
        DiscreteAction.TURN_LEFT
    ),
    expert: bool = True,
    real_frames: bool = True,
) -> tuple[Path, Path]:
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_path = (
        root / f"{name}.jsonl"
    )

    frame_directory = (
        root / f"{name}_frames"
    )

    frame_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_path = (
        frame_directory
        / "step_000000.npy"
    )

    np.save(
        frame_path,
        np.full(
            FRAME_SHAPE,
            fill_value=0.25,
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    relative_frame_path = (
        frame_path.relative_to(root)
        .as_posix()
    )

    episode_id = f"{name}-episode"

    observation = ObservationSnapshot(
        tick=0,
        telemetry=None,
        raw_frame=None,
        policy_frame=FrameSnapshot(
            path=relative_frame_path,
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
                episode_id=episode_id,
                started_at=(
                    "2026-07-19T21:00:00+00:00"
                ),
                metadata={
                    "frame_shape": list(
                        FRAME_SHAPE
                    ),
                    "frame_encoding": "npy",
                    "frame_references_are_real": (
                        real_frames
                    ),
                    "expert_demonstration": (
                        expert
                    ),
                    "demonstration_source": (
                        "human"
                        if expert
                        else "biased_policy"
                    ),
                },
            )
        )

        writer.write_transition(
            AgentTransition(
                episode_id=episode_id,
                step_index=0,
                observation=observation,
                decision=PolicyDecision(
                    action=ActionCommand(
                        action=action,
                        duration_ticks=1,
                    ),
                    policy_name=(
                        "human-demonstration"
                        if expert
                        else "integration-policy"
                    ),
                    policy_version="v1",
                    action_scores={},
                    deterministic=True,
                ),
                reward=0.0,
                next_observation=None,
                terminated=True,
                truncated=False,
                reward_components={},
                info={},
            )
        )

        writer.write_episode_ended(
            EpisodeEnded(
                episode_id=episode_id,
                ended_at=(
                    "2026-07-19T21:00:01+00:00"
                ),
                steps=1,
                terminated=True,
                truncated=False,
                outcome="complete",
            )
        )

    return episode_path, frame_path


def mutate_transition(
    episode_path: Path,
    mutation: Callable[
        [dict[str, Any]],
        None,
    ],
) -> None:
    lines = episode_path.read_text(
        encoding="utf-8"
    ).splitlines()

    payload = json.loads(
        lines[1]
    )

    mutation(payload["data"])

    lines[1] = json.dumps(
        payload,
        separators=(",", ":"),
    )

    episode_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_dataset_loads_expert_sample(
    tmp_path: Path,
) -> None:
    episode_path, frame_path = (
        write_episode(
            tmp_path,
            "expert",
            action=DiscreteAction.FIRE,
        )
    )

    dataset = DemonstrationDataset(
        episode_path
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample.frames.shape == FRAME_SHAPE
    assert sample.frames.dtype == torch.float32
    assert sample.action_index.dtype == torch.long
    assert sample.action_index.ndim == 0
    assert sample.action_index.item() == int(
        DiscreteAction.FIRE
    )
    assert sample.duration_ticks == 1
    assert sample.episode_id == (
        "expert-episode"
    )
    assert sample.step_index == 0
    assert sample.source_episode_path == (
        episode_path.resolve()
    )
    assert sample.source_frame_path == (
        frame_path.resolve()
    )
    assert dataset.action_counts["FIRE"] == 1


def test_dataset_orders_paths_deterministically(
    tmp_path: Path,
) -> None:
    b_path, _ = write_episode(
        tmp_path,
        "b",
        action=DiscreteAction.FIRE,
    )

    a_path, _ = write_episode(
        tmp_path,
        "a",
        action=DiscreteAction.FORWARD,
    )

    dataset = DemonstrationDataset(
        [
            b_path,
            a_path,
        ]
    )

    assert [
        dataset[index].episode_id
        for index in range(len(dataset))
    ] == [
        "a-episode",
        "b-episode",
    ]


def test_dataloader_builds_deterministic_batch(
    tmp_path: Path,
) -> None:
    b_path, _ = write_episode(
        tmp_path,
        "b",
        action=DiscreteAction.FIRE,
    )

    a_path, _ = write_episode(
        tmp_path,
        "a",
        action=DiscreteAction.FORWARD,
    )

    dataset = DemonstrationDataset(
        [
            b_path,
            a_path,
        ]
    )

    loader = make_demonstration_dataloader(
        dataset,
        batch_size=2,
    )

    batch = next(iter(loader))

    assert batch.frames.shape == (
        2,
        *FRAME_SHAPE,
    )
    assert batch.frames.dtype == torch.float32
    assert batch.action_indices.dtype == (
        torch.long
    )
    assert batch.action_indices.tolist() == [
        int(DiscreteAction.FORWARD),
        int(DiscreteAction.FIRE),
    ]
    assert batch.duration_ticks.tolist() == [
        1,
        1,
    ]
    assert batch.episode_ids == (
        "a-episode",
        "b-episode",
    )
    assert batch.step_indices.tolist() == [
        0,
        0,
    ]


def test_dataset_rejects_nonexpert_by_default(
    tmp_path: Path,
) -> None:
    episode_path, _ = write_episode(
        tmp_path,
        "nonexpert",
        expert=False,
    )

    with pytest.raises(
        ValueError,
        match="expert human demonstration",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_allows_explicit_nonexpert(
    tmp_path: Path,
) -> None:
    episode_path, _ = write_episode(
        tmp_path,
        "nonexpert",
        expert=False,
    )

    dataset = DemonstrationDataset(
        episode_path,
        allow_nonexpert=True,
    )

    assert len(dataset) == 1


def test_dataset_rejects_nonreal_frames(
    tmp_path: Path,
) -> None:
    episode_path, _ = write_episode(
        tmp_path,
        "references",
        real_frames=False,
    )

    with pytest.raises(
        ValueError,
        match="real frame references",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_path_escape(
    tmp_path: Path,
) -> None:
    episode_root = (
        tmp_path / "episodes"
    )

    episode_path, _ = write_episode(
        episode_root,
        "escape",
    )

    outside_path = (
        tmp_path / "outside.npy"
    )

    np.save(
        outside_path,
        np.zeros(
            FRAME_SHAPE,
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    mutate_transition(
        episode_path,
        lambda data: data[
            "observation"
        ]["policy_frame"].update(
            {
                "path": "../outside.npy",
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="escapes",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_missing_frame(
    tmp_path: Path,
) -> None:
    episode_path, _ = write_episode(
        tmp_path,
        "missing",
    )

    mutate_transition(
        episode_path,
        lambda data: data[
            "observation"
        ]["policy_frame"].update(
            {
                "path": (
                    "missing_frames/"
                    "absent.npy"
                ),
            }
        ),
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_wrong_frame_shape(
    tmp_path: Path,
) -> None:
    episode_path, frame_path = write_episode(
        tmp_path,
        "shape",
    )

    np.save(
        frame_path,
        np.zeros(
            (4, 3, 84, 84),
            dtype=np.float32,
        ),
        allow_pickle=False,
    )

    with pytest.raises(
        ValueError,
        match="unexpected shape",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_wrong_frame_dtype(
    tmp_path: Path,
) -> None:
    episode_path, frame_path = write_episode(
        tmp_path,
        "dtype",
    )

    np.save(
        frame_path,
        np.zeros(
            FRAME_SHAPE,
            dtype=np.float64,
        ),
        allow_pickle=False,
    )

    with pytest.raises(
        TypeError,
        match="float32",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_nonfinite_frame(
    tmp_path: Path,
) -> None:
    episode_path, frame_path = write_episode(
        tmp_path,
        "nonfinite",
    )

    frame = np.zeros(
        FRAME_SHAPE,
        dtype=np.float32,
    )
    frame[0, 0, 0, 0] = np.inf

    np.save(
        frame_path,
        frame,
        allow_pickle=False,
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_action_mapping_mismatch(
    tmp_path: Path,
) -> None:
    episode_path, _ = write_episode(
        tmp_path,
        "action",
        action=DiscreteAction.TURN_LEFT,
    )

    mutate_transition(
        episode_path,
        lambda data: data[
            "decision"
        ]["action"].update(
            {
                "name": "FORWARD",
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_invalid_duration(
    tmp_path: Path,
) -> None:
    episode_path, _ = write_episode(
        tmp_path,
        "duration",
    )

    mutate_transition(
        episode_path,
        lambda data: data[
            "decision"
        ]["action"].update(
            {
                "duration_ticks": 0,
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        DemonstrationDataset(
            episode_path
        )


def test_dataset_rejects_object_array(
    tmp_path: Path,
) -> None:
    episode_path, frame_path = write_episode(
        tmp_path,
        "object",
    )

    np.save(
        frame_path,
        np.array(
            [object()],
            dtype=object,
        ),
    )

    with pytest.raises(
        ValueError,
        match="safely load",
    ):
        DemonstrationDataset(
            episode_path
        )
