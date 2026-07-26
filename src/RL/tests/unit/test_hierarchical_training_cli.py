"""Tests for the hierarchical behavior-cloning training CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    InspectionJSONLWriter,
    ObservationSnapshot,
    PolicyDecision,
)
from RL.tools.training.train_hierarchical_behavior_cloning import (
    main,
    parse_args,
)
from RL.training.imitation import (
    load_hierarchical_checkpoint,
)


FRAME_SHAPE = (4, 3, 90, 160)


def write_episode(
    root: Path,
) -> Path:
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_path = root / "episode.jsonl"
    frame_directory = root / "frames"
    frame_directory.mkdir(exist_ok=True)

    for step in range(2):
        np.save(
            frame_directory
            / f"policy_{step:06d}.npy",
            np.full(
                FRAME_SHAPE,
                0.1 + step * 0.1,
                dtype=np.float32,
            ),
            allow_pickle=False,
        )

    with InspectionJSONLWriter(
        episode_path
    ) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id="cli-episode",
                started_at=(
                    "2026-07-24T08:00:00+00:00"
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
                    "game_mode": "dm",
                },
            )
        )

        for step in range(2):
            action = (
                DiscreteAction.NO_OP
                if step == 0
                else DiscreteAction.FIRE
            )

            writer.write_transition(
                AgentTransition(
                    episode_id="cli-episode",
                    step_index=step,
                    observation=ObservationSnapshot(
                        tick=step,
                        telemetry=None,
                        raw_frame=None,
                        policy_frame=FrameSnapshot(
                            path=(
                                "frames/"
                                f"policy_{step:06d}.npy"
                            ),
                            shape=FRAME_SHAPE,
                            dtype="float32",
                            encoding="npy",
                            transform=(
                                "RGB; resize=160x90; "
                                "normalize=0..1; stack=4"
                            ),
                        ),
                    ),
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
                            "forward_axis": step,
                            "strafe_axis": 0,
                            "turn_delta_x": (
                                float(step)
                            ),
                            "look_delta_y": 0.0,
                            "fire": step == 1,
                            "jump": False,
                            "weapon_delta": 0,
                            "duration_ticks": 1,
                        },
                        "pressed_keycodes": [],
                        "pressed_buttons": (
                            [1]
                            if step == 1
                            else []
                        ),
                        "timestamp_ns": (
                            1_000_000_000
                            + step
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
                episode_id="cli-episode",
                ended_at=(
                    "2026-07-24T08:00:01+00:00"
                ),
                steps=2,
                terminated=False,
                truncated=True,
                outcome="frame_limit_reached",
            )
        )

    return episode_path


def common_arguments(
    episode: Path,
    output: Path,
) -> list[str]:
    return [
        str(episode),
        "--output-checkpoint",
        str(output),
        "--sequence-length",
        "2",
        "--stride",
        "1",
        "--batch-size",
        "1",
        "--epochs",
        "1",
        "--max-optimizer-steps",
        "1",
        "--log-every-steps",
        "1",
        "--device",
        "cpu",
        "--torch-threads",
        "1",
        "--visual-feature-dim",
        "16",
        "--previous-action-dim",
        "4",
        "--mode-embedding-dim",
        "4",
        "--recurrent-input-dim",
        "16",
        "--recurrent-hidden-dim",
        "16",
        "--recurrent-layers",
        "1",
    ]


def test_parse_args_defaults() -> None:
    args = parse_args(
        [
            "episode.jsonl",
            "--output-checkpoint",
            "output.pt",
        ]
    )

    assert args.sequence_length == 8
    assert args.stride == 4
    assert args.batch_size == 4
    assert args.device == "auto"
    assert args.resume_checkpoint is None
    assert not args.allow_dataset_change
    assert args.fire_positive_weight is None
    assert args.jump_positive_weight is None
    assert args.weapon_previous_weight is None
    assert args.weapon_neutral_weight is None
    assert args.weapon_next_weight is None


def test_parse_args_rejects_invalid_limit() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "episode.jsonl",
                "--output-checkpoint",
                "output.pt",
                "--max-optimizer-steps",
                "0",
            ]
        )


def test_cli_trains_saves_and_resumes(
    tmp_path: Path,
) -> None:
    episode = write_episode(
        tmp_path / "data"
    )

    first_path = tmp_path / "first.pt"

    assert main(
        common_arguments(
            episode,
            first_path,
        )
    ) == 0

    first = load_hierarchical_checkpoint(
        first_path
    )

    assert (
        first.trainer.optimizer_step_count
        == 1
    )
    assert first.metadata[
        "candidate_playable"
    ] is False
    assert first.metadata[
        "tactical_intent_loss_enabled"
    ] is False
    assert first.metadata[
        "final_evaluation"
    ]["fire_positive_count"] == 1

    second_path = tmp_path / "second.pt"

    resume_arguments = common_arguments(
        episode,
        second_path,
    )
    resume_arguments.extend(
        [
            "--resume-checkpoint",
            str(first_path),
        ]
    )

    assert main(resume_arguments) == 0

    second = load_hierarchical_checkpoint(
        second_path
    )

    assert (
        second.trainer.optimizer_step_count
        == 2
    )
    assert second.metadata[
        "starting_optimizer_step_count"
    ] == 1
    assert second.metadata[
        "run_optimizer_steps"
    ] == 1
    assert second.metadata[
        "resumed_from"
    ] == str(first_path.resolve())



def test_cli_persists_and_resumes_class_weights(
    tmp_path: Path,
) -> None:
    episode = write_episode(
        tmp_path / "weighted-data"
    )

    first_path = tmp_path / "weighted-first.pt"

    first_arguments = common_arguments(
        episode,
        first_path,
    )
    first_arguments.extend(
        [
            "--fire-positive-weight",
            "2.5",
            "--jump-positive-weight",
            "4.0",
            "--weapon-previous-weight",
            "6.0",
            "--weapon-neutral-weight",
            "1.0",
            "--weapon-next-weight",
            "5.0",
        ]
    )

    assert main(first_arguments) == 0

    first = load_hierarchical_checkpoint(
        first_path
    )

    assert (
        first.trainer.class_weights.fire_positive
        == 2.5
    )
    assert (
        first.trainer.class_weights.jump_positive
        == 4.0
    )
    assert (
        first.trainer.class_weights.weapon_previous
        == 6.0
    )
    assert (
        first.trainer.class_weights.weapon_neutral
        == 1.0
    )
    assert (
        first.trainer.class_weights.weapon_next
        == 5.0
    )

    final_evaluation = first.metadata[
        "final_evaluation"
    ]

    assert final_evaluation[
        "weapon_previous_positive_count"
    ] == 0
    assert final_evaluation[
        "weapon_neutral_positive_count"
    ] == 2
    assert final_evaluation[
        "weapon_next_positive_count"
    ] == 0

    second_path = tmp_path / "weighted-second.pt"

    resume_arguments = common_arguments(
        episode,
        second_path,
    )
    resume_arguments.extend(
        [
            "--resume-checkpoint",
            str(first_path),
        ]
    )

    assert main(resume_arguments) == 0

    second = load_hierarchical_checkpoint(
        second_path
    )

    assert (
        second.trainer.class_weights
        == first.trainer.class_weights
    )
