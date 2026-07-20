"""Inspectable recording of one bounded neural-agent episode."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from RL.agents.policies.neural_policy import (
    NeuralPolicyAgent,
)
from RL.env.core.contracts import Environment
from RL.episodes.runner import (
    EpisodeRunResult,
    EpisodeTransition,
    run_bounded_episode,
)
from RL.inspection.contracts import (
    AgentTransition,
    FrameSnapshot,
    ObservationSnapshot,
    PolicyDecision,
)
from RL.inspection.jsonl import (
    EpisodeEnded,
    EpisodeStarted,
    InspectionJSONLWriter,
)
from RL.observations.contracts import Observation


@dataclass(frozen=True)
class InspectedNeuralEpisodeResult:
    """Result and storage paths from one inspected episode."""

    episode_id: str
    jsonl_path: Path
    frame_directory: Path
    run_result: EpisodeRunResult


def _timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _nonempty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{field_name} must be a nonempty string"
        )

    return value


def _checkpoint_string(
    value: str | Path | None,
) -> str | None:
    if value is None:
        return None

    text = str(value)

    if not text:
        raise ValueError(
            "checkpoint must be a nonempty path"
        )

    return text


def _frame_array(
    observation: Observation,
    *,
    expected_shape: tuple[int, int, int, int],
) -> np.ndarray:
    if not isinstance(observation, Observation):
        raise TypeError(
            "observation must be an Observation"
        )

    frame = observation.frame

    if isinstance(frame, np.ndarray):
        array = frame

    elif isinstance(frame, torch.Tensor):
        array = (
            frame.detach()
            .cpu()
            .numpy()
        )

    else:
        raise TypeError(
            "observation frame must be a NumPy "
            "array or torch.Tensor"
        )

    actual_shape = tuple(
        int(dimension)
        for dimension in array.shape
    )

    if actual_shape != expected_shape:
        raise ValueError(
            "unexpected observation frame shape: "
            f"expected {expected_shape}, "
            f"received {actual_shape}"
        )

    if not np.issubdtype(
        array.dtype,
        np.floating,
    ):
        raise TypeError(
            "observation frame must use "
            "a floating-point dtype"
        )

    if not bool(np.isfinite(array).all()):
        raise ValueError(
            "observation frame must contain "
            "only finite values"
        )

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


def _atomic_save_array(
    path: Path,
    array: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        "."
        + path.name
        + "."
        + uuid4().hex
        + ".tmp"
    )

    try:
        with temporary_path.open("wb") as file:
            np.save(
                file,
                array,
                allow_pickle=False,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )


class NeuralEpisodeInspectionRecorder:
    """Convert bounded neural transitions into inspection records."""

    def __init__(
        self,
        *,
        writer: InspectionJSONLWriter,
        agent: NeuralPolicyAgent,
        episode_id: str,
        jsonl_path: Path,
        frame_directory: Path,
        checkpoint: str | Path | None = None,
    ) -> None:
        if not isinstance(
            writer,
            InspectionJSONLWriter,
        ):
            raise TypeError(
                "writer must be an InspectionJSONLWriter"
            )

        if not isinstance(
            agent,
            NeuralPolicyAgent,
        ):
            raise TypeError(
                "agent must be a NeuralPolicyAgent"
            )

        self.writer = writer
        self.agent = agent
        self.episode_id = _nonempty_string(
            episode_id,
            field_name="episode_id",
        )
        self.jsonl_path = jsonl_path
        self.frame_directory = frame_directory
        self.checkpoint = _checkpoint_string(
            checkpoint
        )

        self._next_step_index = 0

    @property
    def steps(self) -> int:
        """Return the number of successfully written transitions."""

        return self._next_step_index

    def _relative_path(
        self,
        path: Path,
    ) -> str:
        return Path(
            os.path.relpath(
                path,
                start=self.jsonl_path.parent,
            )
        ).as_posix()

    def _observation_snapshot(
        self,
        observation: Observation,
        *,
        filename: str,
    ) -> ObservationSnapshot:
        array = _frame_array(
            observation,
            expected_shape=(
                self.agent.model.frame_shape
            ),
        )

        frame_path = (
            self.frame_directory
            / filename
        )

        _atomic_save_array(
            frame_path,
            array,
        )

        return ObservationSnapshot(
            tick=observation.tick,
            telemetry=observation.telemetry,
            raw_frame=None,
            policy_frame=FrameSnapshot(
                path=self._relative_path(
                    frame_path
                ),
                shape=tuple(
                    int(dimension)
                    for dimension in array.shape
                ),
                dtype=str(array.dtype),
                encoding="npy",
                transform=(
                    "RGB; resize=160x90; "
                    "normalize=0..1; stack=4"
                ),
            ),
        )

    def record(
        self,
        transition: EpisodeTransition,
    ) -> None:
        """Persist one transition and its frame arrays."""

        if not isinstance(
            transition,
            EpisodeTransition,
        ):
            raise TypeError(
                "transition must be an EpisodeTransition"
            )

        if transition.step_index != (
            self._next_step_index
        ):
            raise ValueError(
                "inspection transition indexes must "
                "be contiguous and begin at zero"
            )

        action_scores = dict(
            self.agent.last_action_scores
        )

        if len(action_scores) != (
            self.agent.model.action_count
        ):
            raise RuntimeError(
                "neural agent action scores are unavailable"
            )

        observation_snapshot = (
            self._observation_snapshot(
                transition.observation,
                filename=(
                    f"step_{transition.step_index:06d}"
                    "_observation.npy"
                ),
            )
        )

        next_observation_snapshot = (
            self._observation_snapshot(
                transition.next_observation,
                filename=(
                    f"step_{transition.step_index:06d}"
                    "_next_observation.npy"
                ),
            )
        )

        raw_reward_components = (
            transition.info.get(
                "reward_components",
                {},
            )
        )

        if not isinstance(
            raw_reward_components,
            Mapping,
        ):
            raise TypeError(
                "reward_components must be a mapping"
            )

        reward_components = {
            str(name): float(value)
            for name, value
            in raw_reward_components.items()
        }

        self.writer.write_transition(
            AgentTransition(
                episode_id=self.episode_id,
                step_index=(
                    transition.step_index
                ),
                observation=(
                    observation_snapshot
                ),
                decision=PolicyDecision(
                    action=transition.action,
                    policy_name=(
                        self.agent.policy_name
                    ),
                    policy_version=(
                        self.agent.policy_version
                    ),
                    checkpoint=self.checkpoint,
                    action_scores=action_scores,
                    deterministic=True,
                ),
                reward=transition.reward,
                next_observation=(
                    next_observation_snapshot
                ),
                terminated=(
                    transition.terminated
                ),
                truncated=(
                    transition.truncated
                ),
                goal=None,
                reward_components=(
                    reward_components
                ),
                info=dict(transition.info),
            )
        )

        self._next_step_index += 1


def run_inspected_neural_episode(
    environment: Environment,
    agent: NeuralPolicyAgent,
    *,
    max_steps: int,
    output_path: str | Path,
    seed: int | None = None,
    episode_id: str | None = None,
    checkpoint: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InspectedNeuralEpisodeResult:
    """Run and persist one strictly bounded neural episode."""

    if not isinstance(
        agent,
        NeuralPolicyAgent,
    ):
        raise TypeError(
            "agent must be a NeuralPolicyAgent"
        )

    jsonl_path = Path(output_path)

    if jsonl_path.exists():
        raise FileExistsError(
            str(jsonl_path)
        )

    resolved_episode_id = (
        episode_id
        if episode_id is not None
        else (
            "neural-episode-"
            + uuid4().hex
        )
    )

    resolved_episode_id = _nonempty_string(
        resolved_episode_id,
        field_name="episode_id",
    )

    resolved_checkpoint = _checkpoint_string(
        checkpoint
    )

    frame_directory = (
        jsonl_path.parent
        / (
            jsonl_path.stem
            + "_frames"
        )
    )

    if (
        frame_directory.exists()
        and any(frame_directory.iterdir())
    ):
        raise FileExistsError(
            str(frame_directory)
        )

    frame_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_metadata: Mapping[str, Any] = (
        {}
        if metadata is None
        else metadata
    )

    if not isinstance(
        raw_metadata,
        Mapping,
    ):
        raise TypeError(
            "metadata must be a mapping"
        )

    started_metadata = dict(
        raw_metadata
    )

    started_metadata.update(
        {
            "policy_name": (
                agent.policy_name
            ),
            "policy_version": (
                agent.policy_version
            ),
            "checkpoint": (
                resolved_checkpoint
            ),
            "frame_shape": list(
                agent.model.frame_shape
            ),
            "frame_encoding": "npy",
            "frame_references_are_real": True,
            "learning_phase": (
                "pre_training_inference"
            ),
        }
    )

    with InspectionJSONLWriter(
        jsonl_path
    ) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id=(
                    resolved_episode_id
                ),
                started_at=_timestamp(),
                metadata=started_metadata,
            )
        )

        recorder = (
            NeuralEpisodeInspectionRecorder(
                writer=writer,
                agent=agent,
                episode_id=(
                    resolved_episode_id
                ),
                jsonl_path=jsonl_path,
                frame_directory=(
                    frame_directory
                ),
                checkpoint=(
                    resolved_checkpoint
                ),
            )
        )

        try:
            run_result = run_bounded_episode(
                environment,
                agent,
                max_steps=max_steps,
                seed=seed,
                on_transition=recorder.record,
            )

        except Exception as error:
            try:
                writer.write_episode_ended(
                    EpisodeEnded(
                        episode_id=(
                            resolved_episode_id
                        ),
                        ended_at=_timestamp(),
                        steps=recorder.steps,
                        terminated=False,
                        truncated=True,
                        outcome="error",
                        summary={
                            "transition_count": (
                                recorder.steps
                            ),
                            "error_type": (
                                type(error).__name__
                            ),
                            "error_message": str(
                                error
                            ),
                        },
                    )
                )
            except Exception:
                pass

            raise

        outcome = (
            "terminated"
            if run_result.terminated
            else "truncated"
            if run_result.truncated
            else "completed"
        )

        writer.write_episode_ended(
            EpisodeEnded(
                episode_id=(
                    resolved_episode_id
                ),
                ended_at=_timestamp(),
                steps=run_result.steps,
                terminated=(
                    run_result.terminated
                ),
                truncated=(
                    run_result.truncated
                ),
                outcome=outcome,
                summary={
                    "transition_count": (
                        run_result.steps
                    ),
                    "total_reward": (
                        run_result.total_reward
                    ),
                    "termination_reason": (
                        run_result.termination_reason
                    ),
                    "frame_file_count": (
                        run_result.steps * 2
                    ),
                },
            )
        )

    return InspectedNeuralEpisodeResult(
        episode_id=resolved_episode_id,
        jsonl_path=jsonl_path,
        frame_directory=frame_directory,
        run_result=run_result,
    )
