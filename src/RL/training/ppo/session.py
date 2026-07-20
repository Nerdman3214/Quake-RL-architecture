"""Strictly bounded and auditable PPO training sessions."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from RL.agents.policies.actor_critic_agent import (
    ActorCriticPolicyAgent,
)
from RL.env.core.contracts import Environment
from RL.training.ppo.checkpoint import (
    PPOTrainingProgress,
    save_ppo_training_checkpoint,
)
from RL.training.ppo.collector import (
    CollectedRolloutResult,
    collect_bounded_rollout,
)
from RL.training.ppo.core import (
    PPOBatchResult,
    PPOMetrics,
    PPOTrainer,
)


EnvironmentFactory = Callable[
    [int],
    Environment,
]


def _positive_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )

    return value


def _nonnegative_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a nonnegative integer"
        )

    return value


def _finite_number(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return float(value)


def _probability(
    value: object,
    *,
    field_name: str,
) -> float:
    number = _finite_number(
        value,
        field_name=field_name,
    )

    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{field_name} must be between zero and one"
        )

    return number


def _json_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any]:
    normalized = dict(value or {})

    try:
        json.dumps(
            normalized,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{field_name} must be JSON-compatible"
        ) from error

    return normalized


def _output_path(
    value: str | Path | None,
    *,
    field_name: str,
) -> Path | None:
    if value is None:
        return None

    path = Path(value)

    if path.exists():
        raise FileExistsError(
            f"{field_name} already exists: {path}"
        )

    return path


@dataclass(frozen=True)
class PPOTrainingSessionConfig:
    """Hard limits and safeguards for one PPO session."""

    rollout_count: int
    max_steps_per_rollout: int
    training_enabled: bool = False
    updates_per_rollout: int = 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    bootstrap_on_environment_truncation: bool = False
    max_absolute_kl: float = 0.05
    stop_on_kl: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rollout_count",
            _positive_integer(
                self.rollout_count,
                field_name="rollout_count",
            ),
        )

        object.__setattr__(
            self,
            "max_steps_per_rollout",
            _positive_integer(
                self.max_steps_per_rollout,
                field_name="max_steps_per_rollout",
            ),
        )

        object.__setattr__(
            self,
            "updates_per_rollout",
            _nonnegative_integer(
                self.updates_per_rollout,
                field_name="updates_per_rollout",
            ),
        )

        if not isinstance(
            self.training_enabled,
            bool,
        ):
            raise TypeError(
                "training_enabled must be bool"
            )

        if (
            self.training_enabled
            and self.updates_per_rollout == 0
        ):
            raise ValueError(
                "training sessions require at least "
                "one update per rollout"
            )

        if (
            not self.training_enabled
            and self.updates_per_rollout != 0
        ):
            raise ValueError(
                "evaluation-only sessions cannot "
                "perform optimizer updates"
            )

        object.__setattr__(
            self,
            "gamma",
            _probability(
                self.gamma,
                field_name="gamma",
            ),
        )

        object.__setattr__(
            self,
            "gae_lambda",
            _probability(
                self.gae_lambda,
                field_name="gae_lambda",
            ),
        )

        if not isinstance(
            self.normalize_advantages,
            bool,
        ):
            raise TypeError(
                "normalize_advantages must be bool"
            )

        if not isinstance(
            self.bootstrap_on_environment_truncation,
            bool,
        ):
            raise TypeError(
                "bootstrap_on_environment_truncation "
                "must be bool"
            )

        maximum_kl = _finite_number(
            self.max_absolute_kl,
            field_name="max_absolute_kl",
        )

        if maximum_kl <= 0.0:
            raise ValueError(
                "max_absolute_kl must be positive"
            )

        object.__setattr__(
            self,
            "max_absolute_kl",
            maximum_kl,
        )

        if not isinstance(
            self.stop_on_kl,
            bool,
        ):
            raise TypeError(
                "stop_on_kl must be bool"
            )

        if (
            self.seed is not None
            and (
                isinstance(self.seed, bool)
                or not isinstance(self.seed, int)
            )
        ):
            raise TypeError(
                "seed must be an integer or None"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "rollout_count": self.rollout_count,
            "max_steps_per_rollout": (
                self.max_steps_per_rollout
            ),
            "training_enabled": (
                self.training_enabled
            ),
            "updates_per_rollout": (
                self.updates_per_rollout
            ),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "normalize_advantages": (
                self.normalize_advantages
            ),
            "bootstrap_on_environment_truncation": (
                self.bootstrap_on_environment_truncation
            ),
            "max_absolute_kl": (
                self.max_absolute_kl
            ),
            "stop_on_kl": self.stop_on_kl,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PPOUpdateAudit:
    """One evaluation or optimizer operation."""

    rollout_index: int
    operation_index: int
    result: PPOBatchResult

    def __post_init__(self) -> None:
        _nonnegative_integer(
            self.rollout_index,
            field_name="rollout_index",
        )

        _nonnegative_integer(
            self.operation_index,
            field_name="operation_index",
        )

        if not isinstance(
            self.result,
            PPOBatchResult,
        ):
            raise TypeError(
                "result must be a PPOBatchResult"
            )

    def to_record(self) -> dict[str, Any]:
        metrics = self.result.metrics

        return {
            "rollout_index": self.rollout_index,
            "operation_index": (
                self.operation_index
            ),
            "optimizer_step": (
                self.result.optimizer_step
            ),
            "optimizer_step_count": (
                self.result.optimizer_step_count
            ),
            "gradient_norm": (
                self.result.gradient_norm
            ),
            "metrics": {
                "total_loss": metrics.total_loss,
                "policy_loss": metrics.policy_loss,
                "value_loss": metrics.value_loss,
                "entropy": metrics.entropy,
                "approximate_kl": (
                    metrics.approximate_kl
                ),
                "clip_fraction": (
                    metrics.clip_fraction
                ),
                "explained_variance": (
                    metrics.explained_variance
                ),
                "sample_count": (
                    metrics.sample_count
                ),
            },
        }


@dataclass(frozen=True)
class PPORolloutAudit:
    """Auditable summary of one collected rollout."""

    rollout_index: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    termination_reason: str
    bootstrap_value: float
    action_counts: Mapping[str, int]
    operations: tuple[PPOUpdateAudit, ...]

    def __post_init__(self) -> None:
        _nonnegative_integer(
            self.rollout_index,
            field_name="rollout_index",
        )

        _positive_integer(
            self.steps,
            field_name="steps",
        )

        _finite_number(
            self.total_reward,
            field_name="total_reward",
        )

        _finite_number(
            self.bootstrap_value,
            field_name="bootstrap_value",
        )

        if not isinstance(
            self.terminated,
            bool,
        ):
            raise TypeError(
                "terminated must be bool"
            )

        if not isinstance(
            self.truncated,
            bool,
        ):
            raise TypeError(
                "truncated must be bool"
            )

        if (
            not isinstance(
                self.termination_reason,
                str,
            )
            or not self.termination_reason
        ):
            raise ValueError(
                "termination_reason must not be empty"
            )

        normalized_counts: dict[str, int] = {}

        for action_name, count in (
            self.action_counts.items()
        ):
            if (
                not isinstance(action_name, str)
                or not action_name
            ):
                raise ValueError(
                    "action names must not be empty"
                )

            normalized_counts[action_name] = (
                _nonnegative_integer(
                    count,
                    field_name=(
                        f"action_counts.{action_name}"
                    ),
                )
            )

        object.__setattr__(
            self,
            "action_counts",
            MappingProxyType(
                normalized_counts
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "rollout_index": self.rollout_index,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": (
                self.termination_reason
            ),
            "bootstrap_value": (
                self.bootstrap_value
            ),
            "action_counts": dict(
                self.action_counts
            ),
            "operations": [
                operation.to_record()
                for operation in self.operations
            ],
        }


@dataclass(frozen=True)
class PPOTrainingSessionResult:
    """Complete result of one bounded session."""

    config: PPOTrainingSessionConfig
    progress: PPOTrainingProgress
    rollouts: tuple[PPORolloutAudit, ...]
    stopped_early: bool
    stop_reason: str
    before_checkpoint_path: Path | None
    after_checkpoint_path: Path | None
    audit_path: Path | None

    @property
    def completed_rollouts(self) -> int:
        return len(self.rollouts)

    @property
    def environment_steps(self) -> int:
        return sum(
            rollout.steps
            for rollout in self.rollouts
        )

    @property
    def optimizer_operations(self) -> int:
        return sum(
            int(operation.result.optimizer_step)
            for rollout in self.rollouts
            for operation in rollout.operations
        )


class _JSONLAuditWriter:
    def __init__(
        self,
        path: Path | None,
    ) -> None:
        self.path = path
        self._file = None

        if path is not None:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._file = path.open(
                "x",
                encoding="utf-8",
            )

    def write(
        self,
        record_type: str,
        data: Mapping[str, Any],
    ) -> None:
        if self._file is None:
            return

        record = {
            "type": record_type,
            "data": dict(data),
        }

        self._file.write(
            json.dumps(
                record,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

        self._file.flush()
        os.fsync(
            self._file.fileno()
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _episode_completed(
    rollout: CollectedRolloutResult,
) -> bool:
    return (
        rollout.terminated
        or (
            rollout.truncated
            and rollout.termination_reason
            != "max_steps_reached"
        )
    )


def run_bounded_ppo_training_session(
    environment_factory: EnvironmentFactory,
    agent: ActorCriticPolicyAgent,
    trainer: PPOTrainer,
    config: PPOTrainingSessionConfig,
    *,
    progress: PPOTrainingProgress | None = None,
    before_checkpoint_path: (
        str | Path | None
    ) = None,
    after_checkpoint_path: (
        str | Path | None
    ) = None,
    audit_path: str | Path | None = None,
    policy_name: str = (
        "xonotic-visual-actor-critic"
    ),
    policy_version: str = "v1",
    checkpoint_metadata: (
        Mapping[str, Any] | None
    ) = None,
) -> PPOTrainingSessionResult:
    """Run a finite number of rollouts and guarded PPO operations."""

    if not callable(
        environment_factory
    ):
        raise TypeError(
            "environment_factory must be callable"
        )

    if not isinstance(
        agent,
        ActorCriticPolicyAgent,
    ):
        raise TypeError(
            "agent must be an ActorCriticPolicyAgent"
        )

    if not isinstance(
        trainer,
        PPOTrainer,
    ):
        raise TypeError(
            "trainer must be a PPOTrainer"
        )

    if not isinstance(
        config,
        PPOTrainingSessionConfig,
    ):
        raise TypeError(
            "config must be PPOTrainingSessionConfig"
        )

    if agent.model is not trainer.model:
        raise ValueError(
            "agent and trainer must share "
            "the same actor-critic model"
        )

    current_progress = (
        progress
        if progress is not None
        else PPOTrainingProgress()
    )

    if not isinstance(
        current_progress,
        PPOTrainingProgress,
    ):
        raise TypeError(
            "progress must be PPOTrainingProgress"
        )

    before_path = _output_path(
        before_checkpoint_path,
        field_name="before_checkpoint_path",
    )

    after_path = _output_path(
        after_checkpoint_path,
        field_name="after_checkpoint_path",
    )

    resolved_audit_path = _output_path(
        audit_path,
        field_name="audit_path",
    )

    if (
        before_path is not None
        and after_path is not None
        and before_path.resolve()
        == after_path.resolve()
    ):
        raise ValueError(
            "before and after checkpoint paths "
            "must be different"
        )

    metadata = _json_mapping(
        checkpoint_metadata,
        field_name="checkpoint_metadata",
    )

    if before_path is not None:
        save_ppo_training_checkpoint(
            before_path,
            trainer,
            progress=current_progress,
            policy_name=policy_name,
            policy_version=policy_version,
            metadata={
                **metadata,
                "session_phase": "before",
            },
        )

    writer = _JSONLAuditWriter(
        resolved_audit_path
    )

    rollout_records: list[
        PPORolloutAudit
    ] = []

    stopped_early = False
    stop_reason = "completed"

    try:
        writer.write(
            "session_started",
            {
                "config": config.to_record(),
                "initial_progress": (
                    current_progress.to_record()
                ),
                "training_enabled": (
                    config.training_enabled
                ),
            },
        )

        for rollout_index in range(
            config.rollout_count
        ):
            environment = (
                environment_factory(
                    rollout_index
                )
            )

            if not isinstance(
                environment,
                Environment,
            ):
                raise TypeError(
                    "environment_factory must return "
                    "an Environment"
                )

            seed = (
                None
                if config.seed is None
                else config.seed
                + rollout_index
            )

            collected = collect_bounded_rollout(
                environment,
                agent,
                max_steps=(
                    config.max_steps_per_rollout
                ),
                seed=seed,
                gamma=config.gamma,
                gae_lambda=(
                    config.gae_lambda
                ),
                normalize_advantages=(
                    config.normalize_advantages
                ),
                bootstrap_on_environment_truncation=(
                    config.bootstrap_on_environment_truncation
                ),
            )

            action_counts = Counter(
                transition.action.action.name
                for transition
                in collected.transitions
            )

            operations: list[
                PPOUpdateAudit
            ] = []

            if config.training_enabled:
                operation_count = (
                    config.updates_per_rollout
                )
            else:
                operation_count = 1

            for operation_index in range(
                operation_count
            ):
                batch_result = (
                    trainer.train_batch(
                        collected.batch
                    )
                    if config.training_enabled
                    else trainer.evaluate_batch(
                        collected.batch
                    )
                )

                operation = PPOUpdateAudit(
                    rollout_index=rollout_index,
                    operation_index=(
                        operation_index
                    ),
                    result=batch_result,
                )

                operations.append(
                    operation
                )

                writer.write(
                    "ppo_operation",
                    operation.to_record(),
                )

                approximate_kl = abs(
                    batch_result.metrics
                    .approximate_kl
                )

                if (
                    config.stop_on_kl
                    and approximate_kl
                    > config.max_absolute_kl
                ):
                    stopped_early = True
                    stop_reason = (
                        "absolute_kl_limit_exceeded"
                    )
                    break

            rollout_record = (
                PPORolloutAudit(
                    rollout_index=rollout_index,
                    steps=collected.steps,
                    total_reward=(
                        collected.total_reward
                    ),
                    terminated=(
                        collected.terminated
                    ),
                    truncated=(
                        collected.truncated
                    ),
                    termination_reason=(
                        collected.termination_reason
                    ),
                    bootstrap_value=(
                        collected.bootstrap_value
                    ),
                    action_counts=action_counts,
                    operations=tuple(
                        operations
                    ),
                )
            )

            rollout_records.append(
                rollout_record
            )

            current_progress = (
                PPOTrainingProgress(
                    rollout_count=(
                        current_progress
                        .rollout_count
                        + 1
                    ),
                    environment_step_count=(
                        current_progress
                        .environment_step_count
                        + collected.steps
                    ),
                    completed_episode_count=(
                        current_progress
                        .completed_episode_count
                        + int(
                            _episode_completed(
                                collected
                            )
                        )
                    ),
                    cumulative_reward=(
                        current_progress
                        .cumulative_reward
                        + collected.total_reward
                    ),
                )
            )

            writer.write(
                "rollout_completed",
                {
                    **rollout_record.to_record(),
                    "progress": (
                        current_progress
                        .to_record()
                    ),
                },
            )

            if stopped_early:
                break

        if after_path is not None:
            save_ppo_training_checkpoint(
                after_path,
                trainer,
                progress=current_progress,
                policy_name=policy_name,
                policy_version=policy_version,
                metadata={
                    **metadata,
                    "session_phase": "after",
                    "stopped_early": (
                        stopped_early
                    ),
                    "stop_reason": (
                        stop_reason
                    ),
                },
            )

        result = PPOTrainingSessionResult(
            config=config,
            progress=current_progress,
            rollouts=tuple(
                rollout_records
            ),
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            before_checkpoint_path=(
                before_path
            ),
            after_checkpoint_path=(
                after_path
            ),
            audit_path=(
                resolved_audit_path
            ),
        )

        writer.write(
            "session_completed",
            {
                "completed_rollouts": (
                    result.completed_rollouts
                ),
                "environment_steps": (
                    result.environment_steps
                ),
                "optimizer_operations": (
                    result.optimizer_operations
                ),
                "stopped_early": (
                    result.stopped_early
                ),
                "stop_reason": (
                    result.stop_reason
                ),
                "progress": (
                    result.progress.to_record()
                ),
            },
        )

        return result
    except Exception as error:
        writer.write(
            "session_failed",
            {
                "error_type": (
                    type(error).__name__
                ),
                "error": str(error),
            },
        )
        raise
    finally:
        writer.close()
