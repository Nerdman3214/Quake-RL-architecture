"""Bounded, auditable death-aware PPO training sessions."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import (
    dataclass,
    fields,
    replace,
)
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from RL.agents import (
    ActorCriticPolicyAgent,
    VisualActorCriticNetwork,
)
from RL.env.core.contracts import (
    Environment,
)
from RL.training.ppo.checkpoint import (
    PPOTrainingProgress,
    load_ppo_training_checkpoint,
    save_ppo_training_checkpoint,
)
from RL.training.ppo.death_aware import (
    DeathAwarePPOConfig,
    DeathAwarePPOResult,
    run_death_aware_ppo_step,
)


DeathAwareEnvironmentFactory = Callable[
    [int],
    Environment,
]

DeathAwareAgentFactory = Callable[
    [VisualActorCriticNetwork, int],
    ActorCriticPolicyAgent,
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


def _json_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping"
        )

    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
            )
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{field_name} must be JSON-compatible"
        ) from error

    if not isinstance(normalized, dict):
        raise ValueError(
            f"{field_name} must normalize to an object"
        )

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


def _validate_distinct_paths(
    paths: Mapping[str, Path | None],
) -> None:
    resolved: dict[Path, str] = {}

    for field_name, path in paths.items():
        if path is None:
            continue

        key = path.resolve()

        if key in resolved:
            raise ValueError(
                f"{field_name} and "
                f"{resolved[key]} must be different"
            )

        resolved[key] = field_name


def _config_record(
    config: DeathAwarePPOConfig,
) -> dict[str, Any]:
    return {
        field.name: getattr(
            config,
            field.name,
        )
        for field in fields(config)
    }


def _state_copy(
    model: VisualActorCriticNetwork,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        .cpu()
        .clone()
        for name, tensor
        in model.state_dict().items()
    }


def _state_difference(
    before: Mapping[str, torch.Tensor],
    model: VisualActorCriticNetwork,
) -> tuple[
    tuple[str, ...],
    float,
]:
    changed: list[str] = []
    maximum_change = 0.0

    current_state = model.state_dict()

    if set(before) != set(current_state):
        raise ValueError(
            "model state fields changed during attempt"
        )

    for name, old_tensor in before.items():
        current = (
            current_state[name]
            .detach()
            .cpu()
        )

        if torch.equal(
            old_tensor,
            current,
        ):
            continue

        changed.append(name)

        difference = torch.max(
            torch.abs(
                current.to(
                    dtype=torch.float64
                )
                - old_tensor.to(
                    dtype=torch.float64
                )
            )
        )

        maximum_change = max(
            maximum_change,
            float(difference.item()),
        )

    return (
        tuple(changed),
        maximum_change,
    )


def _states_equal(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> bool:
    return (
        set(first) == set(second)
        and all(
            torch.equal(
                first[name].detach().cpu(),
                second[name].detach().cpu(),
            )
            for name in first
        )
    )


@dataclass(frozen=True)
class DeathAwarePPOTrainingSessionConfig:
    """Hard limits and promotion rules for one session."""

    attempt_count: int
    step_config: DeathAwarePPOConfig
    require_respawn_evidence: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_count",
            _positive_integer(
                self.attempt_count,
                field_name="attempt_count",
            ),
        )

        if not isinstance(
            self.step_config,
            DeathAwarePPOConfig,
        ):
            raise TypeError(
                "step_config must be "
                "DeathAwarePPOConfig"
            )

        if (
            self.step_config
            .updates_per_signal
            != 1
        ):
            raise ValueError(
                "death-aware promotion sessions "
                "require exactly one update per signal"
            )

        if not isinstance(
            self.require_respawn_evidence,
            bool,
        ):
            raise TypeError(
                "require_respawn_evidence "
                "must be bool"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "attempt_count": self.attempt_count,
            "step_config": _config_record(
                self.step_config
            ),
            "require_respawn_evidence": (
                self.require_respawn_evidence
            ),
        }


@dataclass(frozen=True)
class DeathAwarePPOAttemptAudit:
    """Audit record for one isolated checkpoint attempt."""

    attempt_index: int
    seed: int | None
    result: DeathAwarePPOResult
    source_optimizer_step_count: int
    ending_optimizer_step_count: int
    weights_changed: bool
    changed_state_tensor_names: tuple[
        str,
        ...,
    ]
    maximum_absolute_parameter_change: float
    action_counts: Mapping[str, int]
    accepted: bool
    rejection_reasons: tuple[
        str,
        ...,
    ]

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(
                self.attempt_index,
                int,
            )
            or self.attempt_index < 0
        ):
            raise ValueError(
                "attempt_index must be "
                "a nonnegative integer"
            )

        if not isinstance(
            self.result,
            DeathAwarePPOResult,
        ):
            raise TypeError(
                "result must be DeathAwarePPOResult"
            )

        if not isinstance(
            self.accepted,
            bool,
        ):
            raise TypeError(
                "accepted must be bool"
            )

        if (
            self.accepted
            and self.rejection_reasons
        ):
            raise ValueError(
                "accepted attempts cannot have "
                "rejection reasons"
            )

        if (
            not self.accepted
            and not self.rejection_reasons
        ):
            raise ValueError(
                "rejected attempts require "
                "at least one reason"
            )

        object.__setattr__(
            self,
            "action_counts",
            MappingProxyType(
                dict(self.action_counts)
            ),
        )

    @property
    def rollout(self):
        return self.result.rollout

    def to_record(self) -> dict[str, Any]:
        rollout = self.rollout

        updates = []

        for update in self.result.updates:
            metrics = update.metrics

            updates.append(
                {
                    "optimizer_step": (
                        update.optimizer_step
                    ),
                    "optimizer_step_count": (
                        update.optimizer_step_count
                    ),
                    "gradient_norm": (
                        update.gradient_norm
                    ),
                    "total_loss": (
                        metrics.total_loss
                    ),
                    "policy_loss": (
                        metrics.policy_loss
                    ),
                    "value_loss": (
                        metrics.value_loss
                    ),
                    "entropy": (
                        metrics.entropy
                    ),
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
                }
            )

        return {
            "attempt_index": self.attempt_index,
            "seed": self.seed,
            "accepted": self.accepted,
            "rejection_reasons": list(
                self.rejection_reasons
            ),
            "steps": rollout.steps,
            "ppo_batch_steps": int(
                rollout.batch.frames.shape[0]
            ),
            "total_reward": (
                rollout.total_reward
            ),
            "meaningful_signal": (
                rollout.meaningful_signal
            ),
            "signal_reason": (
                rollout.signal_reason
            ),
            "death_detected": (
                rollout.death_detected
            ),
            "death_event_confirmed": (
                rollout.death_event_confirmed
            ),
            "death_reward_confirmed": (
                rollout.death_reward_confirmed
            ),
            "death_confirmation_steps": (
                rollout.death_confirmation_steps
            ),
            "confirmed_death_reward": (
                rollout.confirmed_death_reward
            ),
            "respawn_detected": (
                rollout.respawn_detected
            ),
            "respawn_inferred": (
                rollout.respawn_inferred
            ),
            "respawn_signal_reason": (
                rollout.respawn_signal_reason
            ),
            "respawn_wait_steps": (
                rollout.respawn_wait_steps
            ),
            "respawn_fire_actions": (
                rollout.respawn_fire_actions
            ),
            "post_respawn_reward": (
                rollout.post_respawn_reward
            ),
            "environment_terminated": (
                rollout.environment_terminated
            ),
            "environment_truncated": (
                rollout.environment_truncated
            ),
            "hard_limit_reached": (
                rollout.hard_limit_reached
            ),
            "action_counts": dict(
                self.action_counts
            ),
            "optimizer_operations": (
                self.result.optimizer_operations
            ),
            "source_optimizer_step_count": (
                self.source_optimizer_step_count
            ),
            "ending_optimizer_step_count": (
                self.ending_optimizer_step_count
            ),
            "update_skipped_reason": (
                self.result.update_skipped_reason
            ),
            "stopped_on_kl": (
                self.result.stopped_on_kl
            ),
            "weights_changed": (
                self.weights_changed
            ),
            "changed_state_tensor_count": len(
                self.changed_state_tensor_names
            ),
            "changed_state_tensor_names": list(
                self.changed_state_tensor_names
            ),
            "maximum_absolute_parameter_change": (
                self.maximum_absolute_parameter_change
            ),
            "updates": updates,
        }


@dataclass(frozen=True)
class DeathAwarePPOTrainingSessionResult:
    """Complete result of isolated death-aware attempts."""

    config: DeathAwarePPOTrainingSessionConfig
    source_checkpoint_path: Path
    source_optimizer_step_count: int
    progress: PPOTrainingProgress
    attempts: tuple[
        DeathAwarePPOAttemptAudit,
        ...,
    ]
    accepted_attempt_index: int | None
    before_checkpoint_path: Path | None
    promoted_checkpoint_path: Path | None
    audit_path: Path | None

    @property
    def promoted(self) -> bool:
        return (
            self.accepted_attempt_index
            is not None
        )

    @property
    def attempts_completed(self) -> int:
        return len(self.attempts)

    @property
    def accepted_attempt(
        self,
    ) -> DeathAwarePPOAttemptAudit | None:
        if self.accepted_attempt_index is None:
            return None

        return next(
            attempt
            for attempt in self.attempts
            if attempt.attempt_index
            == self.accepted_attempt_index
        )

    @property
    def ending_optimizer_step_count(
        self,
    ) -> int:
        accepted = self.accepted_attempt

        if accepted is None:
            return (
                self.source_optimizer_step_count
            )

        return (
            accepted.ending_optimizer_step_count
        )


class _JSONLAuditWriter:
    """Durable JSONL writer matching PPO session audits."""

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


def _rejection_reasons(
    result: DeathAwarePPOResult,
    *,
    source_optimizer_step_count: int,
    ending_optimizer_step_count: int,
    weights_changed: bool,
    config: DeathAwarePPOTrainingSessionConfig,
) -> tuple[str, ...]:
    rollout = result.rollout
    reasons: list[str] = []

    if not rollout.death_detected:
        reasons.append(
            "death_not_detected"
        )

    if not rollout.death_reward_confirmed:
        reasons.append(
            "death_reward_not_confirmed"
        )

    if (
        rollout.confirmed_death_reward
        > config.step_config
        .death_reward_threshold
    ):
        reasons.append(
            "death_reward_above_threshold"
        )

    respawn_evidence = bool(
        rollout.respawn_detected
        or rollout.respawn_inferred
    )

    if (
        config.require_respawn_evidence
        and not respawn_evidence
    ):
        reasons.append(
            "respawn_evidence_missing"
        )

    if (
        rollout.respawn_inferred
        and rollout.respawn_signal_reason
        != "second_death_proves_respawn"
    ):
        reasons.append(
            "invalid_respawn_inference"
        )

    if (
        rollout.respawn_inferred
        and rollout.post_respawn_reward
        > config.step_config
        .death_reward_threshold
    ):
        reasons.append(
            "inferred_respawn_reward_unconfirmed"
        )

    if result.optimizer_operations != 1:
        reasons.append(
            "optimizer_operation_count_invalid"
        )

    if (
        ending_optimizer_step_count
        != source_optimizer_step_count + 1
    ):
        reasons.append(
            "optimizer_step_increment_invalid"
        )

    if not weights_changed:
        reasons.append(
            "model_weights_unchanged"
        )

    if result.stopped_on_kl:
        reasons.append(
            "absolute_kl_limit_exceeded"
        )

    return tuple(reasons)


def run_bounded_death_aware_ppo_session(
    source_checkpoint_path: str | Path,
    environment_factory: DeathAwareEnvironmentFactory,
    agent_factory: DeathAwareAgentFactory,
    config: DeathAwarePPOTrainingSessionConfig,
    *,
    device: str | torch.device = "cpu",
    before_checkpoint_path: (
        str | Path | None
    ) = None,
    promoted_checkpoint_path: (
        str | Path | None
    ) = None,
    audit_path: str | Path | None = None,
    policy_name: str = (
        "xonotic-death-aware-ppo"
    ),
    policy_version: str = "v1",
    checkpoint_metadata: (
        Mapping[str, Any] | None
    ) = None,
) -> DeathAwarePPOTrainingSessionResult:
    """Run isolated attempts and promote only an accepted death update."""

    source_path = Path(
        source_checkpoint_path
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            f"source checkpoint does not exist: "
            f"{source_path}"
        )

    if not callable(
        environment_factory
    ):
        raise TypeError(
            "environment_factory must be callable"
        )

    if not callable(
        agent_factory
    ):
        raise TypeError(
            "agent_factory must be callable"
        )

    if not isinstance(
        config,
        DeathAwarePPOTrainingSessionConfig,
    ):
        raise TypeError(
            "config must be "
            "DeathAwarePPOTrainingSessionConfig"
        )

    resolved_device = torch.device(
        device
    )

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    before_path = _output_path(
        before_checkpoint_path,
        field_name="before_checkpoint_path",
    )

    promoted_path = _output_path(
        promoted_checkpoint_path,
        field_name=(
            "promoted_checkpoint_path"
        ),
    )

    resolved_audit_path = _output_path(
        audit_path,
        field_name="audit_path",
    )

    _validate_distinct_paths(
        {
            "before_checkpoint_path": (
                before_path
            ),
            "promoted_checkpoint_path": (
                promoted_path
            ),
            "audit_path": (
                resolved_audit_path
            ),
        }
    )

    metadata = _json_mapping(
        checkpoint_metadata,
        field_name="checkpoint_metadata",
    )

    baseline = load_ppo_training_checkpoint(
        source_path,
        device=resolved_device,
    )

    source_optimizer_step_count = (
        baseline.trainer.optimizer_step_count
    )

    initial_progress = baseline.progress

    if before_path is not None:
        save_ppo_training_checkpoint(
            before_path,
            baseline.trainer,
            progress=initial_progress,
            policy_name=policy_name,
            policy_version=policy_version,
            metadata={
                **metadata,
                "session_phase": "before",
                "source_checkpoint": str(
                    source_path
                ),
                "source_optimizer_step_count": (
                    source_optimizer_step_count
                ),
            },
        )

    writer = _JSONLAuditWriter(
        resolved_audit_path
    )

    attempt_records: list[
        DeathAwarePPOAttemptAudit
    ] = []

    accepted_attempt_index: int | None = (
        None
    )

    current_progress = initial_progress

    try:
        writer.write(
            "death_aware_session_started",
            {
                "config": config.to_record(),
                "source_checkpoint": str(
                    source_path
                ),
                "source_optimizer_step_count": (
                    source_optimizer_step_count
                ),
                "initial_progress": (
                    initial_progress.to_record()
                ),
                "before_checkpoint_path": (
                    str(before_path)
                    if before_path is not None
                    else None
                ),
                "requested_promoted_checkpoint_path": (
                    str(promoted_path)
                    if promoted_path is not None
                    else None
                ),
            },
        )

        for attempt_index in range(
            config.attempt_count
        ):
            seed = (
                None
                if config.step_config.seed is None
                else config.step_config.seed
                + attempt_index
            )

            if seed is not None:
                torch.manual_seed(seed)

                if (
                    resolved_device.type
                    == "cuda"
                ):
                    torch.cuda.manual_seed_all(
                        seed
                    )

            loaded = (
                load_ppo_training_checkpoint(
                    source_path,
                    device=resolved_device,
                )
            )

            trainer = loaded.trainer
            model = loaded.model

            if (
                trainer.optimizer_step_count
                != source_optimizer_step_count
            ):
                raise ValueError(
                    "attempt did not reload the "
                    "source optimizer step"
                )

            agent = agent_factory(
                model,
                attempt_index,
            )

            if not isinstance(
                agent,
                ActorCriticPolicyAgent,
            ):
                raise TypeError(
                    "agent_factory must return "
                    "ActorCriticPolicyAgent"
                )

            if agent.model is not model:
                raise ValueError(
                    "agent_factory must bind the "
                    "reloaded checkpoint model"
                )

            environment = (
                environment_factory(
                    attempt_index
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

            starting_state = _state_copy(
                model
            )

            attempt_config = replace(
                config.step_config,
                seed=seed,
            )

            result = (
                run_death_aware_ppo_step(
                    environment,
                    agent,
                    trainer,
                    attempt_config,
                )
            )

            (
                changed_names,
                maximum_change,
            ) = _state_difference(
                starting_state,
                model,
            )

            weights_changed = bool(
                changed_names
            )

            reasons = _rejection_reasons(
                result,
                source_optimizer_step_count=(
                    source_optimizer_step_count
                ),
                ending_optimizer_step_count=(
                    trainer.optimizer_step_count
                ),
                weights_changed=(
                    weights_changed
                ),
                config=config,
            )

            accepted = not reasons

            action_counts = Counter(
                transition.action.action.name
                for transition
                in result.rollout.transitions
            )

            attempt_audit = (
                DeathAwarePPOAttemptAudit(
                    attempt_index=attempt_index,
                    seed=seed,
                    result=result,
                    source_optimizer_step_count=(
                        source_optimizer_step_count
                    ),
                    ending_optimizer_step_count=(
                        trainer.optimizer_step_count
                    ),
                    weights_changed=(
                        weights_changed
                    ),
                    changed_state_tensor_names=(
                        changed_names
                    ),
                    maximum_absolute_parameter_change=(
                        maximum_change
                    ),
                    action_counts=(
                        action_counts
                    ),
                    accepted=accepted,
                    rejection_reasons=reasons,
                )
            )

            attempt_records.append(
                attempt_audit
            )

            writer.write(
                "death_aware_attempt_completed",
                attempt_audit.to_record(),
            )

            if not accepted:
                continue

            accepted_attempt_index = (
                attempt_index
            )

            rollout = result.rollout

            current_progress = (
                PPOTrainingProgress(
                    rollout_count=(
                        initial_progress
                        .rollout_count
                        + 1
                    ),
                    environment_step_count=(
                        initial_progress
                        .environment_step_count
                        + rollout.steps
                    ),
                    completed_episode_count=(
                        initial_progress
                        .completed_episode_count
                        + 1
                    ),
                    cumulative_reward=(
                        initial_progress
                        .cumulative_reward
                        + rollout.total_reward
                    ),
                )
            )

            if promoted_path is not None:
                save_ppo_training_checkpoint(
                    promoted_path,
                    trainer,
                    progress=current_progress,
                    policy_name=policy_name,
                    policy_version=policy_version,
                    metadata={
                        **metadata,
                        "session_phase": (
                            "promoted"
                        ),
                        "source_checkpoint": str(
                            source_path
                        ),
                        "accepted_attempt_index": (
                            attempt_index
                        ),
                        "confirmed_death": True,
                        "confirmed_death_reward": (
                            rollout
                            .confirmed_death_reward
                        ),
                        "respawn_detected": (
                            rollout.respawn_detected
                        ),
                        "respawn_inferred": (
                            rollout.respawn_inferred
                        ),
                        "respawn_signal_reason": (
                            rollout
                            .respawn_signal_reason
                        ),
                    },
                )

                promoted = (
                    load_ppo_training_checkpoint(
                        promoted_path,
                        device=resolved_device,
                    )
                )

                saved_state = {
                    name: tensor.detach()
                    .cpu()
                    for name, tensor
                    in promoted.model
                    .state_dict().items()
                }

                current_state = {
                    name: tensor.detach()
                    .cpu()
                    for name, tensor
                    in model.state_dict().items()
                }

                if not _states_equal(
                    current_state,
                    saved_state,
                ):
                    raise RuntimeError(
                        "promoted checkpoint weights "
                        "did not reload exactly"
                    )

                if (
                    promoted.trainer
                    .optimizer_step_count
                    != source_optimizer_step_count
                    + 1
                ):
                    raise RuntimeError(
                        "promoted checkpoint optimizer "
                        "step is invalid"
                    )

                if (
                    promoted.progress
                    != current_progress
                ):
                    raise RuntimeError(
                        "promoted checkpoint progress "
                        "did not reload exactly"
                    )

            writer.write(
                "death_aware_attempt_promoted",
                {
                    "attempt_index": (
                        attempt_index
                    ),
                    "optimizer_step_count": (
                        trainer.optimizer_step_count
                    ),
                    "progress": (
                        current_progress.to_record()
                    ),
                    "promoted_checkpoint_path": (
                        str(promoted_path)
                        if promoted_path
                        is not None
                        else None
                    ),
                },
            )

            break

        result = (
            DeathAwarePPOTrainingSessionResult(
                config=config,
                source_checkpoint_path=(
                    source_path
                ),
                source_optimizer_step_count=(
                    source_optimizer_step_count
                ),
                progress=current_progress,
                attempts=tuple(
                    attempt_records
                ),
                accepted_attempt_index=(
                    accepted_attempt_index
                ),
                before_checkpoint_path=(
                    before_path
                ),
                promoted_checkpoint_path=(
                    promoted_path
                    if accepted_attempt_index
                    is not None
                    else None
                ),
                audit_path=(
                    resolved_audit_path
                ),
            )
        )

        writer.write(
            "death_aware_session_completed",
            {
                "attempts_completed": (
                    result.attempts_completed
                ),
                "promoted": (
                    result.promoted
                ),
                "accepted_attempt_index": (
                    result.accepted_attempt_index
                ),
                "source_optimizer_step_count": (
                    result
                    .source_optimizer_step_count
                ),
                "ending_optimizer_step_count": (
                    result
                    .ending_optimizer_step_count
                ),
                "progress": (
                    result.progress.to_record()
                ),
                "promoted_checkpoint_path": (
                    str(
                        result
                        .promoted_checkpoint_path
                    )
                    if result
                    .promoted_checkpoint_path
                    is not None
                    else None
                ),
            },
        )

        return result

    except Exception as error:
        writer.write(
            "death_aware_session_failed",
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
