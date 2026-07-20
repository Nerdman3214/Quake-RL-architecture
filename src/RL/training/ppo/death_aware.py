"""Death-aware, signal-gated PPO trajectory collection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.agents.policies.actor_critic_agent import (
    ActorCriticPolicyAgent,
)
from RL.env.core.contracts import (
    Environment,
    StepResult,
)
from RL.observations.contracts import (
    Observation,
)
from RL.training.ppo.collector import (
    CollectedRolloutTransition,
)
from RL.training.ppo.core import (
    PPOBatchResult,
    PPORolloutBatch,
    PPOTrainer,
    RolloutBuffer,
    RolloutTransition,
)


_DEATH_EVENTS = frozenset(
    {
        "controlled_player_death",
        "controlled_player_died",
        "death",
        "player_death",
        "player_died",
        "player_suicide",
        "suicide",
    }
)

_RESPAWN_EVENTS = frozenset(
    {
        "controlled_player_respawned",
        "player_now_playing",
        "player_respawned",
        "respawn",
        "spawn",
    }
)

_MEANINGFUL_EVENTS = frozenset(
    {
        "controlled_player_kill",
        "enemy_killed",
        "first_blood",
        "match_ended",
        "match_won",
        "objective_completed",
        "player_captured_flag",
        "player_scored",
        "round_ended",
    }
)


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
        or not isinstance(
            value,
            (int, float),
        )
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


def _validate_observation(
    value: object,
    *,
    source: str,
) -> Observation:
    if not isinstance(
        value,
        Observation,
    ):
        raise TypeError(
            f"{source} must return an Observation"
        )

    return value


def _validate_step_result(
    value: object,
) -> StepResult:
    if not isinstance(
        value,
        StepResult,
    ):
        raise TypeError(
            "environment step must return a StepResult"
        )

    _validate_observation(
        value.observation,
        source="environment step",
    )

    _finite_number(
        value.reward,
        field_name="environment reward",
    )

    if not isinstance(
        value.terminated,
        bool,
    ):
        raise TypeError(
            "environment terminated must be bool"
        )

    if not isinstance(
        value.truncated,
        bool,
    ):
        raise TypeError(
            "environment truncated must be bool"
        )

    if (
        value.terminated
        and value.truncated
    ):
        raise ValueError(
            "environment step cannot be both "
            "terminated and truncated"
        )

    if not isinstance(
        value.info,
        Mapping,
    ):
        raise TypeError(
            "environment info must be a mapping"
        )

    return value


def _frame_copy(
    observation: Observation,
) -> torch.Tensor:
    frame = observation.frame

    if frame is None:
        raise ValueError(
            "rollout observation frame must not be None"
        )

    expected_shape = (
        4,
        3,
        90,
        160,
    )

    if isinstance(
        frame,
        np.ndarray,
    ):
        if tuple(frame.shape) != expected_shape:
            raise ValueError(
                "rollout frame has an unexpected shape"
            )

        if not np.issubdtype(
            frame.dtype,
            np.floating,
        ):
            raise TypeError(
                "rollout frame must be floating point"
            )

        if not bool(
            np.isfinite(frame).all()
        ):
            raise ValueError(
                "rollout frame must be finite"
            )

        copied = np.array(
            frame,
            dtype=np.float32,
            order="C",
            copy=True,
        )

        return torch.from_numpy(
            copied
        )

    if isinstance(
        frame,
        torch.Tensor,
    ):
        if tuple(frame.shape) != expected_shape:
            raise ValueError(
                "rollout frame has an unexpected shape"
            )

        if not frame.is_floating_point():
            raise TypeError(
                "rollout frame must be floating point"
            )

        if not bool(
            torch.isfinite(frame)
            .all()
            .item()
        ):
            raise ValueError(
                "rollout frame must be finite"
            )

        return (
            frame.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
            .clone()
        )

    raise TypeError(
        "rollout frame must be a NumPy array "
        "or torch.Tensor"
    )


def _event_types(
    info: Mapping[str, Any],
) -> frozenset[str]:
    normalized: set[str] = set()

    raw_event_types = info.get(
        "event_types",
        (),
    )

    if isinstance(
        raw_event_types,
        str,
    ):
        candidates = (
            raw_event_types,
        )
    elif isinstance(
        raw_event_types,
        (
            list,
            tuple,
            set,
            frozenset,
        ),
    ):
        candidates = tuple(
            raw_event_types
        )
    else:
        candidates = ()

    for candidate in candidates:
        if (
            isinstance(candidate, str)
            and candidate.strip()
        ):
            normalized.add(
                candidate.strip().lower()
            )

    raw_events = info.get(
        "events",
        (),
    )

    if isinstance(
        raw_events,
        (
            list,
            tuple,
        ),
    ):
        for event in raw_events:
            if not isinstance(
                event,
                Mapping,
            ):
                continue

            candidate = (
                event.get("event_type")
                or event.get("type")
            )

            if (
                isinstance(candidate, str)
                and candidate.strip()
            ):
                normalized.add(
                    candidate.strip().lower()
                )

    return frozenset(
        normalized
    )


def _truthy_flag(
    info: Mapping[str, Any],
    *names: str,
) -> bool:
    return any(
        info.get(name) is True
        for name in names
    )


def _alive(
    observation: Observation,
) -> bool | None:
    telemetry = observation.telemetry

    if telemetry is None:
        return None

    return bool(
        telemetry.alive
    )


def _death_detected(
    observation: Observation,
    next_observation: Observation,
    *,
    reward: float,
    info: Mapping[str, Any],
    death_reward_threshold: float,
) -> bool:
    previous_alive = _alive(
        observation
    )

    next_alive = _alive(
        next_observation
    )

    alive_transition = (
        previous_alive is True
        and next_alive is False
    )

    reward_signal = (
        reward
        <= death_reward_threshold
    )

    event_signal = bool(
        _event_types(info)
        & _DEATH_EVENTS
    )

    flag_signal = _truthy_flag(
        info,
        "controlled_player_died",
        "controlled_player_death",
        "player_died",
    )

    return bool(
        alive_transition
        or reward_signal
        or event_signal
        or flag_signal
    )


def _respawn_detected(
    previous_observation: Observation,
    next_observation: Observation,
    info: Mapping[str, Any],
) -> bool:
    previous_alive = _alive(
        previous_observation
    )

    next_alive = _alive(
        next_observation
    )

    alive_transition = (
        previous_alive is False
        and next_alive is True
    )

    event_signal = bool(
        _event_types(info)
        & _RESPAWN_EVENTS
    )

    flag_signal = _truthy_flag(
        info,
        "controlled_player_respawned",
        "player_respawned",
    )

    return bool(
        alive_transition
        or event_signal
        or flag_signal
    )


@dataclass(frozen=True)
class DeathAwarePPOConfig:
    """Limits and signals for one life-aware PPO segment."""

    max_steps: int
    max_respawn_wait_steps: int = 80
    updates_per_signal: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    bootstrap_on_environment_truncation: bool = False
    death_reward_threshold: float = -1.0
    signal_reward_epsilon: float = 1e-8
    max_absolute_kl: float = 0.10
    stop_on_kl: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_steps",
            _positive_integer(
                self.max_steps,
                field_name="max_steps",
            ),
        )

        object.__setattr__(
            self,
            "max_respawn_wait_steps",
            _nonnegative_integer(
                self.max_respawn_wait_steps,
                field_name=(
                    "max_respawn_wait_steps"
                ),
            ),
        )

        object.__setattr__(
            self,
            "updates_per_signal",
            _positive_integer(
                self.updates_per_signal,
                field_name=(
                    "updates_per_signal"
                ),
            ),
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

        death_threshold = _finite_number(
            self.death_reward_threshold,
            field_name=(
                "death_reward_threshold"
            ),
        )

        if death_threshold >= 0.0:
            raise ValueError(
                "death_reward_threshold must be negative"
            )

        object.__setattr__(
            self,
            "death_reward_threshold",
            death_threshold,
        )

        epsilon = _finite_number(
            self.signal_reward_epsilon,
            field_name=(
                "signal_reward_epsilon"
            ),
        )

        if epsilon < 0.0:
            raise ValueError(
                "signal_reward_epsilon must be nonnegative"
            )

        object.__setattr__(
            self,
            "signal_reward_epsilon",
            epsilon,
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
                or not isinstance(
                    self.seed,
                    int,
                )
            )
        ):
            raise TypeError(
                "seed must be an integer or None"
            )


@dataclass(frozen=True)
class DeathAwareRolloutResult:
    """One fresh trajectory ending at a signal or hard limit."""

    initial_observation: Observation
    reset_info: Mapping[str, Any]
    transitions: tuple[
        CollectedRolloutTransition,
        ...,
    ]
    batch: PPORolloutBatch
    total_reward: float
    meaningful_signal: bool
    signal_reason: str
    death_detected: bool
    environment_terminated: bool
    environment_truncated: bool
    hard_limit_reached: bool
    bootstrap_value: float
    respawn_detected: bool
    respawn_wait_steps: int
    post_respawn_observation: Observation | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.initial_observation,
            Observation,
        ):
            raise TypeError(
                "initial_observation must be "
                "an Observation"
            )

        if not self.transitions:
            raise ValueError(
                "rollout must contain transitions"
            )

        if not isinstance(
            self.batch,
            PPORolloutBatch,
        ):
            raise TypeError(
                "batch must be a PPORolloutBatch"
            )

        if (
            len(self.transitions)
            != int(self.batch.frames.shape[0])
        ):
            raise ValueError(
                "transition count must match batch size"
            )

        _finite_number(
            self.total_reward,
            field_name="total_reward",
        )

        _finite_number(
            self.bootstrap_value,
            field_name="bootstrap_value",
        )

        for field_name in (
            "meaningful_signal",
            "death_detected",
            "environment_terminated",
            "environment_truncated",
            "hard_limit_reached",
            "respawn_detected",
        ):
            if not isinstance(
                getattr(self, field_name),
                bool,
            ):
                raise TypeError(
                    f"{field_name} must be bool"
                )

        if (
            not isinstance(
                self.signal_reason,
                str,
            )
            or not self.signal_reason
        ):
            raise ValueError(
                "signal_reason must not be empty"
            )

        _nonnegative_integer(
            self.respawn_wait_steps,
            field_name="respawn_wait_steps",
        )

        if (
            self.post_respawn_observation
            is not None
            and not isinstance(
                self.post_respawn_observation,
                Observation,
            )
        ):
            raise TypeError(
                "post_respawn_observation must be "
                "an Observation or None"
            )

        object.__setattr__(
            self,
            "reset_info",
            MappingProxyType(
                dict(self.reset_info)
            ),
        )

    @property
    def steps(self) -> int:
        return len(
            self.transitions
        )

    @property
    def final_observation(
        self,
    ) -> Observation:
        return (
            self.transitions[-1]
            .next_observation
        )


@dataclass(frozen=True)
class DeathAwarePPOResult:
    """Collection plus a guarded PPO decision."""

    rollout: DeathAwareRolloutResult
    evaluation: PPOBatchResult
    updates: tuple[
        PPOBatchResult,
        ...,
    ]
    update_skipped_reason: str | None
    stopped_on_kl: bool

    @property
    def optimizer_operations(
        self,
    ) -> int:
        return sum(
            int(update.optimizer_step)
            for update in self.updates
        )

    @property
    def update_performed(
        self,
    ) -> bool:
        return (
            self.optimizer_operations > 0
        )


def _meaningful_signal(
    *,
    reward: float,
    info: Mapping[str, Any],
    death_detected: bool,
    terminated: bool,
    truncated: bool,
    epsilon: float,
) -> tuple[bool, str]:
    if death_detected:
        return True, "death_detected"

    if abs(reward) > epsilon:
        return True, "reward_signal"

    if bool(
        _event_types(info)
        & _MEANINGFUL_EVENTS
    ):
        return True, "authoritative_event"

    if terminated:
        return True, "environment_terminated"

    if truncated:
        return True, "environment_truncated"

    return False, "zero_signal"


def _wait_for_respawn(
    environment: Environment,
    observation: Observation,
    *,
    max_wait_steps: int,
) -> tuple[
    bool,
    int,
    Observation | None,
]:
    if _alive(observation) is True:
        return True, 0, observation

    current = observation

    for wait_index in range(
        max_wait_steps
    ):
        result = _validate_step_result(
            environment.step(
                ActionCommand(
                    action=(
                        DiscreteAction.NO_OP
                    ),
                    duration_ticks=1,
                )
            )
        )

        next_observation = (
            result.observation
        )

        if _respawn_detected(
            current,
            next_observation,
            result.info,
        ):
            return (
                True,
                wait_index + 1,
                next_observation,
            )

        current = (
            next_observation
        )

        if (
            result.terminated
            or result.truncated
        ):
            break

    return (
        False,
        max_wait_steps,
        None,
    )


def collect_death_aware_rollout(
    environment: Environment,
    agent: ActorCriticPolicyAgent,
    config: DeathAwarePPOConfig,
) -> DeathAwareRolloutResult:
    """Collect fresh on-policy steps until a signal or hard limit."""

    if not isinstance(
        environment,
        Environment,
    ):
        raise TypeError(
            "environment must be an Environment"
        )

    if not isinstance(
        agent,
        ActorCriticPolicyAgent,
    ):
        raise TypeError(
            "agent must be an ActorCriticPolicyAgent"
        )

    if not isinstance(
        config,
        DeathAwarePPOConfig,
    ):
        raise TypeError(
            "config must be DeathAwarePPOConfig"
        )

    buffer = RolloutBuffer(
        max_steps=config.max_steps
    )

    transitions: list[
        CollectedRolloutTransition
    ] = []

    total_reward = 0.0
    meaningful_signal = False
    signal_reason = "zero_signal"
    death_found = False
    environment_terminated = False
    environment_truncated = False
    hard_limit_reached = False
    bootstrap_value = 0.0
    respawn_found = False
    respawn_wait_steps = 0
    post_respawn_observation: (
        Observation | None
    ) = None

    try:
        reset_result = environment.reset(
            seed=config.seed
        )

        if (
            not isinstance(
                reset_result,
                tuple,
            )
            or len(reset_result) != 2
        ):
            raise TypeError(
                "environment reset must return "
                "(Observation, info)"
            )

        initial_observation = (
            _validate_observation(
                reset_result[0],
                source="environment reset",
            )
        )

        if not isinstance(
            reset_result[1],
            Mapping,
        ):
            raise TypeError(
                "environment reset info "
                "must be a mapping"
            )

        reset_info = dict(
            reset_result[1]
        )

        observation = (
            initial_observation
        )

        for step_index in range(
            config.max_steps
        ):
            action = agent.act(
                observation
            )

            if (
                agent.last_log_prob is None
                or agent.last_value is None
                or agent.last_entropy is None
            ):
                raise RuntimeError(
                    "agent did not expose PPO "
                    "decision estimates"
                )

            result = _validate_step_result(
                environment.step(
                    action
                )
            )

            next_observation = (
                result.observation
            )

            reward = float(
                result.reward
            )

            death_on_step = (
                _death_detected(
                    observation,
                    next_observation,
                    reward=reward,
                    info=result.info,
                    death_reward_threshold=(
                        config.death_reward_threshold
                    ),
                )
            )

            (
                step_has_signal,
                step_signal_reason,
            ) = _meaningful_signal(
                reward=reward,
                info=result.info,
                death_detected=(
                    death_on_step
                ),
                terminated=(
                    result.terminated
                ),
                truncated=(
                    result.truncated
                ),
                epsilon=(
                    config.signal_reward_epsilon
                ),
            )

            segment_terminated = bool(
                result.terminated
                or death_on_step
            )

            segment_truncated = bool(
                result.truncated
                and not segment_terminated
            )

            frame = _frame_copy(
                observation
            )

            buffer.append(
                RolloutTransition(
                    frames=frame,
                    action_index=int(
                        action.action
                    ),
                    reward=reward,
                    terminated=(
                        segment_terminated
                    ),
                    truncated=(
                        segment_truncated
                    ),
                    old_log_prob=(
                        agent.last_log_prob
                    ),
                    old_value=(
                        agent.last_value
                    ),
                    duration_ticks=(
                        action.duration_ticks
                    ),
                )
            )

            transitions.append(
                CollectedRolloutTransition(
                    step_index=step_index,
                    observation=observation,
                    action=action,
                    next_observation=(
                        next_observation
                    ),
                    reward=reward,
                    terminated=(
                        segment_terminated
                    ),
                    truncated=(
                        segment_truncated
                    ),
                    old_log_prob=(
                        agent.last_log_prob
                    ),
                    old_value=(
                        agent.last_value
                    ),
                    entropy=(
                        agent.last_entropy
                    ),
                    info=result.info,
                )
            )

            total_reward += reward
            observation = (
                next_observation
            )

            if step_has_signal:
                meaningful_signal = True
                signal_reason = (
                    step_signal_reason
                )
                death_found = (
                    death_on_step
                )
                environment_terminated = (
                    result.terminated
                )
                environment_truncated = (
                    result.truncated
                )
                break

        else:
            hard_limit_reached = True
            signal_reason = (
                "hard_step_limit_zero_signal"
            )

        final_observation = (
            transitions[-1]
            .next_observation
        )

        if (
            death_found
            or environment_terminated
        ):
            bootstrap_value = 0.0
        elif environment_truncated:
            bootstrap_value = (
                agent.estimate_value(
                    final_observation
                )
                if (
                    config.bootstrap_on_environment_truncation
                )
                else 0.0
            )
        else:
            bootstrap_value = (
                agent.estimate_value(
                    final_observation
                )
            )

        batch = buffer.finish(
            last_value=bootstrap_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            normalize_advantages=(
                config.normalize_advantages
            ),
        )

        if (
            death_found
            and not environment_terminated
            and not environment_truncated
        ):
            (
                respawn_found,
                respawn_wait_steps,
                post_respawn_observation,
            ) = _wait_for_respawn(
                environment,
                final_observation,
                max_wait_steps=(
                    config.max_respawn_wait_steps
                ),
            )

        return DeathAwareRolloutResult(
            initial_observation=(
                initial_observation
            ),
            reset_info=reset_info,
            transitions=tuple(
                transitions
            ),
            batch=batch,
            total_reward=total_reward,
            meaningful_signal=(
                meaningful_signal
            ),
            signal_reason=signal_reason,
            death_detected=death_found,
            environment_terminated=(
                environment_terminated
            ),
            environment_truncated=(
                environment_truncated
            ),
            hard_limit_reached=(
                hard_limit_reached
            ),
            bootstrap_value=(
                bootstrap_value
            ),
            respawn_detected=(
                respawn_found
            ),
            respawn_wait_steps=(
                respawn_wait_steps
            ),
            post_respawn_observation=(
                post_respawn_observation
            ),
        )
    finally:
        environment.close()


def run_death_aware_ppo_step(
    environment: Environment,
    agent: ActorCriticPolicyAgent,
    trainer: PPOTrainer,
    config: DeathAwarePPOConfig,
) -> DeathAwarePPOResult:
    """Collect one fresh segment and update only with a signal."""

    if not isinstance(
        trainer,
        PPOTrainer,
    ):
        raise TypeError(
            "trainer must be a PPOTrainer"
        )

    if agent.model is not trainer.model:
        raise ValueError(
            "agent and trainer must share "
            "the same actor-critic model"
        )

    rollout = (
        collect_death_aware_rollout(
            environment,
            agent,
            config,
        )
    )

    evaluation = (
        trainer.evaluate_batch(
            rollout.batch
        )
    )

    if not rollout.meaningful_signal:
        return DeathAwarePPOResult(
            rollout=rollout,
            evaluation=evaluation,
            updates=(),
            update_skipped_reason=(
                "zero_signal_segment"
            ),
            stopped_on_kl=False,
        )

    updates: list[
        PPOBatchResult
    ] = []

    stopped_on_kl = False

    for _ in range(
        config.updates_per_signal
    ):
        update = trainer.train_batch(
            rollout.batch
        )

        updates.append(
            update
        )

        if (
            config.stop_on_kl
            and abs(
                update.metrics.approximate_kl
            )
            > config.max_absolute_kl
        ):
            stopped_on_kl = True
            break

    return DeathAwarePPOResult(
        rollout=rollout,
        evaluation=evaluation,
        updates=tuple(updates),
        update_skipped_reason=None,
        stopped_on_kl=stopped_on_kl,
    )
