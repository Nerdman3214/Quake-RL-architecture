"""Bounded rollout, GAE, PPO objective, and explicit batch trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from RL.actions.contracts import DiscreteAction
from RL.agents.policies.actor_critic import (
    VisualActorCriticNetwork,
)


PPO_FRAME_SHAPE = (
    4,
    3,
    90,
    160,
)

PPO_ACTION_COUNT = len(
    tuple(DiscreteAction)
)


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


def _unit_interval(
    value: object,
    *,
    field_name: str,
    include_zero: bool = True,
) -> float:
    normalized = _finite_number(
        value,
        field_name=field_name,
    )

    lower_valid = (
        normalized >= 0.0
        if include_zero
        else normalized > 0.0
    )

    if not lower_valid or normalized > 1.0:
        operator = "[0, 1]" if include_zero else "(0, 1]"

        raise ValueError(
            f"{field_name} must be in {operator}"
        )

    return normalized


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    *,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantages and value targets.

    True environment termination suppresses bootstrapping. A truncation
    is intentionally allowed to bootstrap from ``last_value``.
    """

    if not isinstance(
        rewards,
        torch.Tensor,
    ):
        raise TypeError(
            "rewards must be a torch.Tensor"
        )

    if not isinstance(
        values,
        torch.Tensor,
    ):
        raise TypeError(
            "values must be a torch.Tensor"
        )

    if not isinstance(
        terminated,
        torch.Tensor,
    ):
        raise TypeError(
            "terminated must be a torch.Tensor"
        )

    if rewards.ndim != 1:
        raise ValueError(
            "rewards must be one-dimensional"
        )

    if tuple(values.shape) != (
        tuple(rewards.shape)
    ):
        raise ValueError(
            "values must match rewards"
        )

    if tuple(terminated.shape) != (
        tuple(rewards.shape)
    ):
        raise ValueError(
            "terminated must match rewards"
        )

    if rewards.numel() <= 0:
        raise ValueError(
            "rollout must not be empty"
        )

    if not rewards.is_floating_point():
        raise TypeError(
            "rewards must be floating point"
        )

    if not values.is_floating_point():
        raise TypeError(
            "values must be floating point"
        )

    if terminated.dtype != torch.bool:
        raise TypeError(
            "terminated must use torch.bool"
        )

    if not bool(
        torch.isfinite(rewards).all().item()
    ):
        raise ValueError(
            "rewards must be finite"
        )

    if not bool(
        torch.isfinite(values).all().item()
    ):
        raise ValueError(
            "values must be finite"
        )

    validated_last_value = _finite_number(
        last_value,
        field_name="last_value",
    )

    validated_gamma = _unit_interval(
        gamma,
        field_name="gamma",
    )

    validated_lambda = _unit_interval(
        gae_lambda,
        field_name="gae_lambda",
    )

    advantages = torch.zeros_like(
        rewards
    )

    next_value = torch.as_tensor(
        validated_last_value,
        dtype=values.dtype,
        device=values.device,
    )

    running_advantage = torch.zeros(
        (),
        dtype=values.dtype,
        device=values.device,
    )

    for index in range(
        int(rewards.shape[0]) - 1,
        -1,
        -1,
    ):
        nonterminal = (
            ~terminated[index]
        ).to(
            dtype=values.dtype
        )

        delta = (
            rewards[index]
            + validated_gamma
            * next_value
            * nonterminal
            - values[index]
        )

        running_advantage = (
            delta
            + validated_gamma
            * validated_lambda
            * nonterminal
            * running_advantage
        )

        advantages[index] = (
            running_advantage
        )

        next_value = values[index]

    returns = advantages + values

    if not bool(
        torch.isfinite(advantages).all().item()
    ):
        raise RuntimeError(
            "computed advantages are not finite"
        )

    if not bool(
        torch.isfinite(returns).all().item()
    ):
        raise RuntimeError(
            "computed returns are not finite"
        )

    return advantages, returns


@dataclass(frozen=True)
class RolloutTransition:
    """One action decision and authoritative reward."""

    frames: torch.Tensor
    action_index: int
    reward: float
    terminated: bool
    truncated: bool
    old_log_prob: float
    old_value: float
    duration_ticks: int = 1

    def __post_init__(self) -> None:
        if not isinstance(
            self.frames,
            torch.Tensor,
        ):
            raise TypeError(
                "frames must be a torch.Tensor"
            )

        if tuple(self.frames.shape) != (
            PPO_FRAME_SHAPE
        ):
            raise ValueError(
                "frames have an unexpected shape"
            )

        if self.frames.dtype != torch.float32:
            raise TypeError(
                "frames must use torch.float32"
            )

        if not bool(
            torch.isfinite(
                self.frames
            ).all().item()
        ):
            raise ValueError(
                "frames must contain only finite values"
            )

        if (
            isinstance(self.action_index, bool)
            or not isinstance(
                self.action_index,
                int,
            )
        ):
            raise TypeError(
                "action_index must be an integer"
            )

        try:
            DiscreteAction(
                self.action_index
            )
        except ValueError as error:
            raise ValueError(
                "action_index is not a known action"
            ) from error

        _finite_number(
            self.reward,
            field_name="reward",
        )

        _finite_number(
            self.old_log_prob,
            field_name="old_log_prob",
        )

        _finite_number(
            self.old_value,
            field_name="old_value",
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

        if self.terminated and self.truncated:
            raise ValueError(
                "transition cannot be both "
                "terminated and truncated"
            )

        _positive_integer(
            self.duration_ticks,
            field_name="duration_ticks",
        )


@dataclass(frozen=True)
class PPORolloutBatch:
    """Tensor batch containing one bounded rollout."""

    frames: torch.Tensor
    action_indices: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(
            self.frames,
            torch.Tensor,
        ):
            raise TypeError(
                "frames must be a torch.Tensor"
            )

        if self.frames.ndim != 5:
            raise ValueError(
                "batch frames must have five dimensions"
            )

        if tuple(self.frames.shape[1:]) != (
            PPO_FRAME_SHAPE
        ):
            raise ValueError(
                "batch frames have an unexpected shape"
            )

        batch_size = int(
            self.frames.shape[0]
        )

        if batch_size <= 0:
            raise ValueError(
                "rollout batch must not be empty"
            )

        if self.frames.dtype != torch.float32:
            raise TypeError(
                "batch frames must use torch.float32"
            )

        vectors = (
            (
                "action_indices",
                self.action_indices,
            ),
            ("rewards", self.rewards),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
            (
                "old_log_probs",
                self.old_log_probs,
            ),
            ("old_values", self.old_values),
            ("advantages", self.advantages),
            ("returns", self.returns),
        )

        for field_name, tensor in vectors:
            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                raise TypeError(
                    f"{field_name} must be a torch.Tensor"
                )

            if tuple(tensor.shape) != (
                batch_size,
            ):
                raise ValueError(
                    f"{field_name} must match batch size"
                )

        if (
            self.action_indices.dtype
            != torch.long
        ):
            raise TypeError(
                "action_indices must use torch.long"
            )

        if self.terminated.dtype != torch.bool:
            raise TypeError(
                "terminated must use torch.bool"
            )

        if self.truncated.dtype != torch.bool:
            raise TypeError(
                "truncated must use torch.bool"
            )

        floating_tensors = (
            self.frames,
            self.rewards,
            self.old_log_probs,
            self.old_values,
            self.advantages,
            self.returns,
        )

        for tensor in floating_tensors:
            if not tensor.is_floating_point():
                raise TypeError(
                    "rollout numeric tensors must "
                    "be floating point"
                )

            if not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise ValueError(
                    "rollout tensors must be finite"
                )

        if bool(
            (
                self.terminated
                & self.truncated
            ).any().item()
        ):
            raise ValueError(
                "a transition cannot be both "
                "terminated and truncated"
            )


class RolloutBuffer:
    """Collect exactly one bounded, ordered rollout."""

    def __init__(
        self,
        max_steps: int,
    ) -> None:
        self.max_steps = _positive_integer(
            max_steps,
            field_name="max_steps",
        )

        self._transitions: list[
            RolloutTransition
        ] = []

        self._closed = False
        self._finished = False

    @property
    def size(self) -> int:
        return len(self._transitions)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def finished(self) -> bool:
        return self._finished

    def append(
        self,
        transition: RolloutTransition,
    ) -> None:
        """Append one validated transition in collection order."""

        if not isinstance(
            transition,
            RolloutTransition,
        ):
            raise TypeError(
                "transition must be a RolloutTransition"
            )

        if self._closed:
            raise RuntimeError(
                "rollout buffer is already closed"
            )

        copied_frames = (
            transition.frames.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
            .clone()
        )

        copied = RolloutTransition(
            frames=copied_frames,
            action_index=(
                transition.action_index
            ),
            reward=float(
                transition.reward
            ),
            terminated=(
                transition.terminated
            ),
            truncated=(
                transition.truncated
            ),
            old_log_prob=float(
                transition.old_log_prob
            ),
            old_value=float(
                transition.old_value
            ),
            duration_ticks=(
                transition.duration_ticks
            ),
        )

        self._transitions.append(
            copied
        )

        if (
            copied.terminated
            or copied.truncated
            or len(self._transitions)
            >= self.max_steps
        ):
            self._closed = True

    def finish(
        self,
        *,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ) -> PPORolloutBatch:
        """Finalize the buffer and compute GAE exactly once."""

        if self._finished:
            raise RuntimeError(
                "rollout buffer was already finished"
            )

        if not self._transitions:
            raise RuntimeError(
                "cannot finish an empty rollout"
            )

        if not isinstance(
            normalize_advantages,
            bool,
        ):
            raise TypeError(
                "normalize_advantages must be bool"
            )

        frames = torch.stack(
            [
                transition.frames
                for transition
                in self._transitions
            ],
            dim=0,
        )

        action_indices = torch.tensor(
            [
                transition.action_index
                for transition
                in self._transitions
            ],
            dtype=torch.long,
        )

        rewards = torch.tensor(
            [
                transition.reward
                for transition
                in self._transitions
            ],
            dtype=torch.float32,
        )

        terminated = torch.tensor(
            [
                transition.terminated
                for transition
                in self._transitions
            ],
            dtype=torch.bool,
        )

        truncated = torch.tensor(
            [
                transition.truncated
                for transition
                in self._transitions
            ],
            dtype=torch.bool,
        )

        old_log_probs = torch.tensor(
            [
                transition.old_log_prob
                for transition
                in self._transitions
            ],
            dtype=torch.float32,
        )

        old_values = torch.tensor(
            [
                transition.old_value
                for transition
                in self._transitions
            ],
            dtype=torch.float32,
        )

        raw_advantages, returns = compute_gae(
            rewards,
            old_values,
            terminated,
            last_value=last_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        advantages = raw_advantages

        if (
            normalize_advantages
            and advantages.numel() > 1
        ):
            mean = advantages.mean()

            standard_deviation = (
                advantages.std(
                    unbiased=False
                )
            )

            advantages = (
                advantages - mean
            ) / (
                standard_deviation
                + 1e-8
            )

        self._closed = True
        self._finished = True

        return PPORolloutBatch(
            frames=frames,
            action_indices=action_indices,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
        )


@dataclass(frozen=True)
class PPOHyperparameters:
    """Validated PPO objective settings."""

    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    value_clip_epsilon: float | None = 0.2

    def __post_init__(self) -> None:
        clip = _unit_interval(
            self.clip_epsilon,
            field_name="clip_epsilon",
            include_zero=False,
        )

        value_coefficient = _finite_number(
            self.value_coefficient,
            field_name="value_coefficient",
        )

        entropy_coefficient = _finite_number(
            self.entropy_coefficient,
            field_name="entropy_coefficient",
        )

        if value_coefficient < 0.0:
            raise ValueError(
                "value_coefficient must be nonnegative"
            )

        if entropy_coefficient < 0.0:
            raise ValueError(
                "entropy_coefficient must be nonnegative"
            )

        object.__setattr__(
            self,
            "clip_epsilon",
            clip,
        )

        object.__setattr__(
            self,
            "value_coefficient",
            value_coefficient,
        )

        object.__setattr__(
            self,
            "entropy_coefficient",
            entropy_coefficient,
        )

        if self.value_clip_epsilon is not None:
            value_clip = _unit_interval(
                self.value_clip_epsilon,
                field_name="value_clip_epsilon",
                include_zero=False,
            )

            object.__setattr__(
                self,
                "value_clip_epsilon",
                value_clip,
            )


@dataclass(frozen=True)
class PPOLossTensors:
    """Differentiable PPO losses plus detached diagnostics."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    explained_variance: torch.Tensor


def ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    batch: PPORolloutBatch,
    *,
    hyperparameters: (
        PPOHyperparameters | None
    ) = None,
) -> PPOLossTensors:
    """Return the clipped PPO objective for one batch."""

    if not isinstance(
        batch,
        PPORolloutBatch,
    ):
        raise TypeError(
            "batch must be a PPORolloutBatch"
        )

    if not isinstance(
        logits,
        torch.Tensor,
    ):
        raise TypeError(
            "logits must be a torch.Tensor"
        )

    if not isinstance(
        values,
        torch.Tensor,
    ):
        raise TypeError(
            "values must be a torch.Tensor"
        )

    batch_size = int(
        batch.frames.shape[0]
    )

    if tuple(logits.shape) != (
        batch_size,
        PPO_ACTION_COUNT,
    ):
        raise ValueError(
            "logits have an unexpected shape"
        )

    if tuple(values.shape) != (
        batch_size,
    ):
        raise ValueError(
            "values have an unexpected shape"
        )

    if not logits.is_floating_point():
        raise TypeError(
            "logits must be floating point"
        )

    if not values.is_floating_point():
        raise TypeError(
            "values must be floating point"
        )

    if not bool(
        torch.isfinite(logits).all().item()
    ):
        raise ValueError(
            "logits must be finite"
        )

    if not bool(
        torch.isfinite(values).all().item()
    ):
        raise ValueError(
            "values must be finite"
        )

    settings = (
        hyperparameters
        if hyperparameters is not None
        else PPOHyperparameters()
    )

    if not isinstance(
        settings,
        PPOHyperparameters,
    ):
        raise TypeError(
            "hyperparameters must be "
            "PPOHyperparameters"
        )

    actions = batch.action_indices.to(
        device=logits.device,
    )

    old_log_probs = (
        batch.old_log_probs.to(
            device=logits.device,
            dtype=logits.dtype,
        )
    )

    old_values = batch.old_values.to(
        device=values.device,
        dtype=values.dtype,
    )

    advantages = batch.advantages.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    returns = batch.returns.to(
        device=values.device,
        dtype=values.dtype,
    )

    distribution = Categorical(
        logits=logits
    )

    new_log_probs = (
        distribution.log_prob(actions)
    )

    ratios = torch.exp(
        new_log_probs - old_log_probs
    )

    unclipped_surrogate = (
        ratios * advantages
    )

    clipped_ratios = torch.clamp(
        ratios,
        1.0 - settings.clip_epsilon,
        1.0 + settings.clip_epsilon,
    )

    clipped_surrogate = (
        clipped_ratios * advantages
    )

    policy_loss = -torch.minimum(
        unclipped_surrogate,
        clipped_surrogate,
    ).mean()

    if settings.value_clip_epsilon is None:
        value_error = (
            values - returns
        ).pow(2)
    else:
        clipped_values = (
            old_values
            + torch.clamp(
                values - old_values,
                -settings.value_clip_epsilon,
                settings.value_clip_epsilon,
            )
        )

        original_error = (
            values - returns
        ).pow(2)

        clipped_error = (
            clipped_values - returns
        ).pow(2)

        value_error = torch.maximum(
            original_error,
            clipped_error,
        )

    value_loss = (
        0.5 * value_error.mean()
    )

    entropy = (
        distribution.entropy().mean()
    )

    total_loss = (
        policy_loss
        + settings.value_coefficient
        * value_loss
        - settings.entropy_coefficient
        * entropy
    )

    approximate_kl = (
        old_log_probs
        - new_log_probs
    ).mean()

    clip_fraction = (
        (
            torch.abs(ratios - 1.0)
            > settings.clip_epsilon
        )
        .to(dtype=logits.dtype)
        .mean()
    )

    return_variance = torch.var(
        returns,
        unbiased=False,
    )

    if float(
        return_variance.detach().item()
    ) <= 1e-12:
        explained_variance = torch.zeros(
            (),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        residual_variance = torch.var(
            returns - values,
            unbiased=False,
        )

        explained_variance = (
            1.0
            - residual_variance
            / return_variance
        )

    for name, tensor in (
        ("total_loss", total_loss),
        ("policy_loss", policy_loss),
        ("value_loss", value_loss),
        ("entropy", entropy),
        ("approximate_kl", approximate_kl),
        ("clip_fraction", clip_fraction),
        (
            "explained_variance",
            explained_variance,
        ),
    ):
        if tensor.ndim != 0:
            raise RuntimeError(
                f"{name} must be scalar"
            )

        if not bool(
            torch.isfinite(tensor).item()
        ):
            raise RuntimeError(
                f"{name} is not finite"
            )

    return PPOLossTensors(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        explained_variance=(
            explained_variance
        ),
    )


@dataclass(frozen=True)
class PPOMetrics:
    """Detached measurements from one PPO batch."""

    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    sample_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "clip_fraction",
            "explained_variance",
        ):
            _finite_number(
                getattr(self, field_name),
                field_name=field_name,
            )

        if self.value_loss < 0.0:
            raise ValueError(
                "value_loss must be nonnegative"
            )

        if self.entropy < 0.0:
            raise ValueError(
                "entropy must be nonnegative"
            )

        if not (
            0.0
            <= self.clip_fraction
            <= 1.0
        ):
            raise ValueError(
                "clip_fraction must be between "
                "zero and one"
            )

        _positive_integer(
            self.sample_count,
            field_name="sample_count",
        )


@dataclass(frozen=True)
class PPOBatchResult:
    """Result of one evaluation or optimizer operation."""

    metrics: PPOMetrics
    optimizer_step: bool
    optimizer_step_count: int
    gradient_norm: float | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.metrics,
            PPOMetrics,
        ):
            raise TypeError(
                "metrics must be PPOMetrics"
            )

        if not isinstance(
            self.optimizer_step,
            bool,
        ):
            raise TypeError(
                "optimizer_step must be bool"
            )

        if (
            isinstance(
                self.optimizer_step_count,
                bool,
            )
            or not isinstance(
                self.optimizer_step_count,
                int,
            )
            or self.optimizer_step_count < 0
        ):
            raise ValueError(
                "optimizer_step_count must be "
                "a nonnegative integer"
            )

        if self.gradient_norm is not None:
            gradient_norm = _finite_number(
                self.gradient_norm,
                field_name="gradient_norm",
            )

            if gradient_norm < 0.0:
                raise ValueError(
                    "gradient_norm must be nonnegative"
                )


def _gradient_norm(
    model: VisualActorCriticNetwork,
) -> float:
    squared_norm = 0.0
    gradient_count = 0

    for parameter in model.parameters():
        gradient = parameter.grad

        if gradient is None:
            continue

        gradient_count += 1

        if not bool(
            torch.isfinite(
                gradient
            ).all().item()
        ):
            raise RuntimeError(
                "model gradients are not finite"
            )

        squared_norm += float(
            gradient.detach()
            .double()
            .pow(2)
            .sum()
            .item()
        )

    if gradient_count == 0:
        raise RuntimeError(
            "no gradients were produced"
        )

    return math.sqrt(
        squared_norm
    )


class PPOTrainer:
    """Perform explicit PPO evaluation or one optimizer batch."""

    def __init__(
        self,
        model: VisualActorCriticNetwork,
        optimizer: torch.optim.Optimizer,
        *,
        device: str | torch.device = "cpu",
        hyperparameters: (
            PPOHyperparameters | None
        ) = None,
        max_gradient_norm: float = 0.5,
        optimizer_step_count: int = 0,
    ) -> None:
        if not isinstance(
            model,
            VisualActorCriticNetwork,
        ):
            raise TypeError(
                "model must be a "
                "VisualActorCriticNetwork"
            )

        if not isinstance(
            optimizer,
            torch.optim.Optimizer,
        ):
            raise TypeError(
                "optimizer must be a torch optimizer"
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

        settings = (
            hyperparameters
            if hyperparameters is not None
            else PPOHyperparameters()
        )

        if not isinstance(
            settings,
            PPOHyperparameters,
        ):
            raise TypeError(
                "hyperparameters must be "
                "PPOHyperparameters"
            )

        if (
            isinstance(max_gradient_norm, bool)
            or not isinstance(
                max_gradient_norm,
                (int, float),
            )
            or not math.isfinite(
                float(max_gradient_norm)
            )
            or float(max_gradient_norm) <= 0.0
        ):
            raise ValueError(
                "max_gradient_norm must be "
                "finite and positive"
            )

        validated_gradient_norm = float(
            max_gradient_norm
        )

        model_parameter_ids = {
            id(parameter)
            for parameter in model.parameters()
        }

        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        if not optimizer_parameter_ids:
            raise ValueError(
                "optimizer contains no parameters"
            )

        if not optimizer_parameter_ids.issubset(
            model_parameter_ids
        ):
            raise ValueError(
                "optimizer contains parameters "
                "from another model"
            )

        self.model = model.to(
            resolved_device
        )

        self.optimizer = optimizer
        self.device = resolved_device
        self.hyperparameters = settings
        self.max_gradient_norm = (
            validated_gradient_norm
        )

        if (
            isinstance(optimizer_step_count, bool)
            or not isinstance(
                optimizer_step_count,
                int,
            )
            or optimizer_step_count < 0
        ):
            raise ValueError(
                "optimizer_step_count must be "
                "a nonnegative integer"
            )

        self._optimizer_step_count = (
            optimizer_step_count
        )

    @property
    def optimizer_step_count(self) -> int:
        return self._optimizer_step_count

    def _move_batch(
        self,
        batch: PPORolloutBatch,
    ) -> PPORolloutBatch:
        if not isinstance(
            batch,
            PPORolloutBatch,
        ):
            raise TypeError(
                "batch must be a PPORolloutBatch"
            )

        return PPORolloutBatch(
            frames=batch.frames.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            action_indices=(
                batch.action_indices.to(
                    device=self.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            ),
            rewards=batch.rewards.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            terminated=(
                batch.terminated.to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
            ),
            truncated=batch.truncated.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            ),
            old_log_probs=(
                batch.old_log_probs.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            ),
            old_values=batch.old_values.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            advantages=batch.advantages.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            returns=batch.returns.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        )

    @staticmethod
    def _metrics(
        losses: PPOLossTensors,
        *,
        sample_count: int,
    ) -> PPOMetrics:
        return PPOMetrics(
            total_loss=float(
                losses.total_loss
                .detach()
                .item()
            ),
            policy_loss=float(
                losses.policy_loss
                .detach()
                .item()
            ),
            value_loss=float(
                losses.value_loss
                .detach()
                .item()
            ),
            entropy=float(
                losses.entropy
                .detach()
                .item()
            ),
            approximate_kl=float(
                losses.approximate_kl
                .detach()
                .item()
            ),
            clip_fraction=float(
                losses.clip_fraction
                .detach()
                .item()
            ),
            explained_variance=float(
                losses.explained_variance
                .detach()
                .item()
            ),
            sample_count=sample_count,
        )

    def evaluate_batch(
        self,
        batch: PPORolloutBatch,
    ) -> PPOBatchResult:
        """Evaluate one rollout without updating parameters."""

        moved = self._move_batch(batch)

        self.model.eval()

        with torch.inference_mode():
            logits, values = self.model(
                moved.frames
            )

            losses = ppo_loss(
                logits,
                values,
                moved,
                hyperparameters=(
                    self.hyperparameters
                ),
            )

        return PPOBatchResult(
            metrics=self._metrics(
                losses,
                sample_count=int(
                    moved.frames.shape[0]
                ),
            ),
            optimizer_step=False,
            optimizer_step_count=(
                self._optimizer_step_count
            ),
            gradient_norm=None,
        )

    def train_batch(
        self,
        batch: PPORolloutBatch,
    ) -> PPOBatchResult:
        """Perform exactly one PPO optimizer operation."""

        moved = self._move_batch(batch)

        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        logits, values = self.model(
            moved.frames
        )

        losses = ppo_loss(
            logits,
            values,
            moved,
            hyperparameters=(
                self.hyperparameters
            ),
        )

        losses.total_loss.backward()

        gradient_norm = _gradient_norm(
            self.model
        )

        clipped_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=(
                    self.max_gradient_norm
                ),
                error_if_nonfinite=True,
            )
        )

        if not bool(
            torch.isfinite(
                clipped_norm
            ).item()
        ):
            raise RuntimeError(
                "clipped gradient norm is not finite"
            )

        self.optimizer.step()

        self._optimizer_step_count += 1

        return PPOBatchResult(
            metrics=self._metrics(
                losses,
                sample_count=int(
                    moved.frames.shape[0]
                ),
            ),
            optimizer_step=True,
            optimizer_step_count=(
                self._optimizer_step_count
            ),
            gradient_norm=gradient_norm,
        )
