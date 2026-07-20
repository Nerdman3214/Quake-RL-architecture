"""Behavior-cloning objective and bounded batch trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F

from RL.actions.contracts import DiscreteAction
from RL.agents.policies.visual_network import (
    VisualPolicyNetwork,
)
from RL.training.imitation.dataset import (
    DemonstrationBatch,
)


ACTION_COUNT = len(
    tuple(DiscreteAction)
)


@dataclass(frozen=True)
class BehaviorCloningMetrics:
    """Detached measurements from one supervised action batch."""

    loss: float
    accuracy: float
    mean_confidence: float
    sample_count: int
    correct_count: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("loss", self.loss),
            ("accuracy", self.accuracy),
            (
                "mean_confidence",
                self.mean_confidence,
            ),
        ):
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

        if self.loss < 0.0:
            raise ValueError(
                "loss must not be negative"
            )

        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(
                "accuracy must be between zero and one"
            )

        if not (
            0.0
            <= self.mean_confidence
            <= 1.0
        ):
            raise ValueError(
                "mean_confidence must be between "
                "zero and one"
            )

        if (
            isinstance(self.sample_count, bool)
            or not isinstance(
                self.sample_count,
                int,
            )
            or self.sample_count <= 0
        ):
            raise ValueError(
                "sample_count must be positive"
            )

        if (
            isinstance(self.correct_count, bool)
            or not isinstance(
                self.correct_count,
                int,
            )
            or not (
                0
                <= self.correct_count
                <= self.sample_count
            )
        ):
            raise ValueError(
                "correct_count is invalid"
            )


@dataclass(frozen=True)
class BehaviorCloningStepResult:
    """Result of one evaluation or optimizer batch."""

    metrics: BehaviorCloningMetrics
    optimizer_step: bool
    optimizer_step_count: int
    gradient_norm: float | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.metrics,
            BehaviorCloningMetrics,
        ):
            raise TypeError(
                "metrics must be BehaviorCloningMetrics"
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
            if (
                isinstance(
                    self.gradient_norm,
                    bool,
                )
                or not isinstance(
                    self.gradient_norm,
                    (int, float),
                )
                or not math.isfinite(
                    float(self.gradient_norm)
                )
                or self.gradient_norm < 0.0
            ):
                raise ValueError(
                    "gradient_norm must be finite "
                    "and nonnegative"
                )


def _validate_logits_and_targets(
    logits: torch.Tensor,
    action_indices: torch.Tensor,
) -> None:
    if not isinstance(
        logits,
        torch.Tensor,
    ):
        raise TypeError(
            "logits must be a torch.Tensor"
        )

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "(batch, action_count)"
        )

    if logits.shape[0] <= 0:
        raise ValueError(
            "logits batch must not be empty"
        )

    if logits.shape[1] != ACTION_COUNT:
        raise ValueError(
            "logits action dimension must match "
            "DiscreteAction"
        )

    if not logits.is_floating_point():
        raise TypeError(
            "logits must use a floating-point dtype"
        )

    if not bool(
        torch.isfinite(logits).all().item()
    ):
        raise ValueError(
            "logits must contain only finite values"
        )

    if not isinstance(
        action_indices,
        torch.Tensor,
    ):
        raise TypeError(
            "action_indices must be a torch.Tensor"
        )

    if (
        action_indices.ndim != 1
        or action_indices.dtype
        != torch.long
    ):
        raise TypeError(
            "action_indices must be a torch.long vector"
        )

    if action_indices.shape[0] != (
        logits.shape[0]
    ):
        raise ValueError(
            "action_indices batch size must "
            "match logits"
        )

    minimum = int(
        action_indices.min().item()
    )
    maximum = int(
        action_indices.max().item()
    )

    if minimum < 0 or maximum >= ACTION_COUNT:
        raise ValueError(
            "action_indices contain an unknown action"
        )


def _validated_class_weights(
    class_weights: (
        torch.Tensor
        | Sequence[float]
        | None
    ),
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if class_weights is None:
        return None

    if isinstance(
        class_weights,
        torch.Tensor,
    ):
        weights = class_weights.detach().to(
            device=device,
            dtype=dtype,
        )
    else:
        try:
            weights = torch.tensor(
                list(class_weights),
                device=device,
                dtype=dtype,
            )
        except (TypeError, ValueError) as error:
            raise TypeError(
                "class_weights must be a tensor "
                "or numeric sequence"
            ) from error

    if weights.ndim != 1:
        raise ValueError(
            "class_weights must be one-dimensional"
        )

    if tuple(weights.shape) != (
        ACTION_COUNT,
    ):
        raise ValueError(
            "class_weights length must match "
            "DiscreteAction"
        )

    if not bool(
        torch.isfinite(weights).all().item()
    ):
        raise ValueError(
            "class_weights must contain only "
            "finite values"
        )

    if bool((weights < 0).any().item()):
        raise ValueError(
            "class_weights must not be negative"
        )

    if not bool((weights > 0).any().item()):
        raise ValueError(
            "at least one class weight must "
            "be positive"
        )

    return weights


def behavior_cloning_loss(
    logits: torch.Tensor,
    action_indices: torch.Tensor,
    *,
    class_weights: (
        torch.Tensor
        | Sequence[float]
        | None
    ) = None,
) -> torch.Tensor:
    """Return strict cross-entropy over demonstrated actions."""

    _validate_logits_and_targets(
        logits,
        action_indices,
    )

    weights = _validated_class_weights(
        class_weights,
        device=logits.device,
        dtype=logits.dtype,
    )

    loss = F.cross_entropy(
        logits,
        action_indices,
        weight=weights,
    )

    if not bool(
        torch.isfinite(loss).item()
    ):
        raise RuntimeError(
            "behavior-cloning loss is not finite"
        )

    return loss


def behavior_cloning_metrics(
    logits: torch.Tensor,
    action_indices: torch.Tensor,
    *,
    loss: torch.Tensor | None = None,
) -> BehaviorCloningMetrics:
    """Return detached classification metrics."""

    _validate_logits_and_targets(
        logits,
        action_indices,
    )

    resolved_loss = (
        behavior_cloning_loss(
            logits,
            action_indices,
        )
        if loss is None
        else loss
    )

    if (
        not isinstance(
            resolved_loss,
            torch.Tensor,
        )
        or resolved_loss.ndim != 0
        or not bool(
            torch.isfinite(
                resolved_loss
            ).item()
        )
    ):
        raise ValueError(
            "loss must be a finite scalar tensor"
        )

    with torch.no_grad():
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidences, predictions = (
            probabilities.max(dim=1)
        )

        correct_count = int(
            (
                predictions
                == action_indices
            ).sum().item()
        )

        sample_count = int(
            logits.shape[0]
        )

    return BehaviorCloningMetrics(
        loss=float(
            resolved_loss.detach().item()
        ),
        accuracy=(
            correct_count / sample_count
        ),
        mean_confidence=float(
            confidences.mean().item()
        ),
        sample_count=sample_count,
        correct_count=correct_count,
    )


def _resolve_device(
    device: str | torch.device,
) -> torch.device:
    resolved = torch.device(device)

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return resolved


def _gradient_norm(
    model: VisualPolicyNetwork,
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
            "no model gradients were produced"
        )

    return math.sqrt(
        squared_norm
    )


class BehaviorCloningTrainer:
    """Execute explicit single-batch supervised updates.

    This class does not contain an epoch loop, dataset traversal,
    checkpoint schedule, or automatic training lifecycle.
    """

    def __init__(
        self,
        model: VisualPolicyNetwork,
        optimizer: torch.optim.Optimizer,
        *,
        device: str | torch.device = "cpu",
        class_weights: (
            torch.Tensor
            | Sequence[float]
            | None
        ) = None,
        max_gradient_norm: float | None = None,
    ) -> None:
        if not isinstance(
            model,
            VisualPolicyNetwork,
        ):
            raise TypeError(
                "model must be a VisualPolicyNetwork"
            )

        if not isinstance(
            optimizer,
            torch.optim.Optimizer,
        ):
            raise TypeError(
                "optimizer must be a torch optimizer"
            )

        if max_gradient_norm is not None:
            if (
                isinstance(
                    max_gradient_norm,
                    bool,
                )
                or not isinstance(
                    max_gradient_norm,
                    (int, float),
                )
                or not math.isfinite(
                    float(max_gradient_norm)
                )
                or max_gradient_norm <= 0
            ):
                raise ValueError(
                    "max_gradient_norm must be "
                    "finite and positive"
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

        self.model = model
        self.optimizer = optimizer
        self.device = _resolve_device(
            device
        )
        self.max_gradient_norm = (
            float(max_gradient_norm)
            if max_gradient_norm is not None
            else None
        )

        self.model.to(
            self.device
        )

        self._class_weights = (
            _validated_class_weights(
                class_weights,
                device=self.device,
                dtype=torch.float32,
            )
        )

        self._optimizer_step_count = 0

    @property
    def optimizer_step_count(self) -> int:
        """Return completed explicit optimizer steps."""

        return self._optimizer_step_count

    def _prepare_batch(
        self,
        batch: DemonstrationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(
            batch,
            DemonstrationBatch,
        ):
            raise TypeError(
                "batch must be a DemonstrationBatch"
            )

        frames = batch.frames.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )

        action_indices = (
            batch.action_indices.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )
        )

        return frames, action_indices

    def evaluate_batch(
        self,
        batch: DemonstrationBatch,
    ) -> BehaviorCloningStepResult:
        """Evaluate exactly one batch without updating weights."""

        frames, action_indices = (
            self._prepare_batch(batch)
        )

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(frames)

            loss = behavior_cloning_loss(
                logits,
                action_indices,
                class_weights=(
                    self._class_weights
                ),
            )

            metrics = (
                behavior_cloning_metrics(
                    logits,
                    action_indices,
                    loss=loss,
                )
            )

        return BehaviorCloningStepResult(
            metrics=metrics,
            optimizer_step=False,
            optimizer_step_count=(
                self._optimizer_step_count
            ),
            gradient_norm=None,
        )

    def train_batch(
        self,
        batch: DemonstrationBatch,
    ) -> BehaviorCloningStepResult:
        """Perform exactly one explicit optimizer batch."""

        frames, action_indices = (
            self._prepare_batch(batch)
        )

        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        logits = self.model(frames)

        loss = behavior_cloning_loss(
            logits,
            action_indices,
            class_weights=(
                self._class_weights
            ),
        )

        metrics = behavior_cloning_metrics(
            logits,
            action_indices,
            loss=loss,
        )

        loss.backward()

        gradient_norm = _gradient_norm(
            self.model
        )

        if self.max_gradient_norm is not None:
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
                    "gradient norm is not finite"
                )

        self.optimizer.step()

        self._optimizer_step_count += 1

        return BehaviorCloningStepResult(
            metrics=metrics,
            optimizer_step=True,
            optimizer_step_count=(
                self._optimizer_step_count
            ),
            gradient_norm=gradient_norm,
        )
