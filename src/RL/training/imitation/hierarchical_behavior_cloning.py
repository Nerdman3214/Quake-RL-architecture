"""Masked multitask behavior cloning for composite recurrent controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from RL.actions.contracts import DiscreteAction
from RL.agents.policies.hierarchical_recurrent import (
    AXIS_CLASS_COUNT,
    WEAPON_CLASS_COUNT,
    HierarchicalPolicyOutput,
    HierarchicalRecurrentPolicy,
)
from RL.training.imitation.composite_sequence_dataset import (
    CompositeSequenceBatch,
)


LEGACY_ACTION_COUNT = len(tuple(DiscreteAction))


@dataclass(frozen=True)
class HierarchicalLossWeights:
    """Weights for authoritative composite-control objectives."""

    forward: float = 1.0
    strafe: float = 1.0
    mouse: float = 0.01
    fire: float = 1.0
    jump: float = 1.0
    weapon: float = 0.5
    legacy_action: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.forward,
            self.strafe,
            self.mouse,
            self.fire,
            self.jump,
            self.weapon,
            self.legacy_action,
        )

        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(
                    "loss weights must be finite "
                    "and nonnegative"
                )

        if not any(value > 0.0 for value in values):
            raise ValueError(
                "at least one loss weight must be positive"
            )


@dataclass(frozen=True)
class HierarchicalClassWeights:
    """Within-head weights for imbalanced control labels."""

    fire_positive: float = 1.0
    jump_positive: float = 1.0
    weapon_previous: float = 1.0
    weapon_neutral: float = 1.0
    weapon_next: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.fire_positive,
            self.jump_positive,
            self.weapon_previous,
            self.weapon_neutral,
            self.weapon_next,
        )

        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(
                    "class weights must be finite "
                    "and positive"
                )


@dataclass(frozen=True)
class HierarchicalBehaviorCloningLosses:
    """Differentiable component and total losses."""

    total: torch.Tensor
    forward: torch.Tensor
    strafe: torch.Tensor
    mouse: torch.Tensor
    fire: torch.Tensor
    jump: torch.Tensor
    weapon: torch.Tensor
    legacy_action: torch.Tensor


@dataclass(frozen=True)
class HierarchicalBehaviorCloningMetrics:
    """Detached measurements for one recurrent batch."""

    loss: float
    forward_accuracy: float
    strafe_accuracy: float
    fire_accuracy: float
    jump_accuracy: float
    weapon_accuracy: float
    legacy_action_accuracy: float
    mouse_mean_absolute_error: float
    valid_step_count: int


@dataclass(frozen=True)
class HierarchicalBehaviorCloningStepResult:
    """Result of one evaluation or optimizer operation."""

    losses: HierarchicalBehaviorCloningLosses
    metrics: HierarchicalBehaviorCloningMetrics
    optimizer_step: bool
    optimizer_step_count: int
    gradient_norm: float | None


def _validate_output_and_batch(
    output: HierarchicalPolicyOutput,
    batch: CompositeSequenceBatch,
) -> tuple[int, int]:
    if not isinstance(
        output,
        HierarchicalPolicyOutput,
    ):
        raise TypeError(
            "output must be HierarchicalPolicyOutput"
        )

    if not isinstance(
        batch,
        CompositeSequenceBatch,
    ):
        raise TypeError(
            "batch must be CompositeSequenceBatch"
        )

    batch_size = int(batch.frames.shape[0])
    sequence_length = int(batch.frames.shape[1])

    expected_shapes = {
        "forward_logits": (
            batch_size,
            sequence_length,
            AXIS_CLASS_COUNT,
        ),
        "strafe_logits": (
            batch_size,
            sequence_length,
            AXIS_CLASS_COUNT,
        ),
        "mouse_deltas": (
            batch_size,
            sequence_length,
            2,
        ),
        "fire_logits": (
            batch_size,
            sequence_length,
        ),
        "jump_logits": (
            batch_size,
            sequence_length,
        ),
        "weapon_logits": (
            batch_size,
            sequence_length,
            WEAPON_CLASS_COUNT,
        ),
        "legacy_action_logits": (
            batch_size,
            sequence_length,
            LEGACY_ACTION_COUNT,
        ),
    }

    for field_name, expected_shape in (
        expected_shapes.items()
    ):
        tensor = getattr(output, field_name)

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{field_name} must be a tensor"
            )

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{field_name} has an unexpected shape"
            )

        if not tensor.is_floating_point():
            raise TypeError(
                f"{field_name} must be floating point"
            )

        if not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(
                f"{field_name} must contain only "
                "finite values"
            )

    return batch_size, sequence_length


def _targets_on_device(
    batch: CompositeSequenceBatch,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "mask": batch.valid_mask.to(
            device=device,
        ),
        "forward": batch.forward_classes.to(
            device=device,
        ),
        "strafe": batch.strafe_classes.to(
            device=device,
        ),
        "mouse": batch.mouse_deltas.to(
            device=device,
        ),
        "fire": batch.fire_targets.to(
            device=device,
        ),
        "jump": batch.jump_targets.to(
            device=device,
        ),
        "weapon": batch.weapon_classes.to(
            device=device,
        ),
        "legacy": (
            batch.legacy_action_indices.to(
                device=device,
            )
        ),
    }


def hierarchical_behavior_cloning_loss(
    output: HierarchicalPolicyOutput,
    batch: CompositeSequenceBatch,
    *,
    weights: HierarchicalLossWeights | None = None,
    class_weights: HierarchicalClassWeights | None = None,
) -> HierarchicalBehaviorCloningLosses:
    """Return masked multitask losses.

    Tactical-intent loss is intentionally absent until authoritative
    tactical-intent labels are collected or safely derived.
    """

    _validate_output_and_batch(
        output,
        batch,
    )

    resolved_weights = (
        HierarchicalLossWeights()
        if weights is None
        else weights
    )

    if not isinstance(
        resolved_weights,
        HierarchicalLossWeights,
    ):
        raise TypeError(
            "weights must be HierarchicalLossWeights"
        )

    resolved_class_weights = (
        HierarchicalClassWeights()
        if class_weights is None
        else class_weights
    )

    if not isinstance(
        resolved_class_weights,
        HierarchicalClassWeights,
    ):
        raise TypeError(
            "class_weights must be "
            "HierarchicalClassWeights"
        )

    device = output.forward_logits.device

    targets = _targets_on_device(
        batch,
        device,
    )
    mask = targets["mask"]

    if not bool(mask.any().item()):
        raise ValueError(
            "batch must contain at least one valid step"
        )

    forward_loss = F.cross_entropy(
        output.forward_logits[mask],
        targets["forward"][mask],
    )
    strafe_loss = F.cross_entropy(
        output.strafe_logits[mask],
        targets["strafe"][mask],
    )
    mouse_loss = F.smooth_l1_loss(
        output.mouse_deltas[mask],
        targets["mouse"][mask],
    )
    fire_positive_weight = torch.tensor(
        resolved_class_weights.fire_positive,
        device=device,
        dtype=output.fire_logits.dtype,
    )
    jump_positive_weight = torch.tensor(
        resolved_class_weights.jump_positive,
        device=device,
        dtype=output.jump_logits.dtype,
    )
    weapon_class_weights = torch.tensor(
        (
            resolved_class_weights.weapon_previous,
            resolved_class_weights.weapon_neutral,
            resolved_class_weights.weapon_next,
        ),
        device=device,
        dtype=output.weapon_logits.dtype,
    )

    fire_loss = F.binary_cross_entropy_with_logits(
        output.fire_logits[mask],
        targets["fire"][mask],
        pos_weight=fire_positive_weight,
    )
    jump_loss = F.binary_cross_entropy_with_logits(
        output.jump_logits[mask],
        targets["jump"][mask],
        pos_weight=jump_positive_weight,
    )
    weapon_loss = F.cross_entropy(
        output.weapon_logits[mask],
        targets["weapon"][mask],
        weight=weapon_class_weights,
    )
    legacy_loss = F.cross_entropy(
        output.legacy_action_logits[mask],
        targets["legacy"][mask],
    )

    total = (
        resolved_weights.forward * forward_loss
        + resolved_weights.strafe * strafe_loss
        + resolved_weights.mouse * mouse_loss
        + resolved_weights.fire * fire_loss
        + resolved_weights.jump * jump_loss
        + resolved_weights.weapon * weapon_loss
        + resolved_weights.legacy_action
        * legacy_loss
    )

    if not bool(torch.isfinite(total).item()):
        raise RuntimeError(
            "hierarchical behavior-cloning loss "
            "is not finite"
        )

    return HierarchicalBehaviorCloningLosses(
        total=total,
        forward=forward_loss,
        strafe=strafe_loss,
        mouse=mouse_loss,
        fire=fire_loss,
        jump=jump_loss,
        weapon=weapon_loss,
        legacy_action=legacy_loss,
    )


def hierarchical_behavior_cloning_metrics(
    output: HierarchicalPolicyOutput,
    batch: CompositeSequenceBatch,
    *,
    losses: HierarchicalBehaviorCloningLosses,
) -> HierarchicalBehaviorCloningMetrics:
    """Return detached metrics over valid recurrent steps."""

    _validate_output_and_batch(
        output,
        batch,
    )

    if not isinstance(
        losses,
        HierarchicalBehaviorCloningLosses,
    ):
        raise TypeError(
            "losses must be "
            "HierarchicalBehaviorCloningLosses"
        )

    device = output.forward_logits.device
    targets = _targets_on_device(
        batch,
        device,
    )
    mask = targets["mask"]

    valid_step_count = int(
        mask.sum().item()
    )

    if valid_step_count <= 0:
        raise ValueError(
            "batch must contain at least one valid step"
        )

    def classification_accuracy(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        prediction = logits[mask].argmax(
            dim=-1
        )

        return float(
            (
                prediction == target[mask]
            ).float().mean().item()
        )

    with torch.no_grad():
        fire_prediction = (
            torch.sigmoid(
                output.fire_logits[mask]
            )
            >= 0.5
        )
        jump_prediction = (
            torch.sigmoid(
                output.jump_logits[mask]
            )
            >= 0.5
        )

        fire_target = (
            targets["fire"][mask] >= 0.5
        )
        jump_target = (
            targets["jump"][mask] >= 0.5
        )

        mouse_mae = float(
            (
                output.mouse_deltas[mask]
                - targets["mouse"][mask]
            ).abs().mean().item()
        )

    return HierarchicalBehaviorCloningMetrics(
        loss=float(
            losses.total.detach().item()
        ),
        forward_accuracy=(
            classification_accuracy(
                output.forward_logits,
                targets["forward"],
            )
        ),
        strafe_accuracy=(
            classification_accuracy(
                output.strafe_logits,
                targets["strafe"],
            )
        ),
        fire_accuracy=float(
            (
                fire_prediction == fire_target
            ).float().mean().item()
        ),
        jump_accuracy=float(
            (
                jump_prediction == jump_target
            ).float().mean().item()
        ),
        weapon_accuracy=(
            classification_accuracy(
                output.weapon_logits,
                targets["weapon"],
            )
        ),
        legacy_action_accuracy=(
            classification_accuracy(
                output.legacy_action_logits,
                targets["legacy"],
            )
        ),
        mouse_mean_absolute_error=mouse_mae,
        valid_step_count=valid_step_count,
    )


class HierarchicalBehaviorCloningTrainer:
    """Bounded optimizer for recurrent composite behavior cloning."""

    def __init__(
        self,
        model: HierarchicalRecurrentPolicy,
        optimizer: torch.optim.Optimizer,
        *,
        device: str | torch.device = "cpu",
        weights: HierarchicalLossWeights | None = None,
        class_weights: HierarchicalClassWeights | None = None,
        max_gradient_norm: float = 1.0,
        optimizer_step_count: int = 0,
    ) -> None:
        if not isinstance(
            model,
            HierarchicalRecurrentPolicy,
        ):
            raise TypeError(
                "model must be HierarchicalRecurrentPolicy"
            )

        if not isinstance(
            optimizer,
            torch.optim.Optimizer,
        ):
            raise TypeError(
                "optimizer must be a torch optimizer"
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
            or max_gradient_norm <= 0.0
        ):
            raise ValueError(
                "max_gradient_norm must be finite "
                "and positive"
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

        model_parameter_ids = {
            id(parameter)
            for parameter in model.parameters()
        }
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        if model_parameter_ids != (
            optimizer_parameter_ids
        ):
            raise ValueError(
                "optimizer must reference exactly "
                "the model parameters"
            )

        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.weights = (
            HierarchicalLossWeights()
            if weights is None
            else weights
        )
        self.class_weights = (
            HierarchicalClassWeights()
            if class_weights is None
            else class_weights
        )

        if not isinstance(
            self.class_weights,
            HierarchicalClassWeights,
        ):
            raise TypeError(
                "class_weights must be "
                "HierarchicalClassWeights"
            )

        self.max_gradient_norm = float(
            max_gradient_norm
        )
        self.optimizer_step_count = (
            optimizer_step_count
        )

        self.model.to(self.device)

    def _forward(
        self,
        batch: CompositeSequenceBatch,
    ) -> tuple[
        HierarchicalPolicyOutput,
        HierarchicalBehaviorCloningLosses,
        HierarchicalBehaviorCloningMetrics,
    ]:
        telemetry_features = getattr(
            batch,
            "telemetry_features",
            None,
        )
        telemetry_weapon_indices = getattr(
            batch,
            "telemetry_weapon_indices",
            None,
        )

        output = self.model(
            batch.frames.to(
                device=self.device,
            ),
            batch.previous_action_features.to(
                device=self.device,
            ),
            batch.mode_indices.to(
                device=self.device,
            ),
            telemetry_features=(
                telemetry_features.to(
                    device=self.device,
                )
                if telemetry_features is not None
                else None
            ),
            telemetry_weapon_indices=(
                telemetry_weapon_indices.to(
                    device=self.device,
                )
                if telemetry_weapon_indices
                is not None
                else None
            ),
        )

        losses = hierarchical_behavior_cloning_loss(
            output,
            batch,
            weights=self.weights,
            class_weights=self.class_weights,
        )

        metrics = (
            hierarchical_behavior_cloning_metrics(
                output,
                batch,
                losses=losses,
            )
        )

        return output, losses, metrics

    def train_batch(
        self,
        batch: CompositeSequenceBatch,
    ) -> HierarchicalBehaviorCloningStepResult:
        self.model.train()
        self.optimizer.zero_grad(
            set_to_none=True
        )

        _, losses, metrics = self._forward(
            batch
        )

        losses.total.backward()

        gradient_norm_tensor = (
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_gradient_norm,
            )
        )

        gradient_norm = float(
            gradient_norm_tensor.detach().item()
        )

        if not math.isfinite(gradient_norm):
            self.optimizer.zero_grad(
                set_to_none=True
            )
            raise RuntimeError(
                "gradient norm is not finite"
            )

        self.optimizer.step()
        self.optimizer_step_count += 1

        return HierarchicalBehaviorCloningStepResult(
            losses=losses,
            metrics=metrics,
            optimizer_step=True,
            optimizer_step_count=(
                self.optimizer_step_count
            ),
            gradient_norm=gradient_norm,
        )

    def evaluate_batch(
        self,
        batch: CompositeSequenceBatch,
    ) -> HierarchicalBehaviorCloningStepResult:
        self.model.eval()

        with torch.no_grad():
            _, losses, metrics = self._forward(
                batch
            )

        return HierarchicalBehaviorCloningStepResult(
            losses=losses,
            metrics=metrics,
            optimizer_step=False,
            optimizer_step_count=(
                self.optimizer_step_count
            ),
            gradient_norm=None,
        )
