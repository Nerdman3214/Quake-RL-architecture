#!/usr/bin/env python3
"""Train or resume hierarchical behavior cloning from demonstrations."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from RL.agents import HierarchicalRecurrentPolicy
from RL.training.imitation import (
    CompositeSequenceBatch,
    CompositeSequenceDataset,
    HierarchicalBehaviorCloningTrainer,
    HierarchicalClassWeights,
    HierarchicalLossWeights,
    hierarchical_behavior_cloning_loss,
    load_hierarchical_checkpoint,
    make_composite_sequence_dataloader,
    save_hierarchical_checkpoint,
)

from RL.agents.policies.hierarchical_recurrent import (
    DEFAULT_TELEMETRY_WEAPON_COUNT,
    TELEMETRY_FEATURE_COUNT,
)
from RL.training.imitation.telemetry_sequence_dataset import (
    TelemetryCompositeSequenceDataset,
    make_telemetry_composite_sequence_dataloader,
)


DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_POLICY_NAME = "xonotic-hierarchical-bc"
DEFAULT_POLICY_VERSION = "v1"


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be an integer"
        ) from error

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "must be positive"
        )

    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be an integer"
        ) from error

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "must be nonnegative"
        )

    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a number"
        ) from error

    if (
        not math.isfinite(parsed)
        or parsed <= 0.0
    ):
        raise argparse.ArgumentTypeError(
            "must be finite and positive"
        )

    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a number"
        ) from error

    if (
        not math.isfinite(parsed)
        or parsed < 0.0
    ):
        raise argparse.ArgumentTypeError(
            "must be finite and nonnegative"
        )

    return parsed


@dataclass(frozen=True)
class DatasetEvaluation:
    """Whole-dataset metrics weighted by valid recurrent steps."""

    loss: float
    valid_step_count: int

    forward_accuracy: float
    strafe_accuracy: float
    weapon_accuracy: float

    weapon_previous_precision: float | None
    weapon_previous_recall: float | None
    weapon_previous_positive_count: int

    weapon_neutral_precision: float | None
    weapon_neutral_recall: float | None
    weapon_neutral_positive_count: int

    weapon_next_precision: float | None
    weapon_next_recall: float | None
    weapon_next_positive_count: int

    legacy_action_accuracy: float

    fire_accuracy: float
    fire_precision: float | None
    fire_recall: float | None
    fire_positive_count: int

    jump_accuracy: float
    jump_precision: float | None
    jump_recall: float | None
    jump_positive_count: int

    mouse_x_mean_absolute_error: float
    mouse_y_mean_absolute_error: float

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or resume the hierarchical recurrent "
            "behavior-cloning policy using authoritative "
            "human composite-control demonstrations."
        )
    )

    parser.add_argument(
        "episodes",
        nargs="+",
        type=Path,
        help=(
            "One or more authoritative demonstration "
            "episode JSONL files."
        ),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        required=True,
        help="Atomic destination for the training checkpoint.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume model, Adam state, loss weights, and "
            "optimizer count from this checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-dataset-change",
        action="store_true",
        help=(
            "Allow resumed training to use different episode "
            "paths or sequence settings."
        ),
    )

    parser.add_argument(
        "--sequence-length",
        type=_positive_integer,
        default=8,
    )
    parser.add_argument(
        "--stride",
        type=_positive_integer,
        default=4,
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=_positive_integer,
        default=1,
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=_positive_integer,
        default=100,
        help=(
            "Maximum additional optimizer operations during "
            "this invocation."
        ),
    )
    parser.add_argument(
        "--log-every-steps",
        type=_positive_integer,
        default=10,
    )
    parser.add_argument(
        "--num-workers",
        type=_nonnegative_integer,
        default=0,
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--torch-threads",
        type=_positive_integer,
        default=max(
            1,
            min(4, os.cpu_count() or 1),
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260724,
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=None,
        help=(
            "For new training, defaults to 0.0003. For "
            "resume mode, omission preserves the saved rate."
        ),
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=_positive_float,
        default=None,
        help=(
            "Defaults to 1.0 for new training. Omission in "
            "resume mode preserves the checkpoint setting."
        ),
    )

    parser.add_argument(
        "--visual-feature-dim",
        type=_positive_integer,
        default=256,
    )
    parser.add_argument(
        "--previous-action-dim",
        type=_positive_integer,
        default=32,
    )
    parser.add_argument(
        "--mode-embedding-dim",
        type=_positive_integer,
        default=16,
    )
    parser.add_argument(
        "--telemetry-feature-dim",
        type=_positive_integer,
        default=32,
    )
    parser.add_argument(
        "--telemetry-weapon-count",
        type=_positive_integer,
        default=(
            DEFAULT_TELEMETRY_WEAPON_COUNT
        ),
    )
    parser.add_argument(
        "--telemetry-weapon-embedding-dim",
        type=_positive_integer,
        default=8,
    )

    parser.add_argument(
        "--recurrent-input-dim",
        type=_positive_integer,
        default=256,
    )
    parser.add_argument(
        "--recurrent-hidden-dim",
        type=_positive_integer,
        default=256,
    )
    parser.add_argument(
        "--recurrent-layers",
        type=_positive_integer,
        default=2,
    )

    parser.add_argument(
        "--forward-loss-weight",
        type=_nonnegative_float,
        default=None,
    )
    parser.add_argument(
        "--strafe-loss-weight",
        type=_nonnegative_float,
        default=None,
    )
    parser.add_argument(
        "--mouse-loss-weight",
        type=_nonnegative_float,
        default=None,
    )
    parser.add_argument(
        "--fire-loss-weight",
        type=_nonnegative_float,
        default=None,
    )
    parser.add_argument(
        "--jump-loss-weight",
        type=_nonnegative_float,
        default=None,
    )
    parser.add_argument(
        "--weapon-loss-weight",
        type=_nonnegative_float,
        default=None,
    )
    parser.add_argument(
        "--legacy-loss-weight",
        type=_nonnegative_float,
        default=None,
    )

    parser.add_argument(
        "--fire-positive-weight",
        type=_positive_float,
        default=None,
    )
    parser.add_argument(
        "--jump-positive-weight",
        type=_positive_float,
        default=None,
    )
    parser.add_argument(
        "--weapon-previous-weight",
        type=_positive_float,
        default=None,
    )
    parser.add_argument(
        "--weapon-neutral-weight",
        type=_positive_float,
        default=None,
    )
    parser.add_argument(
        "--weapon-next-weight",
        type=_positive_float,
        default=None,
    )

    parser.add_argument(
        "--policy-name",
        default=None,
    )
    parser.add_argument(
        "--policy-version",
        default=None,
    )

    return parser.parse_args(argv)


def _resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    resolved = torch.device(requested)

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return resolved


def _seed_everything(
    seed: int,
    device: torch.device,
) -> None:
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _resolved_loss_weights(
    args: argparse.Namespace,
    *,
    base: HierarchicalLossWeights | None = None,
) -> HierarchicalLossWeights:
    defaults = (
        HierarchicalLossWeights()
        if base is None
        else base
    )

    return HierarchicalLossWeights(
        forward=(
            defaults.forward
            if args.forward_loss_weight is None
            else args.forward_loss_weight
        ),
        strafe=(
            defaults.strafe
            if args.strafe_loss_weight is None
            else args.strafe_loss_weight
        ),
        mouse=(
            defaults.mouse
            if args.mouse_loss_weight is None
            else args.mouse_loss_weight
        ),
        fire=(
            defaults.fire
            if args.fire_loss_weight is None
            else args.fire_loss_weight
        ),
        jump=(
            defaults.jump
            if args.jump_loss_weight is None
            else args.jump_loss_weight
        ),
        weapon=(
            defaults.weapon
            if args.weapon_loss_weight is None
            else args.weapon_loss_weight
        ),
        legacy_action=(
            defaults.legacy_action
            if args.legacy_loss_weight is None
            else args.legacy_loss_weight
        ),
    )


def _resolved_class_weights(
    args: argparse.Namespace,
    *,
    base: HierarchicalClassWeights | None = None,
) -> HierarchicalClassWeights:
    defaults = (
        HierarchicalClassWeights()
        if base is None
        else base
    )

    return HierarchicalClassWeights(
        fire_positive=(
            defaults.fire_positive
            if args.fire_positive_weight is None
            else args.fire_positive_weight
        ),
        jump_positive=(
            defaults.jump_positive
            if args.jump_positive_weight is None
            else args.jump_positive_weight
        ),
        weapon_previous=(
            defaults.weapon_previous
            if args.weapon_previous_weight is None
            else args.weapon_previous_weight
        ),
        weapon_neutral=(
            defaults.weapon_neutral
            if args.weapon_neutral_weight is None
            else args.weapon_neutral_weight
        ),
        weapon_next=(
            defaults.weapon_next
            if args.weapon_next_weight is None
            else args.weapon_next_weight
        ),
    )


def _ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return numerator / denominator


def _optional_ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator <= 0:
        return None

    return numerator / denominator


def evaluate_dataset(
    model: HierarchicalRecurrentPolicy,
    batches: Iterable[CompositeSequenceBatch],
    *,
    device: torch.device,
    weights: HierarchicalLossWeights,
    class_weights: HierarchicalClassWeights,
) -> DatasetEvaluation:
    model.eval()

    valid_total = 0
    weighted_loss_total = 0.0

    forward_correct = 0
    strafe_correct = 0
    weapon_correct = 0
    legacy_correct = 0

    weapon_true_positive = [0, 0, 0]
    weapon_predicted_count = [0, 0, 0]
    weapon_actual_count = [0, 0, 0]

    fire_correct = 0
    fire_true_positive = 0
    fire_predicted_positive = 0
    fire_actual_positive = 0

    jump_correct = 0
    jump_true_positive = 0
    jump_predicted_positive = 0
    jump_actual_positive = 0

    mouse_x_absolute_total = 0.0
    mouse_y_absolute_total = 0.0

    with torch.no_grad():
        for batch in batches:
            frames = batch.frames.to(
                device=device,
            )
            previous = (
                batch.previous_action_features.to(
                    device=device,
                )
            )
            modes = batch.mode_indices.to(
                device=device,
            )
            telemetry_features = (
                batch.telemetry_features.to(
                    device=device,
                )
            )
            telemetry_weapon_indices = (
                batch.telemetry_weapon_indices.to(
                    device=device,
                )
            )

            output = model(
                frames,
                previous,
                modes,
                telemetry_features=(
                    telemetry_features
                ),
                telemetry_weapon_indices=(
                    telemetry_weapon_indices
                ),
            )

            losses = (
                hierarchical_behavior_cloning_loss(
                    output,
                    batch,
                    weights=weights,
                    class_weights=class_weights,
                )
            )

            mask = batch.valid_mask.to(
                device=device,
            )
            valid_count = int(
                mask.sum().item()
            )

            if valid_count <= 0:
                continue

            valid_total += valid_count
            weighted_loss_total += (
                float(losses.total.item())
                * valid_count
            )

            forward_target = (
                batch.forward_classes.to(
                    device=device,
                )[mask]
            )
            strafe_target = (
                batch.strafe_classes.to(
                    device=device,
                )[mask]
            )
            weapon_target = (
                batch.weapon_classes.to(
                    device=device,
                )[mask]
            )
            legacy_target = (
                batch.legacy_action_indices.to(
                    device=device,
                )[mask]
            )

            forward_correct += int(
                (
                    output.forward_logits[
                        mask
                    ].argmax(dim=-1)
                    == forward_target
                ).sum().item()
            )
            strafe_correct += int(
                (
                    output.strafe_logits[
                        mask
                    ].argmax(dim=-1)
                    == strafe_target
                ).sum().item()
            )
            weapon_prediction = (
                output.weapon_logits[
                    mask
                ].argmax(dim=-1)
            )

            weapon_correct += int(
                (
                    weapon_prediction
                    == weapon_target
                ).sum().item()
            )

            for class_index in range(3):
                predicted_class = (
                    weapon_prediction
                    == class_index
                )
                actual_class = (
                    weapon_target
                    == class_index
                )

                weapon_true_positive[
                    class_index
                ] += int(
                    (
                        predicted_class
                        & actual_class
                    ).sum().item()
                )
                weapon_predicted_count[
                    class_index
                ] += int(
                    predicted_class.sum().item()
                )
                weapon_actual_count[
                    class_index
                ] += int(
                    actual_class.sum().item()
                )
            legacy_correct += int(
                (
                    output.legacy_action_logits[
                        mask
                    ].argmax(dim=-1)
                    == legacy_target
                ).sum().item()
            )

            fire_prediction = (
                torch.sigmoid(
                    output.fire_logits[mask]
                )
                >= 0.5
            )
            fire_target = (
                batch.fire_targets.to(
                    device=device,
                )[mask]
                >= 0.5
            )

            fire_correct += int(
                (
                    fire_prediction
                    == fire_target
                ).sum().item()
            )
            fire_true_positive += int(
                (
                    fire_prediction
                    & fire_target
                ).sum().item()
            )
            fire_predicted_positive += int(
                fire_prediction.sum().item()
            )
            fire_actual_positive += int(
                fire_target.sum().item()
            )

            jump_prediction = (
                torch.sigmoid(
                    output.jump_logits[mask]
                )
                >= 0.5
            )
            jump_target = (
                batch.jump_targets.to(
                    device=device,
                )[mask]
                >= 0.5
            )

            jump_correct += int(
                (
                    jump_prediction
                    == jump_target
                ).sum().item()
            )
            jump_true_positive += int(
                (
                    jump_prediction
                    & jump_target
                ).sum().item()
            )
            jump_predicted_positive += int(
                jump_prediction.sum().item()
            )
            jump_actual_positive += int(
                jump_target.sum().item()
            )

            mouse_target = (
                batch.mouse_deltas.to(
                    device=device,
                )[mask]
            )
            mouse_error = (
                output.mouse_deltas[mask]
                - mouse_target
            ).abs()

            mouse_x_absolute_total += float(
                mouse_error[:, 0].sum().item()
            )
            mouse_y_absolute_total += float(
                mouse_error[:, 1].sum().item()
            )

    if valid_total <= 0:
        raise ValueError(
            "evaluation contained no valid steps"
        )

    return DatasetEvaluation(
        loss=(
            weighted_loss_total
            / valid_total
        ),
        valid_step_count=valid_total,
        forward_accuracy=_ratio(
            forward_correct,
            valid_total,
        ),
        strafe_accuracy=_ratio(
            strafe_correct,
            valid_total,
        ),
        weapon_accuracy=_ratio(
            weapon_correct,
            valid_total,
        ),
        weapon_previous_precision=_optional_ratio(
            weapon_true_positive[0],
            weapon_predicted_count[0],
        ),
        weapon_previous_recall=_optional_ratio(
            weapon_true_positive[0],
            weapon_actual_count[0],
        ),
        weapon_previous_positive_count=(
            weapon_actual_count[0]
        ),
        weapon_neutral_precision=_optional_ratio(
            weapon_true_positive[1],
            weapon_predicted_count[1],
        ),
        weapon_neutral_recall=_optional_ratio(
            weapon_true_positive[1],
            weapon_actual_count[1],
        ),
        weapon_neutral_positive_count=(
            weapon_actual_count[1]
        ),
        weapon_next_precision=_optional_ratio(
            weapon_true_positive[2],
            weapon_predicted_count[2],
        ),
        weapon_next_recall=_optional_ratio(
            weapon_true_positive[2],
            weapon_actual_count[2],
        ),
        weapon_next_positive_count=(
            weapon_actual_count[2]
        ),
        legacy_action_accuracy=_ratio(
            legacy_correct,
            valid_total,
        ),
        fire_accuracy=_ratio(
            fire_correct,
            valid_total,
        ),
        fire_precision=_optional_ratio(
            fire_true_positive,
            fire_predicted_positive,
        ),
        fire_recall=_optional_ratio(
            fire_true_positive,
            fire_actual_positive,
        ),
        fire_positive_count=(
            fire_actual_positive
        ),
        jump_accuracy=_ratio(
            jump_correct,
            valid_total,
        ),
        jump_precision=_optional_ratio(
            jump_true_positive,
            jump_predicted_positive,
        ),
        jump_recall=_optional_ratio(
            jump_true_positive,
            jump_actual_positive,
        ),
        jump_positive_count=(
            jump_actual_positive
        ),
        mouse_x_mean_absolute_error=(
            mouse_x_absolute_total
            / valid_total
        ),
        mouse_y_mean_absolute_error=(
            mouse_y_absolute_total
            / valid_total
        ),
    )


def _format_optional(
    value: float | None,
) -> str:
    if value is None:
        return "not_observed"

    return f"{value:.6f}"


def _print_evaluation(
    prefix: str,
    evaluation: DatasetEvaluation,
) -> None:
    print(
        f"{prefix}_loss="
        f"{evaluation.loss:.6f}"
    )
    print(
        f"{prefix}_valid_steps="
        f"{evaluation.valid_step_count}"
    )
    print(
        f"{prefix}_forward_accuracy="
        f"{evaluation.forward_accuracy:.6f}"
    )
    print(
        f"{prefix}_strafe_accuracy="
        f"{evaluation.strafe_accuracy:.6f}"
    )
    print(
        f"{prefix}_weapon_accuracy="
        f"{evaluation.weapon_accuracy:.6f}"
    )
    print(
        f"{prefix}_weapon_previous_precision="
        f"{_format_optional(evaluation.weapon_previous_precision)}"
    )
    print(
        f"{prefix}_weapon_previous_recall="
        f"{_format_optional(evaluation.weapon_previous_recall)}"
    )
    print(
        f"{prefix}_weapon_previous_positive_steps="
        f"{evaluation.weapon_previous_positive_count}"
    )
    print(
        f"{prefix}_weapon_neutral_precision="
        f"{_format_optional(evaluation.weapon_neutral_precision)}"
    )
    print(
        f"{prefix}_weapon_neutral_recall="
        f"{_format_optional(evaluation.weapon_neutral_recall)}"
    )
    print(
        f"{prefix}_weapon_neutral_positive_steps="
        f"{evaluation.weapon_neutral_positive_count}"
    )
    print(
        f"{prefix}_weapon_next_precision="
        f"{_format_optional(evaluation.weapon_next_precision)}"
    )
    print(
        f"{prefix}_weapon_next_recall="
        f"{_format_optional(evaluation.weapon_next_recall)}"
    )
    print(
        f"{prefix}_weapon_next_positive_steps="
        f"{evaluation.weapon_next_positive_count}"
    )
    print(
        f"{prefix}_legacy_accuracy="
        f"{evaluation.legacy_action_accuracy:.6f}"
    )

    print(
        f"{prefix}_fire_accuracy="
        f"{evaluation.fire_accuracy:.6f}"
    )
    print(
        f"{prefix}_fire_precision="
        f"{_format_optional(evaluation.fire_precision)}"
    )
    print(
        f"{prefix}_fire_recall="
        f"{_format_optional(evaluation.fire_recall)}"
    )
    print(
        f"{prefix}_fire_positive_steps="
        f"{evaluation.fire_positive_count}"
    )

    print(
        f"{prefix}_jump_accuracy="
        f"{evaluation.jump_accuracy:.6f}"
    )
    print(
        f"{prefix}_jump_precision="
        f"{_format_optional(evaluation.jump_precision)}"
    )
    print(
        f"{prefix}_jump_recall="
        f"{_format_optional(evaluation.jump_recall)}"
    )
    print(
        f"{prefix}_jump_positive_steps="
        f"{evaluation.jump_positive_count}"
    )

    print(
        f"{prefix}_mouse_x_mae="
        f"{evaluation.mouse_x_mean_absolute_error:.6f}"
    )
    print(
        f"{prefix}_mouse_y_mae="
        f"{evaluation.mouse_y_mean_absolute_error:.6f}"
    )


def _dataset_metadata(
    dataset: TelemetryCompositeSequenceDataset,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "episode_paths": [
            str(path)
            for path in dataset.episode_paths
        ],
        "sequence_length": (
            dataset.sequence_length
        ),
        "stride": dataset.stride,
        "telemetry_schema_version": (
            dataset.telemetry_schema_version
        ),
        "telemetry_feature_count": (
            dataset.telemetry_feature_count
        ),
        "telemetry_weapon_count": (
            dataset.telemetry_weapon_count
        ),
        "sequence_count": len(dataset),
        "batch_size": args.batch_size,
        "authoritative_composite_only": True,
        "deterministic_order": True,
    }


def _validate_resume_dataset(
    saved_metadata: object,
    current_metadata: dict[str, object],
    *,
    allow_dataset_change: bool,
) -> None:
    if allow_dataset_change:
        return

    if not isinstance(saved_metadata, dict):
        saved_metadata = dict(
            saved_metadata
        )

    compared_fields = (
        "episode_paths",
        "sequence_length",
        "stride",
        "telemetry_schema_version",
        "telemetry_feature_count",
        "telemetry_weapon_count",
    )

    for field_name in compared_fields:
        if (
            field_name in saved_metadata
            and saved_metadata[field_name]
            != current_metadata[field_name]
        ):
            raise ValueError(
                "resumed dataset does not match "
                f"checkpoint field: {field_name}; "
                "use --allow-dataset-change to override"
            )


def _build_new_trainer(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> HierarchicalBehaviorCloningTrainer:
    model = HierarchicalRecurrentPolicy(
        visual_feature_dim=(
            args.visual_feature_dim
        ),
        previous_action_dim=(
            args.previous_action_dim
        ),
        mode_embedding_dim=(
            args.mode_embedding_dim
        ),
        telemetry_feature_count=(
            TELEMETRY_FEATURE_COUNT
        ),
        telemetry_feature_dim=(
            args.telemetry_feature_dim
        ),
        telemetry_weapon_count=(
            args.telemetry_weapon_count
        ),
        telemetry_weapon_embedding_dim=(
            args.telemetry_weapon_embedding_dim
        ),
        recurrent_input_dim=(
            args.recurrent_input_dim
        ),
        recurrent_hidden_dim=(
            args.recurrent_hidden_dim
        ),
        recurrent_layers=(
            args.recurrent_layers
        ),
    )

    learning_rate = (
        DEFAULT_LEARNING_RATE
        if args.learning_rate is None
        else args.learning_rate
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    return HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
        device=device,
        weights=_resolved_loss_weights(
            args
        ),
        class_weights=_resolved_class_weights(
            args
        ),
        max_gradient_norm=(
            1.0
            if args.max_gradient_norm is None
            else args.max_gradient_norm
        ),
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(argv)

    torch.set_num_threads(
        args.torch_threads
    )

    device = _resolve_device(
        args.device
    )
    _seed_everything(
        args.seed,
        device,
    )

    dataset = TelemetryCompositeSequenceDataset(
        args.episodes,
        sequence_length=args.sequence_length,
        stride=args.stride,
        telemetry_weapon_count=(
            args.telemetry_weapon_count
        ),
    )

    loader = (
        make_telemetry_composite_sequence_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        )
    )

    dataset_metadata = _dataset_metadata(
        dataset,
        args,
    )

    resumed_from: str | None = None

    if args.resume_checkpoint is None:
        trainer = _build_new_trainer(
            args,
            device=device,
        )
        policy_name = (
            DEFAULT_POLICY_NAME
            if args.policy_name is None
            else args.policy_name
        )
        policy_version = (
            DEFAULT_POLICY_VERSION
            if args.policy_version is None
            else args.policy_version
        )
    else:
        loaded = load_hierarchical_checkpoint(
            args.resume_checkpoint,
            device=device,
        )

        _validate_resume_dataset(
            loaded.dataset_metadata,
            dataset_metadata,
            allow_dataset_change=(
                args.allow_dataset_change
            ),
        )

        trainer = loaded.trainer
        trainer.weights = (
            _resolved_loss_weights(
                args,
                base=trainer.weights,
            )
        )
        trainer.class_weights = (
            _resolved_class_weights(
                args,
                base=trainer.class_weights,
            )
        )

        if args.max_gradient_norm is not None:
            trainer.max_gradient_norm = (
                args.max_gradient_norm
            )

        if args.learning_rate is not None:
            for parameter_group in (
                trainer.optimizer.param_groups
            ):
                parameter_group["lr"] = (
                    args.learning_rate
                )

        policy_name = (
            loaded.policy_name
            if args.policy_name is None
            else args.policy_name
        )
        policy_version = (
            loaded.policy_version
            if args.policy_version is None
            else args.policy_version
        )
        resumed_from = str(
            Path(
                args.resume_checkpoint
            ).resolve()
        )

    starting_optimizer_step = (
        trainer.optimizer_step_count
    )

    print(f"training_device={device}")
    print(
        f"torch_threads="
        f"{args.torch_threads}"
    )
    print(
        f"dataset_episode_count="
        f"{len(dataset.episode_paths)}"
    )
    print(
        f"dataset_sequence_count="
        f"{len(dataset)}"
    )
    print(
        f"sequence_length="
        f"{dataset.sequence_length}"
    )
    print(f"stride={dataset.stride}")
    print(
        "telemetry_feature_count="
        f"{dataset.telemetry_feature_count}"
    )
    print(
        "telemetry_weapon_count="
        f"{dataset.telemetry_weapon_count}"
    )
    print(
        "telemetry_model_enabled="
        f"{trainer.model.telemetry_enabled}"
    )
    print(f"batch_size={args.batch_size}")
    print(f"epochs={args.epochs}")
    print(
        "maximum_additional_optimizer_steps="
        f"{args.max_optimizer_steps}"
    )
    print(
        "starting_optimizer_step_count="
        f"{starting_optimizer_step}"
    )
    print(f"resumed_from={resumed_from}")
    print(
        "fire_positive_weight="
        f"{trainer.class_weights.fire_positive}"
    )
    print(
        "jump_positive_weight="
        f"{trainer.class_weights.jump_positive}"
    )
    print(
        "weapon_previous_weight="
        f"{trainer.class_weights.weapon_previous}"
    )
    print(
        "weapon_neutral_weight="
        f"{trainer.class_weights.weapon_neutral}"
    )
    print(
        "weapon_next_weight="
        f"{trainer.class_weights.weapon_next}"
    )
    print(
        "tactical_intent_loss_enabled=False"
    )
    print(
        "candidate_playable=False"
    )

    initial_evaluation = evaluate_dataset(
        trainer.model,
        loader,
        device=device,
        weights=trainer.weights,
        class_weights=trainer.class_weights,
    )

    print()
    print("=== INITIAL DATASET EVALUATION ===")
    _print_evaluation(
        "initial",
        initial_evaluation,
    )

    run_optimizer_steps = 0

    for epoch_index in range(args.epochs):
        for batch in loader:
            result = trainer.train_batch(
                batch
            )
            run_optimizer_steps += 1

            if (
                run_optimizer_steps == 1
                or run_optimizer_steps
                % args.log_every_steps
                == 0
                or run_optimizer_steps
                == args.max_optimizer_steps
            ):
                print(
                    "training_progress "
                    f"epoch={epoch_index + 1} "
                    f"run_step={run_optimizer_steps} "
                    "optimizer_step="
                    f"{result.optimizer_step_count} "
                    f"batch_loss="
                    f"{result.metrics.loss:.6f} "
                    f"gradient_norm="
                    f"{result.gradient_norm:.6f}"
                )

            if (
                run_optimizer_steps
                >= args.max_optimizer_steps
            ):
                break

        if (
            run_optimizer_steps
            >= args.max_optimizer_steps
        ):
            break

    if run_optimizer_steps <= 0:
        raise RuntimeError(
            "training performed no optimizer operations"
        )

    final_evaluation = evaluate_dataset(
        trainer.model,
        loader,
        device=device,
        weights=trainer.weights,
        class_weights=trainer.class_weights,
    )

    ending_optimizer_step = (
        trainer.optimizer_step_count
    )

    if ending_optimizer_step != (
        starting_optimizer_step
        + run_optimizer_steps
    ):
        raise RuntimeError(
            "optimizer step accounting is inconsistent"
        )

    print()
    print("=== FINAL DATASET EVALUATION ===")
    _print_evaluation(
        "final",
        final_evaluation,
    )

    print(
        "absolute_loss_change="
        f"{final_evaluation.loss - initial_evaluation.loss:.6f}"
    )
    print(
        "run_optimizer_steps="
        f"{run_optimizer_steps}"
    )
    print(
        "ending_optimizer_step_count="
        f"{ending_optimizer_step}"
    )

    output_path = (
        save_hierarchical_checkpoint(
            args.output_checkpoint,
            trainer=trainer,
            policy_name=policy_name,
            policy_version=policy_version,
            dataset_metadata=dataset_metadata,
            metadata={
                "training_tool": (
                    "train_hierarchical_behavior_cloning"
                ),
                "seed": args.seed,
                "device": str(device),
                "epochs_requested": args.epochs,
                "run_optimizer_steps": (
                    run_optimizer_steps
                ),
                "starting_optimizer_step_count": (
                    starting_optimizer_step
                ),
                "ending_optimizer_step_count": (
                    ending_optimizer_step
                ),
                "initial_evaluation": (
                    initial_evaluation.to_record()
                ),
                "final_evaluation": (
                    final_evaluation.to_record()
                ),
                "resumed_from": resumed_from,
                "dataset_change_allowed": (
                    args.allow_dataset_change
                ),
                "tactical_intent_loss_enabled": (
                    False
                ),
                "candidate_playable": False,
            },
        )
    )

    verification = (
        load_hierarchical_checkpoint(
            output_path,
            device=device,
        )
    )

    if (
        verification.trainer
        .optimizer_step_count
        != ending_optimizer_step
    ):
        raise RuntimeError(
            "saved checkpoint optimizer count "
            "did not reload exactly"
        )

    if (
        verification.trainer.class_weights
        != trainer.class_weights
    ):
        raise RuntimeError(
            "saved checkpoint class weights "
            "did not reload exactly"
        )

    print()
    print(
        f"checkpoint_path="
        f"{output_path.resolve()}"
    )
    print(
        "checkpoint_optimizer_step_count="
        f"{verification.trainer.optimizer_step_count}"
    )
    print(
        "hierarchical_training_checkpoint_reload="
        "passed"
    )
    print(
        "hierarchical_behavior_cloning_training="
        "completed"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
