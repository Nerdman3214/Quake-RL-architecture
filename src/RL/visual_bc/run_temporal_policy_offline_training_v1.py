from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


SEED = 20260810

BATCH_SIZE = 4
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

VIEW_LOSS_WEIGHT = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(
    path: Path,
    value: Any,
) -> None:
    temporary = Path(
        str(path) + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_torch_save(
    path: Path,
    value: Any,
) -> None:
    temporary = Path(
        str(path) + ".tmp"
    )

    torch.save(
        value,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def load_trainer(
    trainer_path: Path,
):
    spec = importlib.util.spec_from_file_location(
        "visual_bc_temporal_policy_v1",
        trainer_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create trainer import spec."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def binary_metrics_template(
    names: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    return {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "positive_targets": 0,
        }
        for name in names
    }


def finish_binary_metrics(
    raw: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    output = {}

    for name, counts in raw.items():

        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        accuracy = (
            (tp + tn)
            / (tp + tn + fp + fn)
            if (tp + tn + fp + fn) > 0
            else 0.0
        )

        output[name] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    return output


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    binary_loss_fn: nn.Module,
    view_loss_fn: nn.Module,
    trainer: Any,
) -> dict[str, Any]:

    model.eval()

    total_examples = 0

    total_binary_loss = 0.0
    total_view_loss = 0.0
    total_loss = 0.0

    raw_metrics = binary_metrics_template(
        trainer.BINARY_ACTION_NAMES
    )

    view_absolute_error = torch.zeros(
        len(trainer.VIEW_NAMES),
        dtype=torch.float64,
    )

    with torch.inference_mode():

        for (
            features,
            binary_targets,
            view_targets_degrees,
        ) in loader:

            features = (
                features.float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            binary_targets = (
                binary_targets.to(
                    device,
                    non_blocking=True,
                )
            )

            view_targets_degrees = (
                view_targets_degrees.to(
                    device,
                    non_blocking=True,
                )
            )

            view_scale = (
                trainer.VIEW_SCALE
                .to(device)
            )

            normalized_view_targets = (
                view_targets_degrees
                / view_scale
            )

            binary_logits, view_output = (
                model(features)
            )

            binary_loss = binary_loss_fn(
                binary_logits,
                binary_targets,
            )

            view_loss = view_loss_fn(
                view_output,
                normalized_view_targets,
            )

            loss = (
                binary_loss
                + VIEW_LOSS_WEIGHT
                * view_loss
            )

            batch_size = features.shape[0]

            total_examples += batch_size

            total_binary_loss += (
                binary_loss.item()
                * batch_size
            )

            total_view_loss += (
                view_loss.item()
                * batch_size
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            probabilities = torch.sigmoid(
                binary_logits
            )

            predicted = (
                probabilities >= 0.5
            )

            target_binary = (
                binary_targets >= 0.5
            )

            for action_index, name in enumerate(
                trainer.BINARY_ACTION_NAMES
            ):

                p = predicted[
                    :,
                    action_index,
                ]

                t = target_binary[
                    :,
                    action_index,
                ]

                raw_metrics[name]["tp"] += int(
                    (p & t).sum().item()
                )

                raw_metrics[name]["fp"] += int(
                    (p & ~t).sum().item()
                )

                raw_metrics[name]["fn"] += int(
                    (~p & t).sum().item()
                )

                raw_metrics[name]["tn"] += int(
                    (~p & ~t).sum().item()
                )

                raw_metrics[
                    name
                ][
                    "positive_targets"
                ] += int(
                    t.sum().item()
                )

            predicted_view_degrees = (
                view_output
                * view_scale
            )

            view_absolute_error += (
                (
                    predicted_view_degrees
                    - view_targets_degrees
                )
                .abs()
                .sum(dim=0)
                .double()
                .cpu()
            )

    if total_examples <= 0:
        raise RuntimeError(
            "Validation loader produced no examples."
        )

    view_mae = (
        view_absolute_error
        / total_examples
    )

    return {
        "examples": total_examples,

        "binary_loss":
            total_binary_loss
            / total_examples,

        "view_loss":
            total_view_loss
            / total_examples,

        "total_loss":
            total_loss
            / total_examples,

        "binary_metrics":
            finish_binary_metrics(
                raw_metrics
            ),

        "view_mae_degrees": {
            name: float(
                view_mae[index]
            )
            for index, name in enumerate(
                trainer.VIEW_NAMES
            )
        },
    }


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    binary_loss_fn: nn.Module,
    view_loss_fn: nn.Module,
    trainer: Any,
) -> dict[str, float]:

    model.train()

    total_examples = 0

    total_binary_loss = 0.0
    total_view_loss = 0.0
    total_loss = 0.0

    max_observed_grad_norm = 0.0

    for (
        features,
        binary_targets,
        view_targets_degrees,
    ) in loader:

        features = (
            features.float()
            .to(
                device,
                non_blocking=True,
            )
        )

        binary_targets = (
            binary_targets.to(
                device,
                non_blocking=True,
            )
        )

        view_targets_degrees = (
            view_targets_degrees.to(
                device,
                non_blocking=True,
            )
        )

        view_scale = (
            trainer.VIEW_SCALE
            .to(device)
        )

        normalized_view_targets = (
            view_targets_degrees
            / view_scale
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        binary_logits, view_output = (
            model(features)
        )

        binary_loss = binary_loss_fn(
            binary_logits,
            binary_targets,
        )

        view_loss = view_loss_fn(
            view_output,
            normalized_view_targets,
        )

        loss = (
            binary_loss
            + VIEW_LOSS_WEIGHT
            * view_loss
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss."
            )

        loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )
        )

        grad_norm_value = float(
            grad_norm.detach().cpu()
        )

        if not math.isfinite(
            grad_norm_value
        ):
            raise RuntimeError(
                "Non-finite gradient norm."
            )

        max_observed_grad_norm = max(
            max_observed_grad_norm,
            grad_norm_value,
        )

        for parameter in model.parameters():

            if parameter.grad is None:
                continue

            if not torch.isfinite(
                parameter.grad
            ).all():
                raise RuntimeError(
                    "Non-finite gradient tensor."
                )

        optimizer.step()

        batch_size = features.shape[0]

        total_examples += batch_size

        total_binary_loss += (
            binary_loss.item()
            * batch_size
        )

        total_view_loss += (
            view_loss.item()
            * batch_size
        )

        total_loss += (
            loss.item()
            * batch_size
        )

    if total_examples <= 0:
        raise RuntimeError(
            "Training loader produced no examples."
        )

    return {
        "binary_loss":
            total_binary_loss
            / total_examples,

        "view_loss":
            total_view_loss
            / total_examples,

        "total_loss":
            total_loss
            / total_examples,

        "max_preclip_gradient_norm":
            max_observed_grad_norm,
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trainer",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-trainer-sha",
        required=True,
    )

    parser.add_argument(
        "--expected-split-sha",
        required=True,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this offline run."
        )

    actual_trainer_sha = sha256_file(
        args.trainer
    )

    actual_split_sha = sha256_file(
        args.split_manifest
    )

    if (
        actual_trainer_sha
        != args.expected_trainer_sha
    ):
        raise RuntimeError(
            "Trainer SHA changed."
        )

    if (
        actual_split_sha
        != args.expected_split_sha
    ):
        raise RuntimeError(
            "Split SHA changed."
        )

    split = json.loads(
        args.split_manifest.read_text(
            encoding="utf-8"
        )
    )

    if split.get(
        "session_split_ready"
    ) is not True:
        raise RuntimeError(
            "Session split is not ready."
        )

    if split.get(
        "split_strategy"
    ) != "whole_recording_session":
        raise RuntimeError(
            "Whole-session split required."
        )

    if split.get(
        "random_frame_split"
    ) is not False:
        raise RuntimeError(
            "Random frame split is forbidden."
        )

    if split.get(
        "train_sequence_count"
    ) != 656:
        raise RuntimeError(
            "Unexpected V4 train sequence count."
        )

    if split.get(
        "validation_sequence_count"
    ) != 1190:
        raise RuntimeError(
            "Unexpected V5 validation sequence count."
        )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = (
        args.run_dir
        / "metrics.jsonl"
    )

    best_path = (
        args.run_dir
        / "best.pt"
    )

    last_path = (
        args.run_dir
        / "last.pt"
    )

    manifest_path = (
        args.run_dir
        / "run_manifest.json"
    )

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda:0"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer = load_trainer(
        args.trainer
    )

    train_dataset = (
        trainer.CachedSequenceDataset(
            "train",
            split["train"],
        )
    )

    validation_dataset = (
        trainer.CachedSequenceDataset(
            "validation",
            split["validation"],
        )
    )

    if len(train_dataset) != 656:
        raise RuntimeError(
            "Train dataset count changed."
        )

    if len(validation_dataset) != 1190:
        raise RuntimeError(
            "Validation dataset count changed."
        )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    model = (
        trainer.TemporalVisualPolicy()
        .to(device)
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if trainable_parameters != 10667911:
        raise RuntimeError(
            "Unexpected policy parameter count: "
            f"{trainable_parameters}"
        )

    positive_weights = (
        trainer.compute_positive_weights(
            train_dataset
        )
        .to(device)
    )

    binary_loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=positive_weights
        )
    )

    view_loss_fn = (
        nn.SmoothL1Loss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    run_manifest = {
        "format_version": 1,
        "status": "training_in_progress",

        "trainer_sha256":
            actual_trainer_sha,

        "split_manifest_sha256":
            actual_split_sha,

        "train_session_id":
            split["train"]["session_id"],

        "validation_session_id":
            split[
                "validation"
            ][
                "session_id"
            ],

        "train_sequence_count": 656,
        "validation_sequence_count": 1190,

        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "max_gradient_norm":
            MAX_GRAD_NORM,

        "view_loss_weight":
            VIEW_LOSS_WEIGHT,

        "binary_positive_weights":
            positive_weights
            .detach()
            .cpu()
            .tolist(),

        "model_parameters":
            trainable_parameters,

        "mixed_precision": False,

        "best_epoch": None,
        "best_validation_loss": None,
        "last_completed_epoch": 0,

        "training_started": True,
        "policy_training_started": True,

        "live_xonotic_control": False,
        "automatic_action_allowed": False,
        "behavioral_control_gate_passed": False,
        "controls_modified": False,
    }

    atomic_json(
        manifest_path,
        run_manifest,
    )

    print(
        f"device={device}"
    )

    print(
        f"train_sequences="
        f"{len(train_dataset)}"
    )

    print(
        "validation_sequences="
        f"{len(validation_dataset)}"
    )

    print(
        f"batch_size={BATCH_SIZE}"
    )

    print(
        f"max_epochs={MAX_EPOCHS}"
    )

    print(
        "model_trainable_parameters="
        f"{trainable_parameters}"
    )

    print(
        "binary_positive_weights="
        f"{positive_weights.detach().cpu().tolist()}"
    )

    best_validation_loss = math.inf
    best_epoch = None

    epochs_without_improvement = 0
    completed_epochs = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            binary_loss_fn=binary_loss_fn,
            view_loss_fn=view_loss_fn,
            trainer=trainer,
        )

        validation_metrics = evaluate(
            model=model,
            loader=validation_loader,
            device=device,
            binary_loss_fn=binary_loss_fn,
            view_loss_fn=view_loss_fn,
            trainer=trainer,
        )

        validation_loss = float(
            validation_metrics[
                "total_loss"
            ]
        )

        if not math.isfinite(
            validation_loss
        ):
            raise RuntimeError(
                "Non-finite validation loss."
            )

        improved = (
            validation_loss
            < best_validation_loss
            - 1e-8
        )

        if improved:
            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        completed_epochs = epoch

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation":
                validation_metrics,
            "best_so_far": improved,
        }

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    epoch_record,
                    sort_keys=True,
                )
                + "\n"
            )

        checkpoint = {
            "format_version": 1,
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_metrics":
                train_metrics,

            "validation_metrics":
                validation_metrics,

            "trainer_sha256":
                actual_trainer_sha,

            "split_manifest_sha256":
                actual_split_sha,

            "architecture": {
                "sequence_length":
                    trainer.SEQUENCE_LENGTH,

                "tokens_per_frame":
                    trainer.TOKENS_PER_FRAME,

                "feature_dim":
                    trainer.FEATURE_DIM,

                "transformer_layers":
                    trainer.TRANSFORMER_LAYERS,

                "attention_heads":
                    trainer.ATTENTION_HEADS,

                "ffn_dim":
                    trainer.FFN_DIM,
            },

            "automatic_action_allowed":
                False,

            "behavioral_control_gate_passed":
                False,

            "controls_modified":
                False,
        }

        atomic_torch_save(
            last_path,
            checkpoint,
        )

        if improved:
            atomic_torch_save(
                best_path,
                checkpoint,
            )

        run_manifest[
            "last_completed_epoch"
        ] = epoch

        run_manifest[
            "best_epoch"
        ] = best_epoch

        run_manifest[
            "best_validation_loss"
        ] = (
            best_validation_loss
        )

        atomic_json(
            manifest_path,
            run_manifest,
        )

        print(
            "epoch="
            f"{epoch} "
            "train_total_loss="
            f"{train_metrics['total_loss']:.9f} "
            "validation_total_loss="
            f"{validation_loss:.9f} "
            "best="
            f"{improved}"
        )

        print(
            "validation_view_mae_degrees="
            f"{json.dumps(validation_metrics['view_mae_degrees'], sort_keys=True)}"
        )

        for (
            action_name,
            metric,
        ) in validation_metrics[
            "binary_metrics"
        ].items():

            print(
                f"validation_action="
                f"{action_name} "
                f"positives="
                f"{metric['positive_targets']} "
                f"precision="
                f"{metric['precision']:.6f} "
                f"recall="
                f"{metric['recall']:.6f} "
                f"f1="
                f"{metric['f1']:.6f}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "early_stopping_triggered=True"
            )
            break

    if completed_epochs < 1:
        raise RuntimeError(
            "No training epoch completed."
        )

    if best_epoch is None:
        raise RuntimeError(
            "No best checkpoint selected."
        )

    if not best_path.is_file():
        raise RuntimeError(
            "Best checkpoint missing."
        )

    if not last_path.is_file():
        raise RuntimeError(
            "Last checkpoint missing."
        )

    torch.cuda.synchronize()

    peak_gpu_mib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    current_gpu_mib = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    run_manifest[
        "status"
    ] = "training_complete"

    run_manifest[
        "epochs_completed"
    ] = completed_epochs

    run_manifest[
        "best_epoch"
    ] = best_epoch

    run_manifest[
        "best_validation_loss"
    ] = best_validation_loss

    run_manifest[
        "best_checkpoint"
    ] = str(best_path)

    run_manifest[
        "best_checkpoint_sha256"
    ] = sha256_file(
        best_path
    )

    run_manifest[
        "last_checkpoint"
    ] = str(last_path)

    run_manifest[
        "last_checkpoint_sha256"
    ] = sha256_file(
        last_path
    )

    run_manifest[
        "metrics_file"
    ] = str(metrics_path)

    run_manifest[
        "metrics_sha256"
    ] = sha256_file(
        metrics_path
    )

    run_manifest[
        "peak_gpu_allocated_mib"
    ] = peak_gpu_mib

    run_manifest[
        "current_gpu_allocated_mib"
    ] = current_gpu_mib

    run_manifest[
        "training_completed"
    ] = True

    atomic_json(
        manifest_path,
        run_manifest,
    )

    print(
        f"epochs_completed="
        f"{completed_epochs}"
    )

    print(
        f"best_epoch="
        f"{best_epoch}"
    )

    print(
        "best_validation_loss="
        f"{best_validation_loss:.9f}"
    )

    print(
        f"peak_gpu_allocated_mib="
        f"{peak_gpu_mib:.3f}"
    )

    print(
        "best_checkpoint_sha256="
        f"{sha256_file(best_path)}"
    )

    print(
        "last_checkpoint_sha256="
        f"{sha256_file(last_path)}"
    )

    print(
        "offline_training_complete=True"
    )

    print(
        "automatic_action_allowed=False"
    )

    print(
        "behavioral_control_gate_passed=False"
    )

    print(
        "controls_modified=False"
    )


if __name__ == "__main__":
    main()
