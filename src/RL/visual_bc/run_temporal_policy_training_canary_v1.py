from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


PROJECT = Path(
    "/media/steven/WINPE2/Quake-RL-architecture"
)

TRAINER = (
    PROJECT
    / "src/RL/visual_bc/"
      "train_temporal_policy_v1.py"
)

SPLIT = (
    PROJECT
    / "data/cache/visual_bc/"
      "session_split_v4_train_v5_validation_v1/"
      "session_split_manifest_v1.json"
)


def load_trainer_module():
    spec = importlib.util.spec_from_file_location(
        "visual_bc_temporal_policy_v1",
        TRAINER,
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    torch.manual_seed(
        20260810
    )

    torch.cuda.manual_seed_all(
        20260810
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device(
        "cuda:0"
    )

    trainer = load_trainer_module()

    split = trainer.read_json(
        SPLIT
    )

    if split.get(
        "session_split_ready"
    ) is not True:
        raise RuntimeError(
            "Session split gate is not ready."
        )

    if split.get(
        "split_strategy"
    ) != "whole_recording_session":
        raise RuntimeError(
            "Unexpected split strategy."
        )

    train_dataset = (
        trainer.CachedSequenceDataset(
            "train",
            split["train"],
        )
    )

    if len(train_dataset) != 656:
        raise RuntimeError(
            "Unexpected training sequence count."
        )

    # --------------------------------------------------------
    # Select two real action-active examples where possible,
    # rather than relying only on neutral frames.
    # --------------------------------------------------------

    active_indices = []

    for index, sequence in enumerate(
        train_dataset.sequences
    ):
        action = sequence[
            "target_action"
        ]

        binary_active = any(
            float(
                action[name]
            ) > 0.0
            for name in trainer.BINARY_ACTION_NAMES
        )

        view_active = any(
            abs(
                float(
                    action[name]
                )
            ) > 1e-9
            for name in trainer.VIEW_NAMES
        )

        if binary_active or view_active:
            active_indices.append(
                index
            )

        if len(active_indices) == 2:
            break

    if len(active_indices) < 2:
        active_indices = [
            0,
            1,
        ]

    print(
        f"canary_indices={active_indices}"
    )

    features = torch.stack(
        [
            train_dataset[index][0]
            for index in active_indices
        ],
        dim=0,
    ).float().to(
        device
    )

    binary_targets = torch.stack(
        [
            train_dataset[index][1]
            for index in active_indices
        ],
        dim=0,
    ).to(
        device
    )

    view_degrees = torch.stack(
        [
            train_dataset[index][2]
            for index in active_indices
        ],
        dim=0,
    ).to(
        device
    )

    view_scale = trainer.VIEW_SCALE.to(
        device
    )

    view_targets = (
        view_degrees
        / view_scale
    )

    print(
        f"canary_input_shape="
        f"{tuple(features.shape)}"
    )

    print(
        "canary_binary_targets="
        f"{json.dumps(binary_targets.cpu().tolist())}"
    )

    print(
        "canary_view_targets_degrees="
        f"{json.dumps(view_degrees.cpu().tolist())}"
    )

    model = trainer.TemporalVisualPolicy().to(
        device
    )

    model.train()

    pos_weight = (
        trainer.compute_positive_weights(
            train_dataset
        )
        .to(device)
    )

    binary_loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    view_loss_fn = (
        nn.SmoothL1Loss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    policy_before = (
        model.policy_token
        .detach()
        .clone()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    binary_logits, view_output = model(
        features
    )

    binary_loss = binary_loss_fn(
        binary_logits,
        binary_targets,
    )

    view_loss = view_loss_fn(
        view_output,
        view_targets,
    )

    loss = (
        binary_loss
        + view_loss
    )

    print(
        f"canary_binary_loss_before_step="
        f"{binary_loss.item():.9f}"
    )

    print(
        f"canary_view_loss_before_step="
        f"{view_loss.item():.9f}"
    )

    print(
        f"canary_total_loss_before_step="
        f"{loss.item():.9f}"
    )

    if not math.isfinite(
        loss.item()
    ):
        raise RuntimeError(
            "Non-finite canary loss."
        )

    loss.backward()

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
    )

    gradient_norm_value = float(
        gradient_norm.detach().cpu()
    )

    print(
        f"gradient_norm_before_clip="
        f"{gradient_norm_value:.9f}"
    )

    if not math.isfinite(
        gradient_norm_value
    ):
        raise RuntimeError(
            "Non-finite gradient norm."
        )

    finite_gradients = True
    gradient_tensor_count = 0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        gradient_tensor_count += 1

        if not torch.isfinite(
            parameter.grad
        ).all():
            finite_gradients = False
            break

    print(
        f"gradient_tensor_count="
        f"{gradient_tensor_count}"
    )

    print(
        f"all_gradients_finite="
        f"{finite_gradients}"
    )

    if not finite_gradients:
        raise RuntimeError(
            "Non-finite gradients detected."
        )

    optimizer.step()

    policy_delta = float(
        (
            model.policy_token.detach()
            - policy_before
        )
        .abs()
        .sum()
        .cpu()
    )

    print(
        f"policy_token_parameter_delta_l1="
        f"{policy_delta:.12f}"
    )

    if not (
        policy_delta > 0.0
        and math.isfinite(
            policy_delta
        )
    ):
        raise RuntimeError(
            "Optimizer step did not change policy token."
        )

    torch.cuda.synchronize()

    peak_mib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    current_mib = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    print(
        f"peak_gpu_allocated_mib="
        f"{peak_mib:.3f}"
    )

    print(
        f"current_gpu_allocated_mib="
        f"{current_mib:.3f}"
    )

    print(
        "backward_called=True"
    )

    print(
        "optimizer_step_called=True"
    )

    print(
        "optimizer_steps_completed=1"
    )

    print(
        "checkpoint_written=False"
    )

    print(
        "one_step_training_canary_passed=True"
    )


if __name__ == "__main__":
    main()
