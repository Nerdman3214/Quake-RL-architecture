#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)


HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(
        0,
        str(HERE),
    )


import train_temporal_policy_v2 as v2


MODEL_VERSION = "2.2"
SEED = v2.SEED

AUTOMATIC_ACTION_ALLOWED = False
BEHAVIORAL_CONTROL_GATE_PASSED = False
CONTROLS_MODIFIED = False


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def mild_class_weights(
    counts: list[int],
) -> torch.Tensor:

    total = float(sum(counts))

    values = torch.tensor(
        [
            math.sqrt(
                total / max(float(count), 1.0)
            )
            for count in counts
        ],
        dtype=torch.float32,
    )

    values = values / values.mean()

    return torch.clamp(
        values,
        min=0.75,
        max=2.00,
    )


def mild_positive_weight(
    positives: int,
    total: int,
) -> float:

    if positives <= 0:
        raise RuntimeError(
            "Binary target contains zero positives."
        )

    negatives = total - positives

    weight = math.sqrt(
        negatives / positives
    )

    return float(
        min(
            2.50,
            max(
                1.00,
                weight,
            ),
        )
    )


def build_loss_config(
    labels: list[v2.V2Label],
) -> v2.LossConfig:

    total = len(labels)

    forward_counts = [
        sum(
            label.forward == index
            for label in labels
        )
        for index in range(3)
    ]

    strafe_counts = [
        sum(
            label.strafe == index
            for label in labels
        )
        for index in range(3)
    ]

    jump_positive = sum(
        label.jump > 0.5
        for label in labels
    )

    yaw_positive = sum(
        label.yaw_active > 0.5
        for label in labels
    )

    pitch_positive = sum(
        label.pitch_active > 0.5
        for label in labels
    )


    config = v2.LossConfig(
        forward_weights=
            mild_class_weights(
                forward_counts
            ),

        strafe_weights=
            mild_class_weights(
                strafe_counts
            ),

        jump_pos_weight=
            mild_positive_weight(
                jump_positive,
                total,
            ),

        yaw_pos_weight=
            mild_positive_weight(
                yaw_positive,
                total,
            ),

        pitch_pos_weight=
            mild_positive_weight(
                pitch_positive,
                total,
            ),
    )


    print(
        f"forward_class_counts={forward_counts}"
    )

    print(
        "v2_1_forward_weights="
        f"{config.forward_weights.tolist()}"
    )

    print(
        f"strafe_class_counts={strafe_counts}"
    )

    print(
        "v2_1_strafe_weights="
        f"{config.strafe_weights.tolist()}"
    )

    print(
        "v2_1_jump_positive_weight="
        f"{config.jump_pos_weight:.6f}"
    )

    print(
        "v2_1_yaw_positive_weight="
        f"{config.yaw_pos_weight:.6f}"
    )

    print(
        "v2_1_pitch_positive_weight="
        f"{config.pitch_pos_weight:.6f}"
    )


    return config


def build_session_balanced_weights(
    sessions,
) -> torch.Tensor:

    weights = []

    print(
        "event_oversampling=False"
    )

    print(
        "session_equal_probability_mass=True"
    )


    for session in sessions:

        count = len(session)

        if count <= 0:
            raise RuntimeError(
                "Empty training session."
            )

        per_sample = 1.0 / count

        weights.extend(
            [per_sample] * count
        )

        print(
            f"{session.name}_sequence_count={count}"
        )

        print(
            f"{session.name}_raw_sampling_mass="
            f"{per_sample * count:.12f}"
        )


    tensor = torch.tensor(
        weights,
        dtype=torch.double,
    )


    if len(tensor) != sum(len(session) for session in sessions):
        raise RuntimeError(
            "Unexpected session-balanced weight count."
        )


    return tensor


def make_train_loader(
    sessions,
    dataset,
    batch_size: int,
):

    weights = (
        build_session_balanced_weights(
            sessions
        )
    )


    generator = torch.Generator()
    generator.manual_seed(SEED)


    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def average_precision(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> float:

    truth = (
        target >= 0.5
    )

    positive_count = int(
        truth.sum()
    )

    if positive_count <= 0:
        return 0.0


    order = torch.argsort(
        probability,
        descending=True,
    )


    ranked_truth = truth[
        order
    ].to(
        torch.float64
    )


    cumulative = torch.cumsum(
        ranked_truth,
        dim=0,
    )


    rank = torch.arange(
        1,
        len(ranked_truth) + 1,
        dtype=torch.float64,
    )


    precision = cumulative / rank


    return float(
        precision[
            ranked_truth > 0.5
        ].mean()
    )


@torch.inference_mode()
def evaluate(
    model,
    loader,
    loss_config,
    device,
):

    model.eval()

    loss_totals = {}
    examples = 0

    collected = {
        "forward_logits": [],
        "strafe_logits": [],
        "jump_logits": [],
        "yaw_activity_logits": [],
        "yaw_magnitude": [],
        "pitch_activity_logits": [],
        "pitch_magnitude": [],
        "forward_target": [],
        "strafe_target": [],
        "jump_target": [],
        "yaw_activity_target": [],
        "yaw_target": [],
        "pitch_activity_target": [],
        "pitch_target": [],
    }


    for raw_batch in loader:

        batch = v2.unpack_batch(
            raw_batch,
            device,
        )

        output = model(
            batch[0]
        )

        losses = v2.compute_losses(
            output,
            batch,
            loss_config,
        )

        batch_size = int(
            batch[0].shape[0]
        )

        examples += batch_size


        for name, value in losses.items():

            loss_totals[name] = (
                loss_totals.get(
                    name,
                    0.0,
                )
                + float(value) * batch_size
            )


        collected[
            "forward_logits"
        ].append(
            output[
                "forward_logits"
            ].cpu()
        )

        collected[
            "strafe_logits"
        ].append(
            output[
                "strafe_logits"
            ].cpu()
        )

        collected[
            "jump_logits"
        ].append(
            output[
                "jump_logit"
            ].cpu()
        )

        collected[
            "yaw_activity_logits"
        ].append(
            output[
                "yaw_activity_logit"
            ].cpu()
        )

        collected[
            "yaw_magnitude"
        ].append(
            output[
                "yaw_magnitude"
            ].cpu()
        )

        collected[
            "pitch_activity_logits"
        ].append(
            output[
                "pitch_activity_logit"
            ].cpu()
        )

        collected[
            "pitch_magnitude"
        ].append(
            output[
                "pitch_magnitude"
            ].cpu()
        )


        for name, index in (
            ("forward_target", 1),
            ("strafe_target", 2),
            ("jump_target", 3),
            ("yaw_activity_target", 4),
            ("yaw_target", 5),
            ("pitch_activity_target", 6),
            ("pitch_target", 7),
        ):

            collected[name].append(
                batch[index].cpu()
            )


    value = {
        key: torch.cat(parts)
        for key, parts
        in collected.items()
    }


    forward_metrics = (
        v2.categorical_metrics(
            value[
                "forward_logits"
            ].argmax(dim=1),

            value[
                "forward_target"
            ],

            v2.FORWARD_NAMES,
        )
    )


    strafe_metrics = (
        v2.categorical_metrics(
            value[
                "strafe_logits"
            ].argmax(dim=1),

            value[
                "strafe_target"
            ],

            v2.STRAFE_NAMES,
        )
    )


    jump_metrics = v2.binary_metrics(
        value["jump_logits"],
        value["jump_target"],
    )

    yaw_metrics = v2.binary_metrics(
        value["yaw_activity_logits"],
        value["yaw_activity_target"],
    )

    pitch_metrics = v2.binary_metrics(
        value["pitch_activity_logits"],
        value["pitch_activity_target"],
    )


    jump_ap = average_precision(
        torch.sigmoid(
            value["jump_logits"]
        ),
        value["jump_target"],
    )

    yaw_ap = average_precision(
        torch.sigmoid(
            value[
                "yaw_activity_logits"
            ]
        ),
        value[
            "yaw_activity_target"
        ],
    )

    pitch_ap = average_precision(
        torch.sigmoid(
            value[
                "pitch_activity_logits"
            ]
        ),
        value[
            "pitch_activity_target"
        ],
    )


    yaw_mask = (
        value[
            "yaw_activity_target"
        ] > 0.5
    )

    pitch_mask = (
        value[
            "pitch_activity_target"
        ] > 0.5
    )


    yaw_mae = (
        float(
            (
                (
                    value["yaw_magnitude"][
                        yaw_mask
                    ]
                    -
                    value["yaw_target"][
                        yaw_mask
                    ]
                ).abs()
                * v2.YAW_SCALE
            ).mean()
        )
        if bool(yaw_mask.any())
        else None
    )


    pitch_mae = (
        float(
            (
                (
                    value[
                        "pitch_magnitude"
                    ][pitch_mask]
                    -
                    value[
                        "pitch_target"
                    ][pitch_mask]
                ).abs()
                * v2.PITCH_SCALE
            ).mean()
        )
        if bool(pitch_mask.any())
        else None
    )


    yaw_penalty = min(
        1.0,
        yaw_mae / v2.YAW_SCALE,
    )

    pitch_penalty = min(
        1.0,
        pitch_mae / v2.PITCH_SCALE,
    )


    selection_score = (
        0.30
        * forward_metrics[
            "macro_f1"
        ]

        + 0.25
        * strafe_metrics[
            "macro_f1"
        ]

        + 0.10
        * jump_ap

        + 0.15
        * yaw_ap

        + 0.10
        * pitch_ap

        - 0.025
        * (
            yaw_penalty
            + pitch_penalty
        )
    )


    result = {
        key:
            total / examples

        for key, total
        in loss_totals.items()
    }


    result.update(
        {
            "examples":
                examples,

            "forward_metrics":
                forward_metrics,

            "strafe_metrics":
                strafe_metrics,

            "jump_metrics":
                jump_metrics,

            "yaw_activity_metrics":
                yaw_metrics,

            "pitch_activity_metrics":
                pitch_metrics,

            "jump_average_precision":
                jump_ap,

            "yaw_activity_average_precision":
                yaw_ap,

            "pitch_activity_average_precision":
                pitch_ap,

            "yaw_active_mae_degrees":
                yaw_mae,

            "pitch_active_mae_degrees":
                pitch_mae,

            "selection_score":
                selection_score,
        }
    )


    return result



def make_datasets_v2_2(
    contract_path,
):

    contract = v2.read_json(
        contract_path
    )


    if contract.get(
        "session_split_ready"
    ) is not True:

        raise RuntimeError(
            "V2.2 whole-session contract not ready."
        )


    if contract.get(
        "split_strategy"
    ) != "whole_recording_session":

        raise RuntimeError(
            "Unexpected V2.2 split strategy."
        )


    if contract.get(
        "random_frame_split"
    ) is not False:

        raise RuntimeError(
            "Random frame splitting forbidden."
        )


    if contract.get(
        "training_started"
    ) is not False:

        raise RuntimeError(
            "Contract already marks training started."
        )


    if contract.get(
        "automatic_action_allowed"
    ) is not False:

        raise RuntimeError(
            "Automatic action flag invalid."
        )


    if contract.get(
        "behavioral_control_gate_passed"
    ) is not False:

        raise RuntimeError(
            "Behavioral-control gate unexpectedly passed."
        )


    if contract.get(
        "controls_modified"
    ) is not False:

        raise RuntimeError(
            "Control modification flag invalid."
        )


    if contract.get(
        "V8_consumed"
    ) is not False:

        raise RuntimeError(
            "V8 has been consumed."
        )


    expected_train = [
        "V4",
        "V6",
        "V7",
        "V9",
        "V10",
    ]


    train_names = list(
        contract[
            "train_session_order"
        ]
    )


    if train_names != expected_train:

        raise RuntimeError(
            f"Unexpected V2.2 train order: "
            f"{train_names}"
        )


    if list(
        contract[
            "train_sessions"
        ]
    ) != expected_train:

        raise RuntimeError(
            "train_sessions does not match "
            "train_session_order."
        )


    validation_name = str(
        contract[
            "validation_session"
        ]
    )


    if validation_name != "V5":

        raise RuntimeError(
            "V5 must remain validation."
        )


    if contract.get(
        "untouched_final_test_session"
    ) != "V8":

        raise RuntimeError(
            "V8 final-test contract failed."
        )


    if set(train_names) & {
        validation_name
    }:

        raise RuntimeError(
            "Training/validation overlap."
        )


    specs = contract[
        "sessions"
    ]


    expected_session_set = {
        "V4",
        "V5",
        "V6",
        "V7",
        "V9",
        "V10",
    }


    if set(specs) != expected_session_set:

        raise RuntimeError(
            f"Unexpected runtime session set: "
            f"{sorted(specs)}"
        )


    train_sessions = [
        v2.FrozenSequenceSession(
            name,
            specs[name],
        )
        for name in train_names
    ]


    validation_session = (
        v2.FrozenSequenceSession(
            validation_name,
            specs[
                validation_name
            ],
        )
    )


    train_dataset = (
        v2.ConcatDataset(
            train_sessions
        )
    )


    expected_train_count = int(
        contract[
            "train_sequence_count"
        ]
    )

    expected_validation_count = int(
        contract[
            "validation_sequence_count"
        ]
    )


    if expected_train_count != 10517:

        raise RuntimeError(
            "V2.2 contract does not specify "
            "10517 training sequences."
        )


    if expected_validation_count != 1190:

        raise RuntimeError(
            "V2.2 contract does not specify "
            "1190 validation sequences."
        )


    if len(
        train_dataset
    ) != expected_train_count:

        raise RuntimeError(
            f"Expected {expected_train_count} "
            f"combined V2.2 training sequences; "
            f"got {len(train_dataset)}."
        )


    if len(
        validation_session
    ) != expected_validation_count:

        raise RuntimeError(
            f"Expected {expected_validation_count} "
            f"V2.2 validation sequences; "
            f"got {len(validation_session)}."
        )


    for session in train_sessions:

        expected = int(
            specs[
                session.name
            ][
                "sequence_count"
            ]
        )

        if len(session) != expected:

            raise RuntimeError(
                f"{session.name}: runtime "
                "sequence-count mismatch."
            )


    print(
        "v2_2_train_session_order="
        + ",".join(
            train_names
        )
    )

    print(
        f"v2_2_train_sequences="
        f"{len(train_dataset)}"
    )

    print(
        f"v2_2_validation_sequences="
        f"{len(validation_session)}"
    )

    print(
        "v2_2_dataset_constructor_passed=True"
    )


    return (
        contract,
        train_sessions,
        train_dataset,
        validation_session,
    )


def smoke_only(
    contract_path: Path,
):

    (
        _contract,
        sessions,
        train_dataset,
        validation_dataset,
    ) = make_datasets_v2_2(
        contract_path
    )


    labels = v2.collect_labels(
        sessions
    )

    config = build_loss_config(
        labels
    )

    weights = (
        build_session_balanced_weights(
            sessions
        )
    )


    if not (
        config.jump_pos_weight
        <= 2.5
    ):
        raise RuntimeError(
            "Jump weight is not sufficiently softened."
        )


    first = train_dataset[0]

    batch = tuple(
        item.unsqueeze(0)
        for item in first
    )


    model = (
        v2.TemporalVisualPolicyV2()
    )

    model.eval()


    with torch.inference_mode():

        output = model(
            batch[0]
        )

        losses = v2.compute_losses(
            output,
            batch,
            config,
        )


    if not bool(
        torch.isfinite(
            losses["total_loss"]
        )
    ):
        raise RuntimeError(
            "Non-finite V2.1 smoke loss."
        )


    print(
        f"train_sequences={len(train_dataset)}"
    )

    print(
        "validation_sequences="
        f"{len(validation_dataset)}"
    )

    print(
        f"sampling_weight_count={len(weights)}"
    )

    print(
        "event_oversampling=False"
    )

    print(
        "architecture_reused_from_V2=True"
    )

    print(
        "optimizer_created=False"
    )

    print(
        "checkpoint_written=False"
    )

    print(
        "v2_1_smoke_passed=True"
    )


def canary_only(
    contract_path: Path,
    batch_size: int,
    learning_rate: float,
):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )


    (
        _contract,
        sessions,
        train_dataset,
        _validation_dataset,
    ) = make_datasets_v2_2(
        contract_path
    )


    labels = v2.collect_labels(
        sessions
    )

    config = build_loss_config(
        labels
    )

    loader = make_train_loader(
        sessions,
        train_dataset,
        batch_size,
    )


    device = torch.device(
        "cuda:0"
    )

    model = (
        v2.TemporalVisualPolicyV2()
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4,
    )


    before = (
        model.policy_token
        .detach()
        .clone()
    )


    raw_batch = next(
        iter(loader)
    )

    batch = v2.unpack_batch(
        raw_batch,
        device,
    )


    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        batch[0]
    )

    losses = v2.compute_losses(
        output,
        batch,
        config,
    )

    loss = losses[
        "total_loss"
    ]


    if not bool(
        torch.isfinite(loss)
    ):
        raise RuntimeError(
            "Non-finite canary loss."
        )


    loss.backward()


    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
    )


    optimizer.step()


    delta = float(
        (
            model.policy_token.detach()
            - before
        ).abs().sum()
    )


    if not (
        math.isfinite(delta)
        and delta > 0
    ):
        raise RuntimeError(
            "Canary parameter update failed."
        )


    print(
        f"canary_total_loss={float(loss.detach()):.9f}"
    )

    print(
        "gradient_norm_before_clip="
        f"{float(gradient_norm):.9f}"
    )

    print(
        "policy_token_parameter_delta_l1="
        f"{delta:.12f}"
    )

    print(
        "optimizer_steps_completed=1"
    )

    print(
        "checkpoint_written=False"
    )

    print(
        "v2_1_canary_passed=True"
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    trainer_sha: str,
    contract_sha: str,
    train_metrics,
    validation_metrics,
):

    payload = {
        "format_version":
            21,

        "model_version":
            MODEL_VERSION,

        "epoch":
            epoch,

        "trainer_sha256":
            trainer_sha,

        "parent_v2_trainer_sha256":
            sha256_file(
                HERE
                / "train_temporal_policy_v2.py"
            ),

        "contract_sha256":
            contract_sha,

        "architecture":
            v2.architecture_metadata(),

        "training_recipe": {
            "event_oversampling":
                False,

            "sampling":
                "equal_session_probability_mass",

            "movement_weight_clamp":
                [0.75, 2.0],

            "binary_positive_weight_clamp":
                [1.0, 2.5],

            "weight_decay":
                5e-4,

            "checkpoint_selection":
                "threshold_independent_AP_plus_movement_macro_f1",
        },

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_metrics":
            train_metrics,

        "validation_metrics":
            validation_metrics,

        "automatic_action_allowed":
            False,

        "behavioral_control_gate_passed":
            False,

        "controls_modified":
            False,
    }


    temporary = Path(
        str(path)
        + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    temporary.replace(path)


def train_full(
    contract_path: Path,
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )


    if output_dir.exists():
        raise RuntimeError(
            "Output directory already exists."
        )


    (
        contract,
        sessions,
        train_dataset,
        validation_dataset,
    ) = make_datasets_v2_2(
        contract_path
    )


    labels = v2.collect_labels(
        sessions
    )

    config = build_loss_config(
        labels
    )


    train_loader = make_train_loader(
        sessions,
        train_dataset,
        batch_size,
    )


    validation_loader = (
        v2.make_validation_loader(
            validation_dataset,
            batch_size,
        )
    )


    device = torch.device(
        "cuda:0"
    )


    model = (
        v2.TemporalVisualPolicyV2()
        .to(device)
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=5e-4,
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    trainer_sha = sha256_file(
        Path(__file__)
    )

    contract_sha = sha256_file(
        contract_path
    )


    metrics_path = (
        output_dir
        / "metrics.jsonl"
    )

    best_path = (
        output_dir
        / "best.pt"
    )

    last_path = (
        output_dir
        / "last.pt"
    )

    candidate_dir = (
        output_dir
        / "candidates"
    )


    manifest = {
        "format_version":
            21,

        "model_version":
            MODEL_VERSION,

        "trainer":
            str(
                Path(__file__).resolve()
            ),

        "trainer_sha256":
            trainer_sha,

        "parent_v2_trainer_sha256":
            sha256_file(
                HERE
                / "train_temporal_policy_v2.py"
            ),

        "contract":
            str(
                contract_path.resolve()
            ),

        "contract_sha256":
            contract_sha,

        "train_sessions":
            ["V4", "V6", "V7", "V9", "V10"],

        "validation_session":
            "V5",

        "untouched_final_test_session":
            "V8",

        "train_sequence_count":
            len(train_dataset),

        "validation_sequence_count":
            len(validation_dataset),

        "epochs_requested":
            epochs,

        "patience":
            patience,

        "batch_size":
            batch_size,

        "learning_rate":
            learning_rate,

        "weight_decay":
            5e-4,

        "event_oversampling":
            False,

        "sampling":
            "equal_session_probability_mass",

        "checkpoint_selection":
            "threshold_independent_AP_plus_movement_macro_f1",

        "automatic_action_allowed":
            False,

        "behavioral_control_gate_passed":
            False,

        "controls_modified":
            False,

        "live_xonotic_control":
            False,
    }


    (
        output_dir
        / "run_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    best_score = float("-inf")
    best_epoch = None
    without_improvement = 0


    for epoch in range(
        1,
        epochs + 1,
    ):

        train_metrics = v2.train_epoch(
            model,
            train_loader,
            optimizer,
            config,
            device,
        )


        validation = evaluate(
            model,
            validation_loader,
            config,
            device,
        )


        score = float(
            validation[
                "selection_score"
            ]
        )


        improved = score > best_score


        row = {
            "epoch":
                epoch,

            "best_so_far":
                improved,

            "train":
                train_metrics,

            "validation":
                validation,
        }


        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

            f.flush()


        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            trainer_sha,
            contract_sha,
            train_metrics,
            validation,
        )


        if improved:

            best_score = score
            best_epoch = epoch
            without_improvement = 0

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                trainer_sha,
                contract_sha,
                train_metrics,
                validation,
            )

            candidate_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            candidate_path = (
                candidate_dir
                / f"best_epoch_{epoch:03d}.pt"
            )

            save_checkpoint(
                candidate_path,
                model,
                optimizer,
                epoch,
                trainer_sha,
                contract_sha,
                train_metrics,
                validation,
            )

            print(
                "candidate_checkpoint="
                f"{candidate_path}"
            )

        else:

            without_improvement += 1


        print(
            f"epoch={epoch} "
            f"train_loss="
            f"{train_metrics['total_loss']:.6f} "
            f"validation_loss="
            f"{validation['total_loss']:.6f} "
            f"selection_score="
            f"{score:.6f} "
            f"best_so_far={improved}"
        )


        print(
            "forward_macro_f1="
            f"{validation['forward_metrics']['macro_f1']:.6f} "
            "strafe_macro_f1="
            f"{validation['strafe_metrics']['macro_f1']:.6f} "
            "jump_ap="
            f"{validation['jump_average_precision']:.6f} "
            "yaw_ap="
            f"{validation['yaw_activity_average_precision']:.6f} "
            "pitch_ap="
            f"{validation['pitch_activity_average_precision']:.6f}"
        )


        print(
            "jump_default_f1="
            f"{validation['jump_metrics']['f1']:.6f} "
            "yaw_default_f1="
            f"{validation['yaw_activity_metrics']['f1']:.6f} "
            "pitch_default_f1="
            f"{validation['pitch_activity_metrics']['f1']:.6f}"
        )


        if (
            without_improvement
            >= patience
        ):

            print(
                "early_stopping_triggered=True"
            )

            break


    if best_epoch is None:
        raise RuntimeError(
            "No V2.1 best checkpoint."
        )


    print(
        f"best_epoch={best_epoch}"
    )

    print(
        f"best_selection_score={best_score:.9f}"
    )

    print(
        "v2_1_training_complete=True"
    )

    print(
        "automatic_action_allowed=False"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--smoke-only",
        action="store_true",
    )

    mode.add_argument(
        "--canary-only",
        action="store_true",
    )

    mode.add_argument(
        "--train",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )

    args = parser.parse_args()


    random.seed(SEED)
    torch.manual_seed(SEED)


    if args.smoke_only:

        smoke_only(
            args.contract
        )

    elif args.canary_only:

        canary_only(
            args.contract,
            args.batch_size,
            args.learning_rate,
        )

    else:

        if args.output_dir is None:
            raise RuntimeError(
                "--output-dir required."
            )

        train_full(
            args.contract,
            args.output_dir,
            args.epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
        )


if __name__ == "__main__":
    main()
