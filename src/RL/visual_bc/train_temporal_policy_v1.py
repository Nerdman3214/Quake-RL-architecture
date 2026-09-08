from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import Dataset


# ============================================================
# Visual BC temporal-policy V1
# ============================================================
#
# Input:
#   4 frames
#   x 41 frozen DINOv3 tokens per frame
#   x 384 dimensions per token
#
# Architecture:
#   6-layer Transformer encoder
#   d_model = 384
#   attention heads = 6
#   FFN = 1536
#
# V1 supervised outputs:
#
#   Binary / held-state heads:
#       forward positive
#       forward negative
#       strafe positive
#       strafe negative
#       jump
#
#   Continuous heads:
#       yaw delta
#       pitch delta
#
# Attack/use/crouch/secondary-attack are intentionally NOT
# trained in V1 because the current synchronized recordings do
# not provide validated labels for those channels.
#
# This file supports a --smoke-only mode which performs no
# optimizer step and writes no model checkpoint.
# ============================================================


SEQUENCE_LENGTH = 4

TOKENS_PER_FRAME = 41
FEATURE_DIM = 384

TRANSFORMER_LAYERS = 6
ATTENTION_HEADS = 6
FFN_DIM = 1536

DROPOUT = 0.10

BINARY_ACTION_NAMES = (
    "forward_positive_fraction",
    "forward_negative_fraction",
    "strafe_positive_fraction",
    "strafe_negative_fraction",
    "jump_fraction",
)

UNSUPPORTED_OPTIONAL_ACTIONS = (
    "attack_fraction",
    "crouch_fraction",
    "secondary_attack_fraction",
    "use_fraction",
)

VIEW_NAMES = (
    "view_yaw_delta_degrees",
    "view_pitch_delta_degrees",
)

# Fixed engineering scales.
#
# These are not learned from validation data.
# The model predicts normalized values which can later be
# converted back to degrees.
VIEW_SCALE = torch.tensor(
    [32.0, 16.0],
    dtype=torch.float32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"{path}: invalid JSONL "
                    f"line {line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise RuntimeError(
                    f"{path}: JSONL line {line_number} "
                    "is not an object."
                )

            rows.append(value)

    return rows


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


class CachedSequenceDataset(Dataset):
    """
    Dataset backed exclusively by the frozen DINO feature cache.

    No images are decoded during policy training.
    """

    def __init__(
        self,
        role: str,
        split_spec: dict[str, Any],
    ) -> None:
        super().__init__()

        self.role = role
        self.spec = split_spec

        self.sequence_path = Path(
            split_spec["sequence_index_file"]
        )

        self.cache_manifest_path = Path(
            split_spec["feature_cache_manifest"]
        )

        self.cache_dir = (
            self.cache_manifest_path.parent
        )

        self.sequences = read_jsonl(
            self.sequence_path
        )

        self.cache_manifest = read_json(
            self.cache_manifest_path
        )

        self.chunk_entries = (
            self.cache_manifest.get(
                "completed_chunks",
                [],
            )
        )

        if not isinstance(
            self.chunk_entries,
            list,
        ):
            raise RuntimeError(
                f"{role}: completed_chunks is not a list."
            )

        self.chunk_by_filename: dict[
            str,
            dict[str, Any],
        ] = {}

        for entry in self.chunk_entries:

            filename = entry[
                "feature_file"
            ]

            if filename in self.chunk_by_filename:
                raise RuntimeError(
                    f"{role}: duplicate feature chunk "
                    f"{filename}"
                )

            self.chunk_by_filename[
                filename
            ] = entry

        self._loaded_chunks: dict[
            str,
            torch.Tensor,
        ] = {}

        self.binary_targets: list[
            torch.Tensor
        ] = []

        self.view_targets: list[
            torch.Tensor
        ] = []

        self._audit_contract()


    def _audit_contract(self) -> None:

        expected_count = int(
            self.spec["sequence_count"]
        )

        if len(self.sequences) != expected_count:
            raise RuntimeError(
                f"{self.role}: expected "
                f"{expected_count} sequences, "
                f"found {len(self.sequences)}."
            )

        if (
            sha256_file(
                self.sequence_path
            )
            != self.spec[
                "sequence_index_sha256"
            ]
        ):
            raise RuntimeError(
                f"{self.role}: sequence-index SHA mismatch."
            )

        if (
            sha256_file(
                self.cache_manifest_path
            )
            != self.spec[
                "feature_cache_manifest_sha256"
            ]
        ):
            raise RuntimeError(
                f"{self.role}: cache-manifest SHA mismatch."
            )

        cache_shape = self.cache_manifest.get(
            "feature_shape_per_frame"
        )

        if cache_shape != [
            TOKENS_PER_FRAME,
            FEATURE_DIM,
        ]:
            raise RuntimeError(
                f"{self.role}: unexpected feature shape "
                f"{cache_shape}."
            )

        if (
            self.cache_manifest.get("status")
            != "cache_complete"
        ):
            raise RuntimeError(
                f"{self.role}: feature cache incomplete."
            )

        unsupported_non_null = {
            name: 0
            for name in UNSUPPORTED_OPTIONAL_ACTIONS
        }

        mouse_sources: dict[str, int] = {}

        for sequence_position, record in enumerate(
            self.sequences
        ):

            if record.get(
                "future_outcome_used_as_feature"
            ) is not False:
                raise RuntimeError(
                    f"{self.role}: future outcome leakage "
                    f"at sequence {sequence_position}."
                )

            if record.get(
                "automatic_action_allowed"
            ) is not False:
                raise RuntimeError(
                    f"{self.role}: automatic-action flag "
                    "unexpectedly enabled."
                )

            if record.get(
                "controls_modified"
            ) is not False:
                raise RuntimeError(
                    f"{self.role}: controls-modified flag "
                    "unexpectedly enabled."
                )

            if record.get(
                "raw_mouse_device_delta_captured"
            ) is not False:
                raise RuntimeError(
                    f"{self.role}: unexpected raw-mouse "
                    "label contract."
                )

            mouse_source = record.get(
                "mouse_label_source"
            )

            mouse_sources[mouse_source] = (
                mouse_sources.get(
                    mouse_source,
                    0,
                )
                + 1
            )

            if mouse_source != "view_angle_delta":
                raise RuntimeError(
                    f"{self.role}: unsupported mouse label "
                    f"source {mouse_source!r}."
                )

            feature_refs = record.get(
                "feature_refs"
            )

            filtered_indices = record.get(
                "filtered_indices"
            )

            source_sample_ids = record.get(
                "source_sample_ids"
            )

            if not (
                isinstance(feature_refs, list)
                and len(feature_refs)
                == SEQUENCE_LENGTH
            ):
                raise RuntimeError(
                    f"{self.role}: invalid feature_refs at "
                    f"sequence {sequence_position}."
                )

            if not (
                isinstance(filtered_indices, list)
                and len(filtered_indices)
                == SEQUENCE_LENGTH
            ):
                raise RuntimeError(
                    f"{self.role}: invalid filtered_indices "
                    f"at sequence {sequence_position}."
                )

            if not (
                isinstance(source_sample_ids, list)
                and len(source_sample_ids)
                == SEQUENCE_LENGTH
            ):
                raise RuntimeError(
                    f"{self.role}: invalid source_sample_ids "
                    f"at sequence {sequence_position}."
                )

            if record.get(
                "target_filtered_index"
            ) != filtered_indices[-1]:
                raise RuntimeError(
                    f"{self.role}: target filtered index is "
                    "not the endpoint frame."
                )

            if record.get(
                "target_source_sample_id"
            ) != source_sample_ids[-1]:
                raise RuntimeError(
                    f"{self.role}: target source ID is not "
                    "the endpoint frame."
                )

            for frame_position, (
                ref,
                filtered_index,
            ) in enumerate(
                zip(
                    feature_refs,
                    filtered_indices,
                )
            ):

                if not isinstance(ref, dict):
                    raise RuntimeError(
                        f"{self.role}: feature ref is not a "
                        "dictionary."
                    )

                cache_index = int(
                    ref["cache_index"]
                )

                feature_offset = int(
                    ref["feature_offset"]
                )

                feature_file = str(
                    ref["feature_file"]
                )

                if cache_index != int(
                    filtered_index
                ):
                    raise RuntimeError(
                        f"{self.role}: cache index != "
                        "filtered index."
                    )

                chunk = self.chunk_by_filename.get(
                    feature_file
                )

                if chunk is None:
                    raise RuntimeError(
                        f"{self.role}: referenced chunk "
                        f"{feature_file} not found."
                    )

                sample_start = int(
                    chunk["sample_start"]
                )

                sample_end = int(
                    chunk["sample_end"]
                )

                if not (
                    sample_start
                    <= cache_index
                    <= sample_end
                ):
                    raise RuntimeError(
                        f"{self.role}: cache index outside "
                        "referenced chunk."
                    )

                expected_offset = (
                    cache_index
                    - sample_start
                )

                if (
                    feature_offset
                    != expected_offset
                ):
                    raise RuntimeError(
                        f"{self.role}: invalid feature "
                        f"offset at sequence "
                        f"{sequence_position}, frame "
                        f"{frame_position}."
                    )

            action = record.get(
                "target_action"
            )

            if not isinstance(action, dict):
                raise RuntimeError(
                    f"{self.role}: target_action missing."
                )

            binary_values = []

            for name in BINARY_ACTION_NAMES:

                value = action.get(name)

                if value is None:
                    raise RuntimeError(
                        f"{self.role}: required action "
                        f"{name} is None."
                    )

                value = float(value)

                if not (
                    0.0 <= value <= 1.0
                ):
                    raise RuntimeError(
                        f"{self.role}: {name} outside "
                        f"[0,1]: {value}"
                    )

                binary_values.append(
                    value
                )

            view_values = []

            for name in VIEW_NAMES:

                value = action.get(name)

                if value is None:
                    raise RuntimeError(
                        f"{self.role}: required view label "
                        f"{name} is None."
                    )

                value = float(value)

                if not math.isfinite(value):
                    raise RuntimeError(
                        f"{self.role}: non-finite "
                        f"{name}."
                    )

                view_values.append(
                    value
                )

            for name in UNSUPPORTED_OPTIONAL_ACTIONS:

                if action.get(name) is not None:
                    unsupported_non_null[
                        name
                    ] += 1

            self.binary_targets.append(
                torch.tensor(
                    binary_values,
                    dtype=torch.float32,
                )
            )

            self.view_targets.append(
                torch.tensor(
                    view_values,
                    dtype=torch.float32,
                )
            )

        print(
            f"{self.role}_mouse_label_sources="
            f"{json.dumps(mouse_sources, sort_keys=True)}"
        )

        print(
            f"{self.role}_unsupported_action_non_null="
            f"{json.dumps(unsupported_non_null, sort_keys=True)}"
        )

        if any(
            unsupported_non_null.values()
        ):
            raise RuntimeError(
                f"{self.role}: optional action labels are "
                "present; V1 must be extended before training."
            )

        print(
            f"{self.role}_sequence_contract_valid=True"
        )


    def _load_chunk(
        self,
        filename: str,
    ) -> torch.Tensor:

        existing = self._loaded_chunks.get(
            filename
        )

        if existing is not None:
            return existing

        entry = self.chunk_by_filename[
            filename
        ]

        path = (
            self.cache_dir
            / filename
        )

        if not path.is_file():
            raise RuntimeError(
                f"{self.role}: missing feature file "
                f"{path}"
            )

        if (
            sha256_file(path)
            != entry["feature_sha256"]
        ):
            raise RuntimeError(
                f"{self.role}: feature SHA mismatch "
                f"for {filename}."
            )

        tensors = load_file(
            str(path),
            device="cpu",
        )

        if "features" not in tensors:
            raise RuntimeError(
                f"{self.role}: tensor key 'features' "
                f"missing in {filename}."
            )

        features = tensors[
            "features"
        ]

        expected_shape = tuple(
            entry["feature_shape"]
        )

        if tuple(
            features.shape
        ) != expected_shape:
            raise RuntimeError(
                f"{self.role}: feature shape mismatch "
                f"for {filename}: "
                f"{tuple(features.shape)} "
                f"!= {expected_shape}"
            )

        if features.dtype != torch.float16:
            raise RuntimeError(
                f"{self.role}: expected FP16 cache."
            )

        self._loaded_chunks[
            filename
        ] = features

        return features


    def __len__(self) -> int:
        return len(
            self.sequences
        )


    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        record = self.sequences[
            index
        ]

        frame_features = []

        for ref in record[
            "feature_refs"
        ]:

            chunk = self._load_chunk(
                str(
                    ref[
                        "feature_file"
                    ]
                )
            )

            offset = int(
                ref[
                    "feature_offset"
                ]
            )

            frame = chunk[
                offset
            ]

            if tuple(frame.shape) != (
                TOKENS_PER_FRAME,
                FEATURE_DIM,
            ):
                raise RuntimeError(
                    f"{self.role}: per-frame feature "
                    f"shape mismatch: {tuple(frame.shape)}"
                )

            frame_features.append(
                frame
            )

        features = torch.stack(
            frame_features,
            dim=0,
        )

        binary_target = (
            self.binary_targets[
                index
            ]
        )

        view_target_degrees = (
            self.view_targets[
                index
            ]
        )

        return (
            features,
            binary_target,
            view_target_degrees,
        )


    def binary_target_matrix(
        self,
    ) -> torch.Tensor:

        return torch.stack(
            self.binary_targets,
            dim=0,
        )


class TemporalVisualPolicy(nn.Module):
    """
    Six-layer Transformer over all cached spatial+temporal tokens.

    4 frames x 41 DINO tokens = 164 observation tokens.

    A learned policy token is prepended, producing 165 total
    Transformer tokens. Its final representation feeds the
    action heads.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.policy_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                FEATURE_DIM,
            )
        )

        self.spatial_position = nn.Parameter(
            torch.zeros(
                1,
                1,
                TOKENS_PER_FRAME,
                FEATURE_DIM,
            )
        )

        self.temporal_position = nn.Parameter(
            torch.zeros(
                1,
                SEQUENCE_LENGTH,
                1,
                FEATURE_DIM,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=FEATURE_DIM,
                nhead=ATTENTION_HEADS,
                dim_feedforward=FFN_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(
                FEATURE_DIM
            ),
        )

        self.binary_head = nn.Linear(
            FEATURE_DIM,
            len(
                BINARY_ACTION_NAMES
            ),
        )

        self.view_head = nn.Linear(
            FEATURE_DIM,
            len(
                VIEW_NAMES
            ),
        )

        self._initialize_parameters()


    def _initialize_parameters(
        self,
    ) -> None:

        nn.init.normal_(
            self.policy_token,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.spatial_position,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.temporal_position,
            mean=0.0,
            std=0.02,
        )


    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        expected = (
            SEQUENCE_LENGTH,
            TOKENS_PER_FRAME,
            FEATURE_DIM,
        )

        if tuple(
            features.shape[1:]
        ) != expected:
            raise RuntimeError(
                "Expected [B,4,41,384], "
                f"got {tuple(features.shape)}."
            )

        batch_size = (
            features.shape[0]
        )

        x = (
            features
            + self.spatial_position
            + self.temporal_position
        )

        x = x.reshape(
            batch_size,
            SEQUENCE_LENGTH
            * TOKENS_PER_FRAME,
            FEATURE_DIM,
        )

        policy_token = self.policy_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            (
                policy_token,
                x,
            ),
            dim=1,
        )

        x = self.encoder(
            x
        )

        policy_state = x[
            :,
            0,
            :,
        ]

        binary_logits = self.binary_head(
            policy_state
        )

        normalized_view = self.view_head(
            policy_state
        )

        return (
            binary_logits,
            normalized_view,
        )


def compute_positive_weights(
    train_dataset: CachedSequenceDataset,
) -> torch.Tensor:

    targets = (
        train_dataset.binary_target_matrix()
    )

    positive_mass = targets.sum(
        dim=0
    )

    negative_mass = (
        targets.shape[0]
        - positive_mass
    )

    if torch.any(
        positive_mass <= 0
    ):
        raise RuntimeError(
            "At least one binary action has zero positive "
            "training examples."
        )

    weights = (
        negative_mass
        / positive_mass
    )

    # Prevent an extremely rare action from dominating
    # the first experiment.
    weights = torch.clamp(
        weights,
        min=1.0,
        max=25.0,
    )

    return weights


def smoke_test(
    split_manifest: Path,
) -> None:

    split = read_json(
        split_manifest
    )

    if split.get(
        "session_split_ready"
    ) is not True:
        raise RuntimeError(
            "Whole-session split is not ready."
        )

    if split.get(
        "split_strategy"
    ) != "whole_recording_session":
        raise RuntimeError(
            "Unexpected split strategy."
        )

    if split.get(
        "random_frame_split"
    ) is not False:
        raise RuntimeError(
            "Random frame split is not allowed."
        )

    if split.get(
        "training_started"
    ) is not False:
        raise RuntimeError(
            "Split manifest says training already started."
        )

    train_dataset = CachedSequenceDataset(
        "train",
        split["train"],
    )

    validation_dataset = (
        CachedSequenceDataset(
            "validation",
            split["validation"],
        )
    )

    print(
        f"train_sequences="
        f"{len(train_dataset)}"
    )

    print(
        f"validation_sequences="
        f"{len(validation_dataset)}"
    )

    if len(train_dataset) != 656:
        raise RuntimeError(
            "Unexpected train sequence count."
        )

    if len(validation_dataset) != 1190:
        raise RuntimeError(
            "Unexpected validation sequence count."
        )

    if (
        train_dataset.spec[
            "session_id"
        ]
        == validation_dataset.spec[
            "session_id"
        ]
    ):
        raise RuntimeError(
            "Train and validation session IDs overlap."
        )

    pos_weight = compute_positive_weights(
        train_dataset
    )

    print(
        "binary_action_names="
        f"{json.dumps(BINARY_ACTION_NAMES)}"
    )

    print(
        "binary_positive_weights="
        f"{json.dumps(pos_weight.tolist())}"
    )

    train_indices = [
        0,
        len(train_dataset) - 1,
    ]

    validation_indices = [
        0,
        len(validation_dataset) - 1,
    ]

    train_batch = torch.stack(
        [
            train_dataset[index][0]
            for index in train_indices
        ],
        dim=0,
    ).float()

    validation_batch = torch.stack(
        [
            validation_dataset[index][0]
            for index in validation_indices
        ],
        dim=0,
    ).float()

    print(
        f"train_smoke_input_shape="
        f"{tuple(train_batch.shape)}"
    )

    print(
        "validation_smoke_input_shape="
        f"{tuple(validation_batch.shape)}"
    )

    model = TemporalVisualPolicy()
    model.eval()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"model_total_parameters="
        f"{total_parameters}"
    )

    print(
        f"model_trainable_parameters="
        f"{trainable_parameters}"
    )

    print(
        "transformer_layers="
        f"{TRANSFORMER_LAYERS}"
    )

    print(
        "attention_heads="
        f"{ATTENTION_HEADS}"
    )

    print(
        f"model_width={FEATURE_DIM}"
    )

    print(
        f"ffn_width={FFN_DIM}"
    )

    print(
        "observation_tokens="
        f"{SEQUENCE_LENGTH * TOKENS_PER_FRAME}"
    )

    print(
        "transformer_tokens_with_policy_token="
        f"{SEQUENCE_LENGTH * TOKENS_PER_FRAME + 1}"
    )

    with torch.inference_mode():

        train_logits, train_view = model(
            train_batch
        )

        validation_logits, validation_view = (
            model(
                validation_batch
            )
        )

    print(
        "train_binary_logits_shape="
        f"{tuple(train_logits.shape)}"
    )

    print(
        "train_view_output_shape="
        f"{tuple(train_view.shape)}"
    )

    print(
        "validation_binary_logits_shape="
        f"{tuple(validation_logits.shape)}"
    )

    print(
        "validation_view_output_shape="
        f"{tuple(validation_view.shape)}"
    )

    if tuple(
        train_logits.shape
    ) != (
        len(train_indices),
        len(BINARY_ACTION_NAMES),
    ):
        raise RuntimeError(
            "Unexpected train binary-head shape."
        )

    if tuple(
        train_view.shape
    ) != (
        len(train_indices),
        len(VIEW_NAMES),
    ):
        raise RuntimeError(
            "Unexpected train view-head shape."
        )

    if tuple(
        validation_logits.shape
    ) != (
        len(validation_indices),
        len(BINARY_ACTION_NAMES),
    ):
        raise RuntimeError(
            "Unexpected validation binary-head shape."
        )

    if tuple(
        validation_view.shape
    ) != (
        len(validation_indices),
        len(VIEW_NAMES),
    ):
        raise RuntimeError(
            "Unexpected validation view-head shape."
        )

    if not torch.isfinite(
        train_logits
    ).all():
        raise RuntimeError(
            "Non-finite train smoke logits."
        )

    if not torch.isfinite(
        train_view
    ).all():
        raise RuntimeError(
            "Non-finite train smoke view outputs."
        )

    if not torch.isfinite(
        validation_logits
    ).all():
        raise RuntimeError(
            "Non-finite validation smoke logits."
        )

    if not torch.isfinite(
        validation_view
    ).all():
        raise RuntimeError(
            "Non-finite validation smoke view outputs."
        )

    # Demonstrate the intended loss contract without
    # backward() and without an optimizer.
    #
    # BCE targets are the five held-state fractions.
    # View targets use fixed engineering normalization.

    binary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    view_loss_fn = nn.SmoothL1Loss()

    train_binary_targets = torch.stack(
        [
            train_dataset[index][1]
            for index in train_indices
        ]
    )

    train_view_degrees = torch.stack(
        [
            train_dataset[index][2]
            for index in train_indices
        ]
    )

    normalized_view_targets = (
        train_view_degrees
        / VIEW_SCALE
    )

    binary_loss = binary_loss_fn(
        train_logits,
        train_binary_targets,
    )

    view_loss = view_loss_fn(
        train_view,
        normalized_view_targets,
    )

    total_loss = (
        binary_loss
        + view_loss
    )

    print(
        f"smoke_binary_loss="
        f"{binary_loss.item():.9f}"
    )

    print(
        f"smoke_view_loss="
        f"{view_loss.item():.9f}"
    )

    print(
        f"smoke_total_loss="
        f"{total_loss.item():.9f}"
    )

    if not math.isfinite(
        total_loss.item()
    ):
        raise RuntimeError(
            "Smoke loss is not finite."
        )

    print(
        "optimizer_created=False"
    )

    print(
        "backward_called=False"
    )

    print(
        "optimizer_step_called=False"
    )

    print(
        "checkpoint_written=False"
    )

    print(
        "temporal_policy_smoke_passed=True"
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--smoke-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.smoke_only:
        raise RuntimeError(
            "V1 preparation gate currently permits only "
            "--smoke-only. Training remains explicitly "
            "disabled until the next validated step."
        )

    smoke_test(
        args.split_manifest
    )


if __name__ == "__main__":
    main()
