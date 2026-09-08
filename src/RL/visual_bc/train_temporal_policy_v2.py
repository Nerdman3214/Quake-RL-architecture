#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from safetensors.torch import load_file

from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)


HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(
        0,
        str(HERE),
    )

import train_temporal_policy_v1 as v1


FEATURE_DIM = int(v1.FEATURE_DIM)
SEQUENCE_LENGTH = int(v1.SEQUENCE_LENGTH)
TOKENS_PER_FRAME = int(v1.TOKENS_PER_FRAME)
TRANSFORMER_LAYERS = int(v1.TRANSFORMER_LAYERS)
ATTENTION_HEADS = int(v1.ATTENTION_HEADS)
FFN_DIM = int(v1.FFN_DIM)
DROPOUT = float(
    getattr(
        v1,
        "DROPOUT",
        0.1,
    )
)

FORWARD_NAMES = (
    "backward",
    "neutral",
    "forward",
)

STRAFE_NAMES = (
    "left",
    "neutral",
    "right",
)

SEED = 20260822

AUTOMATIC_ACTION_ALLOWED = False
BEHAVIORAL_CONTROL_GATE_PASSED = False
CONTROLS_MODIFIED = False


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(
    path: Path,
) -> dict[str, Any]:

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Expected object JSON: {path}"
        )

    return value


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    rows: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, raw in enumerate(
            handle,
            start=1,
        ):

            if not raw.strip():
                continue

            value = json.loads(raw)

            if not isinstance(
                value,
                dict,
            ):
                raise RuntimeError(
                    f"{path}:{line_number}: "
                    "JSON object expected."
                )

            rows.append(value)

    return rows


def _view_scales() -> tuple[
    float,
    float,
]:

    names = list(v1.VIEW_NAMES)

    raw = v1.VIEW_SCALE

    if torch.is_tensor(raw):

        values = [
            float(value)
            for value in raw.detach()
            .cpu()
            .flatten()
            .tolist()
        ]

    elif isinstance(
        raw,
        (
            tuple,
            list,
        ),
    ):

        values = [
            float(value)
            for value in raw
        ]

    elif isinstance(
        raw,
        dict,
    ):

        values = [
            float(raw[name])
            for name in names
        ]

    else:

        values = [
            float(raw)
            for _ in names
        ]


    if len(values) != len(names):
        raise RuntimeError(
            "Unexpected V1 VIEW_SCALE shape."
        )


    mapping = dict(
        zip(
            names,
            values,
        )
    )


    yaw = float(
        mapping[
            "view_yaw_delta_degrees"
        ]
    )

    pitch = float(
        mapping[
            "view_pitch_delta_degrees"
        ]
    )


    if (
        not math.isfinite(yaw)
        or not math.isfinite(pitch)
        or yaw <= 0
        or pitch <= 0
    ):
        raise RuntimeError(
            "Invalid V1 engineering view scales."
        )


    return (
        yaw,
        pitch,
    )


YAW_SCALE, PITCH_SCALE = (
    _view_scales()
)


def movement_class(
    value: float,
) -> int:

    if value < 0:
        return 0

    if value > 0:
        return 2

    return 1


@dataclass(frozen=True)
class V2Label:
    forward: int
    strafe: int
    jump: float
    yaw_active: float
    yaw_normalized: float
    pitch_active: float
    pitch_normalized: float


class FrozenSequenceSession(
    Dataset,
):
    """
    Read-only temporal sequence dataset backed by
    precomputed DINO feature chunks.

    No images are decoded.
    No game process is touched.
    No action is emitted.
    """

    def __init__(
        self,
        name: str,
        spec: dict[str, Any],
    ) -> None:

        self.name = name
        self.spec = dict(spec)

        self.index_path = Path(
            self.spec[
                "sequence_index_file"
            ]
        )

        self.cache_dir = Path(
            self.spec[
                "cache_directory"
            ]
        )

        self.cache_manifest_path = Path(
            self.spec[
                "cache_manifest"
            ]
        )

        self.expected_count = int(
            self.spec[
                "sequence_count"
            ]
        )

        self.expected_index_sha = str(
            self.spec[
                "sequence_index_sha256"
            ]
        )


        if sha256_file(
            self.index_path
        ) != self.expected_index_sha:

            raise RuntimeError(
                f"{name}: sequence-index SHA mismatch."
            )


        if sha256_file(
            self.cache_manifest_path
        ) != self.spec[
            "cache_manifest_sha256"
        ]:

            raise RuntimeError(
                f"{name}: cache-manifest SHA mismatch."
            )


        self.rows = read_jsonl(
            self.index_path
        )

        if len(
            self.rows
        ) != self.expected_count:

            raise RuntimeError(
                f"{name}: sequence count mismatch."
            )


        manifest = read_json(
            self.cache_manifest_path
        )

        if manifest.get(
            "status"
        ) != "cache_complete":

            raise RuntimeError(
                f"{name}: feature cache incomplete."
            )

        if manifest.get(
            "feature_shape_per_frame"
        ) != [41, 384]:

            raise RuntimeError(
                f"{name}: feature-shape mismatch."
            )

        if manifest.get(
            "automatic_action_allowed"
        ) is not False:

            raise RuntimeError(
                f"{name}: cache control flag invalid."
            )

        if manifest.get(
            "controls_modified"
        ) is not False:

            raise RuntimeError(
                f"{name}: cache controls modified."
            )


        self.chunk_by_file = {
            str(entry["feature_file"]):
                dict(entry)

            for entry in manifest[
                "completed_chunks"
            ]
        }


        self.loaded_chunks: dict[
            str,
            torch.Tensor,
        ] = {}


        self.labels: list[
            V2Label
        ] = []


        for index, row in enumerate(
            self.rows
        ):

            self._validate_row(
                index,
                row,
            )

            action = row[
                "target_action"
            ]

            yaw = float(
                action[
                    "view_yaw_delta_degrees"
                ]
            )

            pitch = float(
                action[
                    "view_pitch_delta_degrees"
                ]
            )


            self.labels.append(
                V2Label(
                    forward=movement_class(
                        float(
                            action[
                                "forward_axis_mean"
                            ]
                        )
                    ),

                    strafe=movement_class(
                        float(
                            action[
                                "strafe_axis_mean"
                            ]
                        )
                    ),

                    jump=float(
                        float(
                            action[
                                "jump_fraction"
                            ]
                        )
                        > 0.0
                    ),

                    yaw_active=float(
                        abs(yaw) > 1e-12
                    ),

                    yaw_normalized=(
                        yaw / YAW_SCALE
                    ),

                    pitch_active=float(
                        abs(pitch) > 1e-12
                    ),

                    pitch_normalized=(
                        pitch / PITCH_SCALE
                    ),
                )
            )


        print(
            f"{self.name}_sequence_contract_valid=True"
        )

        print(
            f"{self.name}_sequences={len(self.rows)}"
        )


    def _validate_row(
        self,
        index: int,
        row: dict[str, Any],
    ) -> None:

        for key in (
            "automatic_action_allowed",
            "controls_modified",
            "future_outcome_used_as_feature",
        ):

            if row.get(key) is not False:

                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    f"{key} must be false."
                )


        if row.get(
            "raw_mouse_device_delta_captured"
        ) is not False:

            raise RuntimeError(
                f"{self.name}[{index}]: "
                "unexpected raw mouse label."
            )


        if row.get(
            "mouse_label_source"
        ) != "view_angle_delta":

            raise RuntimeError(
                f"{self.name}[{index}]: "
                "mouse-label source mismatch."
            )


        refs = row.get(
            "feature_refs"
        )

        if (
            not isinstance(
                refs,
                list,
            )
            or len(refs)
            != SEQUENCE_LENGTH
        ):

            raise RuntimeError(
                f"{self.name}[{index}]: "
                "invalid feature_refs."
            )


        for ref in refs:

            filename = str(
                ref[
                    "feature_file"
                ]
            )

            offset = int(
                ref[
                    "feature_offset"
                ]
            )

            cache_index = int(
                ref[
                    "cache_index"
                ]
            )


            entry = self.chunk_by_file.get(
                filename
            )

            if entry is None:
                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    f"unknown chunk {filename}."
                )


            sample_start = int(
                entry[
                    "sample_start"
                ]
            )

            sample_end = int(
                entry[
                    "sample_end"
                ]
            )


            if not (
                0
                <= offset
                < int(
                    entry[
                        "sample_count"
                    ]
                )
            ):
                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    "feature offset invalid."
                )


            if (
                sample_start
                + offset
                != cache_index
            ):
                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    "cache-index/offset mismatch."
                )


            if not (
                sample_start
                <= cache_index
                <= sample_end
            ):
                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    "cache index outside chunk."
                )


        action = row.get(
            "target_action"
        )

        if not isinstance(
            action,
            dict,
        ):
            raise RuntimeError(
                f"{self.name}[{index}]: "
                "target_action missing."
            )


        required = (
            "forward_axis_mean",
            "strafe_axis_mean",
            "jump_fraction",
            "view_yaw_delta_degrees",
            "view_pitch_delta_degrees",
        )


        for key in required:

            value = action.get(key)

            if value is None:
                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    f"{key} missing."
                )

            if not math.isfinite(
                float(value)
            ):
                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    f"{key} non-finite."
                )


        for key in (
            "attack_fraction",
            "secondary_attack_fraction",
            "crouch_fraction",
            "use_fraction",
        ):

            if action.get(key) is not None:
                raise RuntimeError(
                    f"{self.name}[{index}]: "
                    f"{key} unexpectedly populated."
                )


    def _load_chunk(
        self,
        filename: str,
    ) -> torch.Tensor:

        cached = self.loaded_chunks.get(
            filename
        )

        if cached is not None:
            return cached


        entry = self.chunk_by_file[
            filename
        ]

        path = (
            self.cache_dir
            / filename
        )


        if not path.is_file():

            raise RuntimeError(
                f"Missing feature chunk: {path}"
            )


        if sha256_file(
            path
        ) != entry[
            "feature_sha256"
        ]:

            raise RuntimeError(
                f"Feature chunk SHA mismatch: {path}"
            )


        tensors = load_file(
            str(path),
            device="cpu",
        )

        if list(
            tensors.keys()
        ) != ["features"]:

            raise RuntimeError(
                f"Unexpected tensor keys: {path}"
            )


        features = tensors[
            "features"
        ]


        if list(
            features.shape
        ) != list(
            entry[
                "feature_shape"
            ]
        ):

            raise RuntimeError(
                f"Feature shape mismatch: {path}"
            )


        if features.dtype != torch.float16:

            raise RuntimeError(
                f"Feature dtype mismatch: {path}"
            )


        if not bool(
            torch.isfinite(
                features
            ).all()
        ):

            raise RuntimeError(
                f"Non-finite feature tensor: {path}"
            )


        self.loaded_chunks[
            filename
        ] = features

        return features


    def __len__(
        self,
    ) -> int:

        return len(
            self.rows
        )


    def __getitem__(
        self,
        index: int,
    ):

        row = self.rows[
            index
        ]

        label = self.labels[
            index
        ]


        frames = []

        for ref in row[
            "feature_refs"
        ]:

            chunk = self._load_chunk(
                str(
                    ref[
                        "feature_file"
                    ]
                )
            )

            feature = chunk[
                int(
                    ref[
                        "feature_offset"
                    ]
                )
            ]

            if list(
                feature.shape
            ) != [
                TOKENS_PER_FRAME,
                FEATURE_DIM,
            ]:

                raise RuntimeError(
                    "Per-frame feature shape mismatch."
                )

            frames.append(feature)


        features = torch.stack(
            frames,
            dim=0,
        ).float()


        return (
            features,

            torch.tensor(
                label.forward,
                dtype=torch.long,
            ),

            torch.tensor(
                label.strafe,
                dtype=torch.long,
            ),

            torch.tensor(
                label.jump,
                dtype=torch.float32,
            ),

            torch.tensor(
                label.yaw_active,
                dtype=torch.float32,
            ),

            torch.tensor(
                label.yaw_normalized,
                dtype=torch.float32,
            ),

            torch.tensor(
                label.pitch_active,
                dtype=torch.float32,
            ),

            torch.tensor(
                label.pitch_normalized,
                dtype=torch.float32,
            ),
        )


class TemporalVisualPolicyV2(
    nn.Module,
):
    """
    Preserves the V1 temporal Transformer backbone.

    Replaces V1's five independent held-state logits
    and unconditional two-value view regression with:

      forward: 3-class categorical
      strafe:  3-class categorical
      jump:    binary
      yaw:     activity + active-only magnitude
      pitch:   activity + active-only magnitude
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


        layer = nn.TransformerEncoderLayer(
            d_model=FEATURE_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )


        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(
                FEATURE_DIM
            ),
        )


        self.forward_head = nn.Linear(
            FEATURE_DIM,
            3,
        )

        self.strafe_head = nn.Linear(
            FEATURE_DIM,
            3,
        )

        self.jump_head = nn.Linear(
            FEATURE_DIM,
            1,
        )

        self.yaw_activity_head = nn.Linear(
            FEATURE_DIM,
            1,
        )

        self.yaw_magnitude_head = nn.Linear(
            FEATURE_DIM,
            1,
        )

        self.pitch_activity_head = nn.Linear(
            FEATURE_DIM,
            1,
        )

        self.pitch_magnitude_head = nn.Linear(
            FEATURE_DIM,
            1,
        )


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


    def encode(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

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


        batch = int(
            features.shape[0]
        )


        x = (
            features
            + self.spatial_position
            + self.temporal_position
        )


        x = x.reshape(
            batch,
            SEQUENCE_LENGTH
            * TOKENS_PER_FRAME,
            FEATURE_DIM,
        )


        token = self.policy_token.expand(
            batch,
            -1,
            -1,
        )


        x = torch.cat(
            (
                token,
                x,
            ),
            dim=1,
        )


        x = self.encoder(x)

        return x[
            :,
            0,
            :,
        ]


    def forward(
        self,
        features: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        state = self.encode(
            features
        )


        return {
            "forward_logits":
                self.forward_head(state),

            "strafe_logits":
                self.strafe_head(state),

            "jump_logit":
                self.jump_head(
                    state
                ).squeeze(-1),

            "yaw_activity_logit":
                self.yaw_activity_head(
                    state
                ).squeeze(-1),

            "yaw_magnitude":
                self.yaw_magnitude_head(
                    state
                ).squeeze(-1),

            "pitch_activity_logit":
                self.pitch_activity_head(
                    state
                ).squeeze(-1),

            "pitch_magnitude":
                self.pitch_magnitude_head(
                    state
                ).squeeze(-1),
        }


def collect_labels(
    sessions: list[
        FrozenSequenceSession
    ],
) -> list[V2Label]:

    return [
        label
        for session in sessions
        for label in session.labels
    ]


def gentle_class_weights(
    counts: list[int],
) -> torch.Tensor:

    total = float(
        sum(counts)
    )


    values = torch.tensor(
        [
            math.sqrt(
                total
                / max(
                    float(count),
                    1.0,
                )
            )
            for count in counts
        ],
        dtype=torch.float32,
    )


    values = (
        values
        / values.mean()
    )


    return torch.clamp(
        values,
        min=0.50,
        max=3.00,
    )


def gentle_positive_weight(
    positives: int,
    total: int,
) -> float:

    negatives = (
        total
        - positives
    )


    if positives <= 0:
        raise RuntimeError(
            "Positive class has zero examples."
        )


    value = math.sqrt(
        negatives
        / positives
    )


    return float(
        min(
            5.0,
            max(
                1.0,
                value,
            ),
        )
    )


@dataclass
class LossConfig:
    forward_weights: torch.Tensor
    strafe_weights: torch.Tensor
    jump_pos_weight: float
    yaw_pos_weight: float
    pitch_pos_weight: float


def build_loss_config(
    labels: list[V2Label],
) -> LossConfig:

    total = len(labels)


    forward_counts = [
        sum(
            label.forward == i
            for label in labels
        )
        for i in range(3)
    ]


    strafe_counts = [
        sum(
            label.strafe == i
            for label in labels
        )
        for i in range(3)
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


    config = LossConfig(
        forward_weights=
            gentle_class_weights(
                forward_counts
            ),

        strafe_weights=
            gentle_class_weights(
                strafe_counts
            ),

        jump_pos_weight=
            gentle_positive_weight(
                jump_positive,
                total,
            ),

        yaw_pos_weight=
            gentle_positive_weight(
                yaw_positive,
                total,
            ),

        pitch_pos_weight=
            gentle_positive_weight(
                pitch_positive,
                total,
            ),
    )


    print(
        f"forward_class_counts={forward_counts}"
    )

    print(
        "forward_class_weights="
        f"{config.forward_weights.tolist()}"
    )

    print(
        f"strafe_class_counts={strafe_counts}"
    )

    print(
        "strafe_class_weights="
        f"{config.strafe_weights.tolist()}"
    )

    print(
        "jump_positive_weight="
        f"{config.jump_pos_weight:.6f}"
    )

    print(
        "yaw_positive_weight="
        f"{config.yaw_pos_weight:.6f}"
    )

    print(
        "pitch_positive_weight="
        f"{config.pitch_pos_weight:.6f}"
    )


    return config


def build_sampling_weights(
    sessions: list[
        FrozenSequenceSession
    ],
) -> torch.Tensor:

    labels = collect_labels(
        sessions
    )

    total = len(labels)

    forward_counts = [
        sum(
            label.forward == i
            for label in labels
        )
        for i in range(3)
    ]

    strafe_counts = [
        sum(
            label.strafe == i
            for label in labels
        )
        for i in range(3)
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


    forward_event = [
        min(
            3.0,
            math.sqrt(
                total
                / max(
                    count,
                    1,
                )
            ),
        )
        for count in forward_counts
    ]


    strafe_event = [
        min(
            3.0,
            math.sqrt(
                total
                / max(
                    count,
                    1,
                )
            ),
        )
        for count in strafe_counts
    ]


    jump_event = min(
        3.0,
        math.sqrt(
            total
            / max(
                jump_positive,
                1,
            )
        ),
    )

    yaw_event = min(
        3.0,
        math.sqrt(
            total
            / max(
                yaw_positive,
                1,
            )
        ),
    )

    pitch_event = min(
        3.0,
        math.sqrt(
            total
            / max(
                pitch_positive,
                1,
            )
        ),
    )


    mean_session_count = (
        sum(
            len(session)
            for session in sessions
        )
        / len(sessions)
    )


    values = []


    for session in sessions:

        session_factor = math.sqrt(
            mean_session_count
            / len(session)
        )

        session_factor = min(
            1.5,
            max(
                0.75,
                session_factor,
            ),
        )


        for label in session.labels:

            event_factor = max(
                1.0,
                forward_event[
                    label.forward
                ],
                strafe_event[
                    label.strafe
                ],
                (
                    jump_event
                    if label.jump > 0.5
                    else 1.0
                ),
                (
                    yaw_event
                    if label.yaw_active > 0.5
                    else 1.0
                ),
                (
                    pitch_event
                    if label.pitch_active > 0.5
                    else 1.0
                ),
            )


            weight = (
                session_factor
                * event_factor
            )


            values.append(
                min(
                    3.0,
                    max(
                        0.5,
                        weight,
                    ),
                )
            )


    tensor = torch.tensor(
        values,
        dtype=torch.double,
    )


    print(
        "sampling_weight_min="
        f"{tensor.min().item():.6f}"
    )

    print(
        "sampling_weight_mean="
        f"{tensor.mean().item():.6f}"
    )

    print(
        "sampling_weight_max="
        f"{tensor.max().item():.6f}"
    )


    return tensor


def unpack_batch(
    batch,
    device: torch.device,
):

    return tuple(
        value.to(
            device,
            non_blocking=True,
        )
        for value in batch
    )


def active_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:

    mask = (
        active > 0.5
    )


    if bool(
        mask.any()
    ):

        return nn.functional.smooth_l1_loss(
            prediction[mask],
            target[mask],
        )


    return (
        prediction.sum()
        * 0.0
    )


def compute_losses(
    output: dict[
        str,
        torch.Tensor,
    ],
    batch,
    config: LossConfig,
):

    (
        _features,
        forward_target,
        strafe_target,
        jump_target,
        yaw_active_target,
        yaw_target,
        pitch_active_target,
        pitch_target,
    ) = batch


    device = output[
        "forward_logits"
    ].device


    forward_loss = (
        nn.functional.cross_entropy(
            output[
                "forward_logits"
            ],
            forward_target,
            weight=
                config.forward_weights.to(
                    device
                ),
        )
    )


    strafe_loss = (
        nn.functional.cross_entropy(
            output[
                "strafe_logits"
            ],
            strafe_target,
            weight=
                config.strafe_weights.to(
                    device
                ),
        )
    )


    jump_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            output[
                "jump_logit"
            ],
            jump_target,
            pos_weight=torch.tensor(
                config.jump_pos_weight,
                device=device,
            ),
        )
    )


    yaw_activity_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            output[
                "yaw_activity_logit"
            ],
            yaw_active_target,
            pos_weight=torch.tensor(
                config.yaw_pos_weight,
                device=device,
            ),
        )
    )


    pitch_activity_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            output[
                "pitch_activity_logit"
            ],
            pitch_active_target,
            pos_weight=torch.tensor(
                config.pitch_pos_weight,
                device=device,
            ),
        )
    )


    yaw_magnitude_loss = (
        active_regression_loss(
            output[
                "yaw_magnitude"
            ],
            yaw_target,
            yaw_active_target,
        )
    )


    pitch_magnitude_loss = (
        active_regression_loss(
            output[
                "pitch_magnitude"
            ],
            pitch_target,
            pitch_active_target,
        )
    )


    total = (
        1.00 * forward_loss
        + 1.00 * strafe_loss
        + 0.75 * jump_loss
        + 0.50 * yaw_activity_loss
        + 0.25 * yaw_magnitude_loss
        + 0.50 * pitch_activity_loss
        + 0.25 * pitch_magnitude_loss
    )


    return {
        "total_loss":
            total,

        "forward_loss":
            forward_loss,

        "strafe_loss":
            strafe_loss,

        "jump_loss":
            jump_loss,

        "yaw_activity_loss":
            yaw_activity_loss,

        "yaw_magnitude_loss":
            yaw_magnitude_loss,

        "pitch_activity_loss":
            pitch_activity_loss,

        "pitch_magnitude_loss":
            pitch_magnitude_loss,
    }


def categorical_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_names,
):

    metrics = {}

    f1_values = []


    for class_index, name in enumerate(
        class_names
    ):

        pred_positive = (
            prediction == class_index
        )

        true_positive = (
            target == class_index
        )


        tp = int(
            (
                pred_positive
                & true_positive
            ).sum()
        )

        fp = int(
            (
                pred_positive
                & ~true_positive
            ).sum()
        )

        fn = int(
            (
                ~pred_positive
                & true_positive
            ).sum()
        )


        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            else 0.0
        )


        metrics[name] = {
            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "support":
                int(
                    true_positive.sum()
                ),
        }


        f1_values.append(
            f1
        )


    metrics[
        "macro_f1"
    ] = (
        sum(f1_values)
        / len(f1_values)
    )


    metrics[
        "accuracy"
    ] = float(
        (
            prediction
            == target
        ).float().mean()
    )


    return metrics


def binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
):

    prediction = (
        torch.sigmoid(logits)
        >= 0.5
    )

    truth = (
        target >= 0.5
    )


    tp = int(
        (
            prediction
            & truth
        ).sum()
    )

    fp = int(
        (
            prediction
            & ~truth
        ).sum()
    )

    fn = int(
        (
            ~prediction
            & truth
        ).sum()
    )

    tn = int(
        (
            ~prediction
            & ~truth
        ).sum()
    )


    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )


    return {
        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "positive_targets":
            int(
                truth.sum()
            ),

        "accuracy":
            (
                tp + tn
            )
            / max(
                tp + fp + fn + tn,
                1,
            ),
    }


def make_datasets(
    contract_path: Path,
):

    contract = read_json(
        contract_path
    )


    if contract.get(
        "session_split_ready"
    ) is not True:

        raise RuntimeError(
            "V2 whole-session contract not ready."
        )

    if contract.get(
        "split_strategy"
    ) != "whole_recording_session":

        raise RuntimeError(
            "Unexpected split strategy."
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
        "controls_modified"
    ) is not False:

        raise RuntimeError(
            "Control modification flag invalid."
        )


    specs = contract[
        "sessions"
    ]


    train_names = list(
        contract[
            "train_session_order"
        ]
    )


    validation_name = str(
        contract[
            "validation_session"
        ]
    )


    if set(train_names) & {
        validation_name
    }:

        raise RuntimeError(
            "Training/validation session overlap."
        )


    train_sessions = [
        FrozenSequenceSession(
            name,
            specs[name],
        )
        for name in train_names
    ]


    validation_session = (
        FrozenSequenceSession(
            validation_name,
            specs[
                validation_name
            ],
        )
    )


    train_dataset = ConcatDataset(
        train_sessions
    )


    if len(
        train_dataset
    ) != 5245:

        raise RuntimeError(
            "Expected 5245 combined training sequences."
        )

    if len(
        validation_session
    ) != 1190:

        raise RuntimeError(
            "Expected 1190 validation sequences."
        )


    return (
        contract,
        train_sessions,
        train_dataset,
        validation_session,
    )


def make_train_loader(
    train_sessions,
    train_dataset,
    batch_size: int,
):

    weights = build_sampling_weights(
        train_sessions
    )


    generator = torch.Generator()
    generator.manual_seed(
        SEED
    )


    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(
            train_dataset
        ),
        replacement=True,
        generator=generator,
    )


    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def make_validation_loader(
    validation_dataset,
    batch_size: int,
):

    return DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    loss_config,
    device,
):

    model.train()

    totals: dict[
        str,
        float,
    ] = {}

    examples = 0

    max_preclip_gradient_norm = 0.0


    for raw_batch in loader:

        batch = unpack_batch(
            raw_batch,
            device,
        )

        features = batch[0]


        optimizer.zero_grad(
            set_to_none=True
        )


        output = model(
            features
        )


        losses = compute_losses(
            output,
            batch,
            loss_config,
        )


        total = losses[
            "total_loss"
        ]


        if not bool(
            torch.isfinite(total)
        ):

            raise RuntimeError(
                "Non-finite training loss."
            )


        total.backward()


        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        )


        max_preclip_gradient_norm = max(
            max_preclip_gradient_norm,
            float(
                gradient_norm
            ),
        )


        optimizer.step()


        batch_size = int(
            features.shape[0]
        )

        examples += batch_size


        for name, value in losses.items():

            totals[name] = (
                totals.get(
                    name,
                    0.0,
                )
                + float(
                    value.detach()
                )
                * batch_size
            )


    result = {
        key:
            value
            / examples

        for key, value
        in totals.items()
    }


    result[
        "examples"
    ] = examples

    result[
        "max_preclip_gradient_norm"
    ] = (
        max_preclip_gradient_norm
    )


    return result


@torch.inference_mode()
def evaluate(
    model,
    loader,
    loss_config,
    device,
):

    model.eval()


    totals = {}

    examples = 0


    forward_logits = []
    strafe_logits = []

    jump_logits = []
    jump_targets = []

    yaw_activity_logits = []
    yaw_activity_targets = []
    yaw_magnitudes = []
    yaw_targets = []

    pitch_activity_logits = []
    pitch_activity_targets = []
    pitch_magnitudes = []
    pitch_targets = []


    for raw_batch in loader:

        batch = unpack_batch(
            raw_batch,
            device,
        )

        features = batch[0]

        output = model(
            features
        )


        losses = compute_losses(
            output,
            batch,
            loss_config,
        )


        batch_size = int(
            features.shape[0]
        )

        examples += batch_size


        for name, value in losses.items():

            totals[name] = (
                totals.get(
                    name,
                    0.0,
                )
                + float(
                    value
                )
                * batch_size
            )


        forward_logits.append(
            output[
                "forward_logits"
            ].cpu()
        )

        strafe_logits.append(
            output[
                "strafe_logits"
            ].cpu()
        )

        jump_logits.append(
            output[
                "jump_logit"
            ].cpu()
        )

        jump_targets.append(
            batch[3].cpu()
        )

        yaw_activity_logits.append(
            output[
                "yaw_activity_logit"
            ].cpu()
        )

        yaw_activity_targets.append(
            batch[4].cpu()
        )

        yaw_magnitudes.append(
            output[
                "yaw_magnitude"
            ].cpu()
        )

        yaw_targets.append(
            batch[5].cpu()
        )

        pitch_activity_logits.append(
            output[
                "pitch_activity_logit"
            ].cpu()
        )

        pitch_activity_targets.append(
            batch[6].cpu()
        )

        pitch_magnitudes.append(
            output[
                "pitch_magnitude"
            ].cpu()
        )

        pitch_targets.append(
            batch[7].cpu()
        )


    f_logits = torch.cat(
        forward_logits
    )

    s_logits = torch.cat(
        strafe_logits
    )


    f_target = torch.cat(
        [
            batch[1].cpu()
            for batch in []
        ]
    ) if False else None


    # Re-run target collection cheaply from loader's
    # underlying dataset labels, avoiding any ambiguity
    # caused by sampler state. Validation loader is natural
    # and ordered.
    dataset = loader.dataset

    labels = dataset.labels


    forward_target = torch.tensor(
        [
            label.forward
            for label in labels
        ],
        dtype=torch.long,
    )

    strafe_target = torch.tensor(
        [
            label.strafe
            for label in labels
        ],
        dtype=torch.long,
    )


    j_logits = torch.cat(
        jump_logits
    )

    j_target = torch.cat(
        jump_targets
    )


    ya_logits = torch.cat(
        yaw_activity_logits
    )

    ya_target = torch.cat(
        yaw_activity_targets
    )

    ym = torch.cat(
        yaw_magnitudes
    )

    yt = torch.cat(
        yaw_targets
    )


    pa_logits = torch.cat(
        pitch_activity_logits
    )

    pa_target = torch.cat(
        pitch_activity_targets
    )

    pm = torch.cat(
        pitch_magnitudes
    )

    pt = torch.cat(
        pitch_targets
    )


    forward_metrics = (
        categorical_metrics(
            f_logits.argmax(
                dim=1
            ),
            forward_target,
            FORWARD_NAMES,
        )
    )


    strafe_metrics = (
        categorical_metrics(
            s_logits.argmax(
                dim=1
            ),
            strafe_target,
            STRAFE_NAMES,
        )
    )


    jump_metrics = binary_metrics(
        j_logits,
        j_target,
    )

    yaw_activity_metrics = (
        binary_metrics(
            ya_logits,
            ya_target,
        )
    )

    pitch_activity_metrics = (
        binary_metrics(
            pa_logits,
            pa_target,
        )
    )


    yaw_mask = (
        ya_target > 0.5
    )

    pitch_mask = (
        pa_target > 0.5
    )


    yaw_mae = (
        float(
            (
                (
                    ym[yaw_mask]
                    - yt[yaw_mask]
                ).abs()
                * YAW_SCALE
            ).mean()
        )
        if bool(
            yaw_mask.any()
        )
        else None
    )


    pitch_mae = (
        float(
            (
                (
                    pm[pitch_mask]
                    - pt[pitch_mask]
                ).abs()
                * PITCH_SCALE
            ).mean()
        )
        if bool(
            pitch_mask.any()
        )
        else None
    )


    yaw_penalty = (
        min(
            1.0,
            (
                yaw_mae
                / YAW_SCALE
            ),
        )
        if yaw_mae is not None
        else 1.0
    )


    pitch_penalty = (
        min(
            1.0,
            (
                pitch_mae
                / PITCH_SCALE
            ),
        )
        if pitch_mae is not None
        else 1.0
    )


    behavior_score = (
        0.30
        * forward_metrics[
            "macro_f1"
        ]

        + 0.25
        * strafe_metrics[
            "macro_f1"
        ]

        + 0.10
        * jump_metrics[
            "f1"
        ]

        + 0.20
        * yaw_activity_metrics[
            "f1"
        ]

        + 0.15
        * pitch_activity_metrics[
            "f1"
        ]

        - 0.025
        * (
            yaw_penalty
            + pitch_penalty
        )
    )


    result = {
        key:
            value
            / examples

        for key, value
        in totals.items()
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
                yaw_activity_metrics,

            "pitch_activity_metrics":
                pitch_activity_metrics,

            "yaw_active_mae_degrees":
                yaw_mae,

            "pitch_active_mae_degrees":
                pitch_mae,

            "behavior_score":
                behavior_score,
        }
    )


    return result


def architecture_metadata():

    return {
        "name":
            "TemporalVisualPolicyV2",

        "feature_dim":
            FEATURE_DIM,

        "sequence_length":
            SEQUENCE_LENGTH,

        "tokens_per_frame":
            TOKENS_PER_FRAME,

        "observation_tokens":
            (
                SEQUENCE_LENGTH
                * TOKENS_PER_FRAME
            ),

        "transformer_layers":
            TRANSFORMER_LAYERS,

        "attention_heads":
            ATTENTION_HEADS,

        "ffn_dim":
            FFN_DIM,

        "dropout":
            DROPOUT,

        "forward_classes":
            list(
                FORWARD_NAMES
            ),

        "strafe_classes":
            list(
                STRAFE_NAMES
            ),

        "jump_head":
            "binary",

        "yaw_head":
            "activity_plus_active_only_magnitude",

        "pitch_head":
            "activity_plus_active_only_magnitude",

        "yaw_scale_degrees":
            YAW_SCALE,

        "pitch_scale_degrees":
            PITCH_SCALE,

        "sampling":
            "gentle_event_and_session_balancing",

        "validation_sampling":
            "natural_unmodified_V5",

        "checkpoint_selection":
            "behavior_aware_multi_metric",
    }


def smoke_only(
    contract_path: Path,
):

    (
        _contract,
        train_sessions,
        train_dataset,
        validation_dataset,
    ) = make_datasets(
        contract_path
    )


    labels = collect_labels(
        train_sessions
    )

    loss_config = build_loss_config(
        labels
    )

    sampling = build_sampling_weights(
        train_sessions
    )


    train_examples = [
        train_dataset[0],
        train_dataset[
            len(train_dataset) - 1
        ],
    ]

    validation_examples = [
        validation_dataset[0],
        validation_dataset[
            len(validation_dataset) - 1
        ],
    ]


    train_batch = tuple(
        torch.stack(
            [
                example[index]
                for example
                in train_examples
            ],
            dim=0,
        )
        for index in range(8)
    )


    validation_batch = tuple(
        torch.stack(
            [
                example[index]
                for example
                in validation_examples
            ],
            dim=0,
        )
        for index in range(8)
    )


    model = TemporalVisualPolicyV2()
    model.eval()


    with torch.inference_mode():

        train_output = model(
            train_batch[0]
        )

        validation_output = model(
            validation_batch[0]
        )


        losses = compute_losses(
            train_output,
            train_batch,
            loss_config,
        )


    expected_shapes = {
        "forward_logits":
            [2, 3],

        "strafe_logits":
            [2, 3],

        "jump_logit":
            [2],

        "yaw_activity_logit":
            [2],

        "yaw_magnitude":
            [2],

        "pitch_activity_logit":
            [2],

        "pitch_magnitude":
            [2],
    }


    for name, expected in (
        expected_shapes.items()
    ):

        actual = list(
            train_output[
                name
            ].shape
        )

        if actual != expected:
            raise RuntimeError(
                f"{name}: shape {actual} "
                f"!= {expected}"
            )

        if not bool(
            torch.isfinite(
                train_output[
                    name
                ]
            ).all()
        ):
            raise RuntimeError(
                f"{name}: non-finite output."
            )


    for name, tensor in (
        validation_output.items()
    ):

        if not bool(
            torch.isfinite(
                tensor
            ).all()
        ):
            raise RuntimeError(
                f"validation {name}: "
                "non-finite output."
            )


    for name, value in losses.items():

        if not bool(
            torch.isfinite(
                value
            )
        ):
            raise RuntimeError(
                f"{name}: non-finite smoke loss."
            )


    print(
        f"V1_VIEW_SCALE_yaw={YAW_SCALE}"
    )

    print(
        f"V1_VIEW_SCALE_pitch={PITCH_SCALE}"
    )

    print(
        f"train_sequences={len(train_dataset)}"
    )

    print(
        "validation_sequences="
        f"{len(validation_dataset)}"
    )

    print(
        "sampling_weight_count="
        f"{len(sampling)}"
    )

    print(
        "train_smoke_input_shape="
        f"{tuple(train_batch[0].shape)}"
    )

    print(
        "validation_smoke_input_shape="
        f"{tuple(validation_batch[0].shape)}"
    )

    print(
        "smoke_total_loss="
        f"{float(losses['total_loss']):.9f}"
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
        "v2_temporal_policy_smoke_passed=True"
    )


def canary_only(
    contract_path: Path,
    batch_size: int,
    learning_rate: float,
):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable for V2 canary."
        )


    (
        _contract,
        train_sessions,
        train_dataset,
        _validation_dataset,
    ) = make_datasets(
        contract_path
    )


    labels = collect_labels(
        train_sessions
    )

    loss_config = build_loss_config(
        labels
    )


    loader = make_train_loader(
        train_sessions,
        train_dataset,
        batch_size,
    )


    device = torch.device(
        "cuda:0"
    )


    model = TemporalVisualPolicyV2().to(
        device
    )

    model.train()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )


    before = (
        model.policy_token
        .detach()
        .clone()
    )


    raw_batch = next(
        iter(loader)
    )

    batch = unpack_batch(
        raw_batch,
        device,
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    output = model(
        batch[0]
    )


    losses = compute_losses(
        output,
        batch,
        loss_config,
    )


    loss = losses[
        "total_loss"
    ]


    if not bool(
        torch.isfinite(loss)
    ):
        raise RuntimeError(
            "Canary loss non-finite."
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
        delta > 0.0
        and math.isfinite(delta)
    ):
        raise RuntimeError(
            "Canary parameter update failed."
        )


    print(
        "canary_input_shape="
        f"{tuple(batch[0].shape)}"
    )

    print(
        "canary_total_loss="
        f"{float(loss):.9f}"
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
        "v2_one_step_training_canary_passed=True"
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
            2,

        "epoch":
            epoch,

        "trainer_sha256":
            trainer_sha,

        "contract_sha256":
            contract_sha,

        "architecture":
            architecture_metadata(),

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

    temporary.replace(
        path
    )


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
            f"Output directory exists: {output_dir}"
        )


    (
        contract,
        train_sessions,
        train_dataset,
        validation_dataset,
    ) = make_datasets(
        contract_path
    )


    labels = collect_labels(
        train_sessions
    )

    loss_config = build_loss_config(
        labels
    )


    train_loader = make_train_loader(
        train_sessions,
        train_dataset,
        batch_size,
    )


    validation_loader = (
        make_validation_loader(
            validation_dataset,
            batch_size,
        )
    )


    device = torch.device(
        "cuda:0"
    )


    model = TemporalVisualPolicyV2().to(
        device
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=False,
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

    run_manifest_path = (
        output_dir
        / "run_manifest.json"
    )


    trainer_sha = sha256_file(
        Path(__file__)
    )

    contract_sha = sha256_file(
        contract_path
    )


    run_manifest = {
        "format_version":
            2,

        "trainer":
            str(
                Path(__file__).resolve()
            ),

        "trainer_sha256":
            trainer_sha,

        "contract":
            str(
                contract_path.resolve()
            ),

        "contract_sha256":
            contract_sha,

        "train_sessions":
            contract[
                "train_session_order"
            ],

        "validation_session":
            contract[
                "validation_session"
            ],

        "untouched_final_test_session":
            contract[
                "untouched_final_test_session"
            ],

        "train_sequence_count":
            len(train_dataset),

        "validation_sequence_count":
            len(validation_dataset),

        "epochs_requested":
            epochs,

        "early_stopping_patience":
            patience,

        "batch_size":
            batch_size,

        "learning_rate":
            learning_rate,

        "weight_decay":
            1e-4,

        "gradient_clip_norm":
            1.0,

        "architecture":
            architecture_metadata(),

        "automatic_action_allowed":
            False,

        "behavioral_control_gate_passed":
            False,

        "controls_modified":
            False,

        "live_xonotic_control":
            False,
    }


    run_manifest_path.write_text(
        json.dumps(
            run_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    best_score = float(
        "-inf"
    )

    best_epoch = None

    epochs_without_improvement = 0


    for epoch in range(
        1,
        epochs + 1,
    ):

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_config,
            device,
        )


        validation_metrics = evaluate(
            model,
            validation_loader,
            loss_config,
            device,
        )


        score = float(
            validation_metrics[
                "behavior_score"
            ]
        )


        improved = (
            score
            > best_score
        )


        row = {
            "epoch":
                epoch,

            "best_so_far":
                improved,

            "train":
                train_metrics,

            "validation":
                validation_metrics,
        }


        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )

            handle.flush()


        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            trainer_sha,
            contract_sha,
            train_metrics,
            validation_metrics,
        )


        if improved:

            best_score = score
            best_epoch = epoch

            epochs_without_improvement = 0


            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                trainer_sha,
                contract_sha,
                train_metrics,
                validation_metrics,
            )

        else:

            epochs_without_improvement += 1


        print(
            f"epoch={epoch} "
            f"train_loss="
            f"{train_metrics['total_loss']:.6f} "
            f"validation_loss="
            f"{validation_metrics['total_loss']:.6f} "
            f"behavior_score="
            f"{score:.6f} "
            f"best_so_far={improved}"
        )


        print(
            "validation_forward_macro_f1="
            f"{validation_metrics['forward_metrics']['macro_f1']:.6f} "
            "validation_strafe_macro_f1="
            f"{validation_metrics['strafe_metrics']['macro_f1']:.6f} "
            "validation_jump_f1="
            f"{validation_metrics['jump_metrics']['f1']:.6f} "
            "validation_yaw_activity_f1="
            f"{validation_metrics['yaw_activity_metrics']['f1']:.6f} "
            "validation_pitch_activity_f1="
            f"{validation_metrics['pitch_activity_metrics']['f1']:.6f}"
        )


        print(
            "validation_yaw_active_mae_degrees="
            f"{validation_metrics['yaw_active_mae_degrees']} "
            "validation_pitch_active_mae_degrees="
            f"{validation_metrics['pitch_active_mae_degrees']}"
        )


        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "early_stopping_triggered=True"
            )

            break


    if best_epoch is None:
        raise RuntimeError(
            "No best checkpoint selected."
        )


    print(
        f"best_epoch={best_epoch}"
    )

    print(
        f"best_behavior_score={best_score:.9f}"
    )

    print(
        f"best_checkpoint={best_path}"
    )

    print(
        f"last_checkpoint={last_path}"
    )

    print(
        "policy_training_complete=True"
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
        default=8,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )


    args = parser.parse_args()


    random.seed(SEED)
    torch.manual_seed(SEED)


    if args.smoke_only:

        smoke_only(
            args.contract
        )

        return


    if args.canary_only:

        canary_only(
            args.contract,
            args.batch_size,
            args.learning_rate,
        )

        return


    if args.output_dir is None:

        raise RuntimeError(
            "--output-dir is required with --train."
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
