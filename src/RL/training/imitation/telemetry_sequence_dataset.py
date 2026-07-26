"""Telemetry-aware wrapper for composite recurrent demonstrations."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import (
    dataclass,
    fields,
    is_dataclass,
)
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from RL.agents.policies.hierarchical_recurrent import (
    DEFAULT_TELEMETRY_WEAPON_COUNT,
    TELEMETRY_FEATURE_COUNT,
)
from RL.training.imitation.composite_sequence_dataset import (
    CompositeSequenceBatch,
    CompositeSequenceDataset,
    collate_composite_sequences,
)


TELEMETRY_SCHEMA_VERSION = 1

TELEMETRY_FEATURE_NAMES = (
    "health_normalized",
    "armor_normalized",
    "ammo_normalized",
    "ammo_known",
    "alive",
    "score_normalized",
    "match_time_normalized",
    "telemetry_present",
)


@dataclass(frozen=True)
class TelemetryCompositeSequenceSample:
    """One base sequence plus synchronized telemetry tensors."""

    base_sample: object
    telemetry_features: torch.Tensor
    telemetry_weapon_indices: torch.Tensor
    telemetry_trainable_mask: torch.Tensor


@dataclass(frozen=True)
class TelemetryCompositeSequenceBatch(
    CompositeSequenceBatch
):
    """Composite-control batch with structured telemetry."""

    telemetry_features: torch.Tensor
    telemetry_weapon_indices: torch.Tensor


def _finite_number(
    value: object,
    *,
    default: float = 0.0,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return default

    return float(value)


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def _weapon_index(
    raw_weapon_id: object,
    *,
    weapon_count: int,
) -> int:
    if (
        isinstance(raw_weapon_id, bool)
        or not isinstance(raw_weapon_id, int)
        or raw_weapon_id < 0
    ):
        return 0

    encoded = raw_weapon_id + 1

    if encoded >= weapon_count:
        return 0

    return encoded


def _telemetry_for_transition(
    transition: dict[str, Any] | None,
    *,
    weapon_count: int,
) -> tuple[torch.Tensor, int, bool]:
    zeros = torch.zeros(
        TELEMETRY_FEATURE_COUNT,
        dtype=torch.float32,
    )

    if not isinstance(transition, dict):
        return zeros, 0, True

    observation = transition.get(
        "observation",
        {},
    )
    info = transition.get(
        "info",
        {},
    )

    telemetry = (
        observation.get("telemetry")
        if isinstance(observation, dict)
        else None
    )
    sync = (
        info.get("telemetry_sync")
        if isinstance(info, dict)
        else None
    )

    sync_mapping = (
        sync
        if isinstance(sync, dict)
        else {}
    )

    pre_spawn = (
        sync_mapping.get("pre_spawn") is True
    )

    if isinstance(telemetry, dict):
        health_for_sentinel = _finite_number(
            telemetry.get("health"),
        )

        if health_for_sentinel <= -600.0:
            pre_spawn = True

    fresh = sync_mapping.get("fresh")

    usable = (
        isinstance(telemetry, dict)
        and fresh is not False
        and not pre_spawn
    )

    # Missing or stale telemetry does not invalidate an older
    # visual/action demonstration. Its availability bit remains zero.
    if not usable:
        return zeros, 0, not pre_spawn

    health = _finite_number(
        telemetry.get("health"),
    )
    armor = _finite_number(
        telemetry.get("armor"),
    )
    ammo = _finite_number(
        telemetry.get("ammo"),
        default=-1.0,
    )
    score = _finite_number(
        telemetry.get("score"),
    )
    match_time = _finite_number(
        telemetry.get("match_time_seconds"),
    )

    alive_value = telemetry.get("alive")
    alive = (
        1.0
        if alive_value is True
        or alive_value == 1
        else 0.0
    )

    ammo_known = 1.0 if ammo >= 0.0 else 0.0

    features = torch.tensor(
        (
            _clamp(
                health,
                -200.0,
                200.0,
            )
            / 200.0,
            _clamp(
                armor,
                0.0,
                200.0,
            )
            / 200.0,
            (
                _clamp(
                    ammo,
                    0.0,
                    200.0,
                )
                / 200.0
                if ammo_known
                else 0.0
            ),
            ammo_known,
            alive,
            _clamp(
                score,
                -100.0,
                100.0,
            )
            / 100.0,
            _clamp(
                match_time,
                0.0,
                1800.0,
            )
            / 1800.0,
            1.0,
        ),
        dtype=torch.float32,
    )

    weapon = _weapon_index(
        sync_mapping.get("weapon_id"),
        weapon_count=weapon_count,
    )

    return features, weapon, True


def _read_transition_map(
    episode_path: Path,
) -> dict[int, dict[str, Any]]:
    transitions: dict[
        int,
        dict[str, Any],
    ] = {}

    for line in episode_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if not line.strip():
            continue

        record = json.loads(line)

        if (
            not isinstance(record, dict)
            or record.get("type")
            != "agent_transition"
        ):
            continue

        data = record.get("data")

        if not isinstance(data, dict):
            continue

        step_index = data.get("step_index")

        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index < 0
        ):
            raise ValueError(
                "transition step_index must be "
                "a nonnegative integer"
            )

        transitions[step_index] = data

    return transitions


class TelemetryCompositeSequenceDataset:
    """Wrap the proven composite dataset with telemetry inputs."""

    def __init__(
        self,
        episode_paths: Sequence[str | Path],
        *,
        sequence_length: int = 8,
        stride: int = 4,
        telemetry_weapon_count: int = (
            DEFAULT_TELEMETRY_WEAPON_COUNT
        ),
    ) -> None:
        if (
            isinstance(telemetry_weapon_count, bool)
            or not isinstance(
                telemetry_weapon_count,
                int,
            )
            or telemetry_weapon_count <= 1
        ):
            raise ValueError(
                "telemetry_weapon_count must be "
                "an integer greater than one"
            )

        self.base_dataset = CompositeSequenceDataset(
            episode_paths,
            sequence_length=sequence_length,
            stride=stride,
        )

        self.episode_paths = tuple(
            Path(path).resolve()
            for path in self.base_dataset.episode_paths
        )
        self.sequence_length = (
            self.base_dataset.sequence_length
        )
        self.stride = self.base_dataset.stride

        self.telemetry_weapon_count = (
            telemetry_weapon_count
        )
        self.telemetry_feature_count = (
            TELEMETRY_FEATURE_COUNT
        )
        self.telemetry_schema_version = (
            TELEMETRY_SCHEMA_VERSION
        )

        self._transition_maps = {
            path: _read_transition_map(path)
            for path in self.episode_paths
        }

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self,
        index: int,
    ) -> TelemetryCompositeSequenceSample:
        base_sample = self.base_dataset[index]

        if not is_dataclass(base_sample):
            raise TypeError(
                "base composite sample must be "
                "a dataclass instance"
            )

        source_path = Path(
            base_sample.source_episode_path
        ).resolve()

        transitions = self._transition_maps.get(
            source_path
        )

        if transitions is None:
            raise RuntimeError(
                "base sample references an unknown "
                "episode path"
            )

        start_step = int(
            base_sample.start_step_index
        )

        feature_rows: list[torch.Tensor] = []
        weapon_indices: list[int] = []
        telemetry_trainable: list[bool] = []

        for offset in range(
            self.sequence_length
        ):
            transition = transitions.get(
                start_step + offset
            )

            (
                features,
                weapon_index,
                trainable,
            ) = _telemetry_for_transition(
                transition,
                weapon_count=(
                    self.telemetry_weapon_count
                ),
            )

            feature_rows.append(features)
            weapon_indices.append(weapon_index)
            telemetry_trainable.append(trainable)

        telemetry_features = torch.stack(
            feature_rows,
            dim=0,
        )

        telemetry_weapon_indices = torch.tensor(
            weapon_indices,
            dtype=torch.long,
        )

        telemetry_trainable_mask = torch.tensor(
            telemetry_trainable,
            dtype=torch.bool,
        )

        return TelemetryCompositeSequenceSample(
            base_sample=base_sample,
            telemetry_features=(
                telemetry_features
            ),
            telemetry_weapon_indices=(
                telemetry_weapon_indices
            ),
            telemetry_trainable_mask=(
                telemetry_trainable_mask
            ),
        )


def collate_telemetry_composite_sequences(
    samples: Sequence[
        TelemetryCompositeSequenceSample
    ],
) -> TelemetryCompositeSequenceBatch:
    """Collate telemetry and the original composite batch."""

    if not samples:
        raise ValueError(
            "samples must not be empty"
        )

    for sample in samples:
        if not isinstance(
            sample,
            TelemetryCompositeSequenceSample,
        ):
            raise TypeError(
                "all samples must be telemetry "
                "composite samples"
            )

    base_batch = collate_composite_sequences(
        [
            sample.base_sample
            for sample in samples
        ]
    )

    base_values = {
        field.name: getattr(
            base_batch,
            field.name,
        )
        for field in fields(
            CompositeSequenceBatch
        )
    }

    telemetry_trainable_mask = torch.stack(
        [
            sample.telemetry_trainable_mask
            for sample in samples
        ],
        dim=0,
    )

    base_values["valid_mask"] = (
        base_batch.valid_mask
        & telemetry_trainable_mask
    )

    return TelemetryCompositeSequenceBatch(
        **base_values,
        telemetry_features=torch.stack(
            [
                sample.telemetry_features
                for sample in samples
            ],
            dim=0,
        ),
        telemetry_weapon_indices=torch.stack(
            [
                sample.telemetry_weapon_indices
                for sample in samples
            ],
            dim=0,
        ),
    )


def make_telemetry_composite_sequence_dataloader(
    dataset: TelemetryCompositeSequenceDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[
    TelemetryCompositeSequenceBatch
]:
    """Build a deterministic telemetry-aware loader."""

    if not isinstance(
        dataset,
        TelemetryCompositeSequenceDataset,
    ):
        raise TypeError(
            "dataset must be a "
            "TelemetryCompositeSequenceDataset"
        )

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            "batch_size must be positive"
        )

    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError(
            "num_workers must be nonnegative"
        )

    if not isinstance(pin_memory, bool):
        raise TypeError(
            "pin_memory must be bool"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=(
            collate_telemetry_composite_sequences
        ),
    )
