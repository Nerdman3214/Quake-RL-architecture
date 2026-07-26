"""Strict recurrent datasets for authoritative composite controls."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from RL.actions.contracts import DiscreteAction
from RL.inspection import InspectionJSONLReader
from RL.recording.match_recorder import (
    COMPACT_POLICY_FRAME_SHAPE,
    COMPACT_POLICY_FRAME_STORAGE,
    LEGACY_POLICY_FRAME_STORAGE,
)
from RL.training.imitation.dataset import (
    POLICY_FRAME_SHAPE,
    _parse_action,
    _require_nonempty_string,
    _require_nonnegative_integer,
    _require_positive_integer,
    _resolve_frame_path,
    _validate_episode_metadata,
    _validate_frame_array,
)


AXIS_CLASS_COUNT = 3
WEAPON_CLASS_COUNT = 3
PREVIOUS_ACTION_FEATURE_COUNT = 8

GAME_MODE_TO_INDEX = MappingProxyType(
    {
        "dm": 0,
        "tdm": 1,
        "ctf": 2,
        "dom": 3,
        "kh": 4,
    }
)


def _axis_class(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in (-1, 0, 1)
    ):
        raise ValueError(
            "axis value must be -1, 0, or 1"
        )

    return value + 1


def _finite_float(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return float(value)


def _required_bool(
    value: object,
    *,
    field_name: str,
) -> bool:
    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be bool"
        )

    return value


def _weapon_class(value: int) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            "weapon_delta must be an integer"
        )

    if value < 0:
        return 0

    if value > 0:
        return 2

    return 1


@dataclass(frozen=True)
class _CompositeStepReference:
    episode_path: Path
    frame_path: Path
    frame_storage_mode: str
    episode_id: str
    step_index: int
    forward_axis: int
    strafe_axis: int
    turn_delta_x: float
    look_delta_y: float
    fire: bool
    jump: bool
    weapon_delta: int
    duration_ticks: int
    legacy_action: DiscreteAction
    timestamp_ns: int


@dataclass(frozen=True)
class _SequenceReference:
    episode_path: Path
    episode_id: str
    mode_index: int
    all_steps: tuple[_CompositeStepReference, ...]
    start_offset: int
    sequence_length: int


def _validate_sequence_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor"
        )

    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} has an unexpected shape"
        )

    if tensor.dtype != dtype:
        raise TypeError(
            f"{name} has an unexpected dtype"
        )

    if tensor.is_floating_point() and not bool(
        torch.isfinite(tensor).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values"
        )


@dataclass(frozen=True)
class CompositeSequenceSample:
    """One fixed-length recurrent composite-control sequence."""

    frames: torch.Tensor
    previous_action_features: torch.Tensor
    forward_classes: torch.Tensor
    strafe_classes: torch.Tensor
    mouse_deltas: torch.Tensor
    fire_targets: torch.Tensor
    jump_targets: torch.Tensor
    weapon_classes: torch.Tensor
    duration_ticks: torch.Tensor
    legacy_action_indices: torch.Tensor
    valid_mask: torch.Tensor
    timestamps_ns: torch.Tensor
    mode_index: torch.Tensor
    episode_id: str
    start_step_index: int
    source_episode_path: Path
    source_frame_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.frames.ndim != 5:
            raise ValueError(
                "frames must have five dimensions"
            )

        sequence_length = int(
            self.frames.shape[0]
        )

        _validate_sequence_tensor(
            self.frames,
            name="frames",
            shape=(
                sequence_length,
                *POLICY_FRAME_SHAPE,
            ),
            dtype=torch.float32,
        )
        _validate_sequence_tensor(
            self.previous_action_features,
            name="previous_action_features",
            shape=(
                sequence_length,
                PREVIOUS_ACTION_FEATURE_COUNT,
            ),
            dtype=torch.float32,
        )
        _validate_sequence_tensor(
            self.forward_classes,
            name="forward_classes",
            shape=(sequence_length,),
            dtype=torch.long,
        )
        _validate_sequence_tensor(
            self.strafe_classes,
            name="strafe_classes",
            shape=(sequence_length,),
            dtype=torch.long,
        )
        _validate_sequence_tensor(
            self.mouse_deltas,
            name="mouse_deltas",
            shape=(sequence_length, 2),
            dtype=torch.float32,
        )
        _validate_sequence_tensor(
            self.fire_targets,
            name="fire_targets",
            shape=(sequence_length,),
            dtype=torch.float32,
        )
        _validate_sequence_tensor(
            self.jump_targets,
            name="jump_targets",
            shape=(sequence_length,),
            dtype=torch.float32,
        )
        _validate_sequence_tensor(
            self.weapon_classes,
            name="weapon_classes",
            shape=(sequence_length,),
            dtype=torch.long,
        )
        _validate_sequence_tensor(
            self.duration_ticks,
            name="duration_ticks",
            shape=(sequence_length,),
            dtype=torch.long,
        )
        _validate_sequence_tensor(
            self.legacy_action_indices,
            name="legacy_action_indices",
            shape=(sequence_length,),
            dtype=torch.long,
        )
        _validate_sequence_tensor(
            self.valid_mask,
            name="valid_mask",
            shape=(sequence_length,),
            dtype=torch.bool,
        )
        _validate_sequence_tensor(
            self.timestamps_ns,
            name="timestamps_ns",
            shape=(sequence_length,),
            dtype=torch.long,
        )
        _validate_sequence_tensor(
            self.mode_index,
            name="mode_index",
            shape=(),
            dtype=torch.long,
        )

        if not bool(self.valid_mask.all().item()):
            raise ValueError(
                "fixed sequences must contain only valid steps"
            )

        if not self.episode_id:
            raise ValueError(
                "episode_id must not be empty"
            )

        if (
            isinstance(self.start_step_index, bool)
            or not isinstance(
                self.start_step_index,
                int,
            )
            or self.start_step_index < 0
        ):
            raise ValueError(
                "start_step_index must be nonnegative"
            )

        if len(self.source_frame_paths) != (
            sequence_length
        ):
            raise ValueError(
                "source_frame_paths length must match "
                "the sequence length"
            )


@dataclass(frozen=True)
class CompositeSequenceBatch:
    """A deterministic batch of recurrent composite sequences."""

    frames: torch.Tensor
    previous_action_features: torch.Tensor
    forward_classes: torch.Tensor
    strafe_classes: torch.Tensor
    mouse_deltas: torch.Tensor
    fire_targets: torch.Tensor
    jump_targets: torch.Tensor
    weapon_classes: torch.Tensor
    duration_ticks: torch.Tensor
    legacy_action_indices: torch.Tensor
    valid_mask: torch.Tensor
    timestamps_ns: torch.Tensor
    mode_indices: torch.Tensor
    episode_ids: tuple[str, ...]
    start_step_indices: torch.Tensor
    source_episode_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.frames.ndim != 6:
            raise ValueError(
                "batch frames must have six dimensions"
            )

        batch_size = int(self.frames.shape[0])
        sequence_length = int(self.frames.shape[1])

        expected_shapes = {
            "previous_action_features": (
                batch_size,
                sequence_length,
                PREVIOUS_ACTION_FEATURE_COUNT,
            ),
            "forward_classes": (
                batch_size,
                sequence_length,
            ),
            "strafe_classes": (
                batch_size,
                sequence_length,
            ),
            "mouse_deltas": (
                batch_size,
                sequence_length,
                2,
            ),
            "fire_targets": (
                batch_size,
                sequence_length,
            ),
            "jump_targets": (
                batch_size,
                sequence_length,
            ),
            "weapon_classes": (
                batch_size,
                sequence_length,
            ),
            "duration_ticks": (
                batch_size,
                sequence_length,
            ),
            "legacy_action_indices": (
                batch_size,
                sequence_length,
            ),
            "valid_mask": (
                batch_size,
                sequence_length,
            ),
            "timestamps_ns": (
                batch_size,
                sequence_length,
            ),
            "mode_indices": (batch_size,),
            "start_step_indices": (batch_size,),
        }

        for name, expected_shape in expected_shapes.items():
            tensor = getattr(self, name)

            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} has an unexpected shape"
                )

        if tuple(self.frames.shape[2:]) != (
            POLICY_FRAME_SHAPE
        ):
            raise ValueError(
                "batch frames have an unexpected policy shape"
            )

        if self.frames.dtype != torch.float32:
            raise TypeError(
                "batch frames must use torch.float32"
            )

        if self.previous_action_features.dtype != (
            torch.float32
        ):
            raise TypeError(
                "previous_action_features must use float32"
            )

        if self.mouse_deltas.dtype != torch.float32:
            raise TypeError(
                "mouse_deltas must use float32"
            )

        if self.fire_targets.dtype != torch.float32:
            raise TypeError(
                "fire_targets must use float32"
            )

        if self.jump_targets.dtype != torch.float32:
            raise TypeError(
                "jump_targets must use float32"
            )

        for name in (
            "forward_classes",
            "strafe_classes",
            "weapon_classes",
            "duration_ticks",
            "legacy_action_indices",
            "timestamps_ns",
            "mode_indices",
            "start_step_indices",
        ):
            if getattr(self, name).dtype != torch.long:
                raise TypeError(
                    f"{name} must use torch.long"
                )

        if self.valid_mask.dtype != torch.bool:
            raise TypeError(
                "valid_mask must use torch.bool"
            )

        if len(self.episode_ids) != batch_size:
            raise ValueError(
                "episode_ids length must match batch size"
            )

        if len(self.source_episode_paths) != batch_size:
            raise ValueError(
                "source_episode_paths length must match "
                "batch size"
            )



def _validate_compact_frame_array(
    path: Path,
) -> np.ndarray:
    try:
        array = np.load(
            path,
            allow_pickle=False,
            mmap_mode="r",
        )
    except Exception as error:
        raise ValueError(
            "unable to safely load compact "
            f"policy frame: {path}"
        ) from error

    if not isinstance(array, np.ndarray):
        raise TypeError(
            "compact policy frame must be "
            "a NumPy array"
        )

    if tuple(array.shape) != (
        COMPACT_POLICY_FRAME_SHAPE
    ):
        raise ValueError(
            "compact policy frame has an "
            "unexpected shape: "
            f"{tuple(array.shape)}"
        )

    if array.dtype != np.uint8:
        raise TypeError(
            "compact policy frame must use uint8"
        )

    return array


def _metadata_frame_shape(
    metadata: Mapping[str, object],
) -> tuple[int, ...]:
    raw_shape = metadata.get("frame_shape")

    if not isinstance(
        raw_shape,
        (list, tuple),
    ):
        raise ValueError(
            "episode frame_shape is invalid"
        )

    try:
        return tuple(
            int(dimension)
            for dimension in raw_shape
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "episode frame_shape is invalid"
        ) from error


def _validate_composite_episode_metadata(
    metadata: object,
    *,
    allow_nonexpert: bool,
) -> str:
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "episode metadata must be a mapping"
        )

    metadata_shape = _metadata_frame_shape(
        metadata
    )

    if metadata_shape == POLICY_FRAME_SHAPE:
        _validate_episode_metadata(
            metadata,
            allow_nonexpert=allow_nonexpert,
        )

        storage_mode = metadata.get(
            "policy_frame_storage_mode",
            LEGACY_POLICY_FRAME_STORAGE,
        )

        if (
            storage_mode
            != LEGACY_POLICY_FRAME_STORAGE
        ):
            raise ValueError(
                "legacy frame metadata has an "
                "inconsistent storage mode"
            )

        return LEGACY_POLICY_FRAME_STORAGE

    if metadata_shape != (
        COMPACT_POLICY_FRAME_SHAPE
    ):
        raise ValueError(
            "episode frame_shape does not match "
            "a supported recurrent storage format"
        )

    if (
        metadata.get(
            "frame_references_are_real"
        )
        is not True
    ):
        raise ValueError(
            "episode must contain real frame "
            "references"
        )

    if metadata.get("frame_encoding") != "npy":
        raise ValueError(
            "episode frame_encoding must be npy"
        )

    if metadata.get("frame_dtype") != "uint8":
        raise ValueError(
            "compact episode frame_dtype must "
            "be uint8"
        )

    if metadata.get(
        "policy_frame_storage_mode"
    ) != COMPACT_POLICY_FRAME_STORAGE:
        raise ValueError(
            "compact episode storage mode is "
            "invalid"
        )

    reconstructed_shape = metadata.get(
        "reconstructed_policy_frame_shape"
    )

    if not isinstance(
        reconstructed_shape,
        (list, tuple),
    ):
        raise ValueError(
            "compact reconstructed frame shape "
            "is invalid"
        )

    try:
        reconstructed_shape_tuple = tuple(
            int(dimension)
            for dimension in reconstructed_shape
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "compact reconstructed frame shape "
            "is invalid"
        ) from error

    if (
        reconstructed_shape_tuple
        != POLICY_FRAME_SHAPE
    ):
        raise ValueError(
            "compact reconstructed frame shape "
            "does not match the visual policy"
        )

    if not allow_nonexpert:
        if (
            metadata.get(
                "expert_demonstration"
            )
            is not True
            or metadata.get(
                "demonstration_source"
            )
            != "human"
        ):
            raise ValueError(
                "episode is not explicitly "
                "marked as an expert human "
                "demonstration"
            )

    return COMPACT_POLICY_FRAME_STORAGE


def _parse_composite_reference(
    episode_path: Path,
    transition: Mapping[str, object],
) -> tuple[
    Path,
    str,
    int,
    str,
    DiscreteAction,
    int,
]:
    episode_id = _require_nonempty_string(
        transition.get("episode_id"),
        field_name="episode_id",
    )

    step_index = _require_nonnegative_integer(
        transition.get("step_index"),
        field_name="step_index",
    )

    observation = transition.get(
        "observation"
    )

    if not isinstance(observation, Mapping):
        raise ValueError(
            "transition observation must be "
            "a mapping"
        )

    policy_frame = observation.get(
        "policy_frame"
    )

    if not isinstance(policy_frame, Mapping):
        raise ValueError(
            "transition must contain a "
            "policy frame"
        )

    raw_shape = policy_frame.get("shape")

    if not isinstance(
        raw_shape,
        (list, tuple),
    ):
        raise ValueError(
            "policy frame metadata shape "
            "is invalid"
        )

    try:
        frame_shape = tuple(
            int(dimension)
            for dimension in raw_shape
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "policy frame metadata shape "
            "is invalid"
        ) from error

    if policy_frame.get("encoding") != "npy":
        raise ValueError(
            "policy frame encoding must be npy"
        )

    frame_path = _resolve_frame_path(
        episode_path,
        policy_frame.get("path"),
    )

    if (
        frame_shape == POLICY_FRAME_SHAPE
        and policy_frame.get("dtype")
        == "float32"
    ):
        frame_storage_mode = (
            LEGACY_POLICY_FRAME_STORAGE
        )
        _validate_frame_array(frame_path)

    elif (
        frame_shape
        == COMPACT_POLICY_FRAME_SHAPE
        and policy_frame.get("dtype")
        == "uint8"
    ):
        frame_storage_mode = (
            COMPACT_POLICY_FRAME_STORAGE
        )
        _validate_compact_frame_array(
            frame_path
        )

    else:
        raise ValueError(
            "policy frame metadata does not "
            "match a supported storage format"
        )

    action, duration_ticks = _parse_action(
        transition.get("decision")
    )

    return (
        frame_path,
        episode_id,
        step_index,
        frame_storage_mode,
        action,
        duration_ticks,
    )



def _parse_composite_step(
    episode_path: Path,
    transition: Mapping[str, object],
    *,
    label_authority: str,
) -> _CompositeStepReference:
    (
        frame_path,
        episode_id,
        step_index,
        frame_storage_mode,
        legacy_action,
        compatibility_duration,
    ) = _parse_composite_reference(
        episode_path,
        transition,
    )

    info = transition.get("info")

    if not isinstance(info, Mapping):
        raise ValueError(
            "transition info must be a mapping"
        )

    if info.get("label_authority") != (
        label_authority
    ):
        raise ValueError(
            "transition label authority does "
            "not match episode metadata"
        )

    if (
        info.get("derived_label_is_primary")
        is not False
    ):
        raise ValueError(
            "derived discrete label must not "
            "be primary"
        )

    derived_action = _require_nonempty_string(
        info.get("derived_discrete_action"),
        field_name="derived_discrete_action",
    )

    if derived_action != legacy_action.name:
        raise ValueError(
            "derived discrete action does not "
            "match the compatibility action"
        )

    human_input = info.get("human_input")

    if not isinstance(human_input, Mapping):
        raise ValueError(
            "human_input must be a mapping"
        )

    forward_axis = human_input.get(
        "forward_axis"
    )
    strafe_axis = human_input.get(
        "strafe_axis"
    )

    _axis_class(forward_axis)
    _axis_class(strafe_axis)

    turn_delta_x = _finite_float(
        human_input.get("turn_delta_x"),
        field_name="turn_delta_x",
    )

    look_delta_y = _finite_float(
        human_input.get("look_delta_y"),
        field_name="look_delta_y",
    )

    fire = _required_bool(
        human_input.get("fire"),
        field_name="fire",
    )

    jump = _required_bool(
        human_input.get("jump"),
        field_name="jump",
    )

    weapon_delta = human_input.get(
        "weapon_delta"
    )
    _weapon_class(weapon_delta)

    duration_ticks = _require_positive_integer(
        human_input.get("duration_ticks"),
        field_name="human duration_ticks",
    )

    if duration_ticks != compatibility_duration:
        raise ValueError(
            "human and compatibility durations "
            "do not match"
        )

    timestamp_ns = _require_nonnegative_integer(
        info.get("timestamp_ns"),
        field_name="timestamp_ns",
    )

    return _CompositeStepReference(
        episode_path=episode_path,
        frame_path=frame_path,
        frame_storage_mode=(
            frame_storage_mode
        ),
        episode_id=episode_id,
        step_index=step_index,
        forward_axis=forward_axis,
        strafe_axis=strafe_axis,
        turn_delta_x=turn_delta_x,
        look_delta_y=look_delta_y,
        fire=fire,
        jump=jump,
        weapon_delta=weapon_delta,
        duration_ticks=duration_ticks,
        legacy_action=legacy_action,
        timestamp_ns=timestamp_ns,
    )


def _previous_action_features(
    step: _CompositeStepReference | None,
) -> tuple[float, ...]:
    if step is None:
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    weapon_direction = float(
        -1
        if step.weapon_delta < 0
        else 1
        if step.weapon_delta > 0
        else 0
    )

    return (
        float(step.forward_axis),
        float(step.strafe_axis),
        step.turn_delta_x,
        step.look_delta_y,
        float(step.fire),
        float(step.jump),
        weapon_direction,
        float(step.duration_ticks),
    )


class CompositeSequenceDataset(
    Dataset[CompositeSequenceSample]
):
    """Deterministic fixed-window composite demonstration data."""

    def __init__(
        self,
        episode_paths: (
            Sequence[str | Path]
            | str
            | Path
        ),
        *,
        sequence_length: int = 8,
        stride: int = 4,
        allow_nonexpert: bool = False,
    ) -> None:
        self.sequence_length = _require_positive_integer(
            sequence_length,
            field_name="sequence_length",
        )
        self.stride = _require_positive_integer(
            stride,
            field_name="stride",
        )

        if not isinstance(allow_nonexpert, bool):
            raise TypeError(
                "allow_nonexpert must be bool"
            )

        if isinstance(episode_paths, (str, Path)):
            raw_paths = [episode_paths]
        else:
            raw_paths = list(episode_paths)

        if not raw_paths:
            raise ValueError(
                "at least one episode path is required"
            )

        resolved_paths = sorted(
            (
                Path(path).resolve()
                for path in raw_paths
            ),
            key=lambda path: path.as_posix(),
        )

        if len(set(resolved_paths)) != len(
            resolved_paths
        ):
            raise ValueError(
                "duplicate episode paths are not allowed"
            )

        sequences: list[_SequenceReference] = []

        for episode_path in resolved_paths:
            if not episode_path.is_file():
                raise FileNotFoundError(
                    str(episode_path)
                )

            records = InspectionJSONLReader(
                episode_path
            ).read_episode()

            started = records[0].data
            metadata = started.get("metadata")

            episode_storage_mode = (
                _validate_composite_episode_metadata(
                    metadata,
                    allow_nonexpert=(
                        allow_nonexpert
                    ),
                )
            )

            if not isinstance(metadata, Mapping):
                raise ValueError(
                    "episode metadata must be a mapping"
                )

            if metadata.get(
                "composite_action_labels_available"
            ) is not True:
                raise ValueError(
                    "episode does not contain authoritative "
                    "composite action labels"
                )

            label_authority = (
                _require_nonempty_string(
                    metadata.get("label_authority"),
                    field_name="label_authority",
                )
            )

            mode_name = _require_nonempty_string(
                metadata.get("game_mode"),
                field_name="game_mode",
            )

            if mode_name not in GAME_MODE_TO_INDEX:
                raise ValueError(
                    f"unsupported game mode: {mode_name}"
                )

            transitions = [
                record.data
                for record in records
                if record.type == "agent_transition"
            ]

            if not transitions:
                raise ValueError(
                    "demonstration episode contains "
                    "no transitions"
                )

            steps = tuple(
                _parse_composite_step(
                    episode_path,
                    transition,
                    label_authority=label_authority,
                )
                for transition in transitions
            )

            step_storage_modes = {
                step.frame_storage_mode
                for step in steps
            }

            if step_storage_modes != {
                episode_storage_mode
            }:
                raise ValueError(
                    "episode metadata and transition "
                    "frame storage modes do not match"
                )

            for expected_index, step in enumerate(
                steps
            ):
                if step.step_index != expected_index:
                    raise ValueError(
                        "composite sequence steps must be "
                        "contiguous and start at zero"
                    )

                if (
                    expected_index > 0
                    and step.timestamp_ns
                    < steps[
                        expected_index - 1
                    ].timestamp_ns
                ):
                    raise ValueError(
                        "composite timestamps must be monotonic"
                    )

            maximum_start = (
                len(steps) - self.sequence_length
            )

            for start_offset in range(
                0,
                maximum_start + 1,
                self.stride,
            ):
                sequences.append(
                    _SequenceReference(
                        episode_path=episode_path,
                        episode_id=steps[0].episode_id,
                        mode_index=GAME_MODE_TO_INDEX[
                            mode_name
                        ],
                        all_steps=steps,
                        start_offset=start_offset,
                        sequence_length=(
                            self.sequence_length
                        ),
                    )
                )

        if not sequences:
            raise ValueError(
                "no complete composite sequences "
                "could be constructed"
            )

        self._episode_paths = tuple(resolved_paths)
        self._sequences = tuple(sequences)

    @property
    def episode_paths(self) -> tuple[Path, ...]:
        return self._episode_paths

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(
        self,
        index: int,
    ) -> CompositeSequenceSample:
        reference = self._sequences[index]
        start = reference.start_offset
        stop = start + reference.sequence_length

        steps = reference.all_steps[
            start:stop
        ]

        frame_tensors: list[torch.Tensor] = []

        storage_mode = (
            steps[0].frame_storage_mode
        )

        if (
            storage_mode
            == LEGACY_POLICY_FRAME_STORAGE
        ):
            for step in steps:
                array = _validate_frame_array(
                    step.frame_path
                )

                copied = np.array(
                    array,
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )

                frame_tensors.append(
                    torch.from_numpy(copied)
                )

        elif (
            storage_mode
            == COMPACT_POLICY_FRAME_STORAGE
        ):
            compact_frames: dict[
                int,
                np.ndarray,
            ] = {}

            history_start = max(
                0,
                start - 3,
            )

            for absolute_index in range(
                history_start,
                stop,
            ):
                array = (
                    _validate_compact_frame_array(
                        reference.all_steps[
                            absolute_index
                        ].frame_path
                    )
                )

                compact_frames[
                    absolute_index
                ] = (
                    np.array(
                        array,
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )
                    / 255.0
                )

            for absolute_index in range(
                start,
                stop,
            ):
                history_indices = (
                    max(0, absolute_index - 3),
                    max(0, absolute_index - 2),
                    max(0, absolute_index - 1),
                    absolute_index,
                )

                stack = np.stack(
                    tuple(
                        compact_frames[
                            history_index
                        ]
                        for history_index
                        in history_indices
                    ),
                    axis=0,
                ).astype(
                    np.float32,
                    copy=False,
                )

                if tuple(stack.shape) != (
                    POLICY_FRAME_SHAPE
                ):
                    raise RuntimeError(
                        "reconstructed compact "
                        "policy stack has an "
                        "unexpected shape"
                    )

                frame_tensors.append(
                    torch.from_numpy(
                        np.ascontiguousarray(
                            stack
                        )
                    )
                )

        else:
            raise RuntimeError(
                "unsupported policy-frame "
                "storage mode"
            )

        previous_features = []

        for offset in range(
            start,
            stop,
        ):
            previous_step = (
                reference.all_steps[offset - 1]
                if offset > 0
                else None
            )
            previous_features.append(
                _previous_action_features(
                    previous_step
                )
            )

        return CompositeSequenceSample(
            frames=torch.stack(
                frame_tensors,
                dim=0,
            ),
            previous_action_features=torch.tensor(
                previous_features,
                dtype=torch.float32,
            ),
            forward_classes=torch.tensor(
                [
                    _axis_class(
                        step.forward_axis
                    )
                    for step in steps
                ],
                dtype=torch.long,
            ),
            strafe_classes=torch.tensor(
                [
                    _axis_class(
                        step.strafe_axis
                    )
                    for step in steps
                ],
                dtype=torch.long,
            ),
            mouse_deltas=torch.tensor(
                [
                    (
                        step.turn_delta_x,
                        step.look_delta_y,
                    )
                    for step in steps
                ],
                dtype=torch.float32,
            ),
            fire_targets=torch.tensor(
                [
                    float(step.fire)
                    for step in steps
                ],
                dtype=torch.float32,
            ),
            jump_targets=torch.tensor(
                [
                    float(step.jump)
                    for step in steps
                ],
                dtype=torch.float32,
            ),
            weapon_classes=torch.tensor(
                [
                    _weapon_class(
                        step.weapon_delta
                    )
                    for step in steps
                ],
                dtype=torch.long,
            ),
            duration_ticks=torch.tensor(
                [
                    step.duration_ticks
                    for step in steps
                ],
                dtype=torch.long,
            ),
            legacy_action_indices=torch.tensor(
                [
                    int(step.legacy_action)
                    for step in steps
                ],
                dtype=torch.long,
            ),
            valid_mask=torch.ones(
                reference.sequence_length,
                dtype=torch.bool,
            ),
            timestamps_ns=torch.tensor(
                [
                    step.timestamp_ns
                    for step in steps
                ],
                dtype=torch.long,
            ),
            mode_index=torch.tensor(
                reference.mode_index,
                dtype=torch.long,
            ),
            episode_id=reference.episode_id,
            start_step_index=steps[0].step_index,
            source_episode_path=(
                reference.episode_path
            ),
            source_frame_paths=tuple(
                step.frame_path
                for step in steps
            ),
        )


def collate_composite_sequences(
    samples: Sequence[
        CompositeSequenceSample
    ],
) -> CompositeSequenceBatch:
    """Collate ordered recurrent samples."""

    if not samples:
        raise ValueError(
            "cannot collate an empty sequence"
        )

    for sample in samples:
        if not isinstance(
            sample,
            CompositeSequenceSample,
        ):
            raise TypeError(
                "all items must be "
                "CompositeSequenceSample"
            )

    return CompositeSequenceBatch(
        frames=torch.stack(
            [sample.frames for sample in samples],
            dim=0,
        ),
        previous_action_features=torch.stack(
            [
                sample.previous_action_features
                for sample in samples
            ],
            dim=0,
        ),
        forward_classes=torch.stack(
            [
                sample.forward_classes
                for sample in samples
            ],
            dim=0,
        ),
        strafe_classes=torch.stack(
            [
                sample.strafe_classes
                for sample in samples
            ],
            dim=0,
        ),
        mouse_deltas=torch.stack(
            [
                sample.mouse_deltas
                for sample in samples
            ],
            dim=0,
        ),
        fire_targets=torch.stack(
            [
                sample.fire_targets
                for sample in samples
            ],
            dim=0,
        ),
        jump_targets=torch.stack(
            [
                sample.jump_targets
                for sample in samples
            ],
            dim=0,
        ),
        weapon_classes=torch.stack(
            [
                sample.weapon_classes
                for sample in samples
            ],
            dim=0,
        ),
        duration_ticks=torch.stack(
            [
                sample.duration_ticks
                for sample in samples
            ],
            dim=0,
        ),
        legacy_action_indices=torch.stack(
            [
                sample.legacy_action_indices
                for sample in samples
            ],
            dim=0,
        ),
        valid_mask=torch.stack(
            [
                sample.valid_mask
                for sample in samples
            ],
            dim=0,
        ),
        timestamps_ns=torch.stack(
            [
                sample.timestamps_ns
                for sample in samples
            ],
            dim=0,
        ),
        mode_indices=torch.stack(
            [
                sample.mode_index
                for sample in samples
            ],
            dim=0,
        ),
        episode_ids=tuple(
            sample.episode_id
            for sample in samples
        ),
        start_step_indices=torch.tensor(
            [
                sample.start_step_index
                for sample in samples
            ],
            dtype=torch.long,
        ),
        source_episode_paths=tuple(
            sample.source_episode_path
            for sample in samples
        ),
    )


def make_composite_sequence_dataloader(
    dataset: CompositeSequenceDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[CompositeSequenceBatch]:
    """Build a deterministic recurrent loader."""

    if not isinstance(
        dataset,
        CompositeSequenceDataset,
    ):
        raise TypeError(
            "dataset must be a "
            "CompositeSequenceDataset"
        )

    validated_batch_size = (
        _require_positive_integer(
            batch_size,
            field_name="batch_size",
        )
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
        batch_size=validated_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_composite_sequences,
    )
