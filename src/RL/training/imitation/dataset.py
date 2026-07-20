"""Strict offline demonstration datasets for imitation learning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from RL.actions.contracts import DiscreteAction
from RL.inspection import InspectionJSONLReader


POLICY_FRAME_SHAPE = (
    4,
    3,
    90,
    160,
)


@dataclass(frozen=True)
class DemonstrationSample:
    """One validated observation and demonstrated action."""

    frames: torch.Tensor
    action_index: torch.Tensor
    duration_ticks: int
    episode_id: str
    step_index: int
    source_episode_path: Path
    source_frame_path: Path

    def __post_init__(self) -> None:
        if not isinstance(
            self.frames,
            torch.Tensor,
        ):
            raise TypeError(
                "frames must be a torch.Tensor"
            )

        if tuple(self.frames.shape) != (
            POLICY_FRAME_SHAPE
        ):
            raise ValueError(
                "frames have an unexpected shape"
            )

        if self.frames.dtype != torch.float32:
            raise TypeError(
                "frames must use torch.float32"
            )

        if not bool(
            torch.isfinite(self.frames).all().item()
        ):
            raise ValueError(
                "frames must contain only finite values"
            )

        if not isinstance(
            self.action_index,
            torch.Tensor,
        ):
            raise TypeError(
                "action_index must be a torch.Tensor"
            )

        if (
            self.action_index.ndim != 0
            or self.action_index.dtype
            != torch.long
        ):
            raise TypeError(
                "action_index must be a torch.long scalar"
            )

        if (
            isinstance(self.duration_ticks, bool)
            or not isinstance(
                self.duration_ticks,
                int,
            )
            or self.duration_ticks <= 0
        ):
            raise ValueError(
                "duration_ticks must be a positive integer"
            )

        if not self.episode_id:
            raise ValueError(
                "episode_id must not be empty"
            )

        if (
            isinstance(self.step_index, bool)
            or not isinstance(
                self.step_index,
                int,
            )
            or self.step_index < 0
        ):
            raise ValueError(
                "step_index must be a nonnegative integer"
            )


@dataclass(frozen=True)
class DemonstrationBatch:
    """A deterministic batch ready for visual-policy inference."""

    frames: torch.Tensor
    action_indices: torch.Tensor
    duration_ticks: torch.Tensor
    episode_ids: tuple[str, ...]
    step_indices: torch.Tensor
    source_episode_paths: tuple[Path, ...]
    source_frame_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.frames.ndim != 5:
            raise ValueError(
                "batch frames must have five dimensions"
            )

        if tuple(self.frames.shape[1:]) != (
            POLICY_FRAME_SHAPE
        ):
            raise ValueError(
                "batch frames have an unexpected shape"
            )

        batch_size = int(
            self.frames.shape[0]
        )

        if self.frames.dtype != torch.float32:
            raise TypeError(
                "batch frames must use torch.float32"
            )

        if (
            self.action_indices.shape
            != (batch_size,)
            or self.action_indices.dtype
            != torch.long
        ):
            raise TypeError(
                "action_indices must be a torch.long vector"
            )

        if (
            self.duration_ticks.shape
            != (batch_size,)
            or self.duration_ticks.dtype
            != torch.long
        ):
            raise TypeError(
                "duration_ticks must be a torch.long vector"
            )

        if (
            self.step_indices.shape
            != (batch_size,)
            or self.step_indices.dtype
            != torch.long
        ):
            raise TypeError(
                "step_indices must be a torch.long vector"
            )

        if len(self.episode_ids) != batch_size:
            raise ValueError(
                "episode_ids length must match batch size"
            )


@dataclass(frozen=True)
class _SampleReference:
    episode_path: Path
    frame_path: Path
    episode_id: str
    step_index: int
    action: DiscreteAction
    duration_ticks: int


def _require_nonempty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{field_name} must be a nonempty string"
        )

    return value


def _require_positive_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )

    return value


def _require_nonnegative_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a nonnegative integer"
        )

    return value


def _validate_frame_array(
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
            f"unable to safely load policy frame: {path}"
        ) from error

    if not isinstance(array, np.ndarray):
        raise TypeError(
            "policy frame must be a NumPy array"
        )

    if tuple(array.shape) != POLICY_FRAME_SHAPE:
        raise ValueError(
            "policy frame has an unexpected shape: "
            f"{tuple(array.shape)}"
        )

    if array.dtype != np.float32:
        raise TypeError(
            "policy frame must use float32"
        )

    if not bool(np.isfinite(array).all()):
        raise ValueError(
            "policy frame must contain only finite values"
        )

    return array


def _resolve_frame_path(
    episode_path: Path,
    raw_path: object,
) -> Path:
    path_text = _require_nonempty_string(
        raw_path,
        field_name="policy frame path",
    )

    relative_path = Path(path_text)

    if relative_path.is_absolute():
        raise ValueError(
            "policy frame path must be relative"
        )

    episode_directory = (
        episode_path.parent.resolve()
    )

    frame_path = (
        episode_directory
        / relative_path
    ).resolve()

    try:
        frame_path.relative_to(
            episode_directory
        )
    except ValueError as error:
        raise ValueError(
            "policy frame path escapes "
            "the episode directory"
        ) from error

    if not frame_path.is_file():
        raise FileNotFoundError(
            str(frame_path)
        )

    return frame_path


def _validate_episode_metadata(
    metadata: object,
    *,
    allow_nonexpert: bool,
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "episode metadata must be a mapping"
        )

    if (
        metadata.get(
            "frame_references_are_real"
        )
        is not True
    ):
        raise ValueError(
            "episode must contain real frame references"
        )

    raw_shape = metadata.get(
        "frame_shape"
    )

    if not isinstance(
        raw_shape,
        (list, tuple),
    ):
        raise ValueError(
            "episode frame_shape is invalid"
        )

    try:
        metadata_shape = tuple(
            int(dimension)
            for dimension in raw_shape
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "episode frame_shape is invalid"
        ) from error

    if metadata_shape != POLICY_FRAME_SHAPE:
        raise ValueError(
            "episode frame_shape does not match "
            "the visual policy"
        )

    if metadata.get("frame_encoding") != "npy":
        raise ValueError(
            "episode frame_encoding must be npy"
        )

    if allow_nonexpert:
        return

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
            "episode is not explicitly marked as "
            "an expert human demonstration"
        )


def _parse_action(
    decision: object,
) -> tuple[DiscreteAction, int]:
    if not isinstance(decision, Mapping):
        raise ValueError(
            "transition decision must be a mapping"
        )

    action_record = decision.get(
        "action"
    )

    if not isinstance(
        action_record,
        Mapping,
    ):
        raise ValueError(
            "transition action must be a mapping"
        )

    raw_value = action_record.get(
        "value"
    )

    if (
        isinstance(raw_value, bool)
        or not isinstance(raw_value, int)
    ):
        raise ValueError(
            "action value must be an integer"
        )

    try:
        action = DiscreteAction(
            raw_value
        )
    except ValueError as error:
        raise ValueError(
            "action value is not a known DiscreteAction"
        ) from error

    action_name = _require_nonempty_string(
        action_record.get("name"),
        field_name="action name",
    )

    if action_name != action.name:
        raise ValueError(
            "action name and value do not match"
        )

    duration_ticks = (
        _require_positive_integer(
            action_record.get(
                "duration_ticks"
            ),
            field_name="duration_ticks",
        )
    )

    return action, duration_ticks


def _parse_reference(
    episode_path: Path,
    transition: Mapping[str, object],
) -> _SampleReference:
    episode_id = _require_nonempty_string(
        transition.get("episode_id"),
        field_name="episode_id",
    )

    step_index = (
        _require_nonnegative_integer(
            transition.get("step_index"),
            field_name="step_index",
        )
    )

    observation = transition.get(
        "observation"
    )

    if not isinstance(
        observation,
        Mapping,
    ):
        raise ValueError(
            "transition observation must be a mapping"
        )

    policy_frame = observation.get(
        "policy_frame"
    )

    if not isinstance(
        policy_frame,
        Mapping,
    ):
        raise ValueError(
            "transition must contain a policy frame"
        )

    raw_shape = policy_frame.get(
        "shape"
    )

    if not isinstance(
        raw_shape,
        (list, tuple),
    ):
        raise ValueError(
            "policy frame metadata shape is invalid"
        )

    try:
        record_shape = tuple(
            int(dimension)
            for dimension in raw_shape
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "policy frame metadata shape is invalid"
        ) from error

    if record_shape != POLICY_FRAME_SHAPE:
        raise ValueError(
            "policy frame metadata shape does not match "
            "the visual policy"
        )

    if policy_frame.get("dtype") != "float32":
        raise ValueError(
            "policy frame metadata dtype must be float32"
        )

    if policy_frame.get("encoding") != "npy":
        raise ValueError(
            "policy frame metadata encoding must be npy"
        )

    frame_path = _resolve_frame_path(
        episode_path,
        policy_frame.get("path"),
    )

    _validate_frame_array(frame_path)

    action, duration_ticks = _parse_action(
        transition.get("decision")
    )

    return _SampleReference(
        episode_path=episode_path,
        frame_path=frame_path,
        episode_id=episode_id,
        step_index=step_index,
        action=action,
        duration_ticks=duration_ticks,
    )


class DemonstrationDataset(
    Dataset[DemonstrationSample]
):
    """Strict, deterministic expert-demonstration dataset."""

    def __init__(
        self,
        episode_paths: (
            Sequence[str | Path]
            | str
            | Path
        ),
        *,
        allow_nonexpert: bool = False,
    ) -> None:
        if not isinstance(
            allow_nonexpert,
            bool,
        ):
            raise TypeError(
                "allow_nonexpert must be bool"
            )

        if isinstance(
            episode_paths,
            (str, Path),
        ):
            raw_paths = [
                episode_paths
            ]
        else:
            raw_paths = list(
                episode_paths
            )

        if not raw_paths:
            raise ValueError(
                "at least one episode path is required"
            )

        resolved_paths: list[Path] = []

        for raw_path in raw_paths:
            episode_path = Path(
                raw_path
            ).resolve()

            if not episode_path.is_file():
                raise FileNotFoundError(
                    str(episode_path)
                )

            resolved_paths.append(
                episode_path
            )

        if len(set(resolved_paths)) != len(
            resolved_paths
        ):
            raise ValueError(
                "duplicate episode paths are not allowed"
            )

        resolved_paths.sort(
            key=lambda path: path.as_posix()
        )

        references: list[
            _SampleReference
        ] = []

        for episode_path in resolved_paths:
            records = InspectionJSONLReader(
                episode_path
            ).read_episode()

            started = records[0].data

            _validate_episode_metadata(
                started.get("metadata"),
                allow_nonexpert=(
                    allow_nonexpert
                ),
            )

            transitions = [
                record.data
                for record in records
                if record.type
                == "agent_transition"
            ]

            if not transitions:
                raise ValueError(
                    "demonstration episode contains "
                    "no transitions"
                )

            for transition in transitions:
                references.append(
                    _parse_reference(
                        episode_path,
                        transition,
                    )
                )

        references.sort(
            key=lambda reference: (
                reference.episode_path.as_posix(),
                reference.step_index,
            )
        )

        self._episode_paths = tuple(
            resolved_paths
        )
        self._references = tuple(
            references
        )

        counts = Counter(
            reference.action.name
            for reference in references
        )

        self._action_counts = (
            MappingProxyType(
                {
                    action.name: int(
                        counts.get(
                            action.name,
                            0,
                        )
                    )
                    for action in DiscreteAction
                }
            )
        )

    @property
    def episode_paths(
        self,
    ) -> tuple[Path, ...]:
        """Return canonical source paths in dataset order."""

        return self._episode_paths

    @property
    def action_counts(
        self,
    ) -> Mapping[str, int]:
        """Return immutable action-frequency counts."""

        return self._action_counts

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(
        self,
        index: int,
    ) -> DemonstrationSample:
        reference = self._references[
            index
        ]

        array = _validate_frame_array(
            reference.frame_path
        )

        copied_array = np.array(
            array,
            dtype=np.float32,
            order="C",
            copy=True,
        )

        frames = torch.from_numpy(
            copied_array
        )

        action_index = torch.tensor(
            int(reference.action),
            dtype=torch.long,
        )

        return DemonstrationSample(
            frames=frames,
            action_index=action_index,
            duration_ticks=(
                reference.duration_ticks
            ),
            episode_id=(
                reference.episode_id
            ),
            step_index=(
                reference.step_index
            ),
            source_episode_path=(
                reference.episode_path
            ),
            source_frame_path=(
                reference.frame_path
            ),
        )


def collate_demonstration_samples(
    samples: Sequence[
        DemonstrationSample
    ],
) -> DemonstrationBatch:
    """Collate ordered samples into one immutable batch."""

    if not samples:
        raise ValueError(
            "cannot collate an empty sample sequence"
        )

    for sample in samples:
        if not isinstance(
            sample,
            DemonstrationSample,
        ):
            raise TypeError(
                "all samples must be DemonstrationSample"
            )

    return DemonstrationBatch(
        frames=torch.stack(
            [
                sample.frames
                for sample in samples
            ],
            dim=0,
        ),
        action_indices=torch.stack(
            [
                sample.action_index
                for sample in samples
            ],
            dim=0,
        ),
        duration_ticks=torch.tensor(
            [
                sample.duration_ticks
                for sample in samples
            ],
            dtype=torch.long,
        ),
        episode_ids=tuple(
            sample.episode_id
            for sample in samples
        ),
        step_indices=torch.tensor(
            [
                sample.step_index
                for sample in samples
            ],
            dtype=torch.long,
        ),
        source_episode_paths=tuple(
            sample.source_episode_path
            for sample in samples
        ),
        source_frame_paths=tuple(
            sample.source_frame_path
            for sample in samples
        ),
    )


def make_demonstration_dataloader(
    dataset: DemonstrationDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[DemonstrationBatch]:
    """Return a deterministic, non-shuffled batch loader."""

    if not isinstance(
        dataset,
        DemonstrationDataset,
    ):
        raise TypeError(
            "dataset must be a DemonstrationDataset"
        )

    validated_batch_size = (
        _require_positive_integer(
            batch_size,
            field_name="batch_size",
        )
    )

    if (
        isinstance(num_workers, bool)
        or not isinstance(
            num_workers,
            int,
        )
        or num_workers < 0
    ):
        raise ValueError(
            "num_workers must be a nonnegative integer"
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
        collate_fn=(
            collate_demonstration_samples
        ),
    )
