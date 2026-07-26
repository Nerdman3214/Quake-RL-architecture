"""Automatic per-match human demonstration recording."""

from __future__ import annotations

import re
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from RL.actions.composite import (
    CompositeActionCommand,
)
from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.engine.client.x11_window import (
    preprocess_rgb_frame,
)
from RL.inspection.contracts import (
    AgentTransition,
    FrameSnapshot,
    ObservationSnapshot,
    PolicyDecision,
)
from RL.inspection.jsonl import (
    EpisodeEnded,
    EpisodeStarted,
    InspectionJSONLWriter,
)

from RL.observations.contracts import (
    PlayerTelemetry,
)


_POLICY_FRAME_WIDTH = 160
_POLICY_FRAME_HEIGHT = 90
_POLICY_FRAME_STACK = 4
_POLICY_FRAME_SHAPE = (
    _POLICY_FRAME_STACK,
    3,
    _POLICY_FRAME_HEIGHT,
    _POLICY_FRAME_WIDTH,
)

COMPACT_POLICY_FRAME_SHAPE = (
    3,
    _POLICY_FRAME_HEIGHT,
    _POLICY_FRAME_WIDTH,
)

LEGACY_POLICY_FRAME_STORAGE = (
    "legacy_float32_stack"
)

COMPACT_POLICY_FRAME_STORAGE = (
    "compact_uint8_frame"
)

POLICY_FRAME_STORAGE_MODES = frozenset(
    {
        LEGACY_POLICY_FRAME_STORAGE,
        COMPACT_POLICY_FRAME_STORAGE,
    }
)

_MATCH_DIRECTORY_PATTERN = re.compile(
    r"^match_(?P<index>[0-9]{6})_"
    r"(?P<status>recording|complete|interrupted)$"
)


@dataclass(frozen=True)
class MatchRecorderConfig:
    """Configuration for numbered match demonstrations."""

    output_root: Path
    controlled_player: str
    save_raw_frames: bool = False
    policy_frame_storage_mode: str = (
        LEGACY_POLICY_FRAME_STORAGE
    )
    policy_width: int = _POLICY_FRAME_WIDTH
    policy_height: int = _POLICY_FRAME_HEIGHT
    frame_stack: int = _POLICY_FRAME_STACK

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root),
        )

        if not self.controlled_player.strip():
            raise ValueError(
                "controlled_player must not be blank"
            )

        if not isinstance(
            self.save_raw_frames,
            bool,
        ):
            raise TypeError(
                "save_raw_frames must be bool"
            )

        if (
            not isinstance(
                self.policy_frame_storage_mode,
                str,
            )
            or self.policy_frame_storage_mode
            not in POLICY_FRAME_STORAGE_MODES
        ):
            raise ValueError(
                "policy_frame_storage_mode must "
                "be one of: "
                + ", ".join(
                    sorted(
                        POLICY_FRAME_STORAGE_MODES
                    )
                )
            )

        if self.policy_width != _POLICY_FRAME_WIDTH:
            raise ValueError(
                "policy_width must be 160"
            )

        if self.policy_height != _POLICY_FRAME_HEIGHT:
            raise ValueError(
                "policy_height must be 90"
            )

        if self.frame_stack != _POLICY_FRAME_STACK:
            raise ValueError(
                "frame_stack must be 4"
            )


@dataclass(frozen=True)
class RecordedMatchResult:
    """Result of finalizing one match recording."""

    match_index: int
    episode_id: str
    status: str
    outcome: str
    step_count: int
    directory: Path
    episode_path: Path
    action_counts: Mapping[str, int]


def utc_now_text() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def derive_discrete_action(
    command: CompositeActionCommand,
) -> DiscreteAction:
    """Derive one compatibility label from full human input.

    The composite command remains the authoritative label. This
    function only supplies the current single-action behavior-cloning
    pipeline with one deterministic compatibility class.
    """

    if not isinstance(
        command,
        CompositeActionCommand,
    ):
        raise TypeError(
            "command must be a CompositeActionCommand"
        )

    if command.weapon_delta > 0:
        return DiscreteAction.NEXT_WEAPON

    if command.weapon_delta < 0:
        return DiscreteAction.PREVIOUS_WEAPON

    if command.fire:
        return DiscreteAction.FIRE

    if command.jump:
        return DiscreteAction.JUMP

    if command.turn_delta_x < 0.0:
        return DiscreteAction.TURN_LEFT

    if command.turn_delta_x > 0.0:
        return DiscreteAction.TURN_RIGHT

    if command.forward_axis > 0:
        return DiscreteAction.FORWARD

    if command.forward_axis < 0:
        return DiscreteAction.BACKWARD

    if command.strafe_axis > 0:
        return DiscreteAction.STRAFE_RIGHT

    if command.strafe_axis < 0:
        return DiscreteAction.STRAFE_LEFT

    return DiscreteAction.NO_OP


def _event_type_and_data(
    event: object,
) -> tuple[str, Mapping[str, Any]]:
    if isinstance(event, Mapping):
        event_type = event.get("type")
        event_data = event.get(
            "data",
            {},
        )
    else:
        event_type = getattr(
            event,
            "type",
            None,
        )
        event_data = getattr(
            event,
            "data",
            {},
        )

    if not isinstance(
        event_type,
        str,
    ) or not event_type:
        raise ValueError(
            "event type must be a nonempty string"
        )

    if not isinstance(
        event_data,
        Mapping,
    ):
        raise ValueError(
            "event data must be a mapping"
        )

    return event_type, event_data


class AutomaticMatchRecorder:
    """Create one supervised episode for every match boundary."""

    def __init__(
        self,
        config: MatchRecorderConfig,
    ) -> None:
        if not isinstance(
            config,
            MatchRecorderConfig,
        ):
            raise TypeError(
                "config must be a MatchRecorderConfig"
            )

        self.config = config

        self.config.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._writer: InspectionJSONLWriter | None = None
        self._active_directory: Path | None = None
        self._episode_path: Path | None = None
        self._episode_id: str | None = None
        self._match_index: int | None = None
        self._match_data: dict[str, Any] = {}

        self._step_index = 0
        self._action_counts: Counter[str] = Counter()
        self._frame_stack: deque[np.ndarray] = deque(
            maxlen=self.config.frame_stack
        )

        self._completed_results: list[
            RecordedMatchResult
        ] = []

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def active_directory(
        self,
    ) -> Path | None:
        return self._active_directory

    @property
    def active_episode_id(
        self,
    ) -> str | None:
        return self._episode_id

    @property
    def step_count(self) -> int:
        return self._step_index

    @property
    def completed_results(
        self,
    ) -> tuple[RecordedMatchResult, ...]:
        return tuple(
            self._completed_results
        )

    def _next_match_index(self) -> int:
        highest = 0

        for path in self.config.output_root.iterdir():
            if not path.is_dir():
                continue

            match = _MATCH_DIRECTORY_PATTERN.match(
                path.name
            )

            if match is None:
                continue

            highest = max(
                highest,
                int(match.group("index")),
            )

        return highest + 1

    def _reset_active_state(self) -> None:
        self._writer = None
        self._active_directory = None
        self._episode_path = None
        self._episode_id = None
        self._match_index = None
        self._match_data = {}

        self._step_index = 0
        self._action_counts.clear()
        self._frame_stack.clear()

    def start_match(
        self,
        match_data: Mapping[str, Any],
    ) -> Path:
        """Begin recording an authoritative match_started event."""

        if not isinstance(
            match_data,
            Mapping,
        ):
            raise TypeError(
                "match_data must be a mapping"
            )

        incoming_match_id = match_data.get(
            "match_id"
        )

        if self.active:
            current_match_id = self._match_data.get(
                "match_id"
            )

            if (
                incoming_match_id is not None
                and incoming_match_id
                == current_match_id
            ):
                assert self._active_directory is not None
                return self._active_directory

            self.finalize(
                status="interrupted",
                outcome="superseded_by_new_match",
            )

        match_index = self._next_match_index()

        directory = (
            self.config.output_root
            / f"match_{match_index:06d}_recording"
        )

        if directory.exists():
            raise FileExistsError(
                str(directory)
            )

        frames_directory = directory / "frames"
        frames_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        if self.config.save_raw_frames:
            (
                directory
                / "raw_frames"
            ).mkdir(
                parents=True,
                exist_ok=False,
            )

        episode_path = directory / "episode.jsonl"
        episode_id = (
            f"human-match-{match_index:06d}"
        )

        writer = InspectionJSONLWriter(
            episode_path
        )

        if (
            self.config.policy_frame_storage_mode
            == LEGACY_POLICY_FRAME_STORAGE
        ):
            stored_policy_shape = (
                _POLICY_FRAME_SHAPE
            )
            stored_policy_dtype = "float32"
            stored_policy_transform = (
                "RGB; resize=160x90; "
                "normalize=0..1; stack=4"
            )
        else:
            stored_policy_shape = (
                COMPACT_POLICY_FRAME_SHAPE
            )
            stored_policy_dtype = "uint8"
            stored_policy_transform = (
                "RGB; resize=160x90; CHW; "
                "uint8; stack=reconstruct=4"
            )

        metadata = {
            "expert_demonstration": True,
            "demonstration_source": "human",
            "capture_source": (
                "human-demonstration"
            ),
            "personal_recording": True,
            "action_labels_available": True,
            (
                "composite_action_labels_"
                "available"
            ): True,
            "label_authority": (
                "xinput2-raw-events"
            ),
            "controlled_player": (
                self.config.controlled_player
            ),
            "match_index": match_index,
            "match_id": match_data.get(
                "match_id"
            ),
            "game_mode": match_data.get(
                "game_mode"
            ),
            "map_name": match_data.get(
                "map_name"
            ),
            "event_channel": match_data.get(
                "event_channel"
            ),
            "authority_tier": match_data.get(
                "authority_tier"
            ),
            "learning_phase": (
                "supervised_data_collection"
            ),
            "frame_references_are_real": True,
            "frame_shape": list(
                stored_policy_shape
            ),
            "frame_dtype": (
                stored_policy_dtype
            ),
            "frame_encoding": "npy",
            "policy_frame_shape": list(
                stored_policy_shape
            ),
            "policy_frame_dtype": (
                stored_policy_dtype
            ),
            "policy_frame_encoding": "npy",
            "policy_frame_storage_mode": (
                self.config
                .policy_frame_storage_mode
            ),
            "reconstructed_policy_frame_shape": [
                *_POLICY_FRAME_SHAPE,
            ],
            "policy_frame_transform": (
                stored_policy_transform
            ),
            "raw_frames_saved": (
                self.config.save_raw_frames
            ),
        }

        try:
            writer.write_episode_started(
                EpisodeStarted(
                    episode_id=episode_id,
                    started_at=utc_now_text(),
                    metadata=metadata,
                )
            )
        except Exception:
            writer.close()
            raise

        self._writer = writer
        self._active_directory = directory
        self._episode_path = episode_path
        self._episode_id = episode_id
        self._match_index = match_index
        self._match_data = dict(match_data)

        self._step_index = 0
        self._action_counts.clear()
        self._frame_stack.clear()

        return directory

    def _build_policy_stack(
        self,
        rgb_frame: np.ndarray,
    ) -> np.ndarray:
        processed = preprocess_rgb_frame(
            rgb_frame,
            width=self.config.policy_width,
            height=self.config.policy_height,
        )

        if not self._frame_stack:
            for _ in range(
                self.config.frame_stack
            ):
                self._frame_stack.append(
                    processed.copy()
                )
        else:
            self._frame_stack.append(
                processed.copy()
            )

        stacked = np.stack(
            tuple(self._frame_stack),
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        if stacked.shape != _POLICY_FRAME_SHAPE:
            raise RuntimeError(
                "recorded policy frame has "
                "an unexpected shape"
            )

        return stacked

    def _build_compact_policy_frame(
        self,
        rgb_frame: np.ndarray,
    ) -> np.ndarray:
        processed = preprocess_rgb_frame(
            rgb_frame,
            width=self.config.policy_width,
            height=self.config.policy_height,
        )

        compact = np.clip(
            np.rint(processed * 255.0),
            0.0,
            255.0,
        ).astype(
            np.uint8,
            copy=False,
        )

        compact = np.ascontiguousarray(
            compact
        )

        if compact.shape != (
            COMPACT_POLICY_FRAME_SHAPE
        ):
            raise RuntimeError(
                "recorded compact policy frame "
                "has an unexpected shape"
            )

        if compact.dtype != np.uint8:
            raise RuntimeError(
                "recorded compact policy frame "
                "must use uint8"
            )

        return compact

    def record_frame(
        self,
        rgb_frame: np.ndarray,
        command: CompositeActionCommand,
        *,
        pressed_keycodes: Iterable[int] = (),
        pressed_buttons: Iterable[int] = (),
        timestamp_ns: int | None = None,
        raw_event_count: int = 0,
        weapon_previous_event_count: int = 0,
        weapon_next_event_count: int = 0,
        game_events: Iterable[object] = (),
        telemetry: PlayerTelemetry | None = None,
        telemetry_sync: Mapping[str, Any] | None = None,

    ) -> bool:
        """Record one synchronized frame/input sample.

        Returns False while no match is active.
        """

        if not self.active:
            return False

        if not isinstance(
            rgb_frame,
            np.ndarray,
        ):
            raise TypeError(
                "rgb_frame must be a NumPy array"
            )

        if (
            rgb_frame.ndim != 3
            or rgb_frame.shape[2] != 3
            or rgb_frame.dtype != np.uint8
        ):
            raise ValueError(
                "rgb_frame must be HxWx3 uint8 RGB"
            )

        if not isinstance(
            command,
            CompositeActionCommand,
        ):
            raise TypeError(
                "command must be a "
                "CompositeActionCommand"
            )

        keycodes = tuple(
            sorted(
                int(value)
                for value in pressed_keycodes
            )
        )

        buttons = tuple(
            sorted(
                int(value)
                for value in pressed_buttons
            )
        )

        for field_name, value in (
            ("raw_event_count", raw_event_count),
            (
                "weapon_previous_event_count",
                weapon_previous_event_count,
            ),
            (
                "weapon_next_event_count",
                weapon_next_event_count,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a "
                    "nonnegative integer"
                )

        if timestamp_ns is None:
            timestamp_ns = time.time_ns()

        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(
                timestamp_ns,
                int,
            )
            or timestamp_ns < 0
        ):
            raise ValueError(
                "timestamp_ns must be a "
                "nonnegative integer"
            )

        assert self._writer is not None
        assert self._active_directory is not None
        assert self._episode_id is not None

        if (
            self.config.policy_frame_storage_mode
            == LEGACY_POLICY_FRAME_STORAGE
        ):
            policy_array = (
                self._build_policy_stack(
                    rgb_frame
                )
            )
            policy_shape = (
                _POLICY_FRAME_SHAPE
            )
            policy_dtype = "float32"
            policy_transform = (
                "RGB; resize=160x90; "
                "normalize=0..1; stack=4"
            )
        else:
            policy_array = (
                self._build_compact_policy_frame(
                    rgb_frame
                )
            )
            policy_shape = (
                COMPACT_POLICY_FRAME_SHAPE
            )
            policy_dtype = "uint8"
            policy_transform = (
                "RGB; resize=160x90; CHW; "
                "uint8; stack=reconstruct=4"
            )

        policy_relative_path = Path(
            "frames"
        ) / (
            f"policy_{self._step_index:06d}.npy"
        )

        policy_absolute_path = (
            self._active_directory
            / policy_relative_path
        )

        np.save(
            policy_absolute_path,
            policy_array,
            allow_pickle=False,
        )

        raw_snapshot: FrameSnapshot | None = None

        if self.config.save_raw_frames:
            raw_relative_path = Path(
                "raw_frames"
            ) / (
                f"raw_{self._step_index:06d}.png"
            )

            raw_absolute_path = (
                self._active_directory
                / raw_relative_path
            )

            Image.fromarray(
                rgb_frame,
                mode="RGB",
            ).save(
                raw_absolute_path,
                format="PNG",
            )

            raw_snapshot = FrameSnapshot(
                path=raw_relative_path.as_posix(),
                shape=tuple(
                    int(value)
                    for value in rgb_frame.shape
                ),
                dtype="uint8",
                encoding="png",
                transform=None,
            )

        game_event_records: list[dict[str, Any]] = []

        for event in game_events:
            event_type, event_data = (
                _event_type_and_data(event)
            )

            game_event_records.append(
                {
                    "type": event_type,
                    "data": dict(event_data),
                }
            )

        human_input_record = (
            command.to_record()
        )
        human_input_record[
            "weapon_previous_event_count"
        ] = weapon_previous_event_count
        human_input_record[
            "weapon_next_event_count"
        ] = weapon_next_event_count

        compatibility_action = (
            derive_discrete_action(command)
        )

        observation = ObservationSnapshot(
            tick=self._step_index,
            telemetry=telemetry,
            raw_frame=raw_snapshot,
            policy_frame=FrameSnapshot(
                path=(
                    policy_relative_path.as_posix()
                ),
                shape=policy_shape,
                dtype=policy_dtype,
                encoding="npy",
                transform=policy_transform,
            ),
        )

        transition = AgentTransition(
            episode_id=self._episode_id,
            step_index=self._step_index,
            observation=observation,
            decision=PolicyDecision(
                action=ActionCommand(
                    action=compatibility_action,
                    duration_ticks=(
                        command.duration_ticks
                    ),
                ),
                policy_name=(
                    "human-demonstration"
                ),
                policy_version="v1",
                checkpoint=None,
                deterministic=True,
            ),
            reward=0.0,
            next_observation=None,
            terminated=False,
            truncated=False,
            reward_components={},
            info={
                "human_input": (
                    human_input_record
                ),
                "pressed_keycodes": list(
                    keycodes
                ),
                "pressed_buttons": list(
                    buttons
                ),
                "timestamp_ns": timestamp_ns,
                "telemetry_sync": (
                    dict(telemetry_sync)
                    if telemetry_sync is not None
                    else None
                ),
                "game_events": game_event_records,
                "game_event_count": len(
                    game_event_records
                ),
                "raw_event_count": (
                    raw_event_count
                ),
                "label_authority": (
                    "xinput2-raw-events"
                ),
                (
                    "derived_discrete_action"
                ): compatibility_action.name,
                "derived_label_is_primary": False,
            },
        )

        self._writer.write_transition(
            transition
        )

        self._action_counts[
            compatibility_action.name
        ] += 1

        self._step_index += 1
        return True

    def finalize(
        self,
        *,
        status: str,
        outcome: str,
    ) -> RecordedMatchResult | None:
        """Finalize the active match and atomically rename its folder."""

        if not self.active:
            return None

        if status not in {
            "complete",
            "interrupted",
        }:
            raise ValueError(
                "status must be complete or interrupted"
            )

        if not isinstance(
            outcome,
            str,
        ) or not outcome:
            raise ValueError(
                "outcome must be a nonempty string"
            )

        assert self._writer is not None
        assert self._active_directory is not None
        assert self._episode_path is not None
        assert self._episode_id is not None
        assert self._match_index is not None

        writer = self._writer
        active_directory = (
            self._active_directory
        )
        episode_id = self._episode_id
        match_index = self._match_index
        step_count = self._step_index

        action_counts = {
            action.name: int(
                self._action_counts.get(
                    action.name,
                    0,
                )
            )
            for action in DiscreteAction
        }

        writer.write_episode_ended(
            EpisodeEnded(
                episode_id=episode_id,
                ended_at=utc_now_text(),
                steps=step_count,
                terminated=(
                    status == "complete"
                ),
                truncated=(
                    status == "interrupted"
                ),
                outcome=outcome,
                summary={
                    "status": status,
                    "match_index": match_index,
                    "match_id": (
                        self._match_data.get(
                            "match_id"
                        )
                    ),
                    "game_mode": (
                        self._match_data.get(
                            "game_mode"
                        )
                    ),
                    "map_name": (
                        self._match_data.get(
                            "map_name"
                        )
                    ),
                    "transition_count": (
                        step_count
                    ),
                    "action_counts": (
                        action_counts
                    ),
                    (
                        "composite_labels_are_"
                        "authoritative"
                    ): True,
                },
            )
        )

        writer.close()

        final_directory = (
            self.config.output_root
            / (
                f"match_{match_index:06d}_"
                f"{status}"
            )
        )

        if final_directory.exists():
            raise FileExistsError(
                str(final_directory)
            )

        active_directory.rename(
            final_directory
        )

        final_episode_path = (
            final_directory
            / self._episode_path.name
        )

        result = RecordedMatchResult(
            match_index=match_index,
            episode_id=episode_id,
            status=status,
            outcome=outcome,
            step_count=step_count,
            directory=final_directory,
            episode_path=final_episode_path,
            action_counts=action_counts,
        )

        self._completed_results.append(
            result
        )

        self._reset_active_state()
        return result

    def process_events(
        self,
        events: Iterable[object],
    ) -> tuple[RecordedMatchResult, ...]:
        """Apply match boundaries from an authoritative event batch."""

        completed: list[
            RecordedMatchResult
        ] = []

        for event in events:
            event_type, data = (
                _event_type_and_data(event)
            )

            if event_type == "match_started":
                self.start_match(data)
                continue

            if (
                event_type == "match_ended"
                and self.active
            ):
                result = self.finalize(
                    status="complete",
                    outcome="match_ended",
                )

                if result is not None:
                    completed.append(
                        result
                    )

        return tuple(completed)

    def interrupt(
        self,
    ) -> RecordedMatchResult | None:
        """Preserve an active match as interrupted."""

        return self.finalize(
            status="interrupted",
            outcome="operator_interrupted",
        )

    def close(
        self,
    ) -> RecordedMatchResult | None:
        """Safely finalize any still-active recording."""

        return self.interrupt()

    def __enter__(
        self,
    ) -> "AutomaticMatchRecorder":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
