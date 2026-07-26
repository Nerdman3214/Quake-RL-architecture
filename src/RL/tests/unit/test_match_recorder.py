"""Tests for automatic per-match human demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import pytest

from RL.actions.composite import (
    CompositeActionCommand,
)
from RL.actions.contracts import (
    DiscreteAction,
)
from RL.inspection.jsonl import (
    InspectionJSONLReader,
)
from RL.recording.match_recorder import (
    AutomaticMatchRecorder,
    COMPACT_POLICY_FRAME_SHAPE,
    COMPACT_POLICY_FRAME_STORAGE,
    LEGACY_POLICY_FRAME_STORAGE,
    MatchRecorderConfig,
    derive_discrete_action,
)
from RL.training.imitation.composite_sequence_dataset import (
    CompositeSequenceDataset,
)
from RL.training.imitation.dataset import (
    DemonstrationDataset,
)


def make_config(
    tmp_path: Path,
    *,
    save_raw_frames: bool = False,
) -> MatchRecorderConfig:
    return MatchRecorderConfig(
        output_root=tmp_path / "matches",
        controlled_player="Noobnog",
        save_raw_frames=save_raw_frames,
    )


def make_frame(
    value: int = 10,
) -> np.ndarray:
    return np.full(
        (120, 200, 3),
        value,
        dtype=np.uint8,
    )


def make_command(
    **overrides,
) -> CompositeActionCommand:
    values = {
        "forward_axis": 0,
        "strafe_axis": 0,
        "turn_delta_x": 0.0,
        "look_delta_y": 0.0,
        "fire": False,
        "jump": False,
        "weapon_delta": 0,
        "duration_ticks": 1,
    }

    values.update(overrides)

    return CompositeActionCommand(
        **values
    )


def match_started(
    match_id: str = "match-id-001",
) -> dict:
    return {
        "type": "match_started",
        "data": {
            "match_id": match_id,
            "game_mode": "tdm",
            "map_name": "fuse",
            "event_channel": (
                "structured_eventlog"
            ),
            "authority_tier": "primary",
        },
    }


def match_ended() -> dict:
    return {
        "type": "match_ended",
        "data": {
            "event_channel": (
                "structured_eventlog"
            ),
            "authority_tier": "primary",
            "raw_line": ":gameover",
        },
    }


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            make_command(
                weapon_delta=1,
                fire=True,
            ),
            DiscreteAction.NEXT_WEAPON,
        ),
        (
            make_command(
                weapon_delta=-1,
            ),
            DiscreteAction.PREVIOUS_WEAPON,
        ),
        (
            make_command(
                fire=True,
                forward_axis=1,
            ),
            DiscreteAction.FIRE,
        ),
        (
            make_command(
                jump=True,
            ),
            DiscreteAction.JUMP,
        ),
        (
            make_command(
                turn_delta_x=-2.0,
            ),
            DiscreteAction.TURN_LEFT,
        ),
        (
            make_command(
                turn_delta_x=2.0,
            ),
            DiscreteAction.TURN_RIGHT,
        ),
        (
            make_command(
                forward_axis=1,
            ),
            DiscreteAction.FORWARD,
        ),
        (
            make_command(
                forward_axis=-1,
            ),
            DiscreteAction.BACKWARD,
        ),
        (
            make_command(
                strafe_axis=1,
            ),
            DiscreteAction.STRAFE_RIGHT,
        ),
        (
            make_command(
                strafe_axis=-1,
            ),
            DiscreteAction.STRAFE_LEFT,
        ),
        (
            make_command(),
            DiscreteAction.NO_OP,
        ),
    ],
)
def test_derive_discrete_action(
    command,
    expected,
) -> None:
    assert (
        derive_discrete_action(command)
        == expected
    )


def test_ignores_frames_outside_match(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    recorded = recorder.record_frame(
        make_frame(),
        make_command(),
    )

    assert recorded is False
    assert recorder.step_count == 0


def test_records_complete_match_and_dataset_loads(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    recorder.process_events(
        [match_started()]
    )

    command = make_command(
        forward_axis=1,
        strafe_axis=1,
        turn_delta_x=4.5,
        look_delta_y=-2.0,
        fire=True,
    )

    assert recorder.record_frame(
        make_frame(20),
        command,
        pressed_keycodes=(25, 40),
        pressed_buttons=(1,),
        timestamp_ns=123456789,
    )

    results = recorder.process_events(
        [match_ended()]
    )

    assert len(results) == 1

    result = results[0]

    assert result.status == "complete"
    assert result.step_count == 1
    assert result.directory.name == (
        "match_000001_complete"
    )
    assert result.episode_path.is_file()
    assert not recorder.active

    records = InspectionJSONLReader(
        result.episode_path
    ).read_episode()

    assert [
        record.type
        for record in records
    ] == [
        "episode_started",
        "agent_transition",
        "episode_ended",
    ]

    metadata = records[0].data["metadata"]

    assert metadata[
        "expert_demonstration"
    ] is True

    assert metadata[
        "demonstration_source"
    ] == "human"

    assert metadata[
        "capture_source"
    ] == "human-demonstration"

    assert metadata[
        "frame_shape"
    ] == [
        4,
        3,
        90,
        160,
    ]

    assert metadata[
        "frame_encoding"
    ] == "npy"

    transition = records[1].data

    assert transition["decision"][
        "action"
    ]["name"] == "FIRE"

    human_input = transition[
        "info"
    ]["human_input"]

    expected_human_input = (
        command.to_record()
    )
    expected_human_input.update(
        {
            "weapon_previous_event_count": 0,
            "weapon_next_event_count": 0,
        }
    )

    assert human_input == expected_human_input

    assert transition["info"][
        "pressed_keycodes"
    ] == [25, 40]

    assert transition["info"][
        "pressed_buttons"
    ] == [1]

    assert transition["info"][
        "timestamp_ns"
    ] == 123456789

    policy_record = transition[
        "observation"
    ]["policy_frame"]

    assert policy_record["shape"] == [
        4,
        3,
        90,
        160,
    ]

    policy_path = (
        result.directory
        / policy_record["path"]
    )

    policy_frame = np.load(
        policy_path,
        allow_pickle=False,
    )

    assert policy_frame.shape == (
        4,
        3,
        90,
        160,
    )

    assert policy_frame.dtype == np.float32

    dataset = DemonstrationDataset(
        result.episode_path
    )

    assert len(dataset) == 1

    sample = dataset[0]

    assert sample.frames.shape == (
        4,
        3,
        90,
        160,
    )

    assert sample.action_index.item() == int(
        DiscreteAction.FIRE
    )


def test_frame_stack_advances_over_time(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    recorder.process_events(
        [match_started()]
    )

    for value in (
        10,
        20,
        30,
        40,
        50,
    ):
        recorder.record_frame(
            make_frame(value),
            make_command(
                forward_axis=1
            ),
        )

    result = recorder.process_events(
        [match_ended()]
    )[0]

    final_frame = np.load(
        result.directory
        / "frames"
        / "policy_000004.npy",
        allow_pickle=False,
    )

    frame_means = [
        float(frame.mean())
        for frame in final_frame
    ]

    assert frame_means == sorted(
        frame_means
    )

    assert len(set(frame_means)) == 4


def test_raw_frame_storage_is_optional(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(
            tmp_path,
            save_raw_frames=True,
        )
    )

    recorder.process_events(
        [match_started()]
    )

    recorder.record_frame(
        make_frame(),
        make_command(),
    )

    result = recorder.process_events(
        [match_ended()]
    )[0]

    raw_path = (
        result.directory
        / "raw_frames"
        / "raw_000000.png"
    )

    assert raw_path.is_file()

    records = InspectionJSONLReader(
        result.episode_path
    ).read_episode()

    assert records[1].data[
        "observation"
    ]["raw_frame"]["path"] == (
        "raw_frames/raw_000000.png"
    )


def test_interrupt_preserves_partial_match(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    recorder.process_events(
        [match_started()]
    )

    recorder.record_frame(
        make_frame(),
        make_command(
            strafe_axis=-1
        ),
    )

    result = recorder.interrupt()

    assert result is not None
    assert result.status == "interrupted"
    assert result.directory.name == (
        "match_000001_interrupted"
    )
    assert result.episode_path.is_file()
    assert not recorder.active

    records = InspectionJSONLReader(
        result.episode_path
    ).read_episode()

    ended = records[-1].data

    assert ended["terminated"] is False
    assert ended["truncated"] is True
    assert ended["outcome"] == (
        "operator_interrupted"
    )


def test_match_number_continues_from_disk(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matches"

    (
        root
        / "match_000004_complete"
    ).mkdir(
        parents=True
    )

    (
        root
        / "match_000007_interrupted"
    ).mkdir()

    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=root,
            controlled_player="Noobnog",
        )
    )

    directory = recorder.start_match(
        match_started()["data"]
    )

    assert directory.name == (
        "match_000008_recording"
    )

    result = recorder.interrupt()

    assert result is not None
    assert result.match_index == 8


def test_duplicate_match_start_does_not_restart(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    first = recorder.start_match(
        match_started()["data"]
    )

    recorder.record_frame(
        make_frame(),
        make_command(),
    )

    second = recorder.start_match(
        match_started()["data"]
    )

    assert second == first
    assert recorder.step_count == 1

    recorder.interrupt()


def test_new_match_start_preserves_previous_as_interrupted(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    recorder.process_events(
        [match_started("match-a")]
    )

    recorder.record_frame(
        make_frame(),
        make_command(
            forward_axis=1
        ),
    )

    recorder.process_events(
        [match_started("match-b")]
    )

    assert recorder.active

    assert len(
        recorder.completed_results
    ) == 1

    first_result = (
        recorder.completed_results[0]
    )

    assert first_result.status == (
        "interrupted"
    )

    assert first_result.outcome == (
        "superseded_by_new_match"
    )

    assert recorder.active_episode_id == (
        "human-match-000002"
    )

    recorder.interrupt()


def test_episode_file_is_valid_jsonl(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    recorder.process_events(
        [match_started()]
    )

    recorder.record_frame(
        make_frame(),
        make_command(),
    )

    result = recorder.process_events(
        [match_ended()]
    )[0]

    lines = result.episode_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 3

    for line in lines:
        assert isinstance(
            json.loads(line),
            dict,
        )


def test_records_auditable_wheel_event_counts(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        make_config(tmp_path)
    )

    recorder.process_events(
        [match_started()]
    )

    assert recorder.record_frame(
        make_frame(),
        make_command(
            weapon_delta=0,
        ),
        timestamp_ns=123456789,
        raw_event_count=9,
        weapon_previous_event_count=2,
        weapon_next_event_count=3,
    )

    result = recorder.process_events(
        [match_ended()]
    )[0]

    records = InspectionJSONLReader(
        result.episode_path
    ).read_episode()

    transition = records[1].data
    info = transition["info"]
    human_input = info["human_input"]

    assert human_input["weapon_delta"] == 0
    assert (
        human_input[
            "weapon_previous_event_count"
        ]
        == 2
    )
    assert (
        human_input[
            "weapon_next_event_count"
        ]
        == 3
    )
    assert info["raw_event_count"] == 9


def test_compact_recorder_stores_single_uint8_policy_frame(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
            policy_frame_storage_mode=(
                COMPACT_POLICY_FRAME_STORAGE
            ),
        )
    )

    recorder.process_events(
        [match_started()]
    )

    assert recorder.record_frame(
        make_frame(20),
        make_command(
            forward_axis=1,
        ),
    )

    result = recorder.process_events(
        [match_ended()]
    )[0]

    records = InspectionJSONLReader(
        result.episode_path
    ).read_episode()

    metadata = records[0].data["metadata"]

    assert metadata[
        "policy_frame_storage_mode"
    ] == COMPACT_POLICY_FRAME_STORAGE
    assert metadata["frame_shape"] == [
        3,
        90,
        160,
    ]
    assert metadata["frame_dtype"] == "uint8"
    assert metadata[
        "policy_frame_shape"
    ] == [3, 90, 160]
    assert metadata[
        "policy_frame_dtype"
    ] == "uint8"
    assert metadata[
        "reconstructed_policy_frame_shape"
    ] == [4, 3, 90, 160]

    transition = records[1].data
    policy_record = transition[
        "observation"
    ]["policy_frame"]

    assert policy_record["shape"] == [
        3,
        90,
        160,
    ]
    assert policy_record["dtype"] == "uint8"
    assert policy_record["encoding"] == "npy"

    policy_path = (
        result.directory
        / policy_record["path"]
    )

    compact = np.load(
        policy_path,
        allow_pickle=False,
    )

    assert compact.shape == (
        COMPACT_POLICY_FRAME_SHAPE
    )
    assert compact.dtype == np.uint8
    assert compact.flags.c_contiguous
    assert compact.nbytes == 43_200
    assert int(compact.min()) == 20
    assert int(compact.max()) == 20


def test_legacy_storage_remains_default(
    tmp_path: Path,
) -> None:
    config = MatchRecorderConfig(
        output_root=tmp_path / "matches",
        controlled_player="Noobnog",
    )

    assert (
        config.policy_frame_storage_mode
        == LEGACY_POLICY_FRAME_STORAGE
    )


@pytest.mark.parametrize(
    "invalid_mode",
    [
        "",
        "unknown",
        "compact",
        123,
        None,
    ],
)
def test_rejects_invalid_policy_frame_storage_mode(
    tmp_path: Path,
    invalid_mode,
) -> None:
    with pytest.raises(
        ValueError,
        match="policy_frame_storage_mode",
    ):
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
            policy_frame_storage_mode=(
                invalid_mode
            ),
        )


def test_compact_episode_reconstructs_recurrent_frame_history(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
            policy_frame_storage_mode=(
                COMPACT_POLICY_FRAME_STORAGE
            ),
        )
    )

    recorder.process_events(
        [match_started()]
    )

    for value in (10, 20, 30, 40, 50):
        assert recorder.record_frame(
            make_frame(value),
            make_command(
                forward_axis=1,
            ),
        )

    result = recorder.process_events(
        [match_ended()]
    )[0]

    dataset = CompositeSequenceDataset(
        result.episode_path,
        sequence_length=4,
        stride=1,
    )

    assert len(dataset) == 2

    first = dataset[0]
    second = dataset[1]

    assert first.frames.shape == (
        4,
        4,
        3,
        90,
        160,
    )

    assert first.frames.dtype == torch.float32

    first_history = (
        first.frames[
            :,
            :,
            0,
            0,
            0,
        ]
        .numpy()
    )

    np.testing.assert_allclose(
        first_history,
        np.array(
            [
                [10, 10, 10, 10],
                [10, 10, 10, 20],
                [10, 10, 20, 30],
                [10, 20, 30, 40],
            ],
            dtype=np.float32,
        )
        / 255.0,
        rtol=0.0,
        atol=1.0e-7,
    )

    final_history = (
        second.frames[
            -1,
            :,
            0,
            0,
            0,
        ]
        .numpy()
    )

    np.testing.assert_allclose(
        final_history,
        np.array(
            [20, 30, 40, 50],
            dtype=np.float32,
        )
        / 255.0,
        rtol=0.0,
        atol=1.0e-7,
    )
