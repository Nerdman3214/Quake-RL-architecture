"""Regression checks for temporal identity and explicit label exclusions."""
from copy import deepcopy

import pytest

from RL.visual_bc.build_temporal_sequence_index import build_sequences, reject_sealed_path


def source(ids):
    records, refs = [], []
    for index, identity in enumerate(ids):
        action = {key: 0.0 for key in (
            "forward_axis_mean", "strafe_axis_mean", "forward_positive_fraction",
            "forward_negative_fraction", "strafe_positive_fraction", "strafe_negative_fraction",
            "jump_fraction", "view_yaw_delta_degrees", "view_pitch_delta_degrees")}
        action.update({key: None for key in ("attack_fraction", "secondary_attack_fraction", "crouch_fraction", "use_fraction")})
        action["view_yaw_delta_degrees"] = float(identity)
        records.append({"sample_id":identity, "source_sample_id":identity,
                        "frame":f"/tmp/V12/frame_{identity+1:06d}.jpg",
                        "frame_video_time_seconds":identity/10,
                        "telemetry_interval_start":identity/10,
                        "telemetry_interval_end":identity/10+.1,
                        "nearest_telemetry_gap_seconds":.02, "segment":1,
                        "action":action, "mouse_label_source":"view_angle_delta",
                        "raw_mouse_device_delta_captured":False,
                        "future_outcome_used_as_feature":False,
                        "automatic_action_allowed":False,"controls_modified":False})
        refs.append({"cache_index":index,"feature_file":"features.safetensors","feature_offset":index})
    return records,refs


def test_dense_cache_indices_do_not_bridge_source_gap():
    records,refs = source([0,1,2,3,9,10,11,12])
    sequences,audit = build_sequences(records,refs,session="V12")
    assert [row["source_sample_ids"] for row in sequences] == [[0,1,2,3],[9,10,11,12]]
    assert audit["rejected_cross_boundary_windows"] == 3
    assert sequences[-1]["target_action"] == records[-1]["action"]


def test_regular_cadence_still_respects_match_segment():
    records,refs = source(range(8))
    for row in records[4:]:
        row["segment"] = 2
    sequences,_ = build_sequences(records,refs,session="V12")
    assert len(sequences) == 2


def test_explicit_ineligible_context_is_excluded_without_erasing_identity():
    records,refs = source(range(8))
    records[3].update(label_eligible=False, label_rejection_reason="interval_ends_after_segment", action=None)
    sequences,audit = build_sequences(records,refs,session="V12")
    assert len(sequences) == 1
    assert sequences[0]["source_sample_ids"] == [4,5,6,7]
    assert audit["label_eligible_record_count"] == 7
    assert audit["rejected_label_windows"] == 4


@pytest.mark.parametrize("change", ["missing_action","nan_action","missing_segment","false_identity"])
def test_malformed_sources_fail_closed(change):
    records,refs = source(range(4))
    if change == "missing_action":
        records[0]["action"] = None
    elif change == "nan_action":
        records[0]["action"]["jump_fraction"] = float("nan")
    elif change == "missing_segment":
        records[0].pop("segment")
    else:
        records[0]["source_sample_id"] = 10
    with pytest.raises(ValueError):
        build_sequences(records,refs,session="V12")


def test_builder_does_not_mutate_source():
    records,refs = source(range(4))
    original = deepcopy(records)
    sequences,_ = build_sequences(records,refs,session="V12")
    sequences[0]["target_action"]["jump_fraction"] = 1
    assert records == original


def test_sealed_paths_are_rejected_before_file_access():
    with pytest.raises(ValueError,match="V8"):
        reject_sealed_path("/does/not/exist/V8/manifest.json")
