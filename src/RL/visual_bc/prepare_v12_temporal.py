"""Prepare review candidates from frozen V12 alignment and DINO features.

Dry-run is the default. This module neither imports a trainer nor changes frozen
inputs. Explicit output goes into a new directory and is never marked frozen.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

if __package__:
    from . import build_temporal_sequence_index as temporal
else:
    import build_temporal_sequence_index as temporal


INTERVAL_SECONDS = 0.1
LABEL_CONTRACT = "historical_0p1s_same_segment_all_frames_labeled_v1"
UNOBSERVED_INPUTS = (
    "input_attack", "input_fire", "input_primary", "input_attack2",
    "input_secondary", "input_crouch", "input_use",
)
SAFETY = {
    "train_admission": False,
    "training_started": False,
    "agent_control": False,
    "automatic_action_allowed": False,
    "controls_modified": False,
    "V8_consumed": False,
}


def finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: expected a finite number")
    return result


def integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected an integer")
    return value


def close(actual: Any, expected: float, field: str, tolerance: float = 1e-8) -> None:
    if abs(finite(actual, field) - expected) > tolerance:
        raise ValueError(f"{field}: value differs from frozen identity/alignment")


def action_from_interval(interval: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce V11b labels; reject missing inputs instead of inventing zeros."""
    if not interval:
        raise ValueError("empty action interval")
    values: dict[str, list[float]] = {}
    for row in interval:
        if row.get("input_available") != 1:
            raise ValueError("input_available must equal 1")
        if any(row.get(key) is not None for key in UNOBSERVED_INPUTS):
            raise ValueError("optional action inputs require a new label contract")
        for key in ("input_move_forward", "input_move_strafe", "input_jump", "view_yaw", "view_pitch"):
            values.setdefault(key, []).append(finite(row.get(key), key))
        if not values["input_jump"][-1].is_integer():
            raise ValueError("input_jump must be integer-valued")
    n = len(interval)
    forward = values["input_move_forward"]
    strafe = values["input_move_strafe"]
    return {
        "forward_axis_mean": sum(forward) / n,
        "strafe_axis_mean": sum(strafe) / n,
        "forward_positive_fraction": sum(x > 1.0 for x in forward) / n,
        "forward_negative_fraction": sum(x < -1.0 for x in forward) / n,
        "strafe_positive_fraction": sum(x > 1.0 for x in strafe) / n,
        "strafe_negative_fraction": sum(x < -1.0 for x in strafe) / n,
        "jump_fraction": sum(int(x) != 0 for x in values["input_jump"]) / n,
        "attack_fraction": None,
        "secondary_attack_fraction": None,
        "crouch_fraction": None,
        "use_fraction": None,
        "view_yaw_delta_degrees": (values["view_yaw"][-1] - values["view_yaw"][0] + 180.0) % 360.0 - 180.0,
        "view_pitch_delta_degrees": values["view_pitch"][-1] - values["view_pitch"][0],
    }


def build_labeled_records(
    alignment: dict[str, Any],
    alignment_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    frame_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every admitted identity, marking unsupported labels explicitly.

    Telemetry resets are detected in recorded order. Manifest line-number errata
    are reported, while per-frame pointers and segment offsets must verify.
    """
    if alignment.get("dataset_candidate") != "V12" or alignment.get("alignment_frozen") is not True:
        raise ValueError("expected frozen V12 alignment")
    if alignment.get("alignment_model") != "piecewise_constant_video_time_equals_sample_time_plus_offset":
        raise ValueError("unsupported alignment model")
    for key in ("train_admission", "agent_control", "automatic_action_allowed", "controls_modified", "V8_consumed"):
        if alignment.get("safety", {}).get(key) is not False:
            raise ValueError(f"alignment safety field {key} must be false")
    close(alignment["video"]["fps"], 10.0, "fps")
    meta = alignment["telemetry"]
    first = integer(meta["capture_recovered_start_line"], "capture start")
    last = integer(meta["capture_recovered_end_line"], "capture end")
    if not 1 <= first <= last <= len(telemetry_rows):
        raise ValueError("capture line bounds exceed recovered telemetry")
    if meta["recovered_total_lines"] != len(telemetry_rows) or meta["capture_records"] != last - first + 1:
        raise ValueError("telemetry record count mismatch")
    original_offset = integer(meta["capture_original_start_line"], "original start") - first
    if meta["capture_original_end_line"] != last + original_offset:
        raise ValueError("original/recovered line offsets disagree")
    capture = telemetry_rows[first - 1:last]
    times = [finite(row.get("sample_time"), "sample_time") for row in capture]
    resets = [i for i in range(1, len(times)) if times[i] < times[i - 1]]
    bounds = [0, *resets, len(capture)]
    segment_specs = alignment["segments"]
    if set(segment_specs) != {str(i) for i in range(1, len(bounds))}:
        raise ValueError("detected reset count differs from frozen segments")
    segments: dict[int, dict[str, Any]] = {}
    for segment, (start, end) in enumerate(zip(bounds, bounds[1:]), 1):
        spec = segment_specs[str(segment)]
        segment_times = times[start:end]
        if any(b <= a for a, b in zip(segment_times, segment_times[1:])):
            raise ValueError("telemetry time is not strictly increasing within segment")
        if spec["records"] != end - start:
            raise ValueError("frozen segment record count mismatch")
        offset = finite(spec["offset_seconds"], "segment offset")
        close(spec["video_first_seconds"], segment_times[0] + offset, "segment first time")
        close(spec["video_last_seconds"], segment_times[-1] + offset, "segment last time")
        segments[segment] = {"times": segment_times, "rows": capture[start:end], "base": start, "offset": offset}

    errata = []
    for reset in resets:
        observed = {"capture_relative_line_1based": reset + 1,
                    "reset_recovered_line": first + reset,
                    "reset_original_line": first + reset + original_offset}
        if len(resets) == 1:
            observed["recorded_reset_recovered_line"] = meta.get("reset_recovered_line")
            observed["recorded_reset_original_line"] = meta.get("reset_original_line")
            observed["manifest_numbering_mismatch"] = any(meta.get(key) != observed[key] for key in ("reset_recovered_line", "reset_original_line"))
        errata.append(observed)

    admitted_count = alignment["frame_indexing"]["accepted_frames"]
    if not admitted_count or len(alignment_rows) != admitted_count or len(frame_rows) != admitted_count:
        raise ValueError("admitted frame/index count mismatch")
    maximum_gap = finite(alignment["frame_indexing"]["max_telemetry_gap_seconds"], "maximum telemetry gap")
    if maximum_gap > 0.075 or maximum_gap < 0:
        raise ValueError("unsupported telemetry gap threshold")
    records = []
    exclusions = []
    interval_counts: Counter[int] = Counter()
    previous_id = -1
    previous_segment = 0
    for row, frame in zip(alignment_rows, frame_rows):
        source_id = integer(row["frame_index"], "frame index")
        frame_number = source_id + 1
        video_time = source_id / 10.0
        filename = f"frame_{frame_number:06d}.jpg"
        if not previous_id < source_id < alignment["video"]["decoded_frames"]:
            raise ValueError("source identities must increase within original video bounds")
        previous_id = source_id
        if (row["frame_number"] != frame_number or frame.get("admitted") is not True
                or frame.get("frame_index_0based") != source_id
                or frame.get("frame_number_1based") != frame_number
                or frame.get("filename") != filename
                or frame.get("relative_path") != f"V12/frames_game_only_10hz/{filename}"):
            raise ValueError("frame admission identity mismatch")
        close(row["video_time_seconds"], video_time, "alignment frame time")
        close(frame["video_time_seconds"], video_time, "frame index time")
        segment_id = integer(row["segment"], "segment")
        if segment_id not in segments or segment_id < previous_segment:
            raise ValueError("invalid or backward alignment segment")
        previous_segment = segment_id
        segment = segments[segment_id]
        pointer = integer(row["telemetry_recovered_line"], "telemetry pointer")
        local_pointer = pointer - first - segment["base"]
        segment_times = segment["times"]
        if not 0 <= local_pointer < len(segment_times):
            raise ValueError("telemetry pointer crosses frozen segment")
        if row["telemetry_original_source_line"] != pointer + original_offset:
            raise ValueError("original telemetry pointer mismatch")
        pointed = segment["rows"][local_pointer]
        close(row["telemetry_sample_time"], segment_times[local_pointer], "telemetry pointer timestamp")
        mapped = segment_times[local_pointer] + segment["offset"]
        close(row["telemetry_mapped_video_time"], mapped, "mapped telemetry time")
        start = video_time - segment["offset"]
        end = start + INTERVAL_SECONDS
        gap = abs(segment_times[local_pointer] - start)
        close(row["nearest_gap_seconds"], gap, "nearest telemetry gap")
        insert = bisect_left(segment_times, start)
        nearest = min(abs(segment_times[i] - start) for i in (insert - 1, insert) if 0 <= i < len(segment_times))
        close(gap, nearest, "nearest telemetry pointer")
        if gap > maximum_gap + 1e-9:
            raise ValueError("admitted frame exceeds frozen telemetry gap")
        if row.get("input_available") != pointed.get("input_available"):
            raise ValueError("input availability differs from frozen pointer")
        left = bisect_left(segment_times, start)
        right = bisect_left(segment_times, end)
        reason = None
        if start < segment_times[0]:
            reason = "interval_starts_before_segment"
        elif end > segment_times[-1]:
            reason = "interval_ends_after_segment"
        elif left == right:
            reason = "empty_action_interval"
        interval = segment["rows"][left:right]
        action = None if reason else action_from_interval(interval)
        if reason:
            exclusions.append({"source_frame_number_1based": frame_number,
                               "source_sample_id": source_id, "segment": segment_id,
                               "reason": reason, "interval_start": start, "interval_end": end,
                               "segment_first_sample_time": segment_times[0],
                               "segment_last_sample_time": segment_times[-1]})
        else:
            interval_counts[len(interval)] += 1
        records.append({
            "sample_id": source_id, "source_sample_id": source_id,
            "frame": str(frame_directory / filename),
            "frame_video_time_seconds": video_time, "segment": segment_id,
            "telemetry_interval_start": start, "telemetry_interval_end": end,
            "nearest_telemetry_gap_seconds": gap,
            "action": action, "label_eligible": reason is None,
            "label_rejection_reason": reason,
            "label_telemetry_sample_count": len(interval),
            "label_telemetry_first_recovered_line": first + segment["base"] + left if interval else None,
            "label_telemetry_last_recovered_line": first + segment["base"] + right - 1 if interval else None,
            "mouse_label_source": "view_angle_delta",
            "raw_mouse_device_delta_captured": False,
            "future_outcome_used_as_feature": False,
            "automatic_action_allowed": False, "controls_modified": False,
        })
    audit = {
        "format_version": 1, "dataset": "V12", "role": "calibration_validation_only",
        "status": "review_candidate", "validation_dataset_frozen": False,
        "label_contract": LABEL_CONTRACT, "action_interval_seconds": INTERVAL_SECONDS,
        "all_context_frames_require_eligible_labels": True,
        "single_sample_intervals_allowed": True, "interpolation_used": False,
        "source_record_count": len(records), "label_eligible_count": len(records) - len(exclusions),
        "label_exclusions": exclusions, "eligible_interval_sample_counts": dict(sorted(interval_counts.items())),
        "reset_line_number_observations": errata, "frozen_alignment_modified": False,
        **SAFETY,
    }
    return records, audit


def guarded_path(path: Path) -> Path:
    resolved = path.resolve()
    if re.search(r"(?:^|[^a-z0-9])v8(?:$|[^a-z0-9])", str(resolved).lower()):
        raise ValueError("V8 paths are excluded from preparation")
    return resolved


def checked_json(path: Path, expected_sha: str) -> dict[str, Any]:
    path = guarded_path(path)
    if temporal.sha256_file(path) != expected_sha:
        raise ValueError(f"SHA256 mismatch: {path}")
    return json.loads(path.read_text())


def verify_alignment_files(directory: Path, alignment_sha: str) -> None:
    expected = {"alignment_manifest_v1.json", "frame_to_telemetry_index_v1.jsonl",
                "rejected_frames_v1.jsonl", "order_preserving_alignment_audit_v4.py",
                "freeze_alignment_and_build_index_v1.py"}
    entries = {}
    for line in (directory / "sha256s_alignment_v1.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name not in expected or name in entries:
            raise ValueError("unexpected/duplicate frozen alignment checksum entry")
        entries[name] = digest
    if set(entries) != expected or entries["alignment_manifest_v1.json"] != alignment_sha:
        raise ValueError("frozen alignment checksum list mismatch")
    for name, digest in entries.items():
        if temporal.sha256_file(guarded_path(directory / name)) != digest:
            raise ValueError(f"frozen alignment hash mismatch: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--frame-cache-directory", type=Path, required=True, help="Directory containing V12 frame_index_v1.jsonl")
    parser.add_argument("--feature-cache-directory", type=Path, required=True)
    parser.add_argument("--alignment-sha256", required=True)
    parser.add_argument("--frame-index-sha256", required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, help="New review-candidate directory; omit for dry-run")
    args = parser.parse_args(argv)
    run = guarded_path(args.run_directory)
    frames = guarded_path(args.frame_cache_directory)
    cache = guarded_path(args.feature_cache_directory)
    output = guarded_path(args.output_directory) if args.output_directory else None
    if output is not None and output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    alignment_dir = run / "frozen_alignment_v1"
    alignment_path = alignment_dir / "alignment_manifest_v1.json"
    alignment = checked_json(alignment_path, args.alignment_sha256)
    if alignment.get("dataset_candidate") != "V12":
        raise ValueError("preparation is restricted to V12")
    verify_alignment_files(alignment_dir, args.alignment_sha256)
    recovery_path = run / "recovered_context_v1/recovery_manifest_v1.json"
    recovery = checked_json(recovery_path, alignment["source_recovery_manifest_sha256"])
    telemetry_path = guarded_path(run / "recovered_context_v1/live_telemetry_capture_context_recovered_v1.jsonl")
    if guarded_path(Path(alignment["telemetry"]["path"])) != telemetry_path:
        raise ValueError("recovered telemetry path mismatch")
    telemetry_sha = recovery["artifacts"][telemetry_path.name]["sha256"]
    if temporal.sha256_file(telemetry_path) != telemetry_sha:
        raise ValueError("recovered telemetry hash mismatch")
    frame_index = guarded_path(frames / "frame_index_v1.jsonl")
    if temporal.sha256_file(frame_index) != args.frame_index_sha256:
        raise ValueError("frame index hash mismatch")
    cache_manifest_path = cache / "manifest_v1.json"
    manifest = checked_json(cache_manifest_path, args.cache_manifest_sha256)
    if manifest.get("dataset") != "V12" or manifest.get("role") != "calibration_validation_only":
        raise ValueError("V12 cache role mismatch")
    frame_directory = guarded_path(frames / "frames_game_only_10hz")
    if guarded_path(Path(manifest["source_frame_cache"])) != frame_directory:
        raise ValueError("feature/source frame directory mismatch")
    frame_rows = temporal.read_jsonl(frame_index)
    records, labels_audit = build_labeled_records(
        alignment, temporal.read_jsonl(alignment_dir / "frame_to_telemetry_index_v1.jsonl"),
        temporal.read_jsonl(telemetry_path), frame_rows, frame_directory,
    )
    admitted = {row["sample_id"] + 1 for row in records}
    rejected = set(range(1, alignment["video"]["decoded_frames"] + 1)) - admitted
    if sorted(rejected) != manifest.get("rejected_source_frame_numbers_1based"):
        raise ValueError("DINO/alignment rejected identities differ")
    if len(rejected) != alignment["frame_indexing"]["rejected_frames"]:
        raise ValueError("rejected identity count differs from frozen alignment")
    refs = temporal.load_cache_refs(records, cache, manifest)
    sequences, sequence_audit = temporal.build_sequences(records, refs, session="V12")
    sequence_audit.update({"status": "review_candidate", "validation_dataset_frozen": False,
                           "label_contract": LABEL_CONTRACT, **SAFETY})
    inputs = {
        "alignment_manifest": {"path": str(alignment_path), "sha256": args.alignment_sha256},
        "recovery_manifest": {"path": str(recovery_path), "sha256": alignment["source_recovery_manifest_sha256"]},
        "telemetry": {"path": str(telemetry_path), "sha256": telemetry_sha},
        "frame_index": {"path": str(frame_index), "sha256": args.frame_index_sha256},
        "cache_manifest": {"path": str(cache_manifest_path), "sha256": args.cache_manifest_sha256},
        "label_script": {"path": str(Path(__file__).resolve()), "sha256": temporal.sha256_file(Path(__file__))},
        "sequence_script": {"path": str(Path(temporal.__file__).resolve()), "sha256": temporal.sha256_file(Path(temporal.__file__))},
    }
    sequence_audit.update({"inputs": inputs, "cache_directory": str(cache),
                           "cache_manifest": str(cache_manifest_path),
                           "cache_manifest_sha256": args.cache_manifest_sha256})
    labels_audit["inputs"] = inputs
    result = {"status": "dry_run_passed", "labels": labels_audit, "sequences": sequence_audit}
    if output is not None:
        # All computations and input validation finish before exclusive creation.
        source_text = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in records)
        import hashlib
        source_sha = hashlib.sha256(source_text.encode()).hexdigest()
        sequence_audit.update({"source_file": str(output / "aligned_samples_label_candidate_v1.jsonl"), "source_sha256": source_sha})
        temporal.write_outputs(output, sequences, sequence_audit, manifest)
        extras = {
            "aligned_samples_label_candidate_v1.jsonl": source_text,
            "label_audit_v1.json": json.dumps(labels_audit, indent=2, allow_nan=False) + "\n",
            "alignment_line_number_erratum_v1.json": json.dumps({
                "source_alignment_manifest_sha256": args.alignment_sha256,
                "observations": labels_audit["reset_line_number_observations"],
                "frozen_inputs_modified": False,
            }, indent=2) + "\n",
        }
        for name, contents in extras.items():
            with (output / name).open("x") as handle:
                handle.write(contents)
        # A separate complete inventory includes the builder's own inventory.
        with (output / "sha256s_candidate_v1.txt").open("x") as handle:
            for path in sorted(output.iterdir()):
                if path.name != "sha256s_candidate_v1.txt":
                    handle.write(f"{temporal.sha256_file(path)}  {path.name}\n")
        result.update(status="review_candidate_written", output_directory=str(output))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
