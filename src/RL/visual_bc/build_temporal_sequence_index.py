#!/usr/bin/env python3
"""Build four-frame frozen-feature indices without importing a training runtime.

The CLI validates inputs in memory by default. Supplying --output-directory
creates new metadata only, and refuses an existing destination. Source frame
identity, explicit match segments, and telemetry intervals establish continuity;
the dense cache index is only a feature address.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any


SEQUENCE_LENGTH = 4
EXPECTED_STEP_SECONDS = 0.1
STEP_TOLERANCE_SECONDS = 0.002
EXPECTED_SPAN_SECONDS = 0.3
SPAN_TOLERANCE_SECONDS = 0.003
MAX_TELEMETRY_GAP_SECONDS = 0.075
ROLES = {"V11b": "TRAIN_candidate_jump_enrichment", "V12": "calibration_validation_only"}
ALIASES = {
    "start": "sample_start", "end": "sample_end", "count": "sample_count",
    "shape": "feature_shape", "dtype": "feature_dtype",
    "record_file": "records_file", "record_sha256": "records_sha256",
}
NUMERIC_ACTION_KEYS = (
    "forward_axis_mean", "strafe_axis_mean", "forward_positive_fraction",
    "forward_negative_fraction", "strafe_positive_fraction",
    "strafe_negative_fraction", "jump_fraction", "view_yaw_delta_degrees",
    "view_pitch_delta_degrees",
)
NULL_ACTION_KEYS = ("attack_fraction", "secondary_attack_fraction", "crouch_fraction", "use_fraction")
SOURCE_FALSE_FLAGS = ("raw_mouse_device_delta_captured", "future_outcome_used_as_feature",
                      "automatic_action_allowed", "controls_modified")
MANIFEST_FALSE_FLAGS = ("automatic_action_allowed", "controls_modified", "training",
                       "train_admission", "agent_control", "V8_consumed")


def reject_sealed_path(path: Path | str) -> None:
    """Reject V8 paths before opening them, including paths through symlinks."""
    for text in (str(path), str(Path(path).resolve())):
        if re.search(r"(?i)(?:^|[^a-z0-9])v8(?:$|[^a-z0-9])", text):
            raise ValueError("V8 is sealed and unsupported by this preparation tool")


def sha256_file(path: Path) -> str:
    reject_sealed_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    reject_sealed_path(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _frame_number(frame: Any) -> int:
    if not isinstance(frame, str):
        raise ValueError("frame must be a path string")
    reject_sealed_path(frame)
    match = re.fullmatch(r"frame_(\d+)\.(?:jpg|jpeg|png)", Path(frame).name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unexpected frame filename: {Path(frame).name}")
    return int(match.group(1))


def _identity(row: dict[str, Any]) -> tuple[int, str, float]:
    source_id = _integer(row.get("sample_id"), "sample_id")
    for key in ("source_sample_id", "source_frame_index_0based"):
        if key in row and _integer(row[key], key) != source_id:
            raise ValueError(f"Source identity mismatch: {key}")
    frame_number = _frame_number(row.get("frame"))
    if frame_number != source_id + 1:
        raise ValueError("Frame filename and original sample_id disagree")
    if "source_frame_number_1based" in row and row["source_frame_number_1based"] != frame_number:
        raise ValueError("Source frame number identity mismatch")
    timestamp = _number(row.get("frame_video_time_seconds"), "frame_video_time_seconds")
    if timestamp < 0:
        raise ValueError("Negative video time")
    return source_id, row["frame"], timestamp


def _session(session: str) -> None:
    if session not in ROLES:
        raise ValueError("Only V11b and calibration/validation-only V12 are supported; V8 is sealed")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _session(manifest.get("dataset"))
    if manifest.get("role") != ROLES[manifest["dataset"]]:
        raise ValueError("Unexpected dataset role; V12 must remain calibration_validation_only")
    if manifest.get("status") != "cache_complete":
        raise ValueError("Feature cache must be complete")
    if manifest.get("feature_shape_per_frame") != [41, 384] or manifest.get("feature_dtype") != "float16":
        raise ValueError("Expected frozen float16 [41,384] features")
    for key in MANIFEST_FALSE_FLAGS:
        if manifest.get(key) is not False:
            raise ValueError(f"Manifest safety flag must be false: {key}")
    if manifest.get("encoder_frozen") is not True or manifest.get("model_eval_mode") is not True:
        raise ValueError("Expected a frozen encoder in evaluation mode")


def compatibility_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Add loader aliases to a copy; reject contradictory native/alias values."""
    _validate_manifest(manifest)
    result = copy.deepcopy(manifest)
    chunks = result.get("completed_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Manifest has no completed chunks")
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Malformed chunk")
        for native, alias in ALIASES.items():
            if native not in chunk and alias not in chunk:
                raise ValueError(f"Chunk missing {native}/{alias}")
            if native in chunk and alias in chunk and chunk[native] != chunk[alias]:
                raise ValueError(f"Contradictory chunk aliases: {native}/{alias}")
            value = chunk[native] if native in chunk else chunk[alias]
            chunk[native] = copy.deepcopy(value)
            chunk[alias] = copy.deepcopy(value)
    count = _integer(result.get("cache_record_count"), "cache_record_count")
    for key in ("completed_sample_count", "total_source_records"):
        if key in result and result[key] != count:
            raise ValueError(f"Manifest count mismatch: {key}")
    if result.get("completed_chunk_count") != len(chunks):
        raise ValueError("Manifest completed_chunk_count mismatch")
    result["completed_sample_count"] = count
    return result


def _child(cache_dir: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("Chunk filename must be a plain basename")
    path = cache_dir / filename
    reject_sealed_path(path)
    if path.resolve().parent != cache_dir.resolve():
        raise ValueError("Chunk path escapes cache directory")
    return path


def _verify_hash(path: Path, expected: Any) -> None:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"Missing or invalid SHA256 pin for {path.name}")
    if sha256_file(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def _validate_tensor_header(path: Path, count: int) -> None:
    """Check safetensors structure without loading torch or an encoder."""
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 1024 * 1024 or size < 2:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(handle.read(size))
    if set(header) - {"features", "__metadata__"} or "features" not in header:
        raise ValueError("Expected a single features tensor")
    tensor = header["features"]
    byte_count = count * 41 * 384 * 2
    if tensor.get("dtype") != "F16" or tensor.get("shape") != [count, 41, 384]:
        raise ValueError("Safetensors feature dtype/shape mismatch")
    if tensor.get("data_offsets") != [0, byte_count] or path.stat().st_size != 8 + size + byte_count:
        raise ValueError("Safetensors feature byte offsets/size mismatch")


def load_cache_refs(records: list[dict[str, Any]], cache_dir: Path,
                    manifest: dict[str, Any], verify_feature_hashes: bool = True) -> list[dict[str, Any]]:
    """Validate cache sidecars and match references by original source identity.

    Disabling feature hash verification is intended only for callers that have
    already checked those exact files. Record hashes and tensor headers are
    always checked; the CLI always checks feature hashes too.
    """
    reject_sealed_path(cache_dir)
    normalized = compatibility_manifest(manifest)
    if normalized["cache_record_count"] != len(records):
        raise ValueError("Source/cache record count mismatch")
    sources = {}
    for row in records:
        identity = _identity(row)
        if identity[0] in sources:
            raise ValueError("Duplicate source sample_id")
        sources[identity[0]] = identity
    refs = {}
    expected_start = 0
    seen_features: set[str] = set()
    seen_sidecars: set[str] = set()
    chunks = normalized["completed_chunks"]
    for chunk in chunks:
        for key in ("start", "end", "count"):
            _integer(chunk[key], f"chunk {key}")
        start, end, count = chunk["start"], chunk["end"], chunk["count"]
        if count < 1 or start != expected_start or end - start + 1 != count:
            raise ValueError("Non-contiguous or inconsistent dense cache chunk range")
        if chunk["shape"] != [count, 41, 384] or chunk["dtype"] != "float16":
            raise ValueError("Chunk feature shape/dtype mismatch")
        feature_path = _child(cache_dir, chunk["feature_file"])
        records_path = _child(cache_dir, chunk["record_file"])
        if feature_path.name in seen_features or records_path.name in seen_sidecars:
            raise ValueError("Duplicate chunk filename")
        seen_features.add(feature_path.name)
        seen_sidecars.add(records_path.name)
        if verify_feature_hashes:
            _verify_hash(feature_path, chunk.get("feature_sha256"))
        _validate_tensor_header(feature_path, count)
        _verify_hash(records_path, chunk["record_sha256"])
        sidecars = read_jsonl(records_path)
        if len(sidecars) != count:
            raise ValueError("Sidecar record count mismatch")
        for offset, row in enumerate(sidecars):
            if _integer(row.get("cache_index"), "cache_index") != start + offset:
                raise ValueError("Sidecar cache_index mismatch")
            if _integer(row.get("feature_offset"), "feature_offset") != offset:
                raise ValueError("Sidecar feature_offset mismatch")
            identity = _identity(row)
            expected = sources.get(identity[0])
            if expected is None or identity[:2] != expected[:2] or abs(identity[2] - expected[2]) > 1e-9:
                raise ValueError("Sidecar/source identity mismatch")
            if identity[0] in refs:
                raise ValueError("Duplicate sidecar source sample_id")
            refs[identity[0]] = {"cache_index": start + offset, "feature_file": feature_path.name,
                                 "feature_offset": offset}
        expected_start = end + 1
    if expected_start != len(records) or len(refs) != len(records):
        raise ValueError("Cache coverage mismatch")
    return [refs[row["sample_id"]] for row in records]


def _validate_action(row: dict[str, Any], session: str) -> bool:
    eligible = row.get("label_eligible", True)
    if not isinstance(eligible, bool):
        raise ValueError("label_eligible must be boolean")
    if not eligible:
        if session != "V12" or not isinstance(row.get("label_rejection_reason"), str) or not row["label_rejection_reason"].strip():
            raise ValueError("Only V12 supports explicitly rejected labels with a reason")
        if row.get("action") is not None:
            raise ValueError("Ineligible records must not contain placeholder action labels")
        return False
    action = row.get("action")
    if not isinstance(action, dict) or set(action) != set(NUMERIC_ACTION_KEYS + NULL_ACTION_KEYS):
        raise ValueError("Missing or unsupported action labels")
    for key in NUMERIC_ACTION_KEYS:
        value = _number(action[key], f"action.{key}")
        if key.endswith("_fraction") and not 0 <= value <= 1:
            raise ValueError(f"Action fraction outside [0,1]: {key}")
    for key in NULL_ACTION_KEYS:
        if action[key] is not None:
            raise ValueError(f"Unsupported captured action must remain null: {key}")
    return True


def build_sequences(records: list[dict[str, Any]], feature_refs: list[dict[str, Any]], *,
                    session: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure sequence construction with explicit source and label boundaries."""
    _session(session)
    if not records or len(records) != len(feature_refs):
        raise ValueError("Nonempty records and one feature reference per record are required")
    identities = [_identity(row) for row in records]
    if len({identity[0] for identity in identities}) != len(records):
        raise ValueError("Duplicate original source identity")
    explicit_segments = ["segment" in row for row in records]
    if any(explicit_segments) != all(explicit_segments) or (session == "V12" and not all(explicit_segments)):
        raise ValueError("V12 requires explicit segment on every row; partial segment metadata is invalid")
    eligible, quality = [], []
    seen_cache = set()
    for index, (row, ref) in enumerate(zip(records, feature_refs)):
        for key in SOURCE_FALSE_FLAGS:
            if row.get(key) is not False:
                raise ValueError(f"Source safety flag must be false: {key}")
        if row.get("mouse_label_source") != "view_angle_delta":
            raise ValueError("Expected view_angle_delta labels")
        if "role" in row and row["role"] != ROLES[session]:
            raise ValueError("Source row role mismatch")
        if "segment" in row and (isinstance(row["segment"], bool) or not isinstance(row["segment"], (str, int)) or row["segment"] == ""):
            raise ValueError("Invalid explicit segment identity")
        eligible.append(_validate_action(row, session))
        start = _number(row.get("telemetry_interval_start"), "telemetry_interval_start")
        end = _number(row.get("telemetry_interval_end"), "telemetry_interval_end")
        if abs(end - start - EXPECTED_STEP_SECONDS) > 1e-6:
            raise ValueError("Action telemetry interval must span 0.1 seconds")
        gap = _number(row.get("nearest_telemetry_gap_seconds"), "nearest_telemetry_gap_seconds")
        if gap < 0:
            raise ValueError("Negative telemetry gap")
        quality.append(gap <= MAX_TELEMETRY_GAP_SECONDS)
        if set(ref) != {"cache_index", "feature_file", "feature_offset"}:
            raise ValueError("Malformed feature reference")
        cache_index = _integer(ref["cache_index"], "cache_index")
        _integer(ref["feature_offset"], "feature_offset")
        if not isinstance(ref["feature_file"], str) or Path(ref["feature_file"]).name != ref["feature_file"]:
            raise ValueError("Feature reference must use a chunk basename")
        reject_sealed_path(ref["feature_file"])
        if cache_index in seen_cache:
            raise ValueError("Duplicate feature cache reference")
        seen_cache.add(cache_index)

    structural_edges: set[int] = set()
    boundaries = []
    segments = []
    segment_start = 0
    for index in range(1, len(records)):
        previous, current = records[index - 1], records[index]
        left, right = identities[index - 1], identities[index]
        reasons = []
        if right[0] != left[0] + 1:
            reasons.append("sample_id_gap")
        if _frame_number(right[1]) != _frame_number(left[1]) + 1:
            reasons.append("frame_number_gap")
        if abs(right[2] - left[2] - EXPECTED_STEP_SECONDS) > STEP_TOLERANCE_SECONDS:
            reasons.append("video_time_gap")
        if abs(current["telemetry_interval_start"] - previous["telemetry_interval_end"]) > 1e-6:
            reasons.append("telemetry_interval_discontinuity")
        if current.get("segment") != previous.get("segment"):
            reasons.append("explicit_segment_change")
        if reasons:
            structural_edges.add(index)
        if not quality[index - 1] or not quality[index]:
            reasons.append("quality_gap_violation")
        if not eligible[index - 1] or not eligible[index]:
            reasons.append("label_ineligible")
        if reasons:
            segments.append((segment_start, index - 1))
            boundaries.append({"left_filtered_index": index - 1, "right_filtered_index": index,
                               "left_source_sample_id": left[0], "right_source_sample_id": right[0],
                               "left_frame": Path(left[1]).name, "right_frame": Path(right[1]).name,
                               "reasons": reasons})
            segment_start = index
    segments.append((segment_start, len(records) - 1))
    sequence_rows = []
    for segment_id, (begin, end) in enumerate(segments):
        for start in range(begin, end - SEQUENCE_LENGTH + 2):
            target_index = start + SEQUENCE_LENGTH - 1
            indices = list(range(start, target_index + 1))
            if not all(eligible[i] and quality[i] for i in indices):
                continue
            times = [identities[i][2] for i in indices]
            span = times[-1] - times[0]
            if abs(span - EXPECTED_SPAN_SECONDS) > SPAN_TOLERANCE_SECONDS:
                raise ValueError("Generated window span is outside 0.3-second tolerance")
            target = records[target_index]
            sequence_rows.append({
                "sequence_id": len(sequence_rows), "segment_id": segment_id,
                "filtered_indices": indices,
                "source_sample_ids": [identities[i][0] for i in indices],
                "frames": [records[i]["frame"] for i in indices],
                "frame_video_times_seconds": times, "sequence_span_seconds": span,
                "feature_refs": [dict(feature_refs[i]) for i in indices],
                "target_filtered_index": target_index, "target_source_sample_id": target["sample_id"],
                "target_action": copy.deepcopy(target["action"]),
                "mouse_label_source": target["mouse_label_source"],
                "raw_mouse_device_delta_captured": False, "future_outcome_used_as_feature": False,
                "automatic_action_allowed": False, "controls_modified": False,
            })
    rejection_counts: Counter[str] = Counter()
    for start in range(max(0, len(records) - SEQUENCE_LENGTH + 1)):
        indices = range(start, start + SEQUENCE_LENGTH)
        if any(edge in structural_edges for edge in range(start + 1, start + SEQUENCE_LENGTH)):
            rejection_counts["rejected_cross_boundary_windows"] += 1
        elif not all(eligible[i] for i in indices):
            rejection_counts["rejected_label_windows"] += 1
        elif not all(quality[i] for i in indices):
            rejection_counts["rejected_quality_windows"] += 1
    possible = max(0, len(records) - SEQUENCE_LENGTH + 1)
    if len(sequence_rows) + sum(rejection_counts.values()) != possible:
        raise ValueError("Internal window accounting mismatch")
    audit = {
        "format_version": 1, "dataset": session, "role": ROLES[session],
        "status": "temporal_sequence_preparation_passed", "source_record_count": len(records),
        "sequence_length": SEQUENCE_LENGTH, "expected_step_seconds": EXPECTED_STEP_SECONDS,
        "expected_span_seconds": EXPECTED_SPAN_SECONDS,
        "step_tolerance_seconds": STEP_TOLERANCE_SECONDS,
        "span_tolerance_seconds": SPAN_TOLERANCE_SECONDS,
        "max_telemetry_gap_seconds": MAX_TELEMETRY_GAP_SECONDS,
        "contiguous_segment_count": len(segments), "boundary_count": len(boundaries),
        "structural_boundary_count": len(structural_edges), "boundaries": boundaries,
        "possible_windows_without_boundary_checks": possible,
        "valid_sequence_count": len(sequence_rows), "label_eligible_record_count": sum(eligible),
        "label_ineligible_records": [{"filtered_index": i, "source_sample_id": identities[i][0],
                                      "frame": records[i]["frame"],
                                      "reason": records[i]["label_rejection_reason"]}
                                     for i, valid in enumerate(eligible) if not valid],
        "quality_ineligible_record_count": len(quality) - sum(quality),
        "rejected_cross_boundary_windows": rejection_counts["rejected_cross_boundary_windows"],
        "rejected_label_windows": rejection_counts["rejected_label_windows"],
        "rejected_quality_windows": rejection_counts["rejected_quality_windows"],
        "temporal_sequence_audit_passed": True, "dense_cache_index_is_not_temporal_identity": True,
        "training": False, "train_admission": False, "automatic_action_allowed": False,
        "controls_modified": False, "agent_control": False, "V8_consumed": False,
    }
    return sequence_rows, audit


def write_outputs(output_directory: Path, rows: list[dict[str, Any]], audit: dict[str, Any],
                  manifest: dict[str, Any]) -> dict[str, Any]:
    """Write validated metadata to an exclusively created directory."""
    reject_sealed_path(output_directory)
    compat = compatibility_manifest(manifest)
    if audit.get("dataset") != compat["dataset"] or audit.get("valid_sequence_count") != len(rows):
        raise ValueError("Audit/output dataset or count mismatch")
    if audit.get("temporal_sequence_audit_passed") is not True:
        raise ValueError("Cannot write outputs without a passed audit")
    # Serialize before claiming the destination so malformed values cannot leave
    # a partially written output directory.
    index_text = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    compat_text = json.dumps(compat, indent=2, allow_nan=False) + "\n"
    final_audit = copy.deepcopy(audit)
    final_audit.update({
        "sequence_index_file": str(output_directory.resolve() / "sequence_index_v1.jsonl"),
        "sequence_index_sha256": hashlib.sha256(index_text.encode()).hexdigest(),
        "cache_manifest_compat_file": str(output_directory.resolve() / "cache_manifest_compat_v1.json"),
        "cache_manifest_compat_sha256": hashlib.sha256(compat_text.encode()).hexdigest(),
    })
    audit_text = json.dumps(final_audit, indent=2, allow_nan=False) + "\n"
    payloads = {"sequence_index_v1.jsonl": index_text, "cache_manifest_compat_v1.json": compat_text,
                "sequence_audit_v1.json": audit_text}
    payloads["sha256s_temporal_v1.txt"] = "".join(
        f"{hashlib.sha256(value.encode()).hexdigest()}  {name}\n" for name, value in payloads.items())
    output_directory.mkdir(parents=False, exist_ok=False)
    for filename, value in payloads.items():
        with (output_directory / filename).open("x", encoding="utf-8") as handle:
            handle.write(value)
    return final_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=tuple(ROLES), required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, help="Create a new directory; omitted means dry run")
    args = parser.parse_args(argv)
    for path in (args.source_file, args.cache_directory, args.cache_manifest):
        reject_sealed_path(path)
    if args.output_directory is not None:
        reject_sealed_path(args.output_directory)
        if args.output_directory.exists() or args.output_directory.is_symlink():
            raise FileExistsError(f"Refusing existing output: {args.output_directory}")
    _verify_hash(args.source_file, args.source_sha256)
    _verify_hash(args.cache_manifest, args.cache_manifest_sha256)
    manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset") != args.session:
        raise ValueError("Requested session does not match cache dataset")
    records = read_jsonl(args.source_file)
    refs = load_cache_refs(records, args.cache_directory, manifest)
    rows, audit = build_sequences(records, refs, session=args.session)
    audit.update({"source_file": str(args.source_file.resolve()), "source_sha256": args.source_sha256,
                  "cache_directory": str(args.cache_directory.resolve()),
                  "cache_manifest": str(args.cache_manifest.resolve()),
                  "cache_manifest_sha256": args.cache_manifest_sha256,
                  "feature_hashes_verified": True, "record_hashes_verified": True,
                  "dry_run": args.output_directory is None})
    if args.output_directory is not None:
        audit = write_outputs(args.output_directory, rows, audit, manifest)
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
