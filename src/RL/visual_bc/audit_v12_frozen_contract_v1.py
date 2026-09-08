"""Independent read-only V12 verifier; does not import the freezer or builders."""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct


CONTRACT_SHA = "69b98f3c99356316d6e1b7135d6915d687910133372f5ccf1adea4f0ba6a8576"
FREEZE_SHA = "5f8b6fee53e0c44553b4f24fe3d5f74eba07d43a7e99bc635a5c363935154f11"
FREEZER_SHA = "1833adca0a8ee2d634460ee6a568a736c6ce962f1a0352f6400274dc8bbba89f"
BUNDLE_SHA = "8b3f58721cb0bb46683d2cd01f6b064d465a466e25c497321e349e8b1e532593"
EXCLUDED = {4641: "interval_ends_after_segment", 4647: "interval_starts_before_segment"}
FALSE_FLAGS = ("train_admission", "training_started", "agent_control", "automatic_action_allowed", "controls_modified", "V8_consumed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(path):
    path = Path(path).resolve()
    require(not re.search(r"(?:^|[^a-z0-9])v8(?:$|[^a-z0-9])", str(path).lower()), "V8 read forbidden")
    return path


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(safe(path).read_text())


def rows(path):
    return [json.loads(line) for line in safe(path).read_text().splitlines()]


def pinned(path, expected):
    require(sha(path) == expected, f"hash mismatch: {path}")


def inventory(directory, filename, expected_names):
    entries = {}
    for line in (directory / filename).read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        require(name in expected_names and name not in entries, f"unexpected inventory member: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", digest), "invalid digest")
        pinned(directory / name, digest)
        entries[name] = digest
    require(set(entries) == set(expected_names), "incomplete checksum inventory")
    return entries


def near(a, b, name, tolerance=1e-8):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, name)


def independent_action(interval):
    require(bool(interval), "empty action interval")
    for record in interval:
        require(record["input_available"] == 1, "input unavailable")
        for field in ("input_move_forward", "input_move_strafe", "input_jump", "view_yaw", "view_pitch"):
            require(math.isfinite(record[field]), f"nonfinite action: {field}")
    n = len(interval)
    result = {}
    for name, field in (("forward", "input_move_forward"), ("strafe", "input_move_strafe")):
        values = [r[field] for r in interval]
        result[name + "_axis_mean"] = sum(values)/n
        result[name + "_positive_fraction"] = sum(x > 1 for x in values)/n
        result[name + "_negative_fraction"] = sum(x < -1 for x in values)/n
    result.update(jump_fraction=sum(int(r["input_jump"]) != 0 for r in interval)/n,
                  view_yaw_delta_degrees=(interval[-1]["view_yaw"]-interval[0]["view_yaw"]+180)%360-180,
                  view_pitch_delta_degrees=interval[-1]["view_pitch"]-interval[0]["view_pitch"])
    result.update({key: None for key in ("attack_fraction", "secondary_attack_fraction", "crouch_fraction", "use_fraction")})
    return result


def audit(directory):
    directory = safe(directory)
    names = {"freeze_audit_v1.json", "freeze_v12_validation_contract_v1.py", "v12_validation_contract_v1.json", "sha256s_v12_validation_contract_v1.txt"}
    require({p.name for p in directory.iterdir()} == names, "frozen directory must contain exactly four files")
    inventory(directory, "sha256s_v12_validation_contract_v1.txt", names - {"sha256s_v12_validation_contract_v1.txt"})
    for name, digest in (("v12_validation_contract_v1.json", CONTRACT_SHA), ("freeze_audit_v1.json", FREEZE_SHA), ("freeze_v12_validation_contract_v1.py", FREEZER_SHA)):
        pinned(directory/name, digest)
    contract = read(directory / "v12_validation_contract_v1.json")
    frozen_audit = read(directory / "freeze_audit_v1.json")
    require(contract["status"] == "frozen_validation_contract" and contract["validation_dataset_frozen"] is True, "not frozen")
    for key in FALSE_FLAGS + ("training_authorized", "source_candidate_modified"):
        require(contract.get(key) is False, f"contract flag: {key}")
    require(contract["role"] == "calibration_validation_only", "role mismatch")
    require(contract["freeze_audit_sha256"] == FREEZE_SHA and contract["freeze_script"]["sha256"] == FREEZER_SHA, "freeze provenance mismatch")
    candidate = safe(contract["source_candidate_directory"])
    hashes = contract["source_candidate_file_sha256s"]
    require(len(hashes) == 8 and {p.name for p in candidate.iterdir()} == set(hashes), "candidate inventory changed")
    for name, digest in hashes.items():
        require(Path(name).name == name, "invalid candidate basename")
        pinned(candidate/name, digest)
    inventory(candidate, "sha256s_candidate_v1.txt", set(hashes)-{"sha256s_candidate_v1.txt"})
    inventory(candidate, "sha256s_temporal_v1.txt", {"sequence_index_v1.jsonl", "sequence_audit_v1.json", "cache_manifest_compat_v1.json"})
    bundle = hashlib.sha256("".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode()).hexdigest()
    require(bundle == BUNDLE_SHA == contract["source_candidate_bundle_sha256"] == frozen_audit["source_candidate_bundle_sha256"], "candidate bundle changed")
    require(hashes == frozen_audit["source_candidate_file_sha256s"], "freeze audit candidate mismatch")
    inputs = contract["provenance_inputs"]
    require(inputs == frozen_audit["verified_provenance_inputs"], "provenance maps differ")
    for item in [*inputs.values(), contract["DINO_manifest"], contract["label_script"], contract["sequence_script"]]:
        pinned(item["path"], item["sha256"])
    labels = rows(candidate / "aligned_samples_label_candidate_v1.jsonl")
    sequences = rows(candidate / "sequence_index_v1.jsonl")
    label_audit = read(candidate / "label_audit_v1.json")
    sequence_audit = read(candidate / "sequence_audit_v1.json")
    for data in (label_audit, sequence_audit):
        require(data["inputs"] == inputs and data["validation_dataset_frozen"] is False, "candidate provenance/status changed")
        for key in FALSE_FLAGS:
            require(data.get(key) is False, f"candidate flag: {key}")
    require(len(labels) == contract["source_record_count"] == 8995, "source count")
    require(len(sequences) == contract["sequence_count"] == 8987, "sequence count")
    ids = [r["sample_id"] for r in labels]
    expected_ids = [i for i in range(9000) if i not in range(4642,4647)]
    require(ids == expected_ids, "original source identities changed")
    exclusions = {r["sample_id"]: r["label_rejection_reason"] for r in labels if not r["label_eligible"]}
    require(exclusions == EXCLUDED, "unexpected label exclusions")
    require(sum(r["label_eligible"] for r in labels) == contract["label_eligible_count"] == 8993, "eligible count")

    # Derive segments and every label directly from raw recovered telemetry.
    alignment = read(inputs["alignment_manifest"]["path"])
    aligned = rows(Path(inputs["alignment_manifest"]["path"]).parent / "frame_to_telemetry_index_v1.jsonl")
    frame_rows = rows(inputs["frame_index"]["path"])
    require(len(aligned) == len(frame_rows) == len(labels), "frozen admission count")
    telemetry = rows(inputs["telemetry"]["path"])
    first = alignment["telemetry"]["capture_recovered_start_line"]
    capture = telemetry[first-1:alignment["telemetry"]["capture_recovered_end_line"]]
    resets = [i for i in range(1,len(capture)) if capture[i]["sample_time"] < capture[i-1]["sample_time"]]
    require(resets == [8748], "reset location changed")
    segments = [capture[:8748],capture[8748:]]
    require(first+resets[0] == 8848, "reset recovered pointer")
    reconstructed = 0
    for label, identity, frame in zip(labels, aligned, frame_rows):
        source_id = label["sample_id"]
        require(source_id == label["source_sample_id"] == identity["frame_index"] == frame["frame_index_0based"], "label/frame identity")
        require(Path(label["frame"]).name == frame["filename"] == f"frame_{source_id+1:06d}.jpg", "frame filename")
        near(label["frame_video_time_seconds"], source_id/10, "source grid")
        segment_id = identity["segment"]
        require(label["segment"] == segment_id, "segment identity")
        segment = segments[segment_id-1]
        offset = alignment["segments"][str(segment_id)]["offset_seconds"]
        segment_times = [r["sample_time"] for r in segment] if source_id in (0,4647) else segment_times
        start, end = source_id/10-offset, source_id/10-offset+0.1
        near(label["telemetry_interval_start"], start, "interval start")
        near(label["telemetry_interval_end"], end, "interval end")
        pointer = identity["telemetry_recovered_line"]
        near(telemetry[pointer-1]["sample_time"], identity["telemetry_sample_time"], "frozen pointer")
        require(identity["telemetry_original_source_line"] == pointer+233, "original pointer")
        reason = "interval_starts_before_segment" if start < segment_times[0] else "interval_ends_after_segment" if end > segment_times[-1] else None
        require(label["label_rejection_reason"] == reason, "label eligibility not independently reproduced")
        if reason:
            require(label["action"] is None, "ineligible action must remain absent")
            continue
        left, right = bisect.bisect_left(segment_times,start), bisect.bisect_left(segment_times,end)
        interval = segment[left:max(right,left+1)]
        require(label["action"] == independent_action(interval), "reconstructed action mismatch")
        reconstructed += 1

    # Verify every feature/record hash and source projection independently.
    manifest = read(contract["DINO_manifest"]["path"])
    cache = safe(Path(contract["DINO_manifest"]["path"]).parent)
    compat = read(candidate / "cache_manifest_compat_v1.json")
    aliases = {"start":"sample_start", "end":"sample_end", "count":"sample_count", "shape":"feature_shape", "dtype":"feature_dtype", "record_file":"records_file", "record_sha256":"records_sha256"}
    require(len(manifest["completed_chunks"]) == len(compat["completed_chunks"]) == 563, "chunk count")
    refs = []
    for chunk, adapted in zip(manifest["completed_chunks"], compat["completed_chunks"]):
        for key,value in chunk.items():
            require(adapted[key] == value, "compatibility changed native chunk")
        for key,alias in aliases.items():
            require(adapted[alias] == chunk[key], "compatibility alias mismatch")
        require(chunk["start"] == len(refs) and chunk["count"] == chunk["end"]-chunk["start"]+1, "dense chunk range")
        for filekey,hashkey in (("feature_file","feature_sha256"),("record_file","record_sha256")):
            require(Path(chunk[filekey]).name == chunk[filekey], "invalid chunk filename")
            pinned(cache/chunk[filekey],chunk[hashkey])
        with (cache/chunk["feature_file"]).open("rb") as handle:
            header_size = struct.unpack("<Q",handle.read(8))[0]
            require(header_size < 1048576, "invalid tensor header")
            header = json.loads(handle.read(header_size))["features"]
        require(header["dtype"] == "F16" and header["shape"] == [chunk["count"],41,384], "tensor shape")
        sidecars = rows(cache/chunk["record_file"])
        require(len(sidecars) == chunk["count"], "sidecar count")
        for offset,sidecar in enumerate(sidecars):
            label = labels[len(refs)]
            require(sidecar["cache_index"] == len(refs) and sidecar["feature_offset"] == offset, "cache addressing")
            for key in ("sample_id","source_sample_id","frame","frame_video_time_seconds"):
                require(sidecar[key] == label[key], f"sidecar source projection: {key}")
            require(sidecar["source_frame_index_0based"] == label["sample_id"] and sidecar["source_frame_number_1based"] == label["sample_id"]+1, "sidecar original frame")
            refs.append({"cache_index":len(refs),"feature_file":chunk["feature_file"],"feature_offset":offset})
    require(len(refs) == 8995, "feature reference count")
    expected_starts = []
    rejected = Counter()
    for start in range(len(labels)-3):
        window = labels[start:start+4]
        if any(b["sample_id"] != a["sample_id"]+1 or b["segment"] != a["segment"] for a,b in zip(window,window[1:])):
            rejected["structural"] += 1
        elif any(not r["label_eligible"] for r in window):
            rejected["label"] += 1
        else:
            expected_starts.append(start)
    require(dict(rejected) == {"structural":3,"label":2}, "window accounting")
    require(len(expected_starts) == len(sequences), "sequence coverage")
    for index,(start,sequence) in enumerate(zip(expected_starts,sequences)):
        window = labels[start:start+4]
        require(sequence["sequence_id"] == index and sequence["filtered_indices"] == list(range(start,start+4)), "sequence order")
        for key,source_key in (("source_sample_ids","sample_id"),("frames","frame"),("frame_video_times_seconds","frame_video_time_seconds")):
            require(sequence[key] == [r[source_key] for r in window], f"sequence identity: {key}")
        require(sequence["feature_refs"] == refs[start:start+4], "feature refs")
        require(sequence["target_filtered_index"] == start+3 and sequence["target_source_sample_id"] == window[-1]["sample_id"], "target identity")
        require(sequence["target_action"] == window[-1]["action"], "target action")
        near(sequence["sequence_span_seconds"], .3, "sequence span")
        for a,b in zip(window,window[1:]):
            near(b["frame_video_time_seconds"]-a["frame_video_time_seconds"],.1,"sequence cadence")
            near(a["telemetry_interval_end"], b["telemetry_interval_start"],"telemetry continuity")
        for key in ("raw_mouse_device_delta_captured","future_outcome_used_as_feature","automatic_action_allowed","controls_modified"):
            require(sequence[key] is False, f"sequence safety: {key}")
        require(sequence["mouse_label_source"] == "view_angle_delta", "mouse label source")
    require({p.name:sha(p) for p in candidate.iterdir()} == hashes, "candidate changed during audit")
    return {"status":"PASS", "error_count":0, "independent_of_freezer_and_builders":True,
            "contract_sha256":CONTRACT_SHA, "freeze_audit_sha256":FREEZE_SHA,
            "freezer_script_sha256":FREEZER_SHA, "candidate_bundle_sha256":bundle,
            "frozen_file_count":4, "candidate_file_count":8, "source_records":len(labels),
            "reconstructed_eligible_labels":reconstructed, "sequence_count":len(sequences),
            "feature_chunks_hashed":563, "record_chunks_hashed":563,
            "structural_windows_excluded":3, "label_windows_excluded":2,
            "label_exclusions":EXCLUDED, "source_candidate_modified":False,
            "tensor_finiteness_rescanned":False, "frozen_contract_directory":str(directory),
            "candidate_file_sha256s":hashes, "provenance_inputs":inputs,
            "known_reset_numbering_erratum_preserved":True,
            **{key:False for key in FALSE_FLAGS}, "training_authorized":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        require(not args.report.exists(), "report already exists")
    try:
        result = audit(args.contract_directory)
    except Exception as exc:
        result = {"status":"FAIL", "error_count":1, "error":str(exc), "V8_consumed":False}
    text = json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.report:
        with safe(args.report).open("x") as handle:
            handle.write(text)
    print(text)
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
