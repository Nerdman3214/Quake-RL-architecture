#!/usr/bin/env python3
"""Audit the frozen supervised protocol; explicitly launch one fresh run.

Without --train this module reads metadata and prints an audit. It imports no
model, CUDA, training runtime, or game-control code. A training invocation owns
one new output directory and starts the pinned v6 runtime once, without retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAIN_ORDER = ["V4", "V6", "V7", "V9", "V10", "V11b"]
SEQUENCE_COUNTS = {"V4": 656, "V6": 1283, "V7": 3172, "V9": 2408,
                   "V10": 2998, "V11b": 2997, "V5": 1190}
SOURCE_COUNTS = {"V4": 662, "V6": 1289, "V7": 3181, "V9": 2414,
                 "V10": 3004, "V11b": 3000, "V5": 1260}
RECIPE = {"epochs": 12, "patience": 4, "batch_size": 4, "learning_rate": 5e-5,
          "weight_decay": 5e-4, "seed": 20260822, "optimizer": "AdamW",
          "sampler": "equal_session_probability_mass_with_replacement",
          "event_oversampling": False, "repeat_canary": False,
          "initialization": "fresh seed-fixed TemporalVisualPolicyV2"}


def safe_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Expected an explicit filesystem path")
    path = Path(value)
    for candidate in (str(path), str(path.resolve())):
        if re.search(r"(?i)(?:^|[^a-z0-9])v8(?:$|[^a-z0-9])", candidate):
            raise ValueError("V8 is sealed: this launcher cannot open V8 paths")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with safe_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_bytes(path: Path, expected_sha: str) -> bytes:
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError(f"Invalid SHA256 pin for {path.name}")
    payload = safe_path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f"SHA256 mismatch: {path}")
    return payload


def _reference(reference: dict[str, Any]) -> tuple[Path, bytes]:
    if not isinstance(reference, dict):
        raise ValueError("Expected a pinned path/sha256 reference")
    path = safe_path(reference["path"])
    return path, _pinned_bytes(path, reference["sha256"])


def _json(payload: bytes, name: str) -> dict[str, Any]:
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {name}")
    return result


def _expect(actual: Any, expected: Any, message: str) -> None:
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise ValueError(message)


def _false_flags(value: dict[str, Any], names: tuple[str, ...], context: str) -> None:
    for key in names:
        if value.get(key) is not False:
            raise ValueError(f"{context}: {key} must remain false")


def _record_count(payload: bytes, name: str) -> int:
    count = 0
    for line in payload.splitlines():
        if not line.strip() or not isinstance(json.loads(line), dict):
            raise ValueError(f"Malformed JSONL record: {name}:{count + 1}")
        count += 1
    return count


def _manifest_chunks(manifest: dict[str, Any], cache_dir: Path, count: int,
                     *, loader_aliases: bool) -> list[dict[str, Any]]:
    _expect(manifest.get("status"), "cache_complete", "Feature cache is incomplete")
    _expect(manifest.get("feature_shape_per_frame"), [41, 384], "Unexpected feature shape")
    _expect(manifest.get("feature_dtype"), "float16", "Unexpected feature dtype")
    _false_flags(manifest, ("automatic_action_allowed", "controls_modified", "training"), "Feature cache")
    for key in ("cache_record_count", "total_source_records"):
        _expect(manifest.get(key), count, f"Cache {key} mismatch")
    if loader_aliases or "completed_sample_count" in manifest:
        _expect(manifest.get("completed_sample_count"), count, "Cache completed count mismatch")
    chunks = manifest.get("completed_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Missing completed cache chunks")
    _expect(manifest.get("completed_chunk_count"), len(chunks), "Cache chunk count mismatch")
    aliases = {"start": "sample_start", "end": "sample_end", "count": "sample_count",
               "shape": "feature_shape", "dtype": "feature_dtype",
               "record_file": "records_file", "record_sha256": "records_sha256"}
    normalized = []
    next_index = 0
    for chunk in chunks:
        item = dict(chunk)
        for native, alias in aliases.items():
            if native in item and alias in item and item[native] != item[alias]:
                raise ValueError(f"Contradictory cache aliases: {native}/{alias}")
            if loader_aliases and alias not in item:
                raise ValueError(f"Missing runtime cache alias: {alias}")
            if native not in item:
                if alias not in item:
                    raise ValueError(f"Missing chunk field: {native}")
                item[native] = item[alias]
        start, end, chunk_count = item["start"], item["end"], item["count"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end, chunk_count)):
            raise ValueError("Chunk indices/count must be integers")
        if chunk_count <= 0 or start != next_index or end != start + chunk_count - 1:
            raise ValueError("Inconsistent cache chunk coverage")
        _expect(item["shape"], [chunk_count, 41, 384], "Chunk shape mismatch")
        _expect(item["dtype"], "float16", "Chunk dtype mismatch")
        for key in ("feature_file", "record_file"):
            filename = item[key]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Chunk path must be a basename within its cache")
            path = safe_path(cache_dir / filename)
            if path.resolve().parent != cache_dir.resolve():
                raise ValueError("Chunk path escapes its cache directory")
        for key in ("feature_sha256", "record_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(item.get(key))) is None:
                raise ValueError("Missing chunk hash pin")
        normalized.append({key: item[key] for key in (
            "start", "end", "count", "shape", "dtype", "feature_file", "feature_sha256",
            "record_file", "record_sha256")})
        next_index = end + 1
    _expect(next_index, count, "Cache coverage does not match source count")
    return normalized


def validate_protocol(protocol_path: Path, protocol_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen recipe and metadata pins without loading any model."""
    payload = _pinned_bytes(safe_path(protocol_path), protocol_sha256)
    protocol = _json(payload, "protocol")
    _expect(protocol.get("format_version"), 1, "Unsupported protocol format")
    _expect(protocol.get("status"), "frozen_supervised_protocol", "Protocol is not frozen")
    if protocol.get("protocol_frozen") is not True or not protocol.get("authorization_basis"):
        raise ValueError("A frozen protocol with recorded authorization basis is required")
    _false_flags(protocol, ("training_started", "V8_consumed", "agent_control", "automatic_action_allowed"), "Protocol")
    train = protocol["train"]
    _expect(train.get("sessions"), TRAIN_ORDER, "TRAIN must contain the fixed six sessions; V12 is never TRAIN")
    _expect(train.get("session_counts"), {name: SEQUENCE_COUNTS[name] for name in TRAIN_ORDER}, "TRAIN per-session counts changed")
    _expect(train.get("sequence_count"), 13514, "TRAIN count changed")
    _false_flags(train, ("random_frame_split", "V12_admitted"), "TRAIN")
    for key, expected in RECIPE.items():
        _expect(protocol["recipe"].get(key), expected, f"Frozen recipe mismatch: {key}")
    validation = protocol["validation"]
    _expect(validation["V5"].get("sequence_count"), 1190, "V5 validation count changed")
    _expect(validation["V12"].get("sequence_count"), 8987, "V12 validation count changed")
    _false_flags(validation, ("pool_for_selection", "optimizer_uses_validation"), "Validation")
    _false_flags(validation["V12"], ("checkpoint_selection",), "V12")
    _expect(protocol["checkpoint_selection"].get("session"), "V5", "Only V5 selects checkpoints")
    _false_flags(protocol["checkpoint_selection"], ("V12_results_used", "V8_results_used"), "Checkpoint selection")
    _expect(protocol["calibration"].get("session"), "V12", "V12 is reserved for calibration")
    _expect(protocol["final_test"].get("session"), "V8", "Final-test designation changed")
    if protocol["final_test"].get("sealed_until_model_freeze") is not True:
        raise ValueError("V8 must remain sealed until model freeze")
    _false_flags(protocol["live"], ("current_authorization",), "Live control")

    trainer_path, trainer_bytes = _reference(protocol["trainer"])
    parent_path, parent_bytes = _reference(protocol["parent_trainer"])
    _expect(trainer_path.name, "train_temporal_policy_v2_2_runtime_v6.py", "Expected runtime v6")
    _expect(parent_path.name, "train_temporal_policy_v2_2_runtime_v5.py", "Expected preserved runtime v5")
    if parent_path.resolve().parent != trainer_path.resolve().parent:
        raise ValueError("Parent runtime must share the trainer directory")
    before = '["V4", "V6", "V7", "V9", "V10"],'
    after = '["V4", "V6", "V7", "V9", "V10", "V11b"],'
    parent_text = parent_bytes.decode()
    if parent_text.count(before) != 1 or trainer_bytes.decode() != parent_text.replace(before, after, 1):
        raise ValueError("Runtime v6 must contain only the frozen V11b run-manifest correction")
    dependencies = protocol.get("dependencies")
    expected_dependencies = {"train_temporal_policy_v2.py", "train_temporal_policy_v1.py"}
    if not isinstance(dependencies, list) or len(dependencies) != 2:
        raise ValueError("Both runtime dependencies must be pinned")
    dependency_names = set()
    for reference in dependencies:
        path, _ = _reference(reference)
        if path.resolve().parent != trainer_path.resolve().parent:
            raise ValueError("Pinned dependency is outside the trainer import directory")
        dependency_names.add(path.name)
    _expect(dependency_names, expected_dependencies, "Unexpected runtime dependency set")

    _, runtime_bytes = _reference(protocol["runtime_contract"])
    contract = _json(runtime_bytes, "runtime contract")
    _expect(contract.get("train_session_order"), TRAIN_ORDER, "Runtime TRAIN order changed")
    _expect(contract.get("train_sessions"), TRAIN_ORDER, "Runtime TRAIN membership changed")
    _expect(contract.get("validation_session"), "V5", "Runtime validation must remain V5")
    _expect(contract.get("untouched_final_test_session"), "V8", "Runtime final-test designation changed")
    _expect(contract.get("train_sequence_count"), 13514, "Runtime TRAIN count changed")
    _expect(contract.get("validation_sequence_count"), 1190, "Runtime validation count changed")
    _false_flags(contract, ("random_frame_split", "automatic_action_allowed", "behavioral_control_gate_passed",
                            "controls_modified", "V8_consumed"), "Runtime contract")
    specs = contract["sessions"]
    _expect(set(specs), set(TRAIN_ORDER + ["V5"]), "Runtime contains an unexpected session (V12/V8 cannot enter training)")
    session_audits = {}
    for name in TRAIN_ORDER + ["V5"]:
        spec = specs[name]
        _expect(spec.get("role"), "validation" if name == "V5" else "train", f"{name}: runtime role mismatch")
        _expect(spec.get("sequence_count"), SEQUENCE_COUNTS[name], f"{name}: sequence count mismatch")
        _expect(spec.get("source_record_count"), SOURCE_COUNTS[name], f"{name}: source count mismatch")
        _false_flags(spec, ("automatic_action_allowed", "controls_modified", "future_outcome_used_as_feature"), name)
        cache_dir = safe_path(spec["cache_directory"])
        pinned = {}
        for path_key, hash_key in (("source_file", "source_sha256"),
                                   ("sequence_index_file", "sequence_index_sha256"),
                                   ("cache_manifest", "cache_manifest_sha256"),
                                   ("DINO_manifest", "DINO_manifest_sha256"),
                                   ("sequence_audit_file", "sequence_audit_sha256")):
            pinned[path_key] = _pinned_bytes(safe_path(spec[path_key]), spec[hash_key])
        _expect(_record_count(pinned["source_file"], name + " source"), SOURCE_COUNTS[name], f"{name}: actual source count mismatch")
        _expect(_record_count(pinned["sequence_index_file"], name + " index"), SEQUENCE_COUNTS[name], f"{name}: actual sequence count mismatch")
        manifest = _json(pinned["cache_manifest"], name + " cache")
        native = _json(pinned["DINO_manifest"], name + " native DINO cache")
        chunks = _manifest_chunks(manifest, cache_dir, SOURCE_COUNTS[name], loader_aliases=True)
        native_chunks = _manifest_chunks(native, cache_dir, SOURCE_COUNTS[name], loader_aliases=False)
        _expect(chunks, native_chunks, f"{name}: compatibility manifest changed native feature/record references")
        for current in (manifest, native):
            _expect(current.get("source_sha256"), spec["source_sha256"], f"{name}: cache/source SHA binding mismatch")
            _expect(safe_path(current["source_file"]).resolve(), safe_path(spec["source_file"]).resolve(), f"{name}: cache/source path binding mismatch")
        sequence_audit = _json(pinned["sequence_audit_file"], name + " sequence audit")
        if sequence_audit.get("temporal_sequence_audit_passed") is not True:
            raise ValueError(f"{name}: temporal audit has not passed")
        _expect(sequence_audit.get("source_record_count"), SOURCE_COUNTS[name], f"{name}: sequence-audit source count mismatch")
        _expect(sequence_audit.get("valid_sequence_count"), SEQUENCE_COUNTS[name], f"{name}: sequence-audit count mismatch")
        session_audits[name] = {"role": spec["role"], "source_count": SOURCE_COUNTS[name],
                                "sequence_count": SEQUENCE_COUNTS[name], "chunk_count": len(chunks),
                                "source_sha256": spec["source_sha256"],
                                "sequence_index_sha256": spec["sequence_index_sha256"],
                                "cache_manifest_sha256": spec["cache_manifest_sha256"],
                                "DINO_manifest_sha256": spec["DINO_manifest_sha256"],
                                "sequence_audit_sha256": spec["sequence_audit_sha256"]}

    _, v12_bytes = _reference(protocol["v12_contract"])
    v12 = _json(v12_bytes, "V12 frozen validation contract")
    _expect(v12.get("dataset"), "V12", "Unexpected calibration dataset")
    _expect(v12.get("role"), "calibration_validation_only", "V12 must never become TRAIN")
    _expect(v12.get("status"), "frozen_validation_contract", "V12 validation contract is not frozen")
    if v12.get("validation_dataset_frozen") is not True:
        raise ValueError("V12 validation dataset must be frozen")
    _expect(v12.get("sequence_count"), 8987, "V12 frozen sequence count mismatch")
    _expect(v12.get("source_record_count"), 8995, "V12 frozen source count mismatch")
    _false_flags(v12, ("train_admission", "training_authorized", "training_started", "V8_consumed",
                      "agent_control", "automatic_action_allowed", "controls_modified"), "V12")
    audit = {
        "format_version": 1, "status": "supervised_preflight_passed", "preflight_passed": True,
        "protocol": str(protocol_path.resolve()), "protocol_sha256": protocol_sha256,
        "launcher": str(Path(__file__).resolve()), "launcher_sha256": sha256_file(Path(__file__)),
        "trainer": protocol["trainer"], "runtime_contract": protocol["runtime_contract"],
        "v12_contract": protocol["v12_contract"], "train_session_order": TRAIN_ORDER,
        "train_sequence_count": 13514, "validation_session": "V5", "validation_sequence_count": 1190,
        "sessions": session_audits, "recipe": protocol["recipe"],
        "v12_payloads_loaded": False, "V8_consumed": False, "agent_control": False,
        "automatic_action_allowed": False, "training_started": False,
        "feature_payload_validation": "Pinned FrozenSequenceSession verifies feature hashes, shapes and finiteness on load",
        "python_executable": sys.executable,
    }
    return protocol, audit


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(protocol_path: Path, protocol_sha256: str, *, train: bool = False,
        output_directory: Path | None = None) -> int:
    if train and output_directory is None:
        raise ValueError("--train requires a new --output-directory")
    if not train and output_directory is not None:
        raise ValueError("Audit-only mode writes no files; --output-directory requires --train")
    if output_directory is not None:
        output_directory = safe_path(output_directory).resolve()
        if output_directory.exists() or output_directory.is_symlink():
            raise FileExistsError(f"Refusing existing output directory: {output_directory}")
    protocol, audit = validate_protocol(protocol_path, protocol_sha256)
    if not train:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    if protocol.get("training_authorized") is not True:
        raise ValueError("Full persistent supervised training is not authorized by this protocol")
    assert output_directory is not None
    # Re-read the exact pinned protocol before saving its original bytes.
    protocol_bytes = _pinned_bytes(protocol_path, protocol_sha256)
    runtime_path = safe_path(protocol["runtime_contract"]["path"]).resolve()
    trainer_path = safe_path(protocol["trainer"]["path"]).resolve()
    command = [sys.executable, "-u", str(trainer_path), "--train", "--contract", str(runtime_path),
               "--output-dir", str(output_directory / "training"), "--epochs", "12",
               "--patience", "4", "--batch-size", "4", "--learning-rate", "5e-05"]
    audit.update({"mode": "train", "command": command, "launch_requested_at_utc": _now(),
                  "output_directory": str(output_directory), "automatic_retry": False})
    output_directory.mkdir(parents=False, exist_ok=False)
    with (output_directory / "protocol.json").open("xb") as handle:
        handle.write(protocol_bytes)
    _write_json_new(output_directory / "launch_audit_v1.json", audit)
    log_path = output_directory / "training_stdout.log"
    print(f"preflight_passed=True\ntraining_log={log_path}\noutput_directory={output_directory}", flush=True)
    environment = dict(os.environ)
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    process = None
    result: dict[str, Any] = {"format_version": 1, "protocol_sha256": protocol_sha256,
                              "started_at_utc": _now(), "automatic_retry": False,
                              "V8_consumed": False, "agent_control": False,
                              "automatic_action_allowed": False}
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=trainer_path.parent, env=environment, start_new_session=True)
            result["pid"] = process.pid
            result["trainer_return_code"] = process.wait()
        result["status"] = "training_process_completed" if process.returncode == 0 else "training_process_failed"
        if process.returncode == 0:
            training_dir = output_directory / "training"
            manifest = _json((training_dir / "run_manifest.json").read_bytes(), "training run manifest")
            _expect(manifest.get("train_sessions"), TRAIN_ORDER, "Training manifest omitted a TRAIN session")
            _expect(manifest.get("train_sequence_count"), 13514, "Training manifest count mismatch")
            _expect(manifest.get("validation_session"), "V5", "Training manifest validation mismatch")
            result["best_checkpoint_sha256"] = sha256_file(training_dir / "best.pt")
            result["last_checkpoint_sha256"] = sha256_file(training_dir / "last.pt")
        return process.returncode
    except BaseException as exc:
        result.update(status="training_launch_or_verification_failed", error_type=type(exc).__name__, error=str(exc))
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process is not None:
            result["trainer_return_code"] = process.returncode
        raise
    finally:
        result["finished_at_utc"] = _now()
        _write_json_new(output_directory / "training_result_v1.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--train", action="store_true", help="Start one new run after passed preflight")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    return run(args.protocol, args.protocol_sha256, train=args.train, output_directory=args.output_directory)


if __name__ == "__main__":
    raise SystemExit(main())
