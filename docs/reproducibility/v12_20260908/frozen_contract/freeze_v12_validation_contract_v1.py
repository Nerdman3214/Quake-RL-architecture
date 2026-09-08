from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


candidate = Path(sys.argv[1]).resolve()
feature_cache = Path(sys.argv[2]).resolve()
label_script = Path(sys.argv[3]).resolve()
sequence_script = Path(sys.argv[4]).resolve()
output = Path(sys.argv[5]).resolve()


EXPECTED_CANDIDATE_FILES = {
    "aligned_samples_label_candidate_v1.jsonl",
    "alignment_line_number_erratum_v1.json",
    "cache_manifest_compat_v1.json",
    "label_audit_v1.json",
    "sequence_audit_v1.json",
    "sequence_index_v1.jsonl",
    "sha256s_candidate_v1.txt",
    "sha256s_temporal_v1.txt",
}

EXPECTED_V2_SHA = (
    "07616491b73ee13fab51f8e63372da61596a14d09b085c8afa87ecb057518d2c"
)

EXPECTED_SEQUENCE_SCRIPT_SHA = (
    "f3b6a8b057035df6fe6ad39166bafe3a5ed0faf9155dbbf7cd262659e500257c"
)

EXPECTED_DINO_MANIFEST_SHA = (
    "53887ff6967a869380da22d2d9c1dfdd0879942af1d1e7d5495c68973d0c9076"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    count = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"blank JSONL line: {path}:{line_number}"
                )

            json.loads(line)
            count += 1

    return count


def parse_inventory(path: Path) -> dict[str, str]:
    values = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise ValueError(
                f"malformed checksum line: "
                f"{path.name}:{line_number}"
            )

        digest, name = parts
        name = name.lstrip("*")

        if name in values:
            raise ValueError(
                f"duplicate checksum entry: {name}"
            )

        values[name] = digest

    return values


def verify_inventory(
    inventory_path: Path,
    expected_names: set[str],
) -> dict[str, str]:
    values = parse_inventory(inventory_path)

    if set(values) != expected_names:
        raise ValueError(
            f"checksum inventory mismatch: "
            f"{inventory_path.name}"
        )

    for name, expected_sha in values.items():
        path = candidate / name

        if not path.is_file():
            raise FileNotFoundError(path)

        actual_sha = sha256_file(path)

        if actual_sha != expected_sha:
            raise ValueError(
                f"checksum mismatch: {name}"
            )

    return values


if output.exists():
    raise FileExistsError(
        f"refusing to overwrite frozen contract: {output}"
    )

if not candidate.is_dir():
    raise FileNotFoundError(candidate)

actual_candidate_files = {
    path.name
    for path in candidate.iterdir()
    if path.is_file()
}

if actual_candidate_files != EXPECTED_CANDIDATE_FILES:
    raise ValueError(
        "review-candidate file inventory changed"
    )


# ------------------------------------------------------------------
# Verify existing candidate inventories
# ------------------------------------------------------------------

temporal_inventory = verify_inventory(
    candidate / "sha256s_temporal_v1.txt",
    {
        "sequence_index_v1.jsonl",
        "cache_manifest_compat_v1.json",
        "sequence_audit_v1.json",
    },
)

candidate_inventory = verify_inventory(
    candidate / "sha256s_candidate_v1.txt",
    EXPECTED_CANDIDATE_FILES
    - {"sha256s_candidate_v1.txt"},
)


# Hash ALL eight source candidate files, including both inventories.
candidate_file_hashes = {
    name: sha256_file(candidate / name)
    for name in sorted(EXPECTED_CANDIDATE_FILES)
}

bundle_text = "".join(
    f"{candidate_file_hashes[name]}  {name}\n"
    for name in sorted(candidate_file_hashes)
)

candidate_bundle_sha256 = sha256_bytes(
    bundle_text.encode("utf-8")
)


# ------------------------------------------------------------------
# Load and independently verify candidate claims
# ------------------------------------------------------------------

label_audit = load_json(
    candidate / "label_audit_v1.json"
)

sequence_audit = load_json(
    candidate / "sequence_audit_v1.json"
)

erratum = load_json(
    candidate / "alignment_line_number_erratum_v1.json"
)

label_count = count_jsonl(
    candidate / "aligned_samples_label_candidate_v1.jsonl"
)

sequence_count = count_jsonl(
    candidate / "sequence_index_v1.jsonl"
)


if label_count != 8995:
    raise ValueError(
        f"unexpected label record count: {label_count}"
    )

if sequence_count != 8987:
    raise ValueError(
        f"unexpected sequence count: {sequence_count}"
    )

if label_audit.get("label_eligible_count") != 8993:
    raise ValueError(
        "unexpected eligible-label count"
    )

if sequence_audit.get("valid_sequence_count") != 8987:
    raise ValueError(
        "sequence audit count mismatch"
    )

if sequence_audit.get(
    "temporal_sequence_audit_passed"
) is not True:
    raise ValueError(
        "source temporal audit did not pass"
    )

if sequence_audit.get(
    "dense_cache_index_is_not_temporal_identity"
) is not True:
    raise ValueError(
        "dense-cache temporal identity safeguard absent"
    )


# ------------------------------------------------------------------
# Exact two exclusions
# ------------------------------------------------------------------

expected_exclusions = [
    {
        "source_frame_number_1based": 4642,
        "source_sample_id": 4641,
        "reason": "interval_ends_after_segment",
    },
    {
        "source_frame_number_1based": 4648,
        "source_sample_id": 4647,
        "reason": "interval_starts_before_segment",
    },
]

actual_exclusions = []

for item in label_audit.get(
    "label_exclusions",
    [],
):
    actual_exclusions.append(
        {
            "source_frame_number_1based":
                item["source_frame_number_1based"],
            "source_sample_id":
                item["source_sample_id"],
            "reason":
                item["reason"],
        }
    )

if actual_exclusions != expected_exclusions:
    raise ValueError(
        "label exclusions differ from audited candidate"
    )


# ------------------------------------------------------------------
# Safety contract
# ------------------------------------------------------------------

for name, obj in (
    ("labels", label_audit),
    ("sequences", sequence_audit),
):
    if obj.get("role") != "calibration_validation_only":
        raise ValueError(
            f"{name}: invalid V12 role"
        )

    if obj.get("validation_dataset_frozen") is not False:
        raise ValueError(
            f"{name}: source candidate unexpectedly frozen"
        )

    for field in (
        "train_admission",
        "training_started",
        "agent_control",
        "automatic_action_allowed",
        "controls_modified",
        "V8_consumed",
    ):
        if obj.get(field) is not False:
            raise ValueError(
                f"{name}: unsafe field {field}"
            )

if sequence_audit.get("training") is not False:
    raise ValueError(
        "source candidate says training occurred"
    )


# ------------------------------------------------------------------
# Bind scripts
# ------------------------------------------------------------------

label_script_sha = sha256_file(label_script)
sequence_script_sha = sha256_file(sequence_script)

if label_script_sha != EXPECTED_V2_SHA:
    raise ValueError(
        "V12 label script SHA changed"
    )

if sequence_script_sha != EXPECTED_SEQUENCE_SCRIPT_SHA:
    raise ValueError(
        "temporal sequence script SHA changed"
    )


# ------------------------------------------------------------------
# Bind DINO manifest and all provenance input hashes
# ------------------------------------------------------------------

dino_manifest = feature_cache / "manifest_v1.json"

if sha256_file(dino_manifest) != EXPECTED_DINO_MANIFEST_SHA:
    raise ValueError(
        "V12 DINO manifest SHA changed"
    )


inputs = sequence_audit.get("inputs", {})

verified_inputs = {}

for key, value in inputs.items():
    path = Path(value["path"]).resolve()
    expected_sha = value["sha256"]

    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = sha256_file(path)

    if actual_sha != expected_sha:
        raise ValueError(
            f"provenance input SHA mismatch: {key}"
        )

    verified_inputs[key] = {
        "path": str(path),
        "sha256": actual_sha,
    }


# ------------------------------------------------------------------
# Reset numbering erratum
# ------------------------------------------------------------------

observations = erratum.get(
    "observations",
    [],
)

if len(observations) != 1:
    raise ValueError(
        "unexpected reset-line erratum count"
    )

if observations[0].get(
    "manifest_numbering_mismatch"
) is not True:
    raise ValueError(
        "expected reset numbering erratum missing"
    )

if erratum.get(
    "frozen_inputs_modified"
) is not False:
    raise ValueError(
        "erratum claims frozen inputs were modified"
    )


# ------------------------------------------------------------------
# Prepare freeze evidence
# ------------------------------------------------------------------

freezer_script_bytes = Path(
    __file__
).read_bytes()

freezer_script_sha = sha256_bytes(
    freezer_script_bytes
)


freeze_audit = {
    "format_version": 1,
    "dataset": "V12",
    "audit_name":
        "v12_validation_freeze_precheck_v1",
    "freeze_precheck_pass": True,

    "source_candidate_directory":
        str(candidate),

    "source_candidate_file_count": 8,
    "source_candidate_file_sha256s":
        candidate_file_hashes,

    "source_candidate_bundle_sha256":
        candidate_bundle_sha256,

    "label_record_count": 8995,
    "label_eligible_count": 8993,
    "sequence_count": 8987,

    "label_exclusions":
        expected_exclusions,

    "sequence_length": 4,
    "expected_step_seconds": 0.1,
    "expected_span_seconds": 0.3,

    "structural_gap": {
        "last_source_id_before_gap": 4641,
        "first_source_id_after_gap": 4647,
        "rejected_source_ids": [
            4642,
            4643,
            4644,
            4645,
            4646,
        ],
        "cross_gap_sequences_allowed": False,
    },

    "reset_line_number_erratum":
        observations,

    "scripts": {
        "label_script": {
            "path": str(label_script),
            "sha256": label_script_sha,
        },
        "sequence_script": {
            "path": str(sequence_script),
            "sha256": sequence_script_sha,
        },
        "freeze_script_sha256":
            freezer_script_sha,
    },

    "DINO_manifest": {
        "path": str(dino_manifest.resolve()),
        "sha256": EXPECTED_DINO_MANIFEST_SHA,
    },

    "verified_provenance_inputs":
        verified_inputs,

    "source_candidate_modified": False,

    "safety": {
        "role": "calibration_validation_only",
        "train_admission": False,
        "training_started": False,
        "training_authorized": False,
        "agent_control": False,
        "automatic_action_allowed": False,
        "controls_modified": False,
        "V8_consumed": False,
    },
}


freeze_audit_text = (
    json.dumps(
        freeze_audit,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n"
)

freeze_audit_sha = sha256_bytes(
    freeze_audit_text.encode("utf-8")
)


# ------------------------------------------------------------------
# Build frozen validation contract
# ------------------------------------------------------------------

contract = {
    "format_version": 1,

    "contract_name":
        "v12_calibration_validation_frozen_v1",

    "dataset": "V12",

    "role":
        "calibration_validation_only",

    "status":
        "frozen_validation_contract",

    "validation_dataset_frozen": True,

    "source_candidate_status":
        "review_candidate",

    "source_candidate_directory":
        str(candidate),

    "source_candidate_bundle_sha256":
        candidate_bundle_sha256,

    "source_candidate_file_sha256s":
        candidate_file_hashes,

    "freeze_audit_file":
        str(output / "freeze_audit_v1.json"),

    "freeze_audit_sha256":
        freeze_audit_sha,

    "label_contract":
        label_audit["label_contract"],

    "source_record_count": 8995,

    "label_eligible_count": 8993,

    "sequence_count": 8987,

    "sequence_length": 4,

    "expected_step_seconds": 0.1,

    "expected_span_seconds": 0.3,

    "label_exclusions":
        expected_exclusions,

    "DINO_manifest": {
        "path": str(dino_manifest.resolve()),
        "sha256": EXPECTED_DINO_MANIFEST_SHA,
    },

    "label_script": {
        "path": str(label_script),
        "sha256": label_script_sha,
    },

    "sequence_script": {
        "path": str(sequence_script),
        "sha256": sequence_script_sha,
    },

    "freeze_script": {
        "file":
            str(
                output
                / "freeze_v12_validation_contract_v1.py"
            ),
        "sha256":
            freezer_script_sha,
    },

    "provenance_inputs":
        verified_inputs,

    "reset_line_number_erratum":
        observations,

    "structural_gap": {
        "last_source_id_before_gap": 4641,
        "first_source_id_after_gap": 4647,
        "rejected_source_ids": [
            4642,
            4643,
            4644,
            4645,
            4646,
        ],
        "cross_gap_sequences_allowed": False,
    },

    "source_candidate_modified": False,

    "train_admission": False,
    "training_started": False,
    "training_authorized": False,
    "agent_control": False,
    "automatic_action_allowed": False,
    "controls_modified": False,
    "V8_consumed": False,
}


contract_text = (
    json.dumps(
        contract,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n"
)

contract_sha = sha256_bytes(
    contract_text.encode("utf-8")
)


# ------------------------------------------------------------------
# Serialize everything BEFORE claiming output directory
# ------------------------------------------------------------------

payloads = {
    "freeze_v12_validation_contract_v1.py":
        freezer_script_bytes,

    "freeze_audit_v1.json":
        freeze_audit_text.encode("utf-8"),

    "v12_validation_contract_v1.json":
        contract_text.encode("utf-8"),
}


inventory_text = "".join(
    f"{sha256_bytes(payloads[name])}  {name}\n"
    for name in sorted(payloads)
)

payloads[
    "sha256s_v12_validation_contract_v1.txt"
] = inventory_text.encode("utf-8")


# ------------------------------------------------------------------
# Write NEW frozen artifact only
# ------------------------------------------------------------------

output.mkdir(
    parents=False,
    exist_ok=False,
)

for name, contents in payloads.items():
    path = output / name

    with path.open("xb") as handle:
        handle.write(contents)


print("=" * 72)
print("V12 VALIDATION FREEZE CONTRACT V1")
print("=" * 72)

print(
    "frozen_contract_directory =",
    output,
)

print(
    "source_candidate_bundle_sha256 =",
    candidate_bundle_sha256,
)

print(
    "freeze_audit_sha256 =",
    freeze_audit_sha,
)

print(
    "contract_sha256 =",
    contract_sha,
)

print(
    "freeze_script_sha256 =",
    freezer_script_sha,
)

print(
    "source_candidate_modified = False"
)

print(
    "validation_dataset_frozen = True"
)

print(
    "role = calibration_validation_only"
)

print(
    "sequence_count = 8987"
)

print(
    "train_admission = False"
)

print(
    "training_authorized = False"
)

print(
    "V8_consumed = False"
)

print(
    "freeze_contract_write_pass = True"
)
