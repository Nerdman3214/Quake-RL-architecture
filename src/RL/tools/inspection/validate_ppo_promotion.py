#!/usr/bin/env python3
"""Validate a PPO checkpoint promotion without modifying artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from RL.training.ppo import (
    load_ppo_training_checkpoint,
)


@dataclass(frozen=True)
class PPOPromotionValidationResult:
    """Summary of a successfully validated PPO promotion."""

    source_checkpoint_path: Path
    before_checkpoint_path: Path
    promoted_checkpoint_path: Path
    audit_path: Path
    source_optimizer_step_count: int
    before_optimizer_step_count: int
    promoted_optimizer_step_count: int
    attempt_count: int
    accepted_attempt_index: int
    changed_state_tensor_names: tuple[str, ...]
    maximum_absolute_parameter_change: float
    confirmed_death_reward: float
    respawn_detected: bool
    respawn_inferred: bool
    respawn_signal_reason: str | None
    source_progress: Mapping[str, Any]
    promoted_progress: Mapping[str, Any]

    @property
    def changed_state_tensor_count(self) -> int:
        return len(
            self.changed_state_tensor_names
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "source_checkpoint_path": str(
                self.source_checkpoint_path
            ),
            "before_checkpoint_path": str(
                self.before_checkpoint_path
            ),
            "promoted_checkpoint_path": str(
                self.promoted_checkpoint_path
            ),
            "audit_path": str(
                self.audit_path
            ),
            "source_optimizer_step_count": (
                self.source_optimizer_step_count
            ),
            "before_optimizer_step_count": (
                self.before_optimizer_step_count
            ),
            "promoted_optimizer_step_count": (
                self.promoted_optimizer_step_count
            ),
            "attempt_count": self.attempt_count,
            "accepted_attempt_index": (
                self.accepted_attempt_index
            ),
            "changed_state_tensor_count": (
                self.changed_state_tensor_count
            ),
            "changed_state_tensor_names": list(
                self.changed_state_tensor_names
            ),
            "maximum_absolute_parameter_change": (
                self.maximum_absolute_parameter_change
            ),
            "confirmed_death_reward": (
                self.confirmed_death_reward
            ),
            "respawn_detected": (
                self.respawn_detected
            ),
            "respawn_inferred": (
                self.respawn_inferred
            ),
            "respawn_signal_reason": (
                self.respawn_signal_reason
            ),
            "source_progress": dict(
                self.source_progress
            ),
            "promoted_progress": dict(
                self.promoted_progress
            ),
            "valid": True,
        }


def _require_file(
    value: str | Path,
    *,
    field_name: str,
) -> Path:
    path = Path(value)

    if not path.is_file():
        raise FileNotFoundError(
            f"{field_name} does not exist: {path}"
        )

    return path


def _validate_distinct_paths(
    paths: Mapping[str, Path],
) -> None:
    resolved: dict[Path, str] = {}

    for field_name, path in paths.items():
        key = path.resolve()

        if key in resolved:
            raise ValueError(
                f"{field_name} and "
                f"{resolved[key]} must be different"
            )

        resolved[key] = field_name


def _require_mapping(
    value: object,
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{context} must be an object"
        )

    return value


def _required(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> Any:
    if key not in mapping:
        raise ValueError(
            f"{context} is missing {key!r}"
        )

    return mapping[key]


def _read_audit(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    try:
        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(
            "audit is not valid UTF-8"
        ) from error

    if not lines:
        raise ValueError(
            "audit must contain at least one record"
        )

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            raise ValueError(
                "audit contains a blank record at "
                f"line {line_number}"
            )

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                "audit contains invalid JSON at "
                f"line {line_number}"
            ) from error

        record_mapping = _require_mapping(
            record,
            context=(
                f"audit record at line {line_number}"
            ),
        )

        record_type = _required(
            record_mapping,
            "type",
            context=(
                f"audit record at line {line_number}"
            ),
        )

        if (
            not isinstance(record_type, str)
            or not record_type
        ):
            raise ValueError(
                "audit record type must be "
                f"nonempty at line {line_number}"
            )

        data = _require_mapping(
            _required(
                record_mapping,
                "data",
                context=(
                    "audit record at line "
                    f"{line_number}"
                ),
            ),
            context=(
                "audit record data at line "
                f"{line_number}"
            ),
        )

        records.append(
            {
                "type": record_type,
                "data": dict(data),
            }
        )

    return records


def _same_path(
    recorded: object,
    expected: Path,
) -> bool:
    if not isinstance(
        recorded,
        (
            str,
            Path,
        ),
    ):
        return False

    return (
        Path(recorded).resolve()
        == expected.resolve()
    )


def _state_copy(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        .cpu()
        .clone()
        for name, tensor
        in model.state_dict().items()
    }


def _states_equal(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> bool:
    return (
        set(first) == set(second)
        and all(
            torch.equal(
                first[name],
                second[name],
            )
            for name in first
        )
    )


def _state_difference(
    source: Mapping[str, torch.Tensor],
    promoted: Mapping[str, torch.Tensor],
) -> tuple[
    tuple[str, ...],
    float,
]:
    if set(source) != set(promoted):
        raise ValueError(
            "source and promoted model state "
            "fields differ"
        )

    changed: list[str] = []
    maximum_change = 0.0

    for name in source:
        old_tensor = source[name]
        new_tensor = promoted[name]

        if torch.equal(
            old_tensor,
            new_tensor,
        ):
            continue

        changed.append(name)

        difference = torch.max(
            torch.abs(
                new_tensor.to(
                    dtype=torch.float64
                )
                - old_tensor.to(
                    dtype=torch.float64
                )
            )
        )

        maximum_change = max(
            maximum_change,
            float(difference.item()),
        )

    return (
        tuple(changed),
        maximum_change,
    )


def _progress_record(
    checkpoint: object,
    *,
    context: str,
) -> dict[str, Any]:
    progress = getattr(
        checkpoint,
        "progress",
        None,
    )

    if progress is None or not hasattr(
        progress,
        "to_record",
    ):
        raise ValueError(
            f"{context} has invalid progress"
        )

    record = progress.to_record()

    return dict(
        _require_mapping(
            record,
            context=f"{context} progress",
        )
    )


def _require_progress_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    integer_fields = (
        "rollout_count",
        "environment_step_count",
        "completed_episode_count",
    )

    for field_name in integer_fields:
        if actual.get(field_name) != expected.get(
            field_name
        ):
            raise ValueError(
                f"{context} {field_name} differs"
            )

    actual_reward = float(
        _required(
            actual,
            "cumulative_reward",
            context=context,
        )
    )

    expected_reward = float(
        _required(
            expected,
            "cumulative_reward",
            context=context,
        )
    )

    if not math.isclose(
        actual_reward,
        expected_reward,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{context} cumulative_reward differs"
        )


def validate_ppo_promotion(
    source_checkpoint_path: str | Path,
    before_checkpoint_path: str | Path,
    promoted_checkpoint_path: str | Path,
    audit_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PPOPromotionValidationResult:
    """Validate checkpoints and a completed promotion audit."""

    source_path = _require_file(
        source_checkpoint_path,
        field_name="source checkpoint",
    )

    before_path = _require_file(
        before_checkpoint_path,
        field_name="before checkpoint",
    )

    promoted_path = _require_file(
        promoted_checkpoint_path,
        field_name="promoted checkpoint",
    )

    resolved_audit_path = _require_file(
        audit_path,
        field_name="audit",
    )

    _validate_distinct_paths(
        {
            "source checkpoint": source_path,
            "before checkpoint": before_path,
            "promoted checkpoint": promoted_path,
            "audit": resolved_audit_path,
        }
    )

    records = _read_audit(
        resolved_audit_path
    )

    if records[0]["type"] != (
        "death_aware_session_started"
    ):
        raise ValueError(
            "audit must begin with "
            "death_aware_session_started"
        )

    if records[-1]["type"] != (
        "death_aware_session_completed"
    ):
        raise ValueError(
            "audit must end with "
            "death_aware_session_completed"
        )

    start = records[0]["data"]
    final = records[-1]["data"]

    attempts = [
        record["data"]
        for record in records
        if record["type"]
        == "death_aware_attempt_completed"
    ]

    if not attempts:
        raise ValueError(
            "audit contains no completed attempts"
        )

    attempt_indices = [
        int(
            _required(
                attempt,
                "attempt_index",
                context="attempt record",
            )
        )
        for attempt in attempts
    ]

    if attempt_indices != list(
        range(len(attempts))
    ):
        raise ValueError(
            "attempt indices must be contiguous "
            "and begin at zero"
        )

    accepted_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("accepted") is True
    ]

    if len(accepted_attempts) != 1:
        raise ValueError(
            "audit must contain exactly one "
            "accepted attempt"
        )

    accepted = accepted_attempts[0]

    accepted_index = int(
        _required(
            accepted,
            "attempt_index",
            context="accepted attempt",
        )
    )

    promoted_records = [
        record["data"]
        for record in records
        if record["type"]
        == "death_aware_attempt_promoted"
    ]

    if len(promoted_records) != 1:
        raise ValueError(
            "audit must contain exactly one "
            "promotion record"
        )

    promotion = promoted_records[0]

    source = load_ppo_training_checkpoint(
        source_path,
        device=device,
    )

    before = load_ppo_training_checkpoint(
        before_path,
        device=device,
    )

    promoted = load_ppo_training_checkpoint(
        promoted_path,
        device=device,
    )

    source_step = int(
        source.trainer.optimizer_step_count
    )
    before_step = int(
        before.trainer.optimizer_step_count
    )
    promoted_step = int(
        promoted.trainer.optimizer_step_count
    )

    if before_step != source_step:
        raise ValueError(
            "before checkpoint optimizer step "
            "does not match source"
        )

    if promoted_step != source_step + 1:
        raise ValueError(
            "promoted checkpoint optimizer step "
            "must equal source plus one"
        )

    source_progress = _progress_record(
        source,
        context="source checkpoint",
    )

    before_progress = _progress_record(
        before,
        context="before checkpoint",
    )

    promoted_progress = _progress_record(
        promoted,
        context="promoted checkpoint",
    )

    _require_progress_equal(
        before_progress,
        source_progress,
        context="before checkpoint progress",
    )

    source_state = _state_copy(
        source.model
    )

    before_state = _state_copy(
        before.model
    )

    promoted_state = _state_copy(
        promoted.model
    )

    if not _states_equal(
        source_state,
        before_state,
    ):
        raise ValueError(
            "before checkpoint model weights "
            "do not match source"
        )

    (
        changed_names,
        maximum_change,
    ) = _state_difference(
        source_state,
        promoted_state,
    )

    if not changed_names:
        raise ValueError(
            "promoted checkpoint contains no "
            "model weight changes"
        )

    config = _require_mapping(
        _required(
            start,
            "config",
            context="session start",
        ),
        context="session config",
    )

    step_config = _require_mapping(
        _required(
            config,
            "step_config",
            context="session config",
        ),
        context="death-aware step config",
    )

    death_threshold = float(
        _required(
            step_config,
            "death_reward_threshold",
            context="death-aware step config",
        )
    )

    configured_attempt_count = int(
        _required(
            config,
            "attempt_count",
            context="session config",
        )
    )

    if (
        configured_attempt_count
        < len(attempts)
    ):
        raise ValueError(
            "completed attempts exceed configured limit"
        )

    if int(
        _required(
            start,
            "source_optimizer_step_count",
            context="session start",
        )
    ) != source_step:
        raise ValueError(
            "session start source optimizer step differs"
        )

    if not _same_path(
        _required(
            start,
            "source_checkpoint",
            context="session start",
        ),
        source_path,
    ):
        raise ValueError(
            "session start source checkpoint path differs"
        )

    if not _same_path(
        _required(
            start,
            "before_checkpoint_path",
            context="session start",
        ),
        before_path,
    ):
        raise ValueError(
            "session start before checkpoint path differs"
        )

    if not _same_path(
        _required(
            start,
            "requested_promoted_checkpoint_path",
            context="session start",
        ),
        promoted_path,
    ):
        raise ValueError(
            "session start promoted checkpoint path differs"
        )

    initial_progress = _require_mapping(
        _required(
            start,
            "initial_progress",
            context="session start",
        ),
        context="session initial progress",
    )

    _require_progress_equal(
        initial_progress,
        source_progress,
        context="audit initial progress",
    )

    if accepted.get(
        "rejection_reasons"
    ) not in (
        [],
        (),
    ):
        raise ValueError(
            "accepted attempt contains "
            "rejection reasons"
        )

    if accepted.get(
        "death_detected"
    ) is not True:
        raise ValueError(
            "accepted attempt did not detect death"
        )

    if accepted.get(
        "death_reward_confirmed"
    ) is not True:
        raise ValueError(
            "accepted attempt did not confirm "
            "the death reward"
        )

    confirmed_reward = float(
        _required(
            accepted,
            "confirmed_death_reward",
            context="accepted attempt",
        )
    )

    if confirmed_reward > death_threshold:
        raise ValueError(
            "accepted death reward is above "
            "the configured threshold"
        )

    respawn_detected = bool(
        accepted.get("respawn_detected")
    )

    respawn_inferred = bool(
        accepted.get("respawn_inferred")
    )

    respawn_reason = accepted.get(
        "respawn_signal_reason"
    )

    require_respawn = bool(
        config.get(
            "require_respawn_evidence",
            True,
        )
    )

    if (
        require_respawn
        and not (
            respawn_detected
            or respawn_inferred
        )
    ):
        raise ValueError(
            "accepted attempt lacks respawn evidence"
        )

    if (
        respawn_inferred
        and respawn_reason
        != "second_death_proves_respawn"
    ):
        raise ValueError(
            "accepted inferred respawn reason "
            "is invalid"
        )

    if (
        respawn_inferred
        and float(
            _required(
                accepted,
                "post_respawn_reward",
                context="accepted attempt",
            )
        )
        > death_threshold
    ):
        raise ValueError(
            "inferred respawn does not contain "
            "a confirmed later death reward"
        )

    if int(
        _required(
            accepted,
            "optimizer_operations",
            context="accepted attempt",
        )
    ) != 1:
        raise ValueError(
            "accepted attempt must contain exactly "
            "one optimizer operation"
        )

    if int(
        _required(
            accepted,
            "source_optimizer_step_count",
            context="accepted attempt",
        )
    ) != source_step:
        raise ValueError(
            "accepted attempt source optimizer "
            "step differs"
        )

    if int(
        _required(
            accepted,
            "ending_optimizer_step_count",
            context="accepted attempt",
        )
    ) != promoted_step:
        raise ValueError(
            "accepted attempt ending optimizer "
            "step differs"
        )

    if accepted.get(
        "weights_changed"
    ) is not True:
        raise ValueError(
            "accepted attempt reports unchanged weights"
        )

    if accepted.get(
        "stopped_on_kl"
    ) is True:
        raise ValueError(
            "accepted attempt stopped on KL"
        )

    recorded_changed_names = tuple(
        _required(
            accepted,
            "changed_state_tensor_names",
            context="accepted attempt",
        )
    )

    recorded_changed_count = int(
        _required(
            accepted,
            "changed_state_tensor_count",
            context="accepted attempt",
        )
    )

    if recorded_changed_count != len(
        changed_names
    ):
        raise ValueError(
            "audit changed-state tensor count differs"
        )

    if set(recorded_changed_names) != set(
        changed_names
    ):
        raise ValueError(
            "audit changed-state tensor names differ"
        )

    recorded_maximum_change = float(
        _required(
            accepted,
            "maximum_absolute_parameter_change",
            context="accepted attempt",
        )
    )

    if not math.isclose(
        recorded_maximum_change,
        maximum_change,
        rel_tol=1e-6,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "audit maximum parameter change differs"
        )

    steps = int(
        _required(
            accepted,
            "steps",
            context="accepted attempt",
        )
    )

    batch_steps = int(
        _required(
            accepted,
            "ppo_batch_steps",
            context="accepted attempt",
        )
    )

    if steps != batch_steps:
        raise ValueError(
            "accepted PPO batch size differs "
            "from rollout steps"
        )

    accepted_reward = float(
        _required(
            accepted,
            "total_reward",
            context="accepted attempt",
        )
    )

    expected_progress = {
        "rollout_count": (
            int(source_progress["rollout_count"])
            + 1
        ),
        "environment_step_count": (
            int(
                source_progress[
                    "environment_step_count"
                ]
            )
            + steps
        ),
        "completed_episode_count": (
            int(
                source_progress[
                    "completed_episode_count"
                ]
            )
            + 1
        ),
        "cumulative_reward": (
            float(
                source_progress[
                    "cumulative_reward"
                ]
            )
            + accepted_reward
        ),
    }

    _require_progress_equal(
        promoted_progress,
        expected_progress,
        context="promoted checkpoint progress",
    )

    if int(
        _required(
            final,
            "attempts_completed",
            context="session completion",
        )
    ) != len(attempts):
        raise ValueError(
            "session completion attempt count differs"
        )

    if final.get("promoted") is not True:
        raise ValueError(
            "session completion does not report promotion"
        )

    if int(
        _required(
            final,
            "accepted_attempt_index",
            context="session completion",
        )
    ) != accepted_index:
        raise ValueError(
            "session completion accepted attempt differs"
        )

    if int(
        _required(
            final,
            "source_optimizer_step_count",
            context="session completion",
        )
    ) != source_step:
        raise ValueError(
            "session completion source optimizer "
            "step differs"
        )

    if int(
        _required(
            final,
            "ending_optimizer_step_count",
            context="session completion",
        )
    ) != promoted_step:
        raise ValueError(
            "session completion ending optimizer "
            "step differs"
        )

    final_progress = _require_mapping(
        _required(
            final,
            "progress",
            context="session completion",
        ),
        context="session completion progress",
    )

    _require_progress_equal(
        final_progress,
        promoted_progress,
        context="session completion progress",
    )

    if not _same_path(
        _required(
            final,
            "promoted_checkpoint_path",
            context="session completion",
        ),
        promoted_path,
    ):
        raise ValueError(
            "session completion promoted path differs"
        )

    if int(
        _required(
            promotion,
            "attempt_index",
            context="promotion record",
        )
    ) != accepted_index:
        raise ValueError(
            "promotion record attempt index differs"
        )

    if int(
        _required(
            promotion,
            "optimizer_step_count",
            context="promotion record",
        )
    ) != promoted_step:
        raise ValueError(
            "promotion record optimizer step differs"
        )

    promotion_progress = _require_mapping(
        _required(
            promotion,
            "progress",
            context="promotion record",
        ),
        context="promotion record progress",
    )

    _require_progress_equal(
        promotion_progress,
        promoted_progress,
        context="promotion record progress",
    )

    if not _same_path(
        _required(
            promotion,
            "promoted_checkpoint_path",
            context="promotion record",
        ),
        promoted_path,
    ):
        raise ValueError(
            "promotion record checkpoint path differs"
        )

    before_metadata = _require_mapping(
        before.metadata,
        context="before checkpoint metadata",
    )

    if before_metadata.get(
        "session_phase"
    ) != "before":
        raise ValueError(
            "before checkpoint metadata phase differs"
        )

    if not _same_path(
        before_metadata.get(
            "source_checkpoint"
        ),
        source_path,
    ):
        raise ValueError(
            "before checkpoint metadata source differs"
        )

    if int(
        before_metadata.get(
            "source_optimizer_step_count",
            -1,
        )
    ) != source_step:
        raise ValueError(
            "before checkpoint metadata optimizer "
            "step differs"
        )

    promoted_metadata = _require_mapping(
        promoted.metadata,
        context="promoted checkpoint metadata",
    )

    if promoted_metadata.get(
        "session_phase"
    ) != "promoted":
        raise ValueError(
            "promoted checkpoint metadata phase differs"
        )

    if not _same_path(
        promoted_metadata.get(
            "source_checkpoint"
        ),
        source_path,
    ):
        raise ValueError(
            "promoted checkpoint metadata source differs"
        )

    if int(
        promoted_metadata.get(
            "accepted_attempt_index",
            -1,
        )
    ) != accepted_index:
        raise ValueError(
            "promoted checkpoint metadata accepted "
            "attempt differs"
        )

    if promoted_metadata.get(
        "confirmed_death"
    ) is not True:
        raise ValueError(
            "promoted metadata does not confirm death"
        )

    if not math.isclose(
        float(
            promoted_metadata.get(
                "confirmed_death_reward"
            )
        ),
        confirmed_reward,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "promoted metadata death reward differs"
        )

    if bool(
        promoted_metadata.get(
            "respawn_detected"
        )
    ) != respawn_detected:
        raise ValueError(
            "promoted metadata respawn detection differs"
        )

    if bool(
        promoted_metadata.get(
            "respawn_inferred"
        )
    ) != respawn_inferred:
        raise ValueError(
            "promoted metadata respawn inference differs"
        )

    if promoted_metadata.get(
        "respawn_signal_reason"
    ) != respawn_reason:
        raise ValueError(
            "promoted metadata respawn reason differs"
        )

    return PPOPromotionValidationResult(
        source_checkpoint_path=source_path,
        before_checkpoint_path=before_path,
        promoted_checkpoint_path=promoted_path,
        audit_path=resolved_audit_path,
        source_optimizer_step_count=source_step,
        before_optimizer_step_count=before_step,
        promoted_optimizer_step_count=(
            promoted_step
        ),
        attempt_count=len(attempts),
        accepted_attempt_index=(
            accepted_index
        ),
        changed_state_tensor_names=(
            changed_names
        ),
        maximum_absolute_parameter_change=(
            maximum_change
        ),
        confirmed_death_reward=(
            confirmed_reward
        ),
        respawn_detected=respawn_detected,
        respawn_inferred=respawn_inferred,
        respawn_signal_reason=respawn_reason,
        source_progress=source_progress,
        promoted_progress=promoted_progress,
    )


def render_validation_text(
    result: PPOPromotionValidationResult,
) -> str:
    """Render a readable successful validation report."""

    lines = [
        "PPO PROMOTION VALIDATION",
        (
            "source_checkpoint="
            f"{result.source_checkpoint_path}"
        ),
        (
            "before_checkpoint="
            f"{result.before_checkpoint_path}"
        ),
        (
            "promoted_checkpoint="
            f"{result.promoted_checkpoint_path}"
        ),
        f"audit_path={result.audit_path}",
        (
            "source_optimizer_step_count="
            f"{result.source_optimizer_step_count}"
        ),
        (
            "before_optimizer_step_count="
            f"{result.before_optimizer_step_count}"
        ),
        (
            "promoted_optimizer_step_count="
            f"{result.promoted_optimizer_step_count}"
        ),
        (
            "attempt_count="
            f"{result.attempt_count}"
        ),
        (
            "accepted_attempt_index="
            f"{result.accepted_attempt_index}"
        ),
        (
            "changed_state_tensor_count="
            f"{result.changed_state_tensor_count}"
        ),
        (
            "maximum_absolute_parameter_change="
            f"{result.maximum_absolute_parameter_change}"
        ),
        (
            "confirmed_death_reward="
            f"{result.confirmed_death_reward}"
        ),
        (
            "respawn_detected="
            f"{result.respawn_detected}"
        ),
        (
            "respawn_inferred="
            f"{result.respawn_inferred}"
        ),
        (
            "respawn_signal_reason="
            f"{result.respawn_signal_reason}"
        ),
        (
            "source_progress="
            f"{dict(result.source_progress)}"
        ),
        (
            "promoted_progress="
            f"{dict(result.promoted_progress)}"
        ),
        "ppo_promotion_validation=passed",
    ]

    return "\n".join(lines)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read and validate source, before, promoted "
            "PPO checkpoints and their promotion audit."
        )
    )

    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--before-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--promoted-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the validation result as JSON.",
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    result = validate_ppo_promotion(
        args.source_checkpoint,
        args.before_checkpoint,
        args.promoted_checkpoint,
        args.audit_path,
        device=args.device,
    )

    if args.json_output:
        print(
            json.dumps(
                result.to_record(),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            render_validation_text(result)
        )

    return 0


def run_cli(
    argv: Sequence[str] | None = None,
) -> int:
    try:
        return main(argv)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        print(
            "Interrupted by operator.",
            file=sys.stderr,
        )
        return 130
    except (
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
