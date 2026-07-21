"""Tests for the read-only PPO promotion validator."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from RL.training.ppo import (
    PPOTrainingProgress,
)
from RL.tools.inspection import (
    validate_ppo_promotion as validator,
)


def make_model(
    value: float,
) -> torch.nn.Module:
    model = torch.nn.Linear(
        1,
        1,
        bias=False,
    )

    with torch.no_grad():
        model.weight.fill_(value)

    return model


def make_checkpoint(
    *,
    model_value: float,
    optimizer_step: int,
    progress: PPOTrainingProgress,
    metadata: dict,
):
    return SimpleNamespace(
        model=make_model(model_value),
        trainer=SimpleNamespace(
            optimizer_step_count=optimizer_step
        ),
        progress=progress,
        metadata=metadata,
    )


def write_records(
    path: Path,
    records: list[dict],
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def build_fixture(
    tmp_path: Path,
    monkeypatch,
):
    source_path = tmp_path / "source.pt"
    before_path = tmp_path / "before.pt"
    promoted_path = tmp_path / "promoted.pt"
    audit_path = tmp_path / "audit.jsonl"

    source_path.write_bytes(b"source")
    before_path.write_bytes(b"before")
    promoted_path.write_bytes(b"promoted")

    source_progress = PPOTrainingProgress(
        rollout_count=3,
        environment_step_count=37,
        completed_episode_count=1,
        cumulative_reward=-1.0,
    )

    promoted_progress = PPOTrainingProgress(
        rollout_count=4,
        environment_step_count=54,
        completed_episode_count=2,
        cumulative_reward=-2.0,
    )

    source = make_checkpoint(
        model_value=1.0,
        optimizer_step=3,
        progress=source_progress,
        metadata={},
    )

    before = make_checkpoint(
        model_value=1.0,
        optimizer_step=3,
        progress=source_progress,
        metadata={
            "session_phase": "before",
            "source_checkpoint": str(
                source_path
            ),
            "source_optimizer_step_count": 3,
        },
    )

    promoted = make_checkpoint(
        model_value=2.0,
        optimizer_step=4,
        progress=promoted_progress,
        metadata={
            "session_phase": "promoted",
            "source_checkpoint": str(
                source_path
            ),
            "accepted_attempt_index": 1,
            "confirmed_death": True,
            "confirmed_death_reward": -1.0,
            "respawn_detected": False,
            "respawn_inferred": True,
            "respawn_signal_reason": (
                "second_death_proves_respawn"
            ),
        },
    )

    records = [
        {
            "type": (
                "death_aware_session_started"
            ),
            "data": {
                "config": {
                    "attempt_count": 3,
                    "step_config": {
                        "death_reward_threshold": (
                            -1.0
                        ),
                    },
                    "require_respawn_evidence": (
                        True
                    ),
                },
                "source_checkpoint": str(
                    source_path
                ),
                "source_optimizer_step_count": 3,
                "initial_progress": (
                    source_progress.to_record()
                ),
                "before_checkpoint_path": str(
                    before_path
                ),
                (
                    "requested_promoted_"
                    "checkpoint_path"
                ): str(promoted_path),
            },
        },
        {
            "type": (
                "death_aware_attempt_completed"
            ),
            "data": {
                "attempt_index": 0,
                "accepted": False,
                "rejection_reasons": [
                    "death_not_detected"
                ],
            },
        },
        {
            "type": (
                "death_aware_attempt_completed"
            ),
            "data": {
                "attempt_index": 1,
                "accepted": True,
                "rejection_reasons": [],
                "steps": 17,
                "ppo_batch_steps": 17,
                "total_reward": -1.0,
                "death_detected": True,
                "death_reward_confirmed": True,
                "confirmed_death_reward": -1.0,
                "respawn_detected": False,
                "respawn_inferred": True,
                "respawn_signal_reason": (
                    "second_death_proves_respawn"
                ),
                "post_respawn_reward": -1.0,
                "optimizer_operations": 1,
                "source_optimizer_step_count": 3,
                "ending_optimizer_step_count": 4,
                "weights_changed": True,
                "changed_state_tensor_count": 1,
                "changed_state_tensor_names": [
                    "weight"
                ],
                (
                    "maximum_absolute_"
                    "parameter_change"
                ): 1.0,
                "stopped_on_kl": False,
            },
        },
        {
            "type": (
                "death_aware_attempt_promoted"
            ),
            "data": {
                "attempt_index": 1,
                "optimizer_step_count": 4,
                "progress": (
                    promoted_progress.to_record()
                ),
                "promoted_checkpoint_path": str(
                    promoted_path
                ),
            },
        },
        {
            "type": (
                "death_aware_session_completed"
            ),
            "data": {
                "attempts_completed": 2,
                "promoted": True,
                "accepted_attempt_index": 1,
                "source_optimizer_step_count": 3,
                "ending_optimizer_step_count": 4,
                "progress": (
                    promoted_progress.to_record()
                ),
                "promoted_checkpoint_path": str(
                    promoted_path
                ),
            },
        },
    ]

    write_records(
        audit_path,
        records,
    )

    checkpoints = {
        source_path.resolve(): source,
        before_path.resolve(): before,
        promoted_path.resolve(): promoted,
    }

    def fake_load(
        path,
        *,
        device="cpu",
    ):
        assert device in (
            "cpu",
            torch.device("cpu"),
        )

        return checkpoints[
            Path(path).resolve()
        ]

    monkeypatch.setattr(
        validator,
        "load_ppo_training_checkpoint",
        fake_load,
    )

    return SimpleNamespace(
        source_path=source_path,
        before_path=before_path,
        promoted_path=promoted_path,
        audit_path=audit_path,
        source=source,
        before=before,
        promoted=promoted,
        records=records,
    )


def validate_fixture(fixture):
    return validator.validate_ppo_promotion(
        fixture.source_path,
        fixture.before_path,
        fixture.promoted_path,
        fixture.audit_path,
    )


def test_valid_promotion_passes(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    result = validate_fixture(fixture)

    assert (
        result.source_optimizer_step_count
        == 3
    )
    assert (
        result.promoted_optimizer_step_count
        == 4
    )
    assert result.attempt_count == 2
    assert result.accepted_attempt_index == 1
    assert result.changed_state_tensor_count == 1
    assert result.changed_state_tensor_names == (
        "weight",
    )
    assert result.respawn_inferred
    assert not result.respawn_detected


def test_validation_is_read_only(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    paths = (
        fixture.source_path,
        fixture.before_path,
        fixture.promoted_path,
        fixture.audit_path,
    )

    before = {
        path: (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in paths
    }

    validate_fixture(fixture)

    after = {
        path: (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in paths
    }

    assert after == before


def test_rejects_before_weight_difference(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    fixture.before.model = make_model(
        1.5
    )

    with pytest.raises(
        ValueError,
        match="before checkpoint model weights",
    ):
        validate_fixture(fixture)


def test_rejects_promoted_optimizer_step(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    fixture.promoted.trainer = (
        SimpleNamespace(
            optimizer_step_count=5
        )
    )

    with pytest.raises(
        ValueError,
        match="source plus one",
    ):
        validate_fixture(fixture)


def test_rejects_multiple_accepted_attempts(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    records = deepcopy(
        fixture.records
    )

    records[1]["data"]["accepted"] = True
    records[1]["data"][
        "rejection_reasons"
    ] = []

    write_records(
        fixture.audit_path,
        records,
    )

    with pytest.raises(
        ValueError,
        match="exactly one accepted attempt",
    ):
        validate_fixture(fixture)


def test_rejects_missing_respawn_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    records = deepcopy(
        fixture.records
    )

    accepted = records[2]["data"]
    accepted["respawn_detected"] = False
    accepted["respawn_inferred"] = False
    accepted["respawn_signal_reason"] = None

    write_records(
        fixture.audit_path,
        records,
    )

    with pytest.raises(
        ValueError,
        match="lacks respawn evidence",
    ):
        validate_fixture(fixture)


def test_rejects_progress_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    fixture.promoted.progress = (
        PPOTrainingProgress(
            rollout_count=4,
            environment_step_count=55,
            completed_episode_count=2,
            cumulative_reward=-2.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="environment_step_count",
    ):
        validate_fixture(fixture)


def test_rejects_incomplete_audit(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    write_records(
        fixture.audit_path,
        fixture.records[:-1],
    )

    with pytest.raises(
        ValueError,
        match="must end",
    ):
        validate_fixture(fixture)


def test_rejects_changed_tensor_name_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    records = deepcopy(
        fixture.records
    )

    records[2]["data"][
        "changed_state_tensor_names"
    ] = ["wrong"]

    write_records(
        fixture.audit_path,
        records,
    )

    with pytest.raises(
        ValueError,
        match="tensor names differ",
    ):
        validate_fixture(fixture)


def test_parse_args_requires_all_paths() -> None:
    with pytest.raises(SystemExit):
        validator.parse_args([])


def test_render_validation_text(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_fixture(
        tmp_path,
        monkeypatch,
    )

    text = validator.render_validation_text(
        validate_fixture(fixture)
    )

    assert (
        "promoted_optimizer_step_count=4"
        in text
    )

    assert (
        "ppo_promotion_validation=passed"
        in text
    )


def test_run_cli_reports_validation_error(
    monkeypatch,
    capsys,
) -> None:
    def raise_error(_):
        raise ValueError("invalid promotion")

    monkeypatch.setattr(
        validator,
        "main",
        raise_error,
    )

    assert validator.run_cli([]) == 1

    captured = capsys.readouterr()

    assert (
        "ERROR: invalid promotion"
        in captured.err
    )


def test_run_cli_suppresses_broken_pipe(
    monkeypatch,
) -> None:
    def raise_broken_pipe(_):
        raise BrokenPipeError

    monkeypatch.setattr(
        validator,
        "main",
        raise_broken_pipe,
    )

    assert validator.run_cli([]) == 0
