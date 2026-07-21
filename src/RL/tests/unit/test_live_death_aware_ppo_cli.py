"""Tests for the bounded live death-aware PPO CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from RL.tools.launch import (
    run_live_death_aware_ppo_session
    as cli,
)


def required_argv(
    tmp_path: Path,
) -> list[str]:
    return [
        "--source-checkpoint",
        str(tmp_path / "source.pt"),
        "--event-session",
        str(tmp_path / "events.jsonl"),
        "--before-checkpoint",
        str(tmp_path / "before.pt"),
        "--promoted-checkpoint",
        str(tmp_path / "promoted.pt"),
        "--audit-path",
        str(tmp_path / "audit.jsonl"),
    ]


def test_parse_args_requires_explicit_paths() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([])


def test_parse_args_accepts_explicit_limits(
    tmp_path,
) -> None:
    args = cli.parse_args(
        required_argv(tmp_path)
        + [
            "--controlled-player",
            "PlayerOne",
            "--attempts",
            "5",
            "--max-steps",
            "200",
            "--max-respawn-wait-steps",
            "300",
            "--death-confirmation-steps",
            "25",
            "--respawn-fire-interval-steps",
            "4",
            "--death-reward-threshold",
            "-2.0",
            "--signal-reward-epsilon",
            "0.001",
            "--max-absolute-kl",
            "0.05",
            "--seed",
            "44",
            "--device",
            "cpu",
            "--tick-seconds",
            "0.1",
            "--turn-pixels-per-tick",
            "3",
            "--max-duration-ticks",
            "12",
            "--duration-ticks",
            "2",
        ]
    )

    assert args.controlled_player == (
        "PlayerOne"
    )
    assert args.attempts == 5
    assert args.max_steps == 200
    assert args.max_respawn_wait_steps == 300
    assert args.death_confirmation_steps == 25
    assert (
        args.respawn_fire_interval_steps
        == 4
    )
    assert args.death_reward_threshold == (
        pytest.approx(-2.0)
    )
    assert args.signal_reward_epsilon == (
        pytest.approx(0.001)
    )
    assert args.max_absolute_kl == (
        pytest.approx(0.05)
    )
    assert args.seed == 44
    assert args.device == "cpu"
    assert args.tick_seconds == (
        pytest.approx(0.1)
    )
    assert args.turn_pixels_per_tick == 3
    assert args.max_duration_ticks == 12
    assert args.duration_ticks == 2


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--attempts", "0"),
        ("--max-steps", "0"),
        (
            "--respawn-fire-interval-steps",
            "0",
        ),
        (
            "--death-reward-threshold",
            "0",
        ),
        ("--max-absolute-kl", "0"),
    ],
)
def test_parse_args_rejects_invalid_limits(
    tmp_path,
    flag,
    value,
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            required_argv(tmp_path)
            + [flag, value]
        )


def test_validate_paths_rejects_existing_output(
    tmp_path,
) -> None:
    args = cli.parse_args(
        required_argv(tmp_path)
    )

    args.source_checkpoint.write_bytes(
        b"checkpoint"
    )
    args.event_session.write_text(
        "{}\n",
        encoding="utf-8",
    )
    args.before_checkpoint.write_bytes(
        b"existing"
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        cli.validate_paths(args)


def test_environment_factory_wires_live_bridge(
    tmp_path,
    monkeypatch,
) -> None:
    args = cli.parse_args(
        required_argv(tmp_path)
        + [
            "--xdotool-command",
            "custom-xdotool",
            "--tick-seconds",
            "0.2",
            "--turn-pixels-per-tick",
            "4",
            "--max-duration-ticks",
            "9",
        ]
    )

    calls = {}
    controller = object()
    executor = object()
    bridge = object()
    processor = object()
    environment = object()

    def fake_controller(**kwargs):
        calls["controller"] = kwargs
        return controller

    def fake_executor(
        received_controller,
        **kwargs,
    ):
        calls["executor"] = (
            received_controller,
            kwargs,
        )
        return executor

    def fake_bridge(**kwargs):
        calls["bridge"] = kwargs
        return bridge

    def fake_processor(
        event_path,
        controlled_player,
    ):
        calls["processor"] = (
            event_path,
            controlled_player,
        )
        return processor

    def fake_environment(
        received_bridge,
        *,
        event_processor,
    ):
        calls["environment"] = (
            received_bridge,
            event_processor,
        )
        return environment

    monkeypatch.setattr(
        cli,
        "X11InputController",
        fake_controller,
    )
    monkeypatch.setattr(
        cli,
        "XonoticActionExecutor",
        fake_executor,
    )
    monkeypatch.setattr(
        cli,
        "XonoticObservationBridge",
        fake_bridge,
    )
    monkeypatch.setattr(
        cli,
        "JSONLEventStepProcessor",
        fake_processor,
    )
    monkeypatch.setattr(
        cli,
        "BridgeEnvironment",
        fake_environment,
    )

    factory = cli.build_environment_factory(
        args
    )

    assert factory(3) is environment
    assert calls["controller"] == {
        "xdotool_command": "custom-xdotool",
    }
    assert calls["executor"] == (
        controller,
        {
            "tick_seconds": 0.2,
            "turn_pixels_per_tick": 4,
            "max_duration_ticks": 9,
        },
    )
    assert calls["bridge"] == {
        "frame_width": 160,
        "frame_height": 90,
        "frame_stack": 4,
        "action_executor": executor,
    }
    assert calls["processor"] == (
        args.event_session,
        "Noobnog",
    )
    assert calls["environment"] == (
        bridge,
        processor,
    )


def test_agent_factory_binds_reloaded_model(
    tmp_path,
    monkeypatch,
) -> None:
    args = cli.parse_args(
        required_argv(tmp_path)
        + [
            "--device",
            "cpu",
            "--duration-ticks",
            "3",
            "--policy-name",
            "test-policy",
            "--policy-version",
            "test-version",
        ]
    )

    calls = {}
    model = object()
    agent = object()

    def fake_agent(
        received_model,
        **kwargs,
    ):
        calls["agent"] = (
            received_model,
            kwargs,
        )
        return agent

    monkeypatch.setattr(
        cli,
        "ActorCriticPolicyAgent",
        fake_agent,
    )

    factory = cli.build_agent_factory(args)

    assert factory(model, 2) is agent
    assert calls["agent"] == (
        model,
        {
            "device": "cpu",
            "deterministic": False,
            "duration_ticks": 3,
            "policy_name": "test-policy",
            "policy_version": "test-version",
        },
    )


def test_execute_invokes_bounded_session(
    tmp_path,
    monkeypatch,
) -> None:
    args = cli.parse_args(
        required_argv(tmp_path)
        + [
            "--attempts",
            "4",
            "--max-steps",
            "88",
            "--device",
            "cpu",
        ]
    )

    args.source_checkpoint.write_bytes(
        b"checkpoint"
    )
    args.event_session.write_text(
        "{}\n",
        encoding="utf-8",
    )

    environment_factory = object()
    agent_factory = object()
    expected_result = object()
    captured = {}

    monkeypatch.setattr(
        cli,
        "preflight_event_session",
        lambda *_: {
            "history_event_count": 10,
            "priming_reason": "active_match",
            "primed_event_count": 4,
            "match_active": True,
            "current_mode": "tdm",
            "controlled_team": "RED",
        },
    )
    monkeypatch.setattr(
        cli,
        "build_environment_factory",
        lambda _: environment_factory,
    )
    monkeypatch.setattr(
        cli,
        "build_agent_factory",
        lambda _: agent_factory,
    )

    def fake_session(
        source_checkpoint,
        received_environment_factory,
        received_agent_factory,
        config,
        **kwargs,
    ):
        captured["source"] = (
            source_checkpoint
        )
        captured["environment_factory"] = (
            received_environment_factory
        )
        captured["agent_factory"] = (
            received_agent_factory
        )
        captured["config"] = config
        captured["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(
        cli,
        "run_bounded_death_aware_ppo_session",
        fake_session,
    )

    assert cli.execute(args) is expected_result

    assert captured["source"] == (
        args.source_checkpoint
    )
    assert (
        captured["environment_factory"]
        is environment_factory
    )
    assert (
        captured["agent_factory"]
        is agent_factory
    )

    config = captured["config"]

    assert config.attempt_count == 4
    assert config.require_respawn_evidence
    assert config.step_config.max_steps == 88
    assert (
        config.step_config.updates_per_signal
        == 1
    )
    assert config.step_config.stop_on_kl

    kwargs = captured["kwargs"]

    assert kwargs["device"] == "cpu"
    assert kwargs["before_checkpoint_path"] == (
        args.before_checkpoint
    )
    assert (
        kwargs["promoted_checkpoint_path"]
        == args.promoted_checkpoint
    )
    assert kwargs["audit_path"] == (
        args.audit_path
    )
    assert (
        kwargs["checkpoint_metadata"]
        ["preflight_mode"]
        == "tdm"
    )


def test_main_returns_zero_when_promoted(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "execute",
        lambda _: SimpleNamespace(
            promoted=True
        ),
    )
    monkeypatch.setattr(
        cli,
        "print_session_result",
        lambda _: None,
    )

    assert cli.main(
        required_argv(tmp_path)
    ) == 0


def test_main_returns_one_without_promotion(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "execute",
        lambda _: SimpleNamespace(
            promoted=False
        ),
    )
    monkeypatch.setattr(
        cli,
        "print_session_result",
        lambda _: None,
    )

    assert cli.main(
        required_argv(tmp_path)
    ) == 1


def test_run_cli_suppresses_broken_pipe(
    monkeypatch,
) -> None:
    def raise_broken_pipe(_):
        raise BrokenPipeError

    monkeypatch.setattr(
        cli,
        "main",
        raise_broken_pipe,
    )

    assert cli.run_cli([]) == 0


def test_run_cli_reports_expected_error(
    monkeypatch,
    capsys,
) -> None:
    def raise_value_error(_):
        raise ValueError("bad input")

    monkeypatch.setattr(
        cli,
        "main",
        raise_value_error,
    )

    assert cli.run_cli([]) == 1

    captured = capsys.readouterr()

    assert "ERROR: bad input" in (
        captured.err
    )


def test_run_cli_reports_keyboard_interrupt(
    monkeypatch,
    capsys,
) -> None:
    def raise_interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        cli,
        "main",
        raise_interrupt,
    )

    assert cli.run_cli([]) == 130

    captured = capsys.readouterr()

    assert "Interrupted by operator" in (
        captured.err
    )
