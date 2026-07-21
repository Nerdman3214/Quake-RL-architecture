"""Run a bounded live death-aware PPO promotion session."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from RL.agents import (
    ActorCriticPolicyAgent,
    VisualActorCriticNetwork,
)
from RL.engine.bridge import (
    XonoticObservationBridge,
)
from RL.engine.client import (
    X11InputController,
    XonoticActionExecutor,
)
from RL.env import (
    BridgeEnvironment,
    Environment,
    JSONLEventStepProcessor,
)
from RL.training.ppo import (
    DeathAwarePPOConfig,
    DeathAwarePPOTrainingSessionConfig,
    DeathAwarePPOTrainingSessionResult,
    run_bounded_death_aware_ppo_session,
)


DEFAULT_CONTROLLED_PLAYER = "Noobnog"
DEFAULT_ATTEMPTS = 3
DEFAULT_MAX_STEPS = 160
DEFAULT_MAX_RESPAWN_WAIT_STEPS = 240
DEFAULT_DEATH_CONFIRMATION_STEPS = 40
DEFAULT_RESPAWN_FIRE_INTERVAL_STEPS = 8
DEFAULT_DEATH_REWARD_THRESHOLD = -1.0
DEFAULT_SIGNAL_REWARD_EPSILON = 1e-8
DEFAULT_MAX_ABSOLUTE_KL = 0.10
DEFAULT_SEED = 20260720
DEFAULT_DEVICE = "cuda:0"
DEFAULT_TICK_SECONDS = 0.05
DEFAULT_TURN_PIXELS_PER_TICK = 1
DEFAULT_MAX_DURATION_TICKS = 20
DEFAULT_ACTION_DURATION_TICKS = 1
DEFAULT_POLICY_NAME = "xonotic-death-aware-ppo"
DEFAULT_POLICY_VERSION = "live-death-aware-v1"


def nonempty_string(value: str) -> str:
    """Parse a command-line value that must contain text."""

    normalized = value.strip()

    if not normalized:
        raise argparse.ArgumentTypeError(
            "value must not be empty"
        )

    return normalized


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, received {value!r}"
        ) from error

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return parsed


def nonnegative_integer(value: str) -> int:
    """Parse an integer greater than or equal to zero."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, received {value!r}"
        ) from error

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must not be negative"
        )

    return parsed


def finite_float(value: str) -> float:
    """Parse a finite floating-point value."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a number, received {value!r}"
        ) from error

    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(
            "value must be finite"
        )

    return parsed


def positive_float(value: str) -> float:
    """Parse a finite number greater than zero."""

    parsed = finite_float(value)

    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return parsed


def nonnegative_float(value: str) -> float:
    """Parse a finite number greater than or equal to zero."""

    parsed = finite_float(value)

    if parsed < 0.0:
        raise argparse.ArgumentTypeError(
            "value must not be negative"
        )

    return parsed


def negative_float(value: str) -> float:
    """Parse a finite number less than zero."""

    parsed = finite_float(value)

    if parsed >= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be negative"
        )

    return parsed


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse explicit bounded live-training arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run isolated live death-aware PPO attempts and "
            "promote only a verified death-learning update."
        )
    )

    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        required=True,
        help=(
            "Existing PPO checkpoint loaded independently "
            "for every attempt."
        ),
    )
    parser.add_argument(
        "--event-session",
        type=Path,
        required=True,
        help=(
            "Active Xonotic JSONL event-session file."
        ),
    )
    parser.add_argument(
        "--controlled-player",
        type=nonempty_string,
        default=DEFAULT_CONTROLLED_PLAYER,
    )
    parser.add_argument(
        "--before-checkpoint",
        type=Path,
        required=True,
        help=(
            "New path for the immutable before-session checkpoint."
        ),
    )
    parser.add_argument(
        "--promoted-checkpoint",
        type=Path,
        required=True,
        help=(
            "New path written only when an attempt is accepted."
        ),
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        required=True,
        help=(
            "New durable JSONL audit path."
        ),
    )

    parser.add_argument(
        "--attempts",
        type=positive_integer,
        default=DEFAULT_ATTEMPTS,
    )
    parser.add_argument(
        "--max-steps",
        type=positive_integer,
        default=DEFAULT_MAX_STEPS,
    )
    parser.add_argument(
        "--max-respawn-wait-steps",
        type=nonnegative_integer,
        default=DEFAULT_MAX_RESPAWN_WAIT_STEPS,
    )
    parser.add_argument(
        "--death-confirmation-steps",
        type=nonnegative_integer,
        default=DEFAULT_DEATH_CONFIRMATION_STEPS,
    )
    parser.add_argument(
        "--respawn-fire-interval-steps",
        type=positive_integer,
        default=DEFAULT_RESPAWN_FIRE_INTERVAL_STEPS,
    )
    parser.add_argument(
        "--death-reward-threshold",
        type=negative_float,
        default=DEFAULT_DEATH_REWARD_THRESHOLD,
    )
    parser.add_argument(
        "--signal-reward-epsilon",
        type=nonnegative_float,
        default=DEFAULT_SIGNAL_REWARD_EPSILON,
    )
    parser.add_argument(
        "--max-absolute-kl",
        type=positive_float,
        default=DEFAULT_MAX_ABSOLUTE_KL,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--device",
        type=nonempty_string,
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        "--xdotool-command",
        type=nonempty_string,
        default="xdotool",
    )
    parser.add_argument(
        "--tick-seconds",
        type=positive_float,
        default=DEFAULT_TICK_SECONDS,
    )
    parser.add_argument(
        "--turn-pixels-per-tick",
        type=positive_integer,
        default=DEFAULT_TURN_PIXELS_PER_TICK,
    )
    parser.add_argument(
        "--max-duration-ticks",
        type=positive_integer,
        default=DEFAULT_MAX_DURATION_TICKS,
    )
    parser.add_argument(
        "--duration-ticks",
        type=positive_integer,
        default=DEFAULT_ACTION_DURATION_TICKS,
    )
    parser.add_argument(
        "--policy-name",
        type=nonempty_string,
        default=DEFAULT_POLICY_NAME,
    )
    parser.add_argument(
        "--policy-version",
        type=nonempty_string,
        default=DEFAULT_POLICY_VERSION,
    )

    return parser.parse_args(argv)


def validate_paths(
    args: argparse.Namespace,
) -> None:
    """Validate input files and protect every output path."""

    if not args.source_checkpoint.is_file():
        raise FileNotFoundError(
            "source checkpoint does not exist: "
            f"{args.source_checkpoint}"
        )

    if not args.event_session.is_file():
        raise FileNotFoundError(
            "event session does not exist: "
            f"{args.event_session}"
        )

    output_paths = {
        "before checkpoint": (
            args.before_checkpoint
        ),
        "promoted checkpoint": (
            args.promoted_checkpoint
        ),
        "audit": args.audit_path,
    }

    for label, path in output_paths.items():
        if path.exists():
            raise FileExistsError(
                f"{label} path already exists: {path}"
            )

    all_paths = {
        "source checkpoint": (
            args.source_checkpoint
        ),
        "event session": args.event_session,
        **output_paths,
    }

    resolved: dict[Path, str] = {}

    for label, path in all_paths.items():
        key = path.resolve()

        if key in resolved:
            raise ValueError(
                f"{label} and {resolved[key]} "
                "must use different paths"
            )

        resolved[key] = label


def _plain_value(
    value: object,
) -> object:
    """Convert enum-like values into JSON-compatible values."""

    if value is None:
        return None

    enum_value = getattr(
        value,
        "value",
        value,
    )

    if isinstance(
        enum_value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return enum_value

    return str(enum_value)


def preflight_event_session(
    event_session: Path,
    controlled_player: str,
) -> dict[str, object]:
    """Verify that the supplied JSONL stream describes an active match."""

    processor = JSONLEventStepProcessor(
        event_session,
        controlled_player,
    )

    processor.reset_episode()

    record = {
        "history_event_count": (
            processor.history_event_count
        ),
        "priming_reason": (
            processor.priming_reason
        ),
        "primed_event_count": (
            processor.primed_event_count
        ),
        "match_active": (
            processor.match_active
        ),
        "current_mode": _plain_value(
            processor.current_mode
        ),
        "controlled_team": _plain_value(
            processor.controlled_team
        ),
    }

    if not record["match_active"]:
        raise RuntimeError(
            "event session does not show an active match"
        )

    return record


def build_environment_factory(
    args: argparse.Namespace,
) -> Callable[[int], Environment]:
    """Build a fresh live bridge environment for every attempt."""

    def environment_factory(
        _: int,
    ) -> Environment:
        controller = X11InputController(
            xdotool_command=(
                args.xdotool_command
            )
        )

        executor = XonoticActionExecutor(
            controller,
            tick_seconds=args.tick_seconds,
            turn_pixels_per_tick=(
                args.turn_pixels_per_tick
            ),
            max_duration_ticks=(
                args.max_duration_ticks
            ),
        )

        bridge = XonoticObservationBridge(
            frame_width=160,
            frame_height=90,
            frame_stack=4,
            action_executor=executor,
        )

        processor = JSONLEventStepProcessor(
            args.event_session,
            args.controlled_player,
        )

        return BridgeEnvironment(
            bridge,
            event_processor=processor,
        )

    return environment_factory


def build_agent_factory(
    args: argparse.Namespace,
) -> Callable[
    [
        VisualActorCriticNetwork,
        int,
    ],
    ActorCriticPolicyAgent,
]:
    """Build a stochastic policy bound to each reloaded model."""

    def agent_factory(
        model: VisualActorCriticNetwork,
        _: int,
    ) -> ActorCriticPolicyAgent:
        return ActorCriticPolicyAgent(
            model,
            device=args.device,
            deterministic=False,
            duration_ticks=(
                args.duration_ticks
            ),
            policy_name=args.policy_name,
            policy_version=(
                args.policy_version
            ),
        )

    return agent_factory


def build_session_config(
    args: argparse.Namespace,
) -> DeathAwarePPOTrainingSessionConfig:
    """Build the strict promotion-session configuration."""

    step_config = DeathAwarePPOConfig(
        max_steps=args.max_steps,
        max_respawn_wait_steps=(
            args.max_respawn_wait_steps
        ),
        death_confirmation_steps=(
            args.death_confirmation_steps
        ),
        respawn_fire_interval_steps=(
            args.respawn_fire_interval_steps
        ),
        updates_per_signal=1,
        death_reward_threshold=(
            args.death_reward_threshold
        ),
        signal_reward_epsilon=(
            args.signal_reward_epsilon
        ),
        max_absolute_kl=(
            args.max_absolute_kl
        ),
        stop_on_kl=True,
        seed=args.seed,
    )

    return DeathAwarePPOTrainingSessionConfig(
        attempt_count=args.attempts,
        step_config=step_config,
        require_respawn_evidence=True,
    )


def execute(
    args: argparse.Namespace,
) -> DeathAwarePPOTrainingSessionResult:
    """Validate, construct, and run one bounded live session."""

    validate_paths(args)

    preflight = preflight_event_session(
        args.event_session,
        args.controlled_player,
    )

    print("=== LIVE EVENT PREFLIGHT ===")

    for key, value in preflight.items():
        print(f"{key}={value}")

    print()
    print("=== BOUNDED DEATH-AWARE PPO SESSION ===")
    print(
        "source_checkpoint="
        f"{args.source_checkpoint}"
    )
    print(
        "event_session="
        f"{args.event_session}"
    )
    print(
        "controlled_player="
        f"{args.controlled_player}"
    )
    print(f"attempt_limit={args.attempts}")
    print(
        "step_limit_per_attempt="
        f"{args.max_steps}"
    )
    print(
        "promoted_checkpoint="
        f"{args.promoted_checkpoint}"
    )
    print(f"audit_path={args.audit_path}")

    return run_bounded_death_aware_ppo_session(
        args.source_checkpoint,
        build_environment_factory(args),
        build_agent_factory(args),
        build_session_config(args),
        device=args.device,
        before_checkpoint_path=(
            args.before_checkpoint
        ),
        promoted_checkpoint_path=(
            args.promoted_checkpoint
        ),
        audit_path=args.audit_path,
        policy_name=args.policy_name,
        policy_version=args.policy_version,
        checkpoint_metadata={
            "live": True,
            "cli": (
                "run_live_death_aware_ppo_session"
            ),
            "controlled_player": (
                args.controlled_player
            ),
            "event_session": str(
                args.event_session
            ),
            "preflight_mode": (
                preflight["current_mode"]
            ),
            "preflight_controlled_team": (
                preflight["controlled_team"]
            ),
        },
    )


def print_session_result(
    result: DeathAwarePPOTrainingSessionResult,
) -> None:
    """Print an operator-readable bounded-session summary."""

    print()
    print("=== ATTEMPT RESULTS ===")

    for attempt in result.attempts:
        rollout = attempt.rollout

        print(
            "attempt="
            f"{attempt.attempt_index} "
            "accepted="
            f"{attempt.accepted} "
            "steps="
            f"{rollout.steps} "
            "reward="
            f"{rollout.total_reward} "
            "death_detected="
            f"{rollout.death_detected} "
            "death_reward_confirmed="
            f"{rollout.death_reward_confirmed} "
            "respawn_detected="
            f"{rollout.respawn_detected} "
            "respawn_inferred="
            f"{rollout.respawn_inferred} "
            "optimizer_operations="
            f"{attempt.result.optimizer_operations} "
            "ending_optimizer_step="
            f"{attempt.ending_optimizer_step_count}"
        )

        if attempt.rejection_reasons:
            print(
                "rejection_reasons="
                + ",".join(
                    attempt.rejection_reasons
                )
            )

    print()
    print("=== SESSION RESULT ===")
    print(
        "source_optimizer_step_count="
        f"{result.source_optimizer_step_count}"
    )
    print(
        "ending_optimizer_step_count="
        f"{result.ending_optimizer_step_count}"
    )
    print(
        "attempts_completed="
        f"{result.attempts_completed}"
    )
    print(f"promoted={result.promoted}")
    print(
        "accepted_attempt_index="
        f"{result.accepted_attempt_index}"
    )
    print(
        "progress="
        f"{result.progress.to_record()}"
    )
    print(
        "before_checkpoint_path="
        f"{result.before_checkpoint_path}"
    )
    print(
        "promoted_checkpoint_path="
        f"{result.promoted_checkpoint_path}"
    )
    print(f"audit_path={result.audit_path}")

    if result.promoted:
        print(
            "live_death_aware_ppo_session=promoted"
        )
    else:
        print(
            "live_death_aware_ppo_session=not_promoted"
        )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the CLI and return a process status."""

    args = parse_args(argv)
    result = execute(args)

    print_session_result(result)

    return 0 if result.promoted else 1


def run_cli(
    argv: Sequence[str] | None = None,
) -> int:
    """Run safely when output is piped or the operator interrupts."""

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
