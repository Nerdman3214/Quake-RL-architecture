"""Readable records for inspecting agent observations and decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from RL.actions.contracts import ActionCommand
from RL.observations.contracts import PlayerTelemetry


@dataclass(frozen=True)
class FrameSnapshot:
    """Reference to a saved raw frame or processed policy input."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    encoding: str
    transform: str | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("frame path must not be empty")

        if not self.shape:
            raise ValueError("frame shape must not be empty")

        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(
                "frame dimensions must be greater than zero"
            )

        if not self.dtype:
            raise ValueError("frame dtype must not be empty")

        if not self.encoding:
            raise ValueError(
                "frame encoding must not be empty"
            )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "encoding": self.encoding,
            "transform": self.transform,
        }


@dataclass(frozen=True)
class ObservationSnapshot:
    """Readable description of one observation seen by an agent."""

    tick: int
    telemetry: PlayerTelemetry | None
    raw_frame: FrameSnapshot | None
    policy_frame: FrameSnapshot | None

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError(
                "observation tick must not be negative"
            )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        telemetry_record = None

        if self.telemetry is not None:
            telemetry_record = {
                "health": self.telemetry.health,
                "armor": self.telemetry.armor,
                "ammo": self.telemetry.ammo,
                "weapon": self.telemetry.weapon,
                "alive": self.telemetry.alive,
                "score": self.telemetry.score,
                "match_time_seconds": (
                    self.telemetry.match_time_seconds
                ),
            }

        return {
            "tick": self.tick,
            "telemetry": telemetry_record,
            "raw_frame": (
                self.raw_frame.to_record()
                if self.raw_frame is not None
                else None
            ),
            "policy_frame": (
                self.policy_frame.to_record()
                if self.policy_frame is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PolicyDecision:
    """The action selected by a policy for one observation."""

    action: ActionCommand
    policy_name: str
    policy_version: str
    checkpoint: str | None = None
    action_scores: Mapping[str, float] = field(
        default_factory=dict
    )
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.policy_name:
            raise ValueError(
                "policy name must not be empty"
            )

        if not self.policy_version:
            raise ValueError(
                "policy version must not be empty"
            )

    def to_record(self) -> dict[str, Any]:
        """Return a readable JSON-compatible representation."""

        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "checkpoint": self.checkpoint,
            "deterministic": self.deterministic,
            "action": {
                "name": self.action.action.name,
                "value": int(self.action.action),
                "duration_ticks": (
                    self.action.duration_ticks
                ),
            },
            "action_scores": {
                str(name): float(score)
                for name, score
                in self.action_scores.items()
            },
        }


@dataclass(frozen=True)
class AgentTransition:
    """One inspectable observation-action-reward transition."""

    episode_id: str
    step_index: int
    observation: ObservationSnapshot
    decision: PolicyDecision
    reward: float
    next_observation: ObservationSnapshot | None
    terminated: bool
    truncated: bool
    reward_components: Mapping[str, float] = field(
        default_factory=dict
    )
    info: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError(
                "episode id must not be empty"
            )

        if self.step_index < 0:
            raise ValueError(
                "step index must not be negative"
            )

    def to_record(self) -> dict[str, Any]:
        """Return a readable JSON-compatible representation."""

        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "observation": self.observation.to_record(),
            "decision": self.decision.to_record(),
            "reward": float(self.reward),
            "reward_components": {
                str(name): float(value)
                for name, value
                in self.reward_components.items()
            },
            "next_observation": (
                self.next_observation.to_record()
                if self.next_observation is not None
                else None
            ),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": dict(self.info),
        }
