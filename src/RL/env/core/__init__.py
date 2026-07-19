"""Core environment contracts and adapters."""

from RL.env.core.bridge_environment import (
    BridgeEnvironment,
)
from RL.env.core.contracts import (
    Environment,
    StepResult,
)

__all__ = [
    "BridgeEnvironment",
    "Environment",
    "StepResult",
]
