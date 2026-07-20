"""Baseline agents used before policy training."""

from RL.agents.baselines.contracts import Agent
from RL.agents.baselines.fixed_action import (
    FixedActionAgent,
)

__all__ = [
    "Agent",
    "FixedActionAgent",
]
