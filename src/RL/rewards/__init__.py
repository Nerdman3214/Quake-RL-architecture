"""Reward subsystem for authoritative event accounting."""

from .contracts import RewardLedger, RewardWeights
from .event_mapper import RewardMapper
from .priming import (
    RewardPrimingPlan,
    build_reward_priming_plan,
)
from .mode_profiles import (
    MODE_REWARD_PROFILES,
    ModeRewardProfile,
    get_mode_reward_profile,
)

__all__ = [
    "MODE_REWARD_PROFILES",
    "ModeRewardProfile",
    "RewardLedger",
    "RewardMapper",
    "RewardPrimingPlan",
    "build_reward_priming_plan",
    "RewardWeights",
    "get_mode_reward_profile",
]
