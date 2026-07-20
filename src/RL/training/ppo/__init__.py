"""Proximal Policy Optimization contracts and utilities."""

from RL.training.ppo.checkpoint import (
    PPO_CHECKPOINT_FORMAT_VERSION,
    LoadedPPOTrainingCheckpoint,
    PPOTrainingProgress,
    load_ppo_training_checkpoint,
    restore_ppo_checkpoint_rng_state,
    save_ppo_training_checkpoint,
)
from RL.training.ppo.collector import (
    CollectedRolloutResult,
    CollectedRolloutTransition,
    collect_bounded_rollout,
)
from RL.training.ppo.death_aware import (
    DeathAwarePPOConfig,
    DeathAwarePPOResult,
    DeathAwareRolloutResult,
    collect_death_aware_rollout,
    run_death_aware_ppo_step,
)
from RL.training.ppo.core import (
    PPO_ACTION_COUNT,
    PPO_FRAME_SHAPE,
    PPOBatchResult,
    PPOHyperparameters,
    PPOLossTensors,
    PPOMetrics,
    PPORolloutBatch,
    PPOTrainer,
    RolloutBuffer,
    RolloutTransition,
    compute_gae,
    ppo_loss,
)
from RL.training.ppo.session import (
    EnvironmentFactory,
    PPORolloutAudit,
    PPOTrainingSessionConfig,
    PPOTrainingSessionResult,
    PPOUpdateAudit,
    run_bounded_ppo_training_session,
)

__all__ = [
    "run_death_aware_ppo_step",
    "collect_death_aware_rollout",
    "DeathAwareRolloutResult",
    "DeathAwarePPOResult",
    "DeathAwarePPOConfig",
    "PPO_ACTION_COUNT",
    "PPO_CHECKPOINT_FORMAT_VERSION",
    "PPO_FRAME_SHAPE",
    "CollectedRolloutResult",
    "CollectedRolloutTransition",
    "EnvironmentFactory",
    "LoadedPPOTrainingCheckpoint",
    "PPOBatchResult",
    "PPOHyperparameters",
    "PPOLossTensors",
    "PPOMetrics",
    "PPORolloutAudit",
    "PPORolloutBatch",
    "PPOTrainer",
    "PPOTrainingProgress",
    "PPOTrainingSessionConfig",
    "PPOTrainingSessionResult",
    "PPOUpdateAudit",
    "RolloutBuffer",
    "RolloutTransition",
    "collect_bounded_rollout",
    "compute_gae",
    "load_ppo_training_checkpoint",
    "ppo_loss",
    "restore_ppo_checkpoint_rng_state",
    "run_bounded_ppo_training_session",
    "save_ppo_training_checkpoint",
]
