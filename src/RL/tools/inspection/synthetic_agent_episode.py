#!/usr/bin/env python3
"""Generate and validate a synthetic goal-aware agent episode."""

from __future__ import annotations

import argparse
from pathlib import Path

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.inspection import (
    AgentTransition,
    EpisodeEnded,
    EpisodeStarted,
    FrameSnapshot,
    GoalSnapshot,
    InspectionJSONLReader,
    InspectionJSONLWriter,
    ObservationSnapshot,
    PolicyDecision,
)
from RL.observations.contracts import PlayerTelemetry


def make_observation(
    tick: int,
    *,
    health: int,
    weapon: str,
) -> ObservationSnapshot:
    return ObservationSnapshot(
        tick=tick,
        telemetry=PlayerTelemetry(
            health=health,
            armor=25,
            ammo=20,
            weapon=weapon,
            alive=health > 0,
            score=0,
            match_time_seconds=tick / 10.0,
        ),
        raw_frame=FrameSnapshot(
            path=f"synthetic/raw/frame_{tick:06d}.png",
            shape=(720, 1280, 3),
            dtype="uint8",
            encoding="png",
            transform=None,
        ),
        policy_frame=FrameSnapshot(
            path=(
                "synthetic/policy/"
                f"frame_{tick:06d}.npy"
            ),
            shape=(4, 84, 84),
            dtype="float32",
            encoding="npy",
            transform=(
                "grayscale; resize=84x84; "
                "normalize=0..1; stack=4"
            ),
        ),
    )


def build_transitions(
    episode_id: str,
) -> list[AgentTransition]:
    observation_0 = make_observation(
        100,
        health=100,
        weapon="shotgun",
    )
    observation_1 = make_observation(
        101,
        health=100,
        weapon="shotgun",
    )
    observation_2 = make_observation(
        102,
        health=92,
        weapon="shotgun",
    )

    return [
        AgentTransition(
            episode_id=episode_id,
            step_index=0,
            observation=observation_0,
            decision=PolicyDecision(
                action=ActionCommand(
                    action=DiscreteAction.TURN_RIGHT,
                ),
                policy_name="synthetic-goal-policy",
                policy_version="v1",
                action_scores={
                    "TURN_RIGHT": 0.80,
                    "FORWARD": 0.20,
                },
            ),
            reward=0.0,
            next_observation=observation_1,
            terminated=False,
            truncated=False,
            goal=GoalSnapshot(
                goal_id="find-opponent-001",
                category="search",
                target=None,
                trigger="opponent_not_visible",
                progress=0.35,
                status="active",
                steps_active=1,
            ),
            info={
                "synthetic": True,
                "enemy_visible": False,
            },
        ),
        AgentTransition(
            episode_id=episode_id,
            step_index=1,
            observation=observation_1,
            decision=PolicyDecision(
                action=ActionCommand(
                    action=DiscreteAction.FORWARD,
                ),
                policy_name="synthetic-goal-policy",
                policy_version="v1",
                action_scores={
                    "FORWARD": 0.72,
                    "FIRE": 0.21,
                },
            ),
            reward=0.0,
            next_observation=observation_2,
            terminated=False,
            truncated=False,
            goal=GoalSnapshot(
                goal_id="establish-line-of-sight-001",
                category="combat",
                target="visible-opponent",
                trigger="opponent_detected",
                progress=1.0,
                status="completed",
                steps_active=2,
                completion_reason=(
                    "opponent_centered_in_policy_frame"
                ),
            ),
            info={
                "synthetic": True,
                "enemy_visible": True,
            },
        ),
        AgentTransition(
            episode_id=episode_id,
            step_index=2,
            observation=observation_2,
            decision=PolicyDecision(
                action=ActionCommand(
                    action=DiscreteAction.FIRE,
                ),
                policy_name="synthetic-goal-policy",
                policy_version="v1",
                action_scores={
                    "FIRE": 0.76,
                    "STRAFE_LEFT": 0.31,
                },
            ),
            reward=0.0,
            next_observation=None,
            terminated=True,
            truncated=False,
            goal=GoalSnapshot(
                goal_id="engage-opponent-001",
                category="combat",
                target="visible-opponent",
                trigger="line_of_sight_established",
                progress=0.40,
                status="abandoned",
                steps_active=1,
                failure_reason="synthetic_episode_ended",
            ),
            info={
                "synthetic": True,
                "enemy_visible": True,
            },
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic inspectable "
            "agent episode."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/tmp/quake-rl-synthetic-agent-episode.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    episode_id = "synthetic-open-arena-001"

    transitions = build_transitions(episode_id)

    with InspectionJSONLWriter(
        output_path
    ) as writer:
        writer.write_episode_started(
            EpisodeStarted(
                episode_id=episode_id,
                started_at=(
                    "2026-07-18T15:00:00+00:00"
                ),
                metadata={
                    "synthetic": True,
                    "curriculum_stage": (
                        "open_arena_1v1"
                    ),
                    "bot_skill": 1,
                    "learning_phase": (
                        "pre_training_inspection"
                    ),
                    "frame_references_are_real": False,
                },
            )
        )

        for transition in transitions:
            writer.write_transition(transition)

        writer.write_episode_ended(
            EpisodeEnded(
                episode_id=episode_id,
                ended_at=(
                    "2026-07-18T15:00:03+00:00"
                ),
                steps=len(transitions),
                terminated=True,
                truncated=False,
                outcome="synthetic_complete",
                summary={
                    "reward_mode": "disabled",
                    "goal_system": (
                        "inspection_only"
                    ),
                    "transition_count": len(
                        transitions
                    ),
                },
            )
        )

    records = InspectionJSONLReader(
        output_path
    ).read_episode()

    print("=" * 72)
    print("SYNTHETIC AI-VIEW EPISODE")
    print("=" * 72)
    print(f"output={output_path}")
    print(f"episode_id={episode_id}")
    print(
        "frame_references_are_real=False"
    )
    print(
        "reward_mode=disabled"
    )
    print(
        "transition_count="
        f"{len(transitions)}"
    )
    print()

    for record in records:
        if record.type != "agent_transition":
            continue

        data = record.data
        observation = data["observation"]
        decision = data["decision"]
        goal = data["goal"]

        print(
            f"step={data['step_index']} "
            f"tick={observation['tick']} "
            f"goal={goal['goal_id']} "
            f"status={goal['status']} "
            f"progress={goal['progress']:.2f} "
            f"action={decision['action']['name']} "
            f"reward={data['reward']:.2f}"
        )
        print(
            "  raw_frame="
            f"{observation['raw_frame']['path']}"
        )
        print(
            "  policy_frame="
            f"{observation['policy_frame']['path']}"
        )

    print()
    print("validation=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
