"""Tests for inspected bounded neural episodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.agents import (
    NeuralPolicyAgent,
    VisualPolicyNetwork,
)
from RL.env.core.contracts import (
    Environment,
    StepResult,
)
from RL.inspection import (
    InspectionJSONLReader,
    run_inspected_neural_episode,
)
from RL.observations.contracts import Observation


def make_observation(
    tick: int,
) -> Observation:
    return Observation(
        frame=np.full(
            (4, 3, 90, 160),
            fill_value=tick / 10.0,
            dtype=np.float32,
        ),
        telemetry=None,
        tick=tick,
    )


def make_agent(
    action: DiscreteAction,
) -> NeuralPolicyAgent:
    model = VisualPolicyNetwork()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.action_head.bias[
            int(action)
        ] = 8.0

    return NeuralPolicyAgent(
        model,
        device="cpu",
        duration_ticks=1,
        policy_name="inspection-test-policy",
        policy_version="v2",
    )


class FakeEnvironment(Environment):
    def __init__(
        self,
        *,
        terminate_at: int | None = None,
        fail_step: bool = False,
        reward: float = 0.0,
    ) -> None:
        self.terminate_at = terminate_at
        self.fail_step = fail_step
        self.reward = reward

        self.step_count = 0
        self.close_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Observation, dict[str, object]]:
        return make_observation(0), {
            "seed": seed,
        }

    def step(
        self,
        action: ActionCommand,
    ) -> StepResult:
        if self.fail_step:
            raise RuntimeError(
                "synthetic step failure"
            )

        self.step_count += 1

        terminated = (
            self.terminate_at is not None
            and self.step_count
            >= self.terminate_at
        )

        return StepResult(
            observation=make_observation(
                self.step_count
            ),
            reward=self.reward,
            terminated=terminated,
            truncated=False,
            info={
                "reward_components": {
                    "frag": self.reward,
                },
                "event_types": (
                    "synthetic_event",
                ),
                "termination_reason": (
                    "synthetic_terminal"
                    if terminated
                    else None
                ),
            },
        )

    def close(self) -> None:
        self.close_count += 1


def transition_records(
    path: Path,
) -> list[dict[str, object]]:
    records = InspectionJSONLReader(
        path
    ).read_episode()

    return [
        record.data
        for record in records
        if record.type == "agent_transition"
    ]


def test_inspected_episode_writes_valid_frames(
    tmp_path: Path,
) -> None:
    output = tmp_path / "episode.jsonl"
    environment = FakeEnvironment()
    agent = make_agent(
        DiscreteAction.TURN_LEFT
    )

    result = run_inspected_neural_episode(
        environment,
        agent,
        max_steps=2,
        output_path=output,
        episode_id="inspection-episode-001",
    )

    records = InspectionJSONLReader(
        output
    ).read_episode()

    assert [
        record.type
        for record in records
    ] == [
        "episode_started",
        "agent_transition",
        "agent_transition",
        "episode_ended",
    ]

    assert result.run_result.steps == 2
    assert environment.close_count == 1

    transitions = transition_records(
        output
    )

    for data in transitions:
        for key in (
            "observation",
            "next_observation",
        ):
            snapshot = data[key]
            frame_record = snapshot[
                "policy_frame"
            ]

            frame_path = (
                output.parent
                / frame_record["path"]
            )

            assert frame_path.is_file()

            array = np.load(
                frame_path,
                allow_pickle=False,
            )

            assert array.shape == (
                4,
                3,
                90,
                160,
            )
            assert array.dtype == np.float32


def test_inspected_episode_records_policy_data(
    tmp_path: Path,
) -> None:
    output = tmp_path / "policy.jsonl"
    agent = make_agent(
        DiscreteAction.FIRE
    )

    run_inspected_neural_episode(
        FakeEnvironment(),
        agent,
        max_steps=1,
        output_path=output,
        checkpoint="checkpoints/policy-v2.pt",
    )

    decision = transition_records(
        output
    )[0]["decision"]

    assert decision["policy_name"] == (
        "inspection-test-policy"
    )
    assert decision["policy_version"] == "v2"
    assert decision["checkpoint"] == (
        "checkpoints/policy-v2.pt"
    )
    assert decision["action"]["name"] == "FIRE"
    assert decision["action_scores"]["FIRE"] == 8.0
    assert len(decision["action_scores"]) == 11


def test_inspected_episode_records_rewards(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reward.jsonl"

    result = run_inspected_neural_episode(
        FakeEnvironment(
            reward=1.5
        ),
        make_agent(
            DiscreteAction.FORWARD
        ),
        max_steps=1,
        output_path=output,
    )

    transition = transition_records(
        output
    )[0]

    assert transition["reward"] == 1.5
    assert transition[
        "reward_components"
    ]["frag"] == 1.5
    assert result.run_result.total_reward == 1.5


def test_inspected_episode_records_max_truncation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "truncated.jsonl"

    run_inspected_neural_episode(
        FakeEnvironment(),
        make_agent(
            DiscreteAction.NO_OP
        ),
        max_steps=2,
        output_path=output,
    )

    records = InspectionJSONLReader(
        output
    ).read_episode()

    ended = records[-1].data
    final_transition = records[-2].data

    assert ended["truncated"]
    assert not ended["terminated"]
    assert ended["outcome"] == "truncated"
    assert ended["summary"][
        "termination_reason"
    ] == "max_steps_reached"
    assert final_transition["truncated"]


def test_inspected_episode_records_terminal_result(
    tmp_path: Path,
) -> None:
    output = tmp_path / "terminal.jsonl"

    result = run_inspected_neural_episode(
        FakeEnvironment(
            terminate_at=1
        ),
        make_agent(
            DiscreteAction.JUMP
        ),
        max_steps=5,
        output_path=output,
    )

    records = InspectionJSONLReader(
        output
    ).read_episode()

    assert result.run_result.steps == 1
    assert result.run_result.terminated
    assert records[-1].data["terminated"]
    assert records[-1].data["outcome"] == (
        "terminated"
    )


def test_inspected_episode_writes_error_end_record(
    tmp_path: Path,
) -> None:
    output = tmp_path / "error.jsonl"
    environment = FakeEnvironment(
        fail_step=True
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic step failure",
    ):
        run_inspected_neural_episode(
            environment,
            make_agent(
                DiscreteAction.FORWARD
            ),
            max_steps=2,
            output_path=output,
        )

    records = InspectionJSONLReader(
        output
    ).read_episode()

    assert environment.close_count == 1
    assert records[-1].data["outcome"] == "error"
    assert records[-1].data["steps"] == 0
    assert records[-1].data["truncated"]
    assert records[-1].data["summary"][
        "error_type"
    ] == "RuntimeError"


def test_inspected_episode_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.jsonl"
    output.write_text(
        "preserve existing data\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
    ):
        run_inspected_neural_episode(
            FakeEnvironment(),
            make_agent(
                DiscreteAction.NO_OP
            ),
            max_steps=1,
            output_path=output,
        )

    assert output.read_text(
        encoding="utf-8"
    ) == "preserve existing data\n"


def test_inspected_episode_rejects_empty_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="checkpoint",
    ):
        run_inspected_neural_episode(
            FakeEnvironment(),
            make_agent(
                DiscreteAction.NO_OP
            ),
            max_steps=1,
            output_path=(
                tmp_path / "invalid.jsonl"
            ),
            checkpoint="",
        )
