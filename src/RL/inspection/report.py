"""Build readable summaries from completed inspection episodes."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from RL.inspection.jsonl import InspectionRecord


def _copy_mapping(
    value: Any,
) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    return {}


def _telemetry_from_observation(
    observation: Any,
) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        return None

    telemetry = observation.get("telemetry")

    if not isinstance(telemetry, dict):
        return None

    return dict(telemetry)


def _numeric_delta(
    start: dict[str, Any] | None,
    end: dict[str, Any] | None,
) -> dict[str, float]:
    if start is None or end is None:
        return {}

    result: dict[str, float] = {}

    for field_name in (
        "health",
        "armor",
        "ammo",
        "score",
        "match_time_seconds",
    ):
        start_value = start.get(field_name)
        end_value = end.get(field_name)

        if (
            isinstance(start_value, (int, float))
            and not isinstance(start_value, bool)
            and isinstance(end_value, (int, float))
            and not isinstance(end_value, bool)
        ):
            delta = (
                float(end_value) - float(start_value)
            )
            rounded_delta = round(delta, 6)

            result[field_name] = (
                0.0
                if rounded_delta == 0.0
                else rounded_delta
            )

    return result


@dataclass(frozen=True)
class EpisodeReport:
    """Summary of one validated agent-inspection episode."""

    episode_id: str
    metadata: dict[str, Any]
    outcome: str | None
    ended_summary: dict[str, Any]
    step_count: int
    terminated: bool
    truncated: bool
    total_reward: float
    action_counts: dict[str, int]
    goal_status_counts: dict[str, int]
    goal_category_counts: dict[str, int]
    goal_timeline: list[dict[str, Any]]
    important_transitions: list[dict[str, Any]]
    frame_references: list[dict[str, Any]]
    start_telemetry: dict[str, Any] | None
    end_telemetry: dict[str, Any] | None
    telemetry_delta: dict[str, float]

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return {
            "episode_id": self.episode_id,
            "metadata": dict(self.metadata),
            "outcome": self.outcome,
            "ended_summary": dict(self.ended_summary),
            "step_count": self.step_count,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "total_reward": self.total_reward,
            "action_counts": dict(self.action_counts),
            "goal_status_counts": dict(
                self.goal_status_counts
            ),
            "goal_category_counts": dict(
                self.goal_category_counts
            ),
            "goal_timeline": list(self.goal_timeline),
            "important_transitions": list(
                self.important_transitions
            ),
            "frame_references": list(
                self.frame_references
            ),
            "start_telemetry": self.start_telemetry,
            "end_telemetry": self.end_telemetry,
            "telemetry_delta": dict(
                self.telemetry_delta
            ),
        }


def build_episode_report(
    records: Sequence[InspectionRecord],
) -> EpisodeReport:
    """Build a report from one validated episode."""

    if not records:
        raise ValueError(
            "inspection records must not be empty"
        )

    if records[0].type != "episode_started":
        raise ValueError(
            "report requires episode_started first"
        )

    if records[-1].type != "episode_ended":
        raise ValueError(
            "report requires episode_ended last"
        )

    transitions = [
        record
        for record in records
        if record.type == "agent_transition"
    ]

    if not transitions:
        raise ValueError(
            "report requires at least one transition"
        )

    started = records[0].data
    ended = records[-1].data

    action_counts: Counter[str] = Counter()
    goal_status_counts: Counter[str] = Counter()
    goal_category_counts: Counter[str] = Counter()

    goal_timeline: list[dict[str, Any]] = []
    important_transitions: list[
        dict[str, Any]
    ] = []
    frame_references: list[dict[str, Any]] = []

    total_reward = 0.0

    for transition in transitions:
        data = transition.data
        step_index = int(data["step_index"])
        reward = float(data["reward"])
        total_reward += reward

        decision = _copy_mapping(
            data.get("decision")
        )
        action = _copy_mapping(
            decision.get("action")
        )
        action_name = str(
            action.get("name", "UNKNOWN")
        )
        action_counts[action_name] += 1

        observation = _copy_mapping(
            data.get("observation")
        )
        raw_frame = _copy_mapping(
            observation.get("raw_frame")
        )
        policy_frame = _copy_mapping(
            observation.get("policy_frame")
        )

        frame_references.append(
            {
                "step_index": step_index,
                "tick": observation.get("tick"),
                "raw_frame": raw_frame.get("path"),
                "policy_frame": (
                    policy_frame.get("path")
                ),
                "policy_transform": (
                    policy_frame.get("transform")
                ),
            }
        )

        goal = data.get("goal")
        reasons: list[str] = []

        if isinstance(goal, dict):
            goal_status = str(
                goal.get("status", "unknown")
            )
            goal_category = str(
                goal.get("category", "unknown")
            )

            goal_status_counts[goal_status] += 1
            goal_category_counts[
                goal_category
            ] += 1

            goal_entry = {
                "step_index": step_index,
                "goal_id": goal.get("goal_id"),
                "category": goal_category,
                "target": goal.get("target"),
                "trigger": goal.get("trigger"),
                "progress": goal.get("progress"),
                "status": goal_status,
                "steps_active": (
                    goal.get("steps_active")
                ),
                "completion_reason": (
                    goal.get("completion_reason")
                ),
                "failure_reason": (
                    goal.get("failure_reason")
                ),
                "action": action_name,
            }
            goal_timeline.append(goal_entry)

            if goal_status != "active":
                reasons.append(
                    f"goal_{goal_status}"
                )

        if reward != 0.0:
            reasons.append("nonzero_reward")

        if data.get("terminated") is True:
            reasons.append("episode_terminated")

        if data.get("truncated") is True:
            reasons.append("episode_truncated")

        current_telemetry = (
            _telemetry_from_observation(
                observation
            )
        )
        next_telemetry = (
            _telemetry_from_observation(
                data.get("next_observation")
            )
        )

        if (
            current_telemetry is not None
            and next_telemetry is not None
            and current_telemetry.get("health")
            != next_telemetry.get("health")
        ):
            reasons.append("health_changed")

        if reasons:
            important_transitions.append(
                {
                    "step_index": step_index,
                    "tick": observation.get("tick"),
                    "action": action_name,
                    "reward": reward,
                    "goal_id": (
                        goal.get("goal_id")
                        if isinstance(goal, dict)
                        else None
                    ),
                    "reasons": reasons,
                }
            )

    first_observation = transitions[0].data.get(
        "observation"
    )
    start_telemetry = _telemetry_from_observation(
        first_observation
    )

    final_transition = transitions[-1].data
    final_observation = final_transition.get(
        "next_observation"
    )

    if final_observation is None:
        final_observation = final_transition.get(
            "observation"
        )

    end_telemetry = _telemetry_from_observation(
        final_observation
    )

    return EpisodeReport(
        episode_id=str(started["episode_id"]),
        metadata=_copy_mapping(
            started.get("metadata")
        ),
        outcome=ended.get("outcome"),
        ended_summary=_copy_mapping(
            ended.get("summary")
        ),
        step_count=len(transitions),
        terminated=bool(ended["terminated"]),
        truncated=bool(ended["truncated"]),
        total_reward=total_reward,
        action_counts=dict(
            sorted(action_counts.items())
        ),
        goal_status_counts=dict(
            sorted(goal_status_counts.items())
        ),
        goal_category_counts=dict(
            sorted(goal_category_counts.items())
        ),
        goal_timeline=goal_timeline,
        important_transitions=important_transitions,
        frame_references=frame_references,
        start_telemetry=start_telemetry,
        end_telemetry=end_telemetry,
        telemetry_delta=_numeric_delta(
            start_telemetry,
            end_telemetry,
        ),
    )


def _format_counts(
    values: dict[str, int],
) -> str:
    if not values:
        return "none"

    return ", ".join(
        f"{name}={count}"
        for name, count in values.items()
    )


def render_episode_report_text(
    report: EpisodeReport,
) -> str:
    """Render a readable text report."""

    metadata = report.metadata

    lines = [
        "=" * 72,
        "AGENT EPISODE INSPECTION REPORT",
        "=" * 72,
        f"episode_id={report.episode_id}",
        (
            "curriculum_stage="
            f"{metadata.get('curriculum_stage')}"
        ),
        f"bot_skill={metadata.get('bot_skill')}",
        (
            "learning_phase="
            f"{metadata.get('learning_phase')}"
        ),
        (
            "frame_references_are_real="
            f"{metadata.get('frame_references_are_real')}"
        ),
        f"outcome={report.outcome}",
        f"steps={report.step_count}",
        f"terminated={report.terminated}",
        f"truncated={report.truncated}",
        f"total_reward={report.total_reward:.6f}",
        "",
        "ACTION COUNTS",
        _format_counts(report.action_counts),
        "",
        "GOAL STATUS COUNTS",
        _format_counts(
            report.goal_status_counts
        ),
        "",
        "GOAL CATEGORY COUNTS",
        _format_counts(
            report.goal_category_counts
        ),
        "",
        "TELEMETRY",
        f"start={report.start_telemetry}",
        f"end={report.end_telemetry}",
        f"delta={report.telemetry_delta}",
        "",
        "GOAL TIMELINE",
    ]

    for goal in report.goal_timeline:
        lines.append(
            "step={step_index} "
            "goal={goal_id} "
            "category={category} "
            "status={status} "
            "progress={progress} "
            "action={action}".format(**goal)
        )

    lines.extend(
        [
            "",
            "IMPORTANT TRANSITIONS",
        ]
    )

    if report.important_transitions:
        for transition in (
            report.important_transitions
        ):
            lines.append(
                "step={step_index} "
                "tick={tick} "
                "action={action} "
                "reward={reward:.2f} "
                "reasons={reasons}".format(
                    **transition
                )
            )
    else:
        lines.append("none")

    lines.extend(
        [
            "",
            "AI-VIEW FRAME REFERENCES",
        ]
    )

    for frame in report.frame_references:
        lines.append(
            "step={step_index} tick={tick}".format(
                **frame
            )
        )
        lines.append(
            f"  raw_frame={frame['raw_frame']}"
        )
        lines.append(
            "  policy_frame="
            f"{frame['policy_frame']}"
        )
        lines.append(
            "  policy_transform="
            f"{frame['policy_transform']}"
        )

    lines.extend(
        [
            "",
            "NOTE",
            (
                "Frame paths are references only. "
                "They are displayable only when "
                "the referenced image or tensor "
                "files actually exist."
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def write_episode_report_files(
    report: EpisodeReport,
    *,
    text_path: str | Path,
    json_path: str | Path,
) -> None:
    """Write readable text and structured JSON reports."""

    resolved_text_path = Path(text_path)
    resolved_json_path = Path(json_path)

    resolved_text_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolved_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_text_path.write_text(
        render_episode_report_text(report),
        encoding="utf-8",
    )

    resolved_json_path.write_text(
        json.dumps(
            report.to_record(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
