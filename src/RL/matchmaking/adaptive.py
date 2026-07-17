"""Persistent, conservative adaptive matchmaking policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STATE_VERSION = 1
DEFAULT_RATING = 4.0
DEFAULT_SKILL = 4
DEFAULT_HISTORY_LIMIT = 10
DEFAULT_DECISION_WINDOW = 3


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Clamp a numeric value to an inclusive range."""

    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class MatchPerformance:
    """Normalized evidence from one completed match.

    score_margin, combat_score, and objective_score use the
    inclusive range -1.0 through 1.0.

    discipline_penalty uses 0.0 through 1.0, where higher values
    indicate more teamkills, suicides, or similar mistakes.
    """

    mode: str
    won: bool
    score_margin: float
    combat_score: float
    objective_score: float
    discipline_penalty: float
    bot_skill: int
    bot_count: int
    session_id: str = ""


@dataclass(frozen=True)
class AdaptiveState:
    """Persistent matchmaking state for the controlled agent."""

    version: int = STATE_VERSION
    rating: float = DEFAULT_RATING
    current_skill: int = DEFAULT_SKILL
    matches: int = 0
    last_adjustment_match: int = 0
    recent_matches: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )


def performance_index(
    performance: MatchPerformance,
) -> float:
    """Return a conservative composite performance score."""

    outcome = 1.0 if performance.won else -1.0

    score_margin = clamp(
        performance.score_margin,
        -1.0,
        1.0,
    )
    combat = clamp(
        performance.combat_score,
        -1.0,
        1.0,
    )
    objective = clamp(
        performance.objective_score,
        -1.0,
        1.0,
    )
    discipline = clamp(
        performance.discipline_penalty,
        0.0,
        1.0,
    )

    score = (
        0.15 * outcome
        + 0.20 * score_margin
        + 0.25 * combat
        + 0.35 * objective
        - 0.15 * discipline
    )

    return clamp(score, -1.0, 1.0)


def load_state(path: Path) -> AdaptiveState:
    """Load matchmaking state or return a new default state."""

    if not path.exists():
        return AdaptiveState()

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to load adaptive state: {path}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Adaptive matchmaking state must be a JSON object"
        )

    history = data.get("recent_matches", [])

    if not isinstance(history, list):
        raise ValueError(
            "recent_matches must be a JSON array"
        )

    normalized_history = tuple(
        dict(item)
        for item in history
        if isinstance(item, dict)
    )

    return AdaptiveState(
        version=int(
            data.get("version", STATE_VERSION)
        ),
        rating=float(
            data.get("rating", DEFAULT_RATING)
        ),
        current_skill=int(
            data.get("current_skill", DEFAULT_SKILL)
        ),
        matches=int(data.get("matches", 0)),
        last_adjustment_match=int(
            data.get("last_adjustment_match", 0)
        ),
        recent_matches=normalized_history,
    )


def save_state(
    state: AdaptiveState,
    path: Path,
) -> None:
    """Atomically save adaptive matchmaking state."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    payload = asdict(state)

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def update_state(
    state: AdaptiveState,
    performance: MatchPerformance,
    *,
    minimum_skill: int = 0,
    maximum_skill: int = 8,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    decision_window: int = DEFAULT_DECISION_WINDOW,
) -> AdaptiveState:
    """Record one match and conservatively adjust difficulty."""

    index = performance_index(performance)

    record: dict[str, Any] = {
        "session_id": performance.session_id,
        "mode": performance.mode,
        "won": performance.won,
        "score_margin": clamp(
            performance.score_margin,
            -1.0,
            1.0,
        ),
        "combat_score": clamp(
            performance.combat_score,
            -1.0,
            1.0,
        ),
        "objective_score": clamp(
            performance.objective_score,
            -1.0,
            1.0,
        ),
        "discipline_penalty": clamp(
            performance.discipline_penalty,
            0.0,
            1.0,
        ),
        "performance_index": index,
        "bot_skill": performance.bot_skill,
        "bot_count": performance.bot_count,
    }

    history = (
        state.recent_matches + (record,)
    )[-history_limit:]

    matches = state.matches + 1

    rating = clamp(
        state.rating + 0.25 * index,
        float(minimum_skill),
        float(maximum_skill),
    )

    current_skill = int(
        clamp(
            float(state.current_skill),
            float(minimum_skill),
            float(maximum_skill),
        )
    )

    last_adjustment = state.last_adjustment_match

    window = history[-decision_window:]

    enough_new_evidence = (
        len(window) == decision_window
        and matches - last_adjustment
        >= decision_window
    )

    if enough_new_evidence:
        window_size = float(len(window))

        average_index = sum(
            float(item.get("performance_index", 0.0))
            for item in window
        ) / window_size

        average_combat = sum(
            float(item.get("combat_score", 0.0))
            for item in window
        ) / window_size

        average_objective = sum(
            float(item.get("objective_score", 0.0))
            for item in window
        ) / window_size

        average_discipline = sum(
            float(
                item.get(
                    "discipline_penalty",
                    0.0,
                )
            )
            for item in window
        ) / window_size

        wins = sum(
            bool(item.get("won", False))
            for item in window
        )
        losses = len(window) - wins

        should_increase = (
            average_index >= 0.35
            and average_combat >= -0.05
            and average_objective >= 0.15
            and average_discipline <= 0.25
            and wins >= 2
        )

        should_decrease = (
            average_index <= -0.35
            and losses >= 2
            and (
                average_combat <= -0.15
                or average_objective <= 0.0
            )
        )

        if should_increase:
            updated_skill = min(
                maximum_skill,
                current_skill + 1,
            )

            if updated_skill != current_skill:
                current_skill = updated_skill
                last_adjustment = matches

        elif should_decrease:
            updated_skill = max(
                minimum_skill,
                current_skill - 1,
            )

            if updated_skill != current_skill:
                current_skill = updated_skill
                last_adjustment = matches

    return AdaptiveState(
        version=STATE_VERSION,
        rating=rating,
        current_skill=current_skill,
        matches=matches,
        last_adjustment_match=last_adjustment,
        recent_matches=history,
    )
