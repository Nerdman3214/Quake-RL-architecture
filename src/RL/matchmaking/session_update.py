"""Safely update adaptive state from completed JSONL matches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adaptive import (
    AdaptiveState,
    load_state,
    save_state,
    update_state,
)
from .performance import (
    extract_match_performance,
    load_jsonl_records,
)


Record = Mapping[str, Any]


@dataclass(frozen=True)
class AdaptiveSessionUpdateResult:
    """Outcome of processing one recorder session."""

    status: str
    completed_matches: int = 0
    updated_matches: int = 0
    duplicate_matches: int = 0
    match_keys: tuple[str, ...] = ()
    previous_rating: float | None = None
    rating: float | None = None
    previous_skill: int | None = None
    current_skill: int | None = None
    state_matches: int | None = None
    error: str | None = None


def _session_metadata(
    records: Sequence[Record],
) -> Mapping[str, Any]:
    return next(
        (
            record.get("data", {})
            for record in records
            if record.get("type") == "session_started"
            and isinstance(
                record.get("data"),
                Mapping,
            )
        ),
        {},
    )


def _completed_match_count(
    records: Sequence[Record],
) -> int:
    starts = [
        index
        for index, record in enumerate(records)
        if record.get("type") == "match_started"
    ]

    completed = 0

    for position, start in enumerate(starts):
        end = (
            starts[position + 1]
            if position + 1 < len(starts)
            else len(records)
        )

        if any(
            record.get("type") == "match_ended"
            for record in records[start:end]
        ):
            completed += 1

    return completed


def update_adaptive_state_from_session(
    *,
    session_path: Path,
    state_path: Path,
    controlled_player: str = "Noobnog",
) -> AdaptiveSessionUpdateResult:
    """Apply every unseen completed adaptive match atomically."""

    try:
        records = load_jsonl_records(session_path)
    except Exception as error:
        return AdaptiveSessionUpdateResult(
            status="extraction_failed",
            error=str(error),
        )

    metadata = _session_metadata(records)

    if (
        str(
            metadata.get(
                "matchmaking",
                "",
            )
        ).casefold()
        != "adaptive"
    ):
        return AdaptiveSessionUpdateResult(
            status="not_adaptive",
        )

    completed_matches = _completed_match_count(
        records
    )

    if completed_matches == 0:
        return AdaptiveSessionUpdateResult(
            status="no_completed_match",
        )

    session_id = str(
        metadata.get("session_id")
        or session_path.stem.removeprefix(
            "session_"
        )
    )

    try:
        state = load_state(state_path)
    except Exception as error:
        return AdaptiveSessionUpdateResult(
            status="state_error",
            completed_matches=completed_matches,
            error=str(error),
        )

    seen = set(state.processed_match_keys)
    duplicates: list[str] = []
    pending = []

    for match_index in range(completed_matches):
        match_key = (
            f"{session_id}:{match_index}"
        )

        if match_key in seen:
            duplicates.append(match_key)
            continue

        try:
            performance = extract_match_performance(
                records,
                controlled_player=controlled_player,
                match_index=match_index,
                fallback_session_id=session_id,
            )
        except Exception as error:
            return AdaptiveSessionUpdateResult(
                status="extraction_failed",
                completed_matches=completed_matches,
                duplicate_matches=len(
                    duplicates
                ),
                match_keys=tuple(duplicates),
                previous_rating=state.rating,
                rating=state.rating,
                previous_skill=state.current_skill,
                current_skill=state.current_skill,
                state_matches=state.matches,
                error=(
                    f"match {match_index}: "
                    f"{error}"
                ),
            )

        pending.append(
            (
                match_key,
                performance,
            )
        )

    if not pending:
        return AdaptiveSessionUpdateResult(
            status="duplicate",
            completed_matches=completed_matches,
            duplicate_matches=len(duplicates),
            match_keys=tuple(duplicates),
            previous_rating=state.rating,
            rating=state.rating,
            previous_skill=state.current_skill,
            current_skill=state.current_skill,
            state_matches=state.matches,
        )

    updated_state: AdaptiveState = state
    updated_keys: list[str] = []

    for match_key, performance in pending:
        updated_state = update_state(
            updated_state,
            performance,
            match_key=match_key,
        )
        updated_keys.append(match_key)

    try:
        save_state(
            updated_state,
            state_path,
        )
    except Exception as error:
        return AdaptiveSessionUpdateResult(
            status="state_error",
            completed_matches=completed_matches,
            duplicate_matches=len(duplicates),
            match_keys=tuple(
                duplicates + updated_keys
            ),
            previous_rating=state.rating,
            rating=state.rating,
            previous_skill=state.current_skill,
            current_skill=state.current_skill,
            state_matches=state.matches,
            error=str(error),
        )

    return AdaptiveSessionUpdateResult(
        status="updated",
        completed_matches=completed_matches,
        updated_matches=len(updated_keys),
        duplicate_matches=len(duplicates),
        match_keys=tuple(
            duplicates + updated_keys
        ),
        previous_rating=state.rating,
        rating=updated_state.rating,
        previous_skill=state.current_skill,
        current_skill=updated_state.current_skill,
        state_matches=updated_state.matches,
    )
