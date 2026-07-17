"""Extract normalized adaptive evidence from completed JSONL matches."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adaptive import MatchPerformance, clamp


Record = Mapping[str, Any]
SUPPORTED_MODES = {"ctf", "dom", "kh"}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number or number in (
        float("inf"),
        float("-inf"),
    ):
        return None

    return number


def _field(data: Record, name: str) -> float:
    fields = data.get("score_fields", {})

    if not isinstance(fields, Mapping):
        return 0.0

    value = _number(fields.get(name))

    return value if value is not None else 0.0


def load_jsonl_records(
    path: Path,
) -> list[dict[str, Any]]:
    """Load normalized records from one JSONL session."""

    records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8",
        ).splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON on line "
                f"{line_number}: {path}"
            ) from error

        if (
            isinstance(record, dict)
            and isinstance(
                record.get("type"),
                str,
            )
            and isinstance(
                record.get("data"),
                dict,
            )
        ):
            records.append(record)

    return records


def _completed_match(
    records: Sequence[Record],
    match_index: int,
) -> list[Record]:
    starts = [
        index
        for index, record in enumerate(records)
        if record.get("type") == "match_started"
    ]

    completed: list[list[Record]] = []

    for position, start in enumerate(starts):
        end = (
            starts[position + 1]
            if position + 1 < len(starts)
            else len(records)
        )

        candidate = list(records[start:end])

        if any(
            record.get("type") == "match_ended"
            for record in candidate
        ):
            completed.append(candidate)

    if not 0 <= match_index < len(completed):
        raise ValueError(
            f"Completed match index "
            f"{match_index} is unavailable"
        )

    return completed[match_index]


def _latest(
    match: Sequence[Record],
    event_type: str,
    key_name: str,
) -> list[dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}

    for record in match:
        if record.get("type") != event_type:
            continue

        data = record.get("data", {})

        if (
            isinstance(data, dict)
            and data.get(key_name) is not None
        ):
            snapshots[
                str(data[key_name])
            ] = data

    return list(snapshots.values())


def _objective(
    mode: str,
    player: Record,
) -> float:
    if mode == "ctf":
        return (
            3.0 * _field(player, "caps")
            + _field(player, "fckills")
            + _field(player, "returns")
            + 0.25 * _field(
                player,
                "pickups",
            )
            - 0.5 * _field(
                player,
                "drops",
            )
        )

    if mode == "dom":
        return (
            _field(player, "takes")
            + 0.25 * _field(
                player,
                "ticks",
            )
        )

    return (
        3.0 * _field(player, "caps")
        + _field(player, "kckills")
        + 0.5 * _field(
            player,
            "destroyed",
        )
        + 0.25 * _field(
            player,
            "pickups",
        )
        + 0.25 * _field(
            player,
            "pushes",
        )
        - 0.5 * _field(
            player,
            "losses",
        )
    )


def _combat(
    player: Record,
) -> float:
    kills = _field(player, "kills")
    deaths = _field(player, "deaths")
    total = kills + deaths

    if total <= 0.0:
        return 0.0

    return clamp(
        (kills - deaths) / total,
        -1.0,
        1.0,
    )


def _discipline(
    player: Record,
) -> float:
    kills = _field(player, "kills")
    deaths = _field(player, "deaths")
    teamkills = _field(
        player,
        "teamkills",
    )
    suicides = _field(
        player,
        "suicides",
    )

    total = (
        kills
        + deaths
        + teamkills
        + suicides
    )

    if total <= 0.0:
        return 0.0

    return clamp(
        (
            2.0 * teamkills
            + suicides
        )
        / total,
        0.0,
        1.0,
    )


def extract_match_performance(
    records: Sequence[Record],
    *,
    controlled_player: str = "Noobnog",
    match_index: int = 0,
    fallback_session_id: str = "",
) -> MatchPerformance:
    """Extract strict scoreboard-grounded evidence."""

    metadata = next(
        (
            record.get("data", {})
            for record in records
            if record.get("type")
            == "session_started"
        ),
        {},
    )

    match = _completed_match(
        records,
        match_index,
    )

    started = next(
        record.get("data", {})
        for record in match
        if record.get("type")
        == "match_started"
    )

    mode = str(
        started.get(
            "game_mode",
            "",
        )
    ).casefold()

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported adaptive mode: "
            f"{mode!r}"
        )

    players = _latest(
        match,
        "player_score_snapshot",
        "player_id",
    )

    teams = _latest(
        match,
        "team_score_snapshot",
        "team_id",
    )

    wanted = controlled_player.casefold()

    controlled = next(
        (
            player
            for player in players
            if str(
                player.get(
                    "player_name",
                    "",
                )
            ).casefold()
            == wanted
        ),
        None,
    )

    if controlled is None:
        raise ValueError(
            f"Final scoreboard for "
            f"{controlled_player!r} "
            "was not found"
        )

    raw_team = str(
        controlled.get(
            "team",
            "",
        )
    ).strip()

    controlled_team = next(
        (
            team
            for team in teams
            if raw_team
            in {
                str(
                    team.get(
                        "team_id",
                        "",
                    )
                ).strip(),
                str(
                    team.get(
                        "team",
                        "",
                    )
                ).strip().upper(),
            }
        ),
        None,
    )

    if controlled_team is None:
        raise ValueError(
            f"Team scoreboard for "
            f"{raw_team!r} was not found"
        )

    team_id = str(
        controlled_team.get(
            "team_id",
            "",
        )
    ).strip()

    team_name = str(
        controlled_team.get(
            "team",
            "",
        )
    ).strip().upper()

    controlled_score = _number(
        controlled_team.get(
            "primary_score"
        )
    )

    if controlled_score is None:
        raise ValueError(
            "Controlled team has no "
            "numeric primary score"
        )

    opponent_scores = [
        score
        for team in teams
        if (
            str(
                team.get(
                    "team_id",
                    "",
                )
            ).strip()
            != team_id
            and str(
                team.get(
                    "team",
                    "",
                )
            ).strip().upper()
            != "NONE"
            and (
                score := _number(
                    team.get(
                        "primary_score"
                    )
                )
            )
            is not None
        )
    ]

    if not opponent_scores:
        raise ValueError(
            "No numeric opponent team "
            "score was found"
        )

    lower_is_better = bool(
        controlled_team.get(
            "primary_lower_is_better",
            False,
        )
    )

    best_opponent = (
        min(opponent_scores)
        if lower_is_better
        else max(opponent_scores)
    )

    difference = (
        best_opponent
        - controlled_score
        if lower_is_better
        else controlled_score
        - best_opponent
    )

    score_margin = clamp(
        difference
        / max(
            abs(controlled_score),
            abs(best_opponent),
            1.0,
        ),
        -1.0,
        1.0,
    )

    team_winners = [
        str(
            record.get(
                "data",
                {},
            ).get(
                "team",
                "",
            )
        ).upper()
        for record in match
        if record.get("type")
        == "team_match_win"
    ]

    player_winners = [
        str(
            record.get(
                "data",
                {},
            ).get(
                "player_name",
                "",
            )
        ).casefold()
        for record in match
        if record.get("type")
        == "player_match_win"
    ]

    if team_winners:
        won = team_name in team_winners
    elif player_winners:
        won = wanted in player_winners
    else:
        won = score_margin > 0.0

    teammates = [
        player
        for player in players
        if str(
            player.get(
                "team",
                "",
            )
        ).strip().upper()
        in {
            team_id.upper(),
            team_name,
        }
    ]

    controlled_objective = max(
        0.0,
        _objective(
            mode,
            controlled,
        ),
    )

    team_objective = sum(
        max(
            0.0,
            _objective(
                mode,
                player,
            ),
        )
        for player in teammates
    )

    objective_score = (
        clamp(
            controlled_objective
            / team_objective,
            0.0,
            1.0,
        )
        if team_objective > 0.0
        else 0.0
    )

    bot_skill = _number(
        metadata.get("bot_skill")
    )

    bot_count = _number(
        metadata.get("bot_count")
    )

    return MatchPerformance(
        mode=mode,
        won=won,
        score_margin=score_margin,
        combat_score=_combat(
            controlled
        ),
        objective_score=objective_score,
        discipline_penalty=_discipline(
            controlled
        ),
        bot_skill=int(
            bot_skill or 0
        ),
        bot_count=int(
            bot_count or 0
        ),
        session_id=str(
            metadata.get("session_id")
            or fallback_session_id
        ),
    )


def extract_match_performance_from_path(
    path: Path,
    *,
    controlled_player: str = "Noobnog",
    match_index: int = 0,
) -> MatchPerformance:
    """Load JSONL and extract one completed match."""

    return extract_match_performance(
        load_jsonl_records(path),
        controlled_player=controlled_player,
        match_index=match_index,
        fallback_session_id=(
            path.stem.removeprefix(
                "session_"
            )
        ),
    )
