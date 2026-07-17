"""Interpret mode-specific Xonotic score labels and snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from RL.events import Event


@dataclass(frozen=True)
class ScoreLabel:
    """Meaning and ordering metadata for one score column."""

    name: str
    raw: str
    primary: bool = False
    secondary: bool = False
    lower_is_better: bool = False


def _parse_value(value: str) -> Any:
    """Convert numeric score values while preserving nonnumeric values."""

    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)

    try:
        return float(value)
    except ValueError:
        return value


def _parse_label(raw: str) -> ScoreLabel:
    """Remove Xonotic ordering flags from a score-label name."""

    suffixes = (
        ("<!!", True, False, True),
        ("!!", True, False, False),
        ("<!", False, True, True),
        ("!", False, True, False),
        ("<", False, False, True),
    )

    for suffix, primary, secondary, lower in suffixes:
        if raw.endswith(suffix):
            return ScoreLabel(
                name=raw[:-len(suffix)],
                raw=raw,
                primary=primary,
                secondary=secondary,
                lower_is_better=lower,
            )

    return ScoreLabel(name=raw, raw=raw)


class XonoticScoreboardTracker:
    """Bind generic score snapshots to their mode-specific labels."""

    def __init__(self) -> None:
        self.player_labels: list[ScoreLabel] = []
        self.team_labels: list[ScoreLabel] = []

    @staticmethod
    def _labels(values: Iterable[str]) -> list[ScoreLabel]:
        return [_parse_label(value) for value in values]

    @staticmethod
    def _enrich(
        event: Event,
        labels: list[ScoreLabel],
    ) -> Event:
        raw_scores = event.data.get("scores", [])

        if not isinstance(raw_scores, list):
            return event

        fields: dict[str, Any] = {}

        for label, value in zip(labels, raw_scores):
            if label.name:
                fields[label.name] = _parse_value(str(value))

        primary = next(
            (label for label in labels if label.primary and label.name),
            None,
        )

        data = dict(event.data)
        data["score_fields"] = fields

        if primary is not None:
            data["primary_score_name"] = primary.name
            data["primary_score"] = fields.get(primary.name)
            data["primary_lower_is_better"] = (
                primary.lower_is_better
            )

        return Event(type=event.type, data=data)

    def process(self, event: Event) -> Event:
        """Update the active schema or enrich a score snapshot."""

        if event.type == "player_score_labels":
            labels = event.data.get("labels", [])

            if isinstance(labels, list):
                self.player_labels = self._labels(labels)

            return event

        if event.type == "team_score_labels":
            labels = event.data.get("labels", [])

            if isinstance(labels, list):
                self.team_labels = self._labels(labels)

            return event

        if event.type == "player_score_snapshot":
            return self._enrich(event, self.player_labels)

        if event.type == "team_score_snapshot":
            return self._enrich(event, self.team_labels)

        return event
