"""Tests for authoritative Xonotic event-source selection."""

from RL.engine.server.unified_event_reader import (
    XonoticUnifiedEventReader,
)


def activate_structured_match(
    reader: XonoticUnifiedEventReader,
) -> None:
    event = reader.parse_line(":gamestart:dm_boil:match1")

    assert event is not None
    assert event.type == "match_started"

    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")
    reader.parse_line(":join:2:2:bot:[BOT]Dominator")


def test_human_kill_is_kept_without_eventlog() -> None:
    reader = XonoticUnifiedEventReader()

    event = reader.parse_line(
        "[BOT]Dominator has been vaporized "
        "by Noobnog's Vortex"
    )

    assert event is not None
    assert event.type == "player_kill"
    assert event.data["killer"] == "Noobnog"
    assert event.data["event_channel"] == "human_console"
    assert event.data["authority_tier"] == "supplemental"


def test_structured_kill_suppresses_human_duplicate() -> None:
    reader = XonoticUnifiedEventReader()
    activate_structured_match(reader)

    structured = reader.parse_line(
        ":kill:frag:1:2:"
        "type=7:items=7-1:victimitems=2-1"
    )

    duplicate = reader.parse_line(
        "[BOT]Dominator has been vaporized "
        "by Noobnog's Vortex"
    )

    assert structured is not None
    assert structured.type == "player_kill"
    assert structured.data["killer"] == "Noobnog"
    assert structured.data["victim"] == "[BOT]Dominator"
    assert structured.data["event_channel"] == (
        "structured_eventlog"
    )
    assert structured.data["authority_tier"] == "primary"

    assert duplicate is not None
    assert duplicate.type == "suppressed_human_event"
    assert duplicate.data["suppressed_event_type"] == (
        "player_kill"
    )
    assert duplicate.data["raw_line"].startswith(
        "[BOT]Dominator"
    )
    assert reader.suppressed_human_count == 1


def test_human_shaping_events_survive_eventlog() -> None:
    reader = XonoticUnifiedEventReader()
    activate_structured_match(reader)

    first_blood = reader.parse_line(
        "Noobnog drew first blood!"
    )
    streak = reader.parse_line(
        "Noobnog made a TRIPLE FRAG!"
    )
    pickup = reader.parse_line(
        "Noobnog picked up Strength"
    )

    assert first_blood is not None
    assert first_blood.type == "first_blood"

    assert streak is not None
    assert streak.type == "frag_streak"

    assert pickup is not None
    assert pickup.type == "item_pickup"

    for event in (first_blood, streak, pickup):
        assert event.data["event_channel"] == "human_console"
        assert event.data["authority_tier"] == "supplemental"


def test_structured_score_snapshot_is_enriched() -> None:
    reader = XonoticUnifiedEventReader()
    activate_structured_match(reader)

    labels = reader.parse_line(
        ":labels:player:score!!,kills,deaths<"
    )

    snapshot = reader.parse_line(
        ":player:see-labels:"
        "30,30,10:520:-1:1:Noobnog"
    )

    assert labels is not None
    assert labels.type == "player_score_labels"

    assert snapshot is not None
    assert snapshot.type == "player_score_snapshot"
    assert snapshot.data["score_fields"] == {
        "score": 30,
        "kills": 30,
        "deaths": 10,
    }
    assert snapshot.data["primary_score"] == 30
    assert snapshot.data["event_channel"] == (
        "structured_eventlog"
    )


def test_unknown_structured_line_remains_primary() -> None:
    reader = XonoticUnifiedEventReader()
    activate_structured_match(reader)

    event = reader.parse_line(":future_mode:event:value")

    assert event is not None
    assert event.type == "eventlog_line"
    assert event.data["raw_line"] == (
        ":future_mode:event:value"
    )
    assert event.data["authority_tier"] == "primary"
