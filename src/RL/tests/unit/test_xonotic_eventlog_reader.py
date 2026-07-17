"""Tests for Xonotic's authoritative structured eventlog."""

from RL.engine.server.eventlog_reader import XonoticEventLogReader


def test_match_start_identifies_mode_and_map() -> None:
    reader = XonoticEventLogReader()
    event = reader.parse_line(":gamestart:ctf_afterslime:match42")

    assert event is not None
    assert event.type == "match_started"
    assert event.data["game_mode"] == "ctf"
    assert event.data["map_name"] == "afterslime"
    assert event.data["match_id"] == "match42"


def test_join_and_frag_resolve_player_names() -> None:
    reader = XonoticEventLogReader()

    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")
    reader.parse_line(":join:2:2:bot:[BOT]Dominator")

    event = reader.parse_line(
        ":kill:frag:1:2:type=7:items=7:victimitems=2"
    )

    assert event is not None
    assert event.type == "player_kill"
    assert event.data["killer"] == "Noobnog"
    assert event.data["victim"] == "[BOT]Dominator"
    assert event.data["kill_kind"] == "frag"
    assert event.data["death_type"] == "7"


def test_team_kill_remains_distinguishable() -> None:
    reader = XonoticEventLogReader()

    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")
    reader.parse_line(":join:2:2:bot:[BOT]Partner")

    event = reader.parse_line(
        ":kill:tk:1:2:type=2:items=2:victimitems=2"
    )

    assert event is not None
    assert event.type == "player_kill"
    assert event.data["kill_kind"] == "tk"


def test_suicide_event() -> None:
    reader = XonoticEventLogReader()
    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")

    event = reader.parse_line(
        ":kill:suicide:1:1:type=4:items=4"
    )

    assert event is not None
    assert event.type == "player_suicide"
    assert event.data["player_name"] == "Noobnog"


def test_ctf_capture() -> None:
    reader = XonoticEventLogReader()
    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")

    event = reader.parse_line(":ctf:capture:14:1")

    assert event is not None
    assert event.type == "ctf_flag_event"
    assert event.data["action"] == "capture"
    assert event.data["flag_color"] == "14"
    assert event.data["player_name"] == "Noobnog"


def test_domination_point_taken() -> None:
    reader = XonoticEventLogReader()
    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")

    event = reader.parse_line(":dom:taken:14:1")

    assert event is not None
    assert event.type == "domination_point_taken"
    assert event.data["previous_team"] == "BLUE"
    assert event.data["player_name"] == "Noobnog"


def test_keyhunt_event() -> None:
    reader = XonoticEventLogReader()
    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")
    reader.parse_line(":join:2:2:bot:[BOT]Dominator")

    event = reader.parse_line(
        ":keyhunt:collect:1:5:2:2:Blue Key"
    )

    assert event is not None
    assert event.type == "keyhunt_event"
    assert event.data["action"] == "collect"
    assert event.data["player_name"] == "Noobnog"
    assert event.data["key_owner_name"] == "[BOT]Dominator"


def test_player_score_snapshot_keeps_mode_specific_values() -> None:
    reader = XonoticEventLogReader()

    event = reader.parse_line(
        ":player:see-labels:20,3,1:400:5:1:Noobnog"
    )

    assert event is not None
    assert event.type == "player_score_snapshot"
    assert event.data["scores"] == ["20", "3", "1"]
    assert event.data["team"] == "5"
    assert event.data["player_name"] == "Noobnog"


def test_race_record() -> None:
    reader = XonoticEventLogReader()
    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")

    event = reader.parse_line(":recordset:1:42.125")

    assert event is not None
    assert event.type == "race_record_set"
    assert event.data["player_name"] == "Noobnog"
    assert event.data["time_seconds"] == "42.125"


def test_match_start_delay_ended() -> None:
    reader = XonoticEventLogReader()
    event = reader.parse_line(":startdelay_ended")

    assert event is not None
    assert event.type == "match_start_delay_ended"
    assert event.data["raw_line"] == ":startdelay_ended"


def test_spectator_team_is_named() -> None:
    reader = XonoticEventLogReader()
    reader.parse_line(":join:1:1:127.0.0.1:Noobnog")

    event = reader.parse_line(":team:1:-1:4")

    assert event is not None
    assert event.type == "player_team_changed"
    assert event.data["player_name"] == "Noobnog"
    assert event.data["team"] == "SPECTATOR"
    assert event.data["join_type"] == "4"



def test_real_ctf_line_uses_final_player_id() -> None:
    from RL.engine.server.eventlog_reader import (
        XonoticEventLogReader,
    )

    reader = XonoticEventLogReader()

    reader.parse_line(
        ":gamestart:ctf_runningmanctf:match1"
    )
    reader.parse_line(
        ":join:1:1:127.0.0.1:Noobnog"
    )

    event = reader.parse_line(
        ":ctf:steal:5:14:1"
    )

    assert event is not None
    assert event.type == "ctf_flag_event"
    assert event.data["player_id"] == "1"
    assert event.data["player_name"] == "Noobnog"


def test_recordset_uses_active_match_context() -> None:
    from RL.engine.server.eventlog_reader import (
        XonoticEventLogReader,
    )

    reader = XonoticEventLogReader()

    reader.parse_line(
        ":gamestart:ctf_runningmanctf:match1"
    )
    ctf_event = reader.parse_line(
        ":recordset:1:22.933319"
    )

    assert ctf_event is not None
    assert ctf_event.type == (
        "ctf_capture_record_set"
    )

    reader.parse_line(
        ":gamestart:rc_runningman:match2"
    )
    race_event = reader.parse_line(
        ":recordset:1:18.000000"
    )

    assert race_event is not None
    assert race_event.type == "race_record_set"
