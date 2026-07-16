# Quake/Xonotic RL Shared AI Handoff

This file is maintained locally by `handoff_manager.py`.

## Coworker protocol

- Steven runs commands, gameplay tests, and approves behavior goals.
- ChatGPT maintains architecture, reviews integration, and plans milestones.
- Odysseus/Qwen performs local inspection and narrow implementation tasks.
- Every assistant must clearly label mocks, placeholders, and unverified claims.

## Current milestone

Establish one real Xonotic observation/event channel and one real action channel before training.

## Permanent facts

- Project root: `/media/steven/WINPE2/Quake-RL-architecture`
- Xonotic root: `/media/steven/WINPE2/Xonotic`
- Manual player name: `Noobnog`
- Local server port previously observed: `26000`
- Current Python smoke tests are mock-only unless explicitly proven otherwise.

## Automatic repository snapshot

<!-- AUTO-SNAPSHOT-START -->
- Updated: `2026-07-15T21:31:35-05:00`
- Project exists: `True`
- Xonotic exists: `True`
- Git branch: `main`
- Git HEAD: `1e11527`

### Git status
```text
M src/RL/docs/AI_HANDOFF.md
 M src/RL/docs/PROJECT_STATE.json
?? PROJECT_FILES.txt
?? events.jsonl
?? jsonl_writer.py
?? src/RL/docs/EVENT_SCHEMA.md
?? src/RL/engine/bridge/event_stream.py
?? src/RL/events/
?? src/RL/jsonl_reader.py
?? src/RL/tests/unit/test_events.py
?? src/RL/tests/unit/test_jsonl_pipeline.py
?? src/RL/tools/inspection/smoke_test_events.py
?? src/RL/tools/launch/live_event_recording_test.py
```

### Recently modified files
- `src/RL/docs/AI_HANDOFF.md` — 2026-07-15T21:31:20-05:00
- `src/RL/docs/PROJECT_STATE.json` — 2026-07-15T21:31:20-05:00
- `src/RL/tools/launch/live_event_recording_test.py` — 2026-07-15T21:15:24-05:00
- `src/RL/engine/bridge/event_stream.py` — 2026-07-15T21:13:39-05:00
- `src/RL/engine/bridge/__pycache__/event_stream.cpython-312.pyc` — 2026-07-15T21:11:19-05:00
- `src/RL/tools/inspection/smoke_test_events.py` — 2026-07-15T20:51:41-05:00
- `src/RL/tests/unit/__pycache__/test_jsonl_pipeline.cpython-312-pytest-7.4.3.pyc` — 2026-07-15T20:46:32-05:00
- `src/RL/tests/unit/test_jsonl_pipeline.py` — 2026-07-15T20:46:28-05:00
- `src/RL/tests/unit/__pycache__/test_events.cpython-312.pyc` — 2026-07-15T14:49:34-05:00
- `src/RL/__pycache__/jsonl_reader.cpython-312.pyc` — 2026-07-15T14:49:34-05:00
- `src/RL/events/__pycache__/schema.cpython-312.pyc` — 2026-07-15T14:49:34-05:00
- `src/RL/events/__pycache__/jsonl_writer.cpython-312.pyc` — 2026-07-15T14:49:34-05:00
- `src/RL/events/__pycache__/jsonl_reader.cpython-312.pyc` — 2026-07-15T14:49:34-05:00
- `src/RL/events/__pycache__/contracts.cpython-312.pyc` — 2026-07-15T14:49:34-05:00
- `src/RL/events/__pycache__/__init__.cpython-312.pyc` — 2026-07-15T14:49:34-05:00

### Xonotic executable check
- `xonotic-linux64-glx`: exists=`True`, executable=`True`
- `xonotic-linux64-sdl`: exists=`True`, executable=`True`
- `xonotic-linux64-dedicated`: exists=`True`, executable=`True`
<!-- AUTO-SNAPSHOT-END -->

## Assistant handoff entries
