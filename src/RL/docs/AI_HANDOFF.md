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
- Updated: `2026-07-14T12:53:52-05:00`
- Project exists: `True`
- Xonotic exists: `True`
- Git branch: `main`
- Git HEAD: `a7bcf5e`

### Git status
```text
M src/RL/docs/AI_HANDOFF.md
 M src/RL/docs/PROJECT_STATE.json
?? event_recording/events.log
?? src/RL/.gitignore
?? src/RL/ARCHITECTURE.md
?? src/RL/__init__.py
?? src/RL/actions/
?? src/RL/agents/
?? src/RL/checkpoints/
?? src/RL/config/
?? src/RL/controller/
?? src/RL/data/
?? src/RL/docs/ODYSSEUS_HANDOFF.md
?? src/RL/docs/__init__.py
?? src/RL/docs/decisions/
?? src/RL/engine/
?? src/RL/env/
?? src/RL/evaluation/
?? src/RL/logs/
?? src/RL/observations/
?? src/RL/rewards/
?? src/RL/runtime/
?? src/RL/tests/
?? src/RL/tools/
?? src/RL/training/
?? test.txt
```

### Recently modified files
- `src/RL/docs/AI_HANDOFF.md` — 2026-07-14T12:53:37-05:00
- `src/RL/docs/PROJECT_STATE.json` — 2026-07-14T12:53:37-05:00
- `src/RL/observations/contracts.py` — 2026-07-10T12:46:19-05:00
- `src/RL/rewards/contracts.py` — 2026-07-10T12:46:19-05:00
- `src/RL/runtime/launch.py` — 2026-07-10T12:46:19-05:00
- `src/RL/runtime/paths.py` — 2026-07-10T12:46:19-05:00
- `src/RL/runtime/validation.py` — 2026-07-10T12:46:19-05:00
- `src/RL/agents/baselines/contracts.py` — 2026-07-10T12:46:19-05:00
- `src/RL/engine/bridge/contracts.py` — 2026-07-10T12:46:19-05:00
- `src/RL/env/core/contracts.py` — 2026-07-10T12:46:19-05:00
- `src/RL/__init__.py` — 2026-07-10T12:46:18-05:00
- `src/RL/actions/contracts.py` — 2026-07-10T12:46:18-05:00
- `src/RL/actions/__init__.py` — 2026-07-10T12:46:18-05:00
- `src/RL/agents/__init__.py` — 2026-07-10T12:46:18-05:00
- `src/RL/checkpoints/__init__.py` — 2026-07-10T12:46:18-05:00

### Xonotic executable check
- `xonotic-linux64-glx`: exists=`True`, executable=`True`
- `xonotic-linux64-sdl`: exists=`True`, executable=`True`
- `xonotic-linux64-dedicated`: exists=`True`, executable=`True`
<!-- AUTO-SNAPSHOT-END -->

## Assistant handoff entries
