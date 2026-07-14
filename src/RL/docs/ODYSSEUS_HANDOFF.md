# Odysseus Handoff: Quake Multiplayer RL

## Current objective
Prepare a multiplayer-first Quake/QuakeWorld RL environment. Do not implement training code until the environment lifecycle is deterministic.

## Existing repository facts
- The repository contains the original Quake/QuakeWorld source tree.
- An empty RL skeleton already existed at `src/RL/controller`, `src/RL/env`, `src/RL/tools`, and `src/RL/training`.
- The architecture has been expanded without deleting those original folders.

## Immediate inspection task for Odysseus
Inspect the exact local engine build and launch path, then report:

1. Which executable is used for the controlled client.
2. Which executable is used for a dedicated server.
3. The working directory and required game-data directories.
4. How console commands can be sent to client and server.
5. Where authoritative frag/death/score events appear.
6. Whether a remote-console, UDP, stdin, log-tail, shared-memory, or engine-hook bridge is most practical.
7. Existing source locations for client input, server scoring/events, screen capture, and networking.

Do not write broad framework code yet. Produce an evidence-based integration note with exact file paths and observed commands/logs.

## Constraints
- Multiplayer is the target, initially local versus bots.
- Start from scratch on the model.
- First prove basic RL works; supervised imitation and preference/RLHF are later phases.
- Preserve fair-play observations by default; privileged state must be an explicit debug mode.
- Keep Python for orchestration/training and C/C++ only where engine hooks or performance require it.
- No modifications to unrelated original engine files during the inspection milestone.

## Required report format
- Current milestone
- Repository/build facts
- Candidate bridge options ranked
- Recommended bridge and why
- Exact files likely to change
- Risks/blockers
- Next single implementation task
