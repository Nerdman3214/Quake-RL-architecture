# Quake Multiplayer RL Architecture

## Project goal
Build a multiplayer-first reinforcement-learning system around Quake/QuakeWorld while keeping the game engine, environment contract, learning algorithms, and experiment data independently replaceable.

The first milestone is deliberately small: one controlled client, one local dedicated server, one repeatable map/match configuration, a discrete action set, stable observations, and rewards derived from authoritative match events.

## Design boundaries

1. **Engine layer** owns Quake client/server integration and transport only.
2. **Environment layer** exposes reset/step semantics and match lifecycle without knowing the learning algorithm.
3. **Observation, action, and reward layers** are separate contracts so each can evolve independently.
4. **Agent and training layers** consume the environment contract and must not directly parse engine memory or console output.
5. **Runtime layer** launches and supervises processes, ports, seeds, logs, and cleanup.
6. **Data layer** stores trajectories and metadata, not source code.
7. **Future imitation learning and preference/RLHF work** remain separate from the first online-RL milestone.

## Directory responsibilities

- `config/` — experiment, map, match, observation, action, reward, and runtime configuration.
- `runtime/` — process lifecycle, session identity, port allocation, health checks, and cleanup.
- `engine/client/` — client-side hooks, command injection, frame/state capture boundaries.
- `engine/server/` — dedicated-server control and authoritative match-event extraction.
- `engine/bridge/` — normalized transport between engine processes and Python.
- `controller/` — high-level match/session coordinator retained from the original skeleton.
- `env/core/` — common environment contract and lifecycle concepts.
- `env/multiplayer/` — multiplayer match reset, player slots, bots, opponents, teams, and termination rules.
- `env/wrappers/` — frame stacking, action repeat, normalization, curriculum, and diagnostics.
- `observations/` — visual and structured-state schemas plus preprocessing boundaries.
- `actions/` — movement, aim, fire, weapon selection, and action-space definitions.
- `rewards/` — scoring-event translation and reward accounting.
- `agents/baselines/` — non-learning/random/scripted agents used to validate the environment.
- `agents/policies/` — trainable policy and value model definitions later.
- `training/online_rl/` — initial PPO-style or equivalent online RL experiments.
- `training/self_play/` — opponent pools, snapshots, matchmaking, and self-play later.
- `training/imitation/` — supervised learning from human or agent trajectories later.
- `training/preferences/` — preference data and RLHF-style optimization later.
- `evaluation/` — fixed scenarios, scorecards, tournaments, regressions, and replays.
- `data/raw/` — immutable captured trajectories/events.
- `data/processed/` — training-ready datasets and indexes.
- `checkpoints/` — model and optimizer snapshots; normally excluded from Git.
- `logs/` — engine, environment, and training logs; normally excluded from Git.
- `tests/unit/` — deterministic contract tests without launching the game.
- `tests/integration/` — local server/client lifecycle and end-to-end smoke tests.
- `tools/inspection/` — utilities for viewing events, observations, trajectories, and match state.
- `tools/launch/` — human-facing launch helpers later.
- `docs/decisions/` — architecture decision records for transport, observations, actions, and rewards.

## Runtime topology

The recommended initial topology is:

`trainer -> environment -> coordinator -> engine bridge -> controlled client`

The coordinator also manages:

`local dedicated server -> opponent bots/clients -> authoritative event stream`

The server is the source of truth for kills, deaths, score, match time, joins, disconnects, and match completion. The controlled client is the source for the agent's observation and executed controls. This avoids rewarding the model from fragile screen text alone.

## Initial contracts

### Observation contract
Start with two channels kept distinct:

- **Pixels:** resized RGB frames, optional frame stack.
- **Structured telemetry:** health, armor, ammo, weapon, alive/dead state, score, match clock, and only data legitimately available to the controlled player.

Avoid privileged enemy positions in the first fair-play agent. They may be available only in a clearly labeled debugging/oracle configuration.

### Action contract
Begin with a compact discrete or multi-discrete space:

- forward/back
- strafe left/right
- turn left/right in coarse increments
- fire
- jump
- weapon next/previous or a small fixed weapon subset
- no-op

Continuous mouse aiming can be introduced after lifecycle and reward correctness are proven.

### Reward contract
Use authoritative event deltas:

- positive: frag, damage contribution if reliable, objective score, match win
- negative: death, suicide/team damage where applicable, disconnect/invalid state
- small shaping only: survival or useful movement, capped so it cannot overpower match score

The reward ledger must record each component separately for debugging.

### Episode contract
An episode is one match or a fixed match segment. Reset must produce a known map, mode, seed, player name/slot, opponent configuration, score limit, and time limit. Termination includes match completion, controlled-client failure, server failure, or a configured timeout.

## Multiplayer stages

1. **Environment validation:** one controlled client versus stationary or easy bots.
2. **Baseline learning:** deathmatch on one map with a small action space.
3. **Generalization:** multiple maps, spawn locations, bot skill levels, and weapon distributions.
4. **Opponent diversity:** scripted bots and frozen historical policies.
5. **Self-play:** policy league and snapshot matchmaking.
6. **Imitation learning:** ingest human demonstrations to improve movement and aiming.
7. **Preference/RLHF:** rank behaviors such as positioning, target selection, and teamwork after reliable trajectory capture exists.

## Odysseus coworker split

Use this document as the shared contract between assistants.

- **ChatGPT:** architecture decisions, integration reviews, milestone planning, failure analysis, and keeping the system aligned with the long-term RL/imitation/preference roadmap.
- **Odysseus:** local repository inspection, command execution, narrow implementation tasks, test runs, and reporting exact diffs/logs.
- **User:** approves behavior goals, runs gameplay tests, records demonstrations, and decides acceptable tradeoffs.

Every handoff should state: current milestone, files changed, commands run, observed output, unresolved issue, and next single task.

## First implementation milestone (no code yet)

The next coding milestone should touch only these areas:

1. `config/` — one local deathmatch profile.
2. `runtime/` — server/client lifecycle plan.
3. `engine/bridge/` — choose and document transport.
4. `env/core/` and `env/multiplayer/` — reset/step contract.
5. `actions/`, `observations/`, and `rewards/` — version-1 schemas.
6. `agents/baselines/` — random/no-op validation agent.
7. `tests/integration/` — one match smoke test.

Do not begin neural-network training until reset, action execution, observations, reward totals, termination, and cleanup are repeatable.
