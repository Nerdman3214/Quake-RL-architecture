# Event schema

The RL event pipeline now uses a dedicated event subsystem under the package.

## Core event shape

Each event is a JSON object with two top-level fields:

- `type`: the event name (for example `player_kill` or `weapon_fire`)
- `data`: a dictionary containing event-specific payload data

## Example

```json
{"type":"player_kill","data":{"killer":"bot1","victim":"bot2","weapon":"rocket"}}
```

## Notes

- Events are intentionally schema-light so they can represent telemetry from the game bridge without coupling to a specific observation or reward implementation.
- Downstream components can consume events via the shared reader and writer helpers in the events package.
