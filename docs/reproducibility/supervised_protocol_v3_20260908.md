# Supervised V11b / V5 / V12 protocol V3

V3 is the current frozen recipe, with `training_authorized=false`.
Recipe freeze and successful audit-only preflight do not authorize optimizer
execution. The selective checkpoint must be committed and pushed, followed by
separate explicit user authorization before any `--train` invocation.
The preserved launcher does not programmatically enforce `training_authorized`;
the authorization decision remains a separate operator gate.

V1 is unchanged historical provenance; its authorization wording overstates the
prior authorization. V2 is unchanged failed-candidate provenance: its new status
was rejected by the launcher's strict frozen-status check before training.
V3 derives from V1, changing only name, authorization wording, execution-gate
wording, and the added false training-authorization flag. Scientific fields,
paths, hashes, data roles, counts, recipe, selection and calibration are unchanged.

TRAIN: V4=656, V6=1283, V7=3172, V9=2408, V10=2998, V11b=2997;
total 13514. V5=1190 exclusively selects checkpoints. V12=8987 is frozen,
calibration/representative validation only, never TRAIN or checkpoint selection.
V8 remains sealed and supplies no selection or calibration results.

Audit-only command (no training/output-directory arguments):

```sh
.venvs/visual-bc-dinov3-clean/bin/python -B src/RL/visual_bc/run_supervised_training_v1.py \
  --protocol src/RL/config/supervised_v11b_v5_v12_protocol_20260908_v3.json \
  --protocol-sha256 7980153f26e5efa6519fe8fb65099dabcce3c3cb66520cae0e22df2b383021b0
```

Audit result: exit 0, `preflight_passed=true`,
`status=supervised_preflight_passed`, TRAIN=13514, validation V5=1190,
`v12_payloads_loaded=false`, `training_started=false`, `V8_consumed=false`,
`agent_control=false`, `automatic_action_allowed=false`.
AST/JSON validation and independent V1-to-V3 semantic comparison passed.
The runtime V5-to-V6 change remains solely the V11b run-manifest session entry.

SHA256 pins:

- Protocol V1: `2d4f3d427d19a74e544cd0eef24890b021578681480abc6b447dc7e46ef164b1`
- Protocol V2: `99f18d1b8a0f3f3e3bf1136e165020465804f74455c690e95ad48fe5feb85604`
- Protocol V3: `7980153f26e5efa6519fe8fb65099dabcce3c3cb66520cae0e22df2b383021b0`
- Launcher V1: `5bb995f3db7c0c43c6631479b440688f1a727faada12dd2069d1dab9366929c6`
- Runtime V6: `657d2fb69f78d4a09aa1c85f5d33630e5ea9ca04a62ea157ca0b405026f63d3e`

The older draft document is excluded because it describes a superseded state.
The untracked evaluator is excluded pending separate review: its no-positive-target
calibration returns `threshold=None`, while the frozen recipe describes
`threshold_1p0_disable_head_and_report_undefined_objective`. No evaluator execution
or inference was performed. This checkpoint does not certify evaluation readiness.
The earlier V10 visibility findings remain unresolved historical data-quality
context; this correction does not modify the frozen data or reopen that audit.
