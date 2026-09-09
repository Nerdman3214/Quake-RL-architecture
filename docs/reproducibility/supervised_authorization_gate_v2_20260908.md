# Supervised launcher V2 authorization gate

Use `src/RL/visual_bc/run_supervised_training_v2.py` for future supervised launches.
It preserves launcher V1 byte-for-byte except for a two-line check in the copied
V2 source: after normal preflight and the audit-only return, training requires
`protocol.get("training_authorized") is True`. Otherwise it raises
`Full persistent supervised training is not authorized by this protocol` before
output creation, training runtime loading, subprocess launch, or artifact writes.
V1 remains historical provenance and does not enforce this new gate.

V3 remains the canonical frozen recipe with `training_authorized=false`.
No authorized protocol was created. Recipe, data roles, runtime, evaluator, and
historical visibility findings were not changed. V8 remains sealed.

Validation with `.venvs/visual-bc-dinov3-clean/bin/python -B`:

- AST/JSON checks passed.
- Real V3 audit-only CLI passed, exit 0: TRAIN=13514, V5=1190,
  V12=8987 calibration only; no V12 payloads loaded.
- Real V3 `--train` refusal passed, exit 1 with the authorization error;
  a new temporary output path stayed absent and its parent stayed empty.
- `src/RL/tests/unit/test_supervised_training_authorization_v2.py` passed:
  audit permits false authorization; missing, false and non-boolean values
  refuse before reread/write/mkdir/subprocess operations.
- No training process was found afterward. No optimizer canary, training,
  evaluation, or game control ran.

SHA256:

- Launcher V2: `28aa910f1913d53be27777067fab3ddced4ea657641e95944f2dcf28151b3189`
- Unchanged V3: `7980153f26e5efa6519fe8fb65099dabcce3c3cb66520cae0e22df2b383021b0`
- Unchanged launcher V1: `5bb995f3db7c0c43c6631479b440688f1a727faada12dd2069d1dab9366929c6`
- Unchanged runtime V6: `657d2fb69f78d4a09aa1c85f5d33630e5ea9ca04a62ea157ca0b405026f63d3e`

The next training decision requires separate explicit user authorization.
`training_authorized=false`, `training_started=false`, `V8_consumed=false`.
