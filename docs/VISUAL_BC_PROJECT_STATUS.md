# Visual behavioral cloning: recovery checkpoint, 2026-09-08

The frozen V12 validation contract passed an independent audit. No supervised
training, optimizer canary, final V8 evaluation, or live control was run by this
checkpoint. The user subsequently requested progression through the supervised
stages until the agent plays Xonotic; the gates below remain sequential.

## Data roles

| Role | Sessions | Sequences |
|---|---|---:|
| TRAIN | V4, V6, V7, V9, V10, V11b | 13,514 |
| Historical validation/calibration | V5 | 1,190 |
| Independent frozen validation/calibration | V12 | 8,987 |
| Sealed final evaluation | V8 | Do not open before model/procedure freeze |

V11b contributes 2,997 TRAIN sequences from 3,000 native 10 Hz frames. V12 must
never enter TRAIN. Exactly one optimizer-step canary occurred historically; it
did not produce a persistent new fully trained model. Preserve the old epoch-7
baseline, SHA256 `6eaab868748cc493dca5810dc1bca4761d7a76edaffda6bfd60493d13e7b9133`.

## Frozen V12 evidence

The independent verifier is
`src/RL/visual_bc/audit_v12_frozen_contract_v1.py`. It does not import or execute
the freezer or either preparation script. It independently recomputes labels
from recovered telemetry, checks every sequence against its source context and
feature addresses, and hashes all 563 feature chunks and 563 record sidecars.
Tensor finiteness was verified by the earlier extraction audit; the independent
contract verifier checks hashes and headers rather than repeating that scan.

The permanent [audit report](reproducibility/v12_20260908/independent_frozen_audit_v1.json)
and [snapshot inventory](reproducibility/v12_20260908/snapshot_inventory_v1.json)
record exact local paths, byte counts, and hashes. Snapshots are byte-for-byte
copies of small metadata. The frozen originals and review candidate remain
untouched. These checksum inventories reference large files that remain local;
they are not a claim that Git contains the dataset.

| Artifact | SHA256 |
|---|---|
| Frozen validation contract | `69b98f3c99356316d6e1b7135d6915d687910133372f5ccf1adea4f0ba6a8576` |
| Freeze audit | `5f8b6fee53e0c44553b4f24fe3d5f74eba07d43a7e99bc635a5c363935154f11` |
| Freezer script | `1833adca0a8ee2d634460ee6a568a736c6ce962f1a0352f6400274dc8bbba89f` |
| Eight-file candidate bundle | `8b3f58721cb0bb46683d2cd01f6b064d465a466e25c497321e349e8b1e532593` |
| Native DINO manifest | `53887ff6967a869380da22d2d9c1dfdd0879942af1d1e7d5495c68973d0c9076` |

Local roots, relative to the repository:

- `data/cache/visual_bc/v12_calibration_validation_frozen_contract_20260908_v1`
- `data/cache/visual_bc/temporal_sequences_v12_calibration_review_candidate_20260907_v1/V12`
- `data/cache/visual_bc/dinov3_v12_calibration_frozen_20260907_v1/V12`
- `data/cache/visual_bc/game_only_v12_calibration_frozen_20260907_v1/V12`
- `data/recordings/visual_bc_calibration_v12_native10hz_candidate_20260907_000209`

All four frozen files and all eight candidate files match the supplied pins.
Counts are 8,995 admitted source records, 8,993 eligible labels, and 8,987
four-frame sequences. Source frames 4643–4647 remain absent. Source frames 4642
and 4648 retain their images/features but lack full same-segment support for
their 0.1-second label interval. No ineligible context is used in a sequence.
Window accounting: 8,992 dense windows minus three structural gap windows minus
two label-ineligible windows equals 8,987; no quality-window exclusions.

The reset-line erratum is preserved: capture-relative reset line 8749 means
recovered-file line 8848 and original-file line 9081. Frozen per-frame pointers
are correct. Do not edit the older alignment manifest in place.

## Proven implementation contracts

- Labels: `prepare_v12_temporal_v2.py`, SHA256
  `07616491b73ee13fab51f8e63372da61596a14d09b085c8afa87ecb057518d2c`.
  Contract `historical_0p1s_same_segment_single_sample_fallback_v2`.
- Sequences: `build_temporal_sequence_index.py`, SHA256
  `f3b6a8b057035df6fe6ad39166bafe3a5ed0faf9155dbbf7cd262659e500257c`.
  The new builder reproduced all 2,997 preserved V11b sequence dictionaries.
- Original `prepare_v12_temporal.py` remains historical evidence. Both versions
  allow intervals containing one telemetry row. V2 additionally restores the
  old fallback to the first following telemetry row when the grid interval
  contains zero rows. Current V12 has no such empty eligible intervals.
- Labels average raw movement inputs, use direction thresholds strictly above
  +1/below -1, and use nonzero integer jump input. Camera labels are wrapped
  first-to-last yaw difference and ordinary pitch difference. These are view
  angles, not physical mouse deltas. Attack/secondary/crouch/use are unobserved
  and remain null; the present policy therefore learns movement/jump/camera.
- Each example is four frames at 10 Hz, spanning 0.3 seconds. Its target is the
  final frame's synchronized action. Dense feature cache indices are addresses;
  original source IDs, frame numbers, time, and match segments establish order.
- Runtime v5 imports `train_temporal_policy_v2`, which imports V1 constants.
  `FrozenSequenceSession` reads `[4,41,384]` frozen features. The compatibility
  manifest supplies native/legacy chunk aliases without modifying tensors.

## Vision path

Human gameplay → telemetry/video alignment → game crop `(18,31,480,270)` →
Lanczos 640×360 JPEG → black 640×400 canvas with 20-pixel top padding → frozen
DINOv3 → CLS plus 40 ordered patch tokens → four-frame temporal policy.

DINO selects rows `[0,6,12,18,24]`, columns `[0,6,11,17,22,28,33,39]` from the
25×40 patch grid; patch tokens start at index 5 after CLS/four registers.
Features are float16 `[41,384]`. The actual model weight SHA is
`208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2`;
the latest prose handoff omitted characters from that SHA. Use frozen manifests.

## Remaining supervised stages

1. Commit/push this audit and small recovery metadata.
2. Freeze a combined V5/V12 validation protocol before looking at new results.
3. Prepare a versioned full-run launcher; preserve trainer v5. Its dataset loader
   includes V11b, but its generated run manifest hardcodes the older five-session
   TRAIN list. Correct provenance in a new version before a full run.
4. Run the supervised recipe, evaluate offline, select the checkpoint and
   calibrate thresholds under the frozen protocol. Preserve the old baseline.
5. Freeze checkpoint, preprocessing, thresholds, postprocessing, and evaluation
   procedure; only then perform the intended single V8 final evaluation.
6. Validate shadow capture/inference and bounded live control, then observe the
   supervised agent playing. RL and broad repository reorganization come later.

## Verification and environment

```bash
.venvs/visual-bc-dinov3-clean/bin/python \
  src/RL/visual_bc/audit_v12_frozen_contract_v1.py \
  --contract-directory data/cache/visual_bc/v12_calibration_validation_frozen_contract_20260908_v1

PYTHONPATH=src python -m pytest -q \
  src/RL/tests/unit/test_prepare_v12_temporal.py \
  src/RL/tests/unit/test_visual_bc_temporal_sequence_builder.py
```

At this checkpoint: independent audit PASS; 31 focused tests pass; all 14 visual
BC Python modules parse; `git diff --check` passes. The clean DINO Python lacks
pytest, so tests use the existing system environment without installing packages.
The sandbox hides NVIDIA devices; an approved check outside it confirms a TITAN
RTX with 24 GiB. GPU execution needs the corresponding sandbox permission.

The repository branch is `main`; configured remote is
`https://github.com/Nerdman3214/Quake-RL-architecture.git`. Remote main matched
local parent `cf6aab260d6dca924d5e966dc792de5169172d6a` before this checkpoint.
Historical hierarchical-BC edits and unrelated untracked work remain local and
are not silently folded into this visual-BC checkpoint. Videos, frame images,
feature tensors, checkpoints, virtual environments, and failed artifacts stay
local. `.gitignore` excludes data/recovery and now also Python environments and
generated caches. No broad cleanup, reset, or force-push is part of this work.
