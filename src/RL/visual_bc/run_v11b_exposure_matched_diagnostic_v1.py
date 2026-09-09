"""Single-arm V11b expected-exposure-matched diagnostic.

Audit first; no production promotion. Only explicit --run executes
optimizer updates, after a pinned preflight and pushed code.

This preserves the original equal-session sampler and extends the
OLD+V11b horizon so each of the six sessions has the same expected
number of draws that each old session had in the 33,790-step OLD-only
reference arm.

No retries, no early stopping, no V12, no V8, no control, no RL.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SEED = 20260822
BATCH = 4

REFERENCE_STEPS = 33790
STEPS = 40548
EVAL_STRIDE = 3379
SCHEDULE = list(range(EVAL_STRIDE, STEPS + 1, EVAL_STRIDE))

OLD = ['V4', 'V6', 'V7', 'V9', 'V10']
FULL = OLD + ['V11b']

ARM = 'old_plus_v11b_exposure_matched'
ARMS = {ARM: FULL}

EXPECTED_DRAWS_PER_SESSION = 27032
COUNTS = dict(V4=656, V6=1283, V7=3172, V9=2408, V10=2998, V11b=2997, V5=1190)
CONTRACT = ROOT / 'data/cache/visual_bc/policy_v2_2_v11b_contract_20260905_v3/v2_2_v11b_runtime_contract_v3.json'
CONTRACT_SHA = '436c3272de5c6837b5b2274f8fbedfd84fbd044ed4ba79974ad8badb792f126f'
PRODUCTION = ROOT / 'data/cache/visual_bc/supervised_v11b_v5_authorized_20260908_v1/training'
HISTORICAL = ROOT / 'data/models/visual_bc_temporal_policy_v2_2_game_only_20260828_v1/best.pt'
PINS = {
    'train_temporal_policy_v1.py': '45f2c2d4dcc05d8cf3e1433f52aeea59feba39f8ce1a883c5fad507f6f92445c',
    'train_temporal_policy_v2.py': '2eb12e664dffa6305f18658e6923b889c96d6fd58bef02925e319da1457f26e6',
    'train_temporal_policy_v2_2_runtime_v6.py': '657d2fb69f78d4a09aa1c85f5d33630e5ea9ca04a62ea157ca0b405026f63d3e',
}
FLAGS = dict(V12_used=False, V8_consumed=False, agent_control=False,
             automatic_action_allowed=False, controls_modified=False, RL_started=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(path):
    path = Path(path).resolve()
    import re
    require(not re.search(r'(?i)(?:^|[^a-z0-9])v(?:8|12)(?:$|[^a-z0-9])', str(path)),
            'Excluded dataset path')
    return path


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def pinned(path, digest):
    require(sha(path) == digest, f'Hash mismatch: {path}')
    return safe(path)


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load_runtime():
    for name, digest in PINS.items():
        pinned(HERE / name, digest)
    sys.path.insert(0, str(HERE))
    import torch
    import numpy as np
    import train_temporal_policy_v2 as v2
    import train_temporal_policy_v2_2_runtime_v6 as v6
    return torch, np, v2, v6


def seed_model(torch, v2):
    random.seed(SEED)
    torch.manual_seed(SEED)
    return v2.TemporalVisualPolicyV2()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def loss_values(config):
    return dict(forward_weights=config.forward_weights.tolist(),
                strafe_weights=config.strafe_weights.tolist(),
                jump_pos_weight=config.jump_pos_weight,
                yaw_pos_weight=config.yaw_pos_weight,
                pitch_pos_weight=config.pitch_pos_weight)


def prepare():
    torch, np, v2, v6 = load_runtime()
    contract = json.loads(pinned(CONTRACT, CONTRACT_SHA).read_text())
    require(set(contract['sessions']) == set(FULL + ['V5']), 'Dataset membership changed')
    datasets = {}
    for name in FULL + ['V5']:
        spec = contract['sessions'][name]
        require(spec['sequence_count'] == COUNTS[name], 'Count mismatch')
        require(spec['role'] == ('validation' if name == 'V5' else 'train'), 'Role mismatch')
        for field in ['sequence_index_file', 'cache_directory', 'cache_manifest']:
            safe(spec[field])
        for field, digest in [('source_file', 'source_sha256'), ('sequence_audit_file', 'sequence_audit_sha256')]:
            pinned(spec[field], spec[digest])
        datasets[name] = v2.FrozenSequenceSession(name, spec)
    # Freeze the successful production loss configuration for BOTH arms.
    # These constants are not re-estimated from either treatment arm.
    config = v6.build_loss_config(v2.collect_labels([datasets[n] for n in FULL]))
    require(config.jump_pos_weight == 2.5, 'Unexpected jump class weight')
    require(v2.SEED == SEED and v6.SEED == SEED, 'Seed changed')
    pinned(PRODUCTION / 'best.pt', '14b466bb128a0d2de017d78c449ec6b25ead79a98cb4bb3f0a2f9667f689adbb')
    pinned(HISTORICAL, '6eaab868748cc493dca5810dc1bca4761d7a76edaffda6bfd60493d13e7b9133')
    manifest = json.loads((PRODUCTION / 'run_manifest.json').read_text())
    history = [json.loads(line) for line in (PRODUCTION / 'metrics.jsonl').read_text().splitlines()]
    require(
        len(history) == 10
        and math.ceil(manifest['train_sequence_count'] / BATCH) * len(history)
        == REFERENCE_STEPS,
        'Production reference horizon changed',
    )
    require((manifest['batch_size'], manifest['learning_rate'], manifest['weight_decay']) == (4, 5e-5, .0005),
            'Scientific recipe changed')
    return torch, np, v2, v6, datasets, config


def preflight():
    torch, np, v2, v6, datasets, config = prepare()

    model = seed_model(torch, v2)
    initial = state_hash(model)

    require(
        STEPS * len(OLD)
        == REFERENCE_STEPS * len(FULL),
        'Expected-exposure matching arithmetic changed',
    )

    require(
        STEPS * BATCH == 162192,
        'Total diagnostic draw budget changed',
    )

    require(
        (STEPS * BATCH) % len(FULL) == 0,
        'Expected per-session draw count is not integral',
    )

    expected_draws_per_session = (
        STEPS * BATCH // len(FULL)
    )

    require(
        expected_draws_per_session
        == EXPECTED_DRAWS_PER_SESSION,
        'Expected per-session exposure changed',
    )

    sessions = {}

    for name in FULL:
        labels = datasets[name].labels

        session_probability = 1.0 / len(FULL)

        jump_probability_within_session = (
            sum(l.jump for l in labels)
            / len(labels)
        )

        left_probability_within_session = (
            sum(l.strafe == 0 for l in labels)
            / len(labels)
        )

        jump_probability = (
            session_probability
            * jump_probability_within_session
        )

        left_probability = (
            session_probability
            * left_probability_within_session
        )

        sessions[name] = dict(
            session_probability=session_probability,
            expected_session_draws=(
                STEPS
                * BATCH
                * session_probability
            ),
            jump_positive_probability=jump_probability,
            expected_jump_draws=(
                STEPS
                * BATCH
                * jump_probability
            ),
            left_probability=left_probability,
            expected_left_draws=(
                STEPS
                * BATCH
                * left_probability
            ),
        )

    exposure = dict(
        sessions=sessions,
        expected_old_jump_draws=sum(
            row['expected_jump_draws']
            for name, row in sessions.items()
            if name in OLD
        ),
        expected_v11b_jump_draws=(
            sessions['V11b']['expected_jump_draws']
        ),
    )

    return dict(
        preflight_passed=True,
        runner_sha256=sha(Path(__file__)),
        contract_sha256=CONTRACT_SHA,
        source_pins=PINS,
        seed=SEED,
        initial_state_hash=initial,
        architecture=v2.architecture_metadata(),
        parameter_count=sum(
            p.numel()
            for p in model.parameters()
        ),
        optimizer=dict(
            name='AdamW',
            learning_rate=5e-5,
            weight_decay=.0005,
            betas=[.9, .999],
            eps=1e-8,
            amsgrad=False,
            gradient_clip_norm=1.0,
        ),
        loss_configuration=loss_values(config),
        loss_coefficients=dict(
            forward=1.,
            strafe=1.,
            jump=.75,
            yaw_active=.5,
            pitch_active=.5,
            yaw_regression=.25,
            pitch_regression=.25,
        ),
        arm=ARM,
        train_sessions=FULL,
        counts=COUNTS,
        reference_steps=REFERENCE_STEPS,
        steps=STEPS,
        evaluation_steps=SCHEDULE,
        batch_size=BATCH,
        samples_total=STEPS * BATCH,
        expected_draws_per_session=(
            expected_draws_per_session
        ),
        exposure=exposure,
        exposure_match_basis=(
            'Expected equal-session probability mass: '
            '40548*4/6 == 33790*4/5 == 27032 '
            'expected draws per session'
        ),
        realized_draws_exactly_matched=False,
        sampler=(
            'equal_session_probability_mass_with_replacement'
        ),
        event_oversampling=False,
        batching=(
            'Continuous replacement draws; every update has '
            'four samples, no partial epoch batch'
        ),
        early_stopping=False,
        retries=False,
        diagnostic_only=True,
        training_started=False,
        **FLAGS,
    )

def binary(y, p, threshold, np):
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(y.sum())-tp; tn = len(y)-tp-fp-fn
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def jump_metrics(y, logits, np):
    p = 1/(1+np.exp(-logits)); pos = p[y == 1]; neg = p[y == 0]
    order = np.argsort(-p, kind='stable'); scores = p[order]; truth = y[order]
    ends = np.r_[np.where(scores[:-1] != scores[1:])[0], len(scores)-1]
    tp = np.cumsum(truth)[ends]; fp = ends+1-tp; fn = y.sum()-tp
    ap = float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/y.sum()) if y.sum() else None
    f1 = 2*tp/(2*tp+fp+fn); best = int(np.argmax(f1))
    return dict(AP=ap, ROC_AUC=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:]))) if len(pos) and len(neg) else None,
                positive_count=len(pos), mean_positive_probability=float(pos.mean()) if len(pos) else None,
                mean_negative_probability=float(neg.mean()) if len(neg) else None,
                positive_negative_gap=float(pos.mean()-neg.mean()) if len(pos) and len(neg) else None,
                best_diagnostic_F1=float(f1[best]), diagnostic_threshold=float(scores[ends[best]]),
                at_0p5=binary(y,p,.5,np))


def evaluate(model, dataset, torch, np, v2, device):
    require(dataset.name in FULL+['V5'], 'Unexpected evaluation dataset')
    model.eval(); outputs = []
    with torch.inference_mode():
        for batch in v2.make_validation_loader(dataset, BATCH):
            out = model(batch[0].to(device))
            require(all(bool(torch.isfinite(v).all()) for v in out.values()), 'Nonfinite inference')
            outputs.append({k:v.detach().cpu().numpy() for k,v in out.items()})
    values = {k:np.concatenate([o[k] for o in outputs]) for k in outputs[0]}
    labels = dataset.labels
    y = np.array([int(l.jump) for l in labels])
    result = dict(jump=jump_metrics(y, values['jump_logit'].astype(float), np))
    for axis in ['forward','strafe']:
        pred = values[axis+'_logits'].argmax(1)
        truth = np.array([getattr(l,axis) for l in labels]); c = np.zeros((3,3), dtype=int)
        for a,b in zip(truth,pred): c[a,b] += 1
        f1 = [2*c[i,i]/(c[i].sum()+c[:,i].sum()) if c[i].sum()+c[:,i].sum() else 0. for i in range(3)]
        result[axis] = dict(macro_F1=float(np.mean(f1)), confusion=c.tolist())
        if axis == 'strafe':
            result[axis].update(left_precision=float(c[0,0]/c[:,0].sum()) if c[:,0].sum() else 0.,
                                left_recall=float(c[0,0]/c[0].sum()) if c[0].sum() else 0., left_F1=float(f1[0]))
    for axis,scale in [('yaw',32),('pitch',16)]:
        target = np.array([int(getattr(l,axis+'_active')) for l in labels])
        metric = jump_metrics(target, values[axis+'_activity_logit'].astype(float), np)
        truth = np.array([getattr(l,axis+'_normalized')*scale for l in labels])
        errors = np.abs(values[axis+'_magnitude']*scale-truth)
        result[axis] = dict(AP=metric['AP'], active_MAE=float(errors[target == 1].mean()) if target.sum() else None)
    return result


def run_arm(arm, directory, audit):
    torch, np, v2, v6, datasets, config = prepare()
    require(torch.cuda.is_available(), 'CUDA unavailable; no optimizer run')
    require(loss_values(config) == audit['loss_configuration'], 'Loss configuration changed')
    model = seed_model(torch, v2)
    initial = state_hash(model)
    require(
        initial == audit['initial_state_hash'],
        'Initial-state mismatch',
    )
    directory.mkdir(exist_ok=False)
    write_new(directory/'started.json', dict(arm=arm, initial_state_hash=initial, automatic_retry=False, **FLAGS))
    device = torch.device('cuda:0'); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.0005,
                                 betas=(.9,.999), eps=1e-8, amsgrad=False)
    completed = 0
    def count_update(optimizer, args, kwargs):
        nonlocal completed
        completed += 1
    hook = optimizer.register_step_post_hook(count_update)
    data = torch.utils.data.ConcatDataset([datasets[n] for n in ARMS[arm]])
    weights = v6.build_session_balanced_weights([datasets[n] for n in ARMS[arm]])
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=STEPS*BATCH, replacement=True,
                                                    generator=torch.Generator().manual_seed(SEED))
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH, sampler=sampler, num_workers=0,
                                        pin_memory=True, drop_last=False)
    stream = iter(loader)
    rows = []; previous = 0
    try:
        for target_step in SCHEDULE:
            train = v2.train_epoch(model, itertools.islice(stream, target_step-previous), optimizer, config, device)
            require(completed == target_step, 'Optimizer-step mismatch')
            require(train['examples'] == (target_step-previous)*BATCH, 'Batch-size mismatch')
            row = dict(global_step=completed, train=train, V5=evaluate(model,datasets['V5'],torch,np,v2,device))
            rows.append(row)
            with (directory/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            print(f"{arm} step={completed} loss={train['total_loss']:.6f} V5_jump_AP={row['V5']['jump']['AP']:.6f} V5_left_F1={row['V5']['strafe']['left_F1']:.6f}", flush=True)
            previous = target_step
        require(next(stream, None) is None and completed == STEPS, 'Budget mismatch')
        final_sessions = {}
        for name in FULL:
            metric = evaluate(model,datasets[name],torch,np,v2,device)['jump']
            final_sessions[name] = dict(in_training=name in ARMS[arm], **metric)
        require(all(bool(torch.isfinite(p).all()) for p in model.state_dict().values()), 'Nonfinite final state')
        checkpoint = directory/'diagnostic_final.pt'
        with checkpoint.open('xb') as f:
            torch.save(dict(model_state_dict=model.state_dict(), architecture=v2.architecture_metadata(),
                            global_step=completed, arm=arm, diagnostic_only=True, initial_state_hash=initial,
                            runner_sha256=audit['runner_sha256'], **FLAGS), f)
        result = dict(status='completed', exit_status=0, global_step=completed,
                      initial_state_hash=initial, final_state_hash=state_hash(model),
                      checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      final_sessions=final_sessions, automatic_retry=False, **FLAGS)
        write_new(directory/'result.json', result)
        return rows, result
    except BaseException as exc:
        write_new(directory/'failure.json', dict(status='failed', completed_steps=completed,
                  error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, **FLAGS))
        raise
    finally:
        hook.remove()


def execute(output, audit_path, audit_sha):
    saved = json.loads(
        pinned(
            audit_path,
            audit_sha,
        ).read_text()
    )

    require(
        saved['preflight_passed'] is True
        and saved['runner_sha256']
        == sha(Path(__file__)),
        'Stale preflight',
    )

    fresh = preflight()

    require(
        fresh == saved,
        'Preflight changed',
    )

    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=ROOT,
        text=True,
    ).strip()

    remote = subprocess.check_output(
        ['git', 'rev-parse', 'origin/main'],
        cwd=ROOT,
        text=True,
    ).strip()

    require(
        head == remote,
        'Code checkpoint not pushed',
    )

    committed = subprocess.check_output(
        [
            'git',
            'show',
            (
                f'{head}:src/RL/visual_bc/'
                'run_v11b_exposure_matched_diagnostic_v1.py'
            ),
        ],
        cwd=ROOT,
    )

    require(
        hashlib.sha256(committed).hexdigest()
        == saved['runner_sha256'],
        'Exposure-matched runner not committed',
    )

    require(
        not output.exists(),
        'Refusing an existing experiment directory',
    )

    output.mkdir(
        parents=False,
        exist_ok=False,
    )

    write_new(
        output / 'preflight.json',
        saved,
    )

    write_new(
        output / 'experiment_started.json',
        dict(
            commit=head,
            arm=ARM,
            train_sessions=FULL,
            steps=STEPS,
            reference_steps=REFERENCE_STEPS,
            expected_draws_per_session=(
                EXPECTED_DRAWS_PER_SESSION
            ),
            expected_exposure_matched=True,
            realized_draws_exactly_matched=False,
            no_retries=True,
            **FLAGS,
        ),
    )

    try:
        rows, result = run_arm(
            ARM,
            output / ARM,
            saved,
        )

        write_new(
            output / 'summary.json',
            dict(
                result=result,
                V5=rows,
                reference=dict(
                    old_only_steps=REFERENCE_STEPS,
                    expected_draws_per_old_session=(
                        EXPECTED_DRAWS_PER_SESSION
                    ),
                ),
                expected_exposure_matched=True,
                realized_draws_exactly_matched=False,
                **FLAGS,
            ),
        )

    except BaseException as exc:
        write_new(
            output / 'experiment_failure.json',
            dict(
                error=repr(exc),
                automatic_retry=False,
                **FLAGS,
            ),
        )
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--audit-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.audit is not None, '--audit path required')
    if args.run:
        require(args.output is not None and args.audit_sha256, 'Pinned audit and new output required')
        execute(safe(args.output), safe(args.audit), args.audit_sha256)
    else:
        require(args.output is None and args.audit_sha256 is None, 'Audit-only arguments invalid')
        report = preflight()
        write_new(safe(args.audit), report)
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
