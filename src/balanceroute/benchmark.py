"""Paired benchmark execution, auditable artifacts, and reference verification."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import platform

import numpy as np

from . import __version__, SchedulerSim
from .strategies import (BalanceRouteConfig, BalanceRoutePolicy, policy_random,
    policy_round_robin, policy_power_of_two, policy_join_shortest_queue, policy_least_token_load)
from .workload import frozen_inputs, generate_lengths, input_digest

BASELINES = {'random': policy_random, 'round_robin': policy_round_robin,
             'power_of_two': policy_power_of_two, 'jsq': policy_join_shortest_queue,
             'least_token_load': policy_least_token_load}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def preset():
    return json.loads(files('balanceroute').joinpath('presets', 'r16.json').read_text(encoding='utf-8'))


def reference():
    return json.loads(files('balanceroute').joinpath('reference', 'r16', 'expected.json').read_text(encoding='utf-8'))


def validate_config(config):
    expected = preset()
    if set(config) != set(expected) or config['schema_version'] != 1:
        raise ValueError('configuration must contain exactly the version-1 preset fields')
    if not isinstance(config['name'], str) or not config['name']:
        raise ValueError('configuration name must be a nonempty string')
    if config['input_source'] not in ['frozen', 'generated']:
        raise ValueError('input_source must be frozen or generated')
    seeds = config['seeds']
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('seeds must be unique nonnegative integers')
    for section in ['workload', 'simulation']:
        if set(config[section]) != set(expected[section]):
            raise ValueError(f'{section} must contain exactly the preset fields')
        if not all(type(v) in [int, float] and np.isfinite(v) for v in config[section].values()):
            raise ValueError(f'{section} values must be finite numbers')
    w, sim = config['workload'], config['simulation']
    for name in ['n_requests', 'min_prompt', 'max_prompt', 'min_output', 'max_output']:
        if type(w[name]) is not int or w[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for kind in ['prompt', 'output']:
        if not w['min_' + kind] < w['mean_' + kind] < w['max_' + kind] or not 0 <= w['sigma_' + kind] <= 5:
            raise ValueError(f'invalid {kind} distribution bounds/mean/sigma')
    if not -1 <= w['correlation'] <= 1:
        raise ValueError('correlation must lie in [-1,1]')
    for name in ['n_workers', 'max_requests_per_worker', 'ready_pool_size']:
        if type(sim[name]) is not int or sim[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if min(sim['fixed_step_seconds'], sim['seconds_per_context_token']) < 0 or sim['fixed_step_seconds'] + sim['seconds_per_context_token'] <= 0:
        raise ValueError('clock coefficients must be nonnegative with positive sum')
    if not 0 <= sim['window_start_fraction'] < sim['window_end_fraction'] <= 1:
        raise ValueError('window fractions must satisfy 0 <= start < end <= 1')
    if config['input_source'] == 'frozen' and (w != expected['workload'] or not set(seeds) <= set(expected['seeds'])):
        raise ValueError('frozen inputs require the reference workload and seeds; use input_source=generated for custom workloads')
    policies = config['policies']
    if not isinstance(policies, list) or not policies:
        raise ValueError('policies must be a nonempty list')
    names = []
    for policy in policies:
        kind = policy.get('kind')
        required = {'name', 'kind', 'parameters'} if kind == 'balance_route' else {'name', 'kind'}
        if set(policy) != required or kind not in [*BASELINES, 'balance_route']:
            raise ValueError('invalid policy specification')
        name = policy['name']
        if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('policy names may contain only ASCII letters, digits, underscores and hyphens')
        names.append(name)
        if kind == 'balance_route':
            if set(policy['parameters']) != set(asdict(BalanceRouteConfig())):
                raise ValueError('every BalanceRoute parameter must be explicit')
            bc = BalanceRouteConfig(**policy['parameters'])
            if bc.horizon == 0 and (bc.alpha != 1 or bc.beta != sim['n_workers']):
                raise ValueError('BR-0 requires alpha=1 and beta=n_workers')
    if len(set(names)) != len(names):
        raise ValueError('policy names must be unique')


def run_one(task):
    config, seed, spec, output_dir = task
    out = Path(output_dir)
    with np.load(out / 'inputs' / f'{seed}.npz', allow_pickle=False) as data:
        s, o = data['s'], data['o']
    params = config['simulation']
    bc = BalanceRouteConfig(**spec['parameters']) if spec['kind'] == 'balance_route' else None
    policy = BalanceRoutePolicy(bc) if bc else BASELINES[spec['kind']]
    sim = SchedulerSim(params['n_workers'], params['max_requests_per_worker'],
                       params['fixed_step_seconds'], params['seconds_per_context_token'],
                       s, o, o, policy, reveal_target=params['ready_pool_size'], random_seed=seed,
                       record_history=True)
    metrics = sim.run()
    loads = np.asarray(sim.hist_sum_lengths)
    maximum = loads.max(axis=1)
    starts = np.bincount(sim.start_step, minlength=sim.step_idx + 1)
    finishes = np.bincount(sim.start_step + o, minlength=sim.step_idx + 1)
    active = np.cumsum(starts - finishes)[:sim.step_idx]
    if int(active.sum()) != int(o.sum()):
        raise RuntimeError('token conservation failed')
    slots = params['n_workers'] * params['max_requests_per_worker']
    ideal_steps = float(o.sum()) / slots
    begin = int(params['window_start_fraction'] * ideal_steps)
    end = int(params['window_end_fraction'] * ideal_steps)
    if not 0 <= begin < end <= sim.step_idx:
        raise ValueError('measurement window is empty or outside the replay; increase workload size or adjust window fractions')
    durations = sim.C + sim.t_length * maximum[begin:end]
    occupancy = active[begin:end] / slots
    remaining = len(s) - np.cumsum(starts)[begin:end]
    trace_name = f'traces/{seed}_{spec["name"]}.npz'
    np.savez_compressed(out / trace_name, maximum=maximum[begin:end], active=active[begin:end],
                        spread=np.ptp(loads[begin:end], axis=1))
    return {'seed': seed, 'policy': spec['name'], 'input_sha256': input_digest(s, o),
            'policy_config': asdict(bc) if bc else None, 'trace': trace_name,
            **{k: v for k, v in metrics.items() if np.isscalar(v)},
            'window_start_step': begin, 'window_end_step_exclusive': end,
            'window_throughput_tokens_per_sec': float(active[begin:end].sum() / durations.sum()),
            'window_step_latency_p95_ms': float(np.percentile(durations, 95) * 1000),
            'window_avg_max_minus_min': float(np.mean(np.ptp(loads[begin:end], axis=1))),
            'window_avg_deficit': float(np.mean(params['n_workers'] * maximum[begin:end] - loads[begin:end].sum(axis=1))),
            'window_mean_slot_occupancy': float(occupancy.mean()),
            'window_min_slot_occupancy': float(occupancy.min()),
            'window_min_not_yet_admitted': int(remaining.min()),
            'window_fully_saturated': bool(np.all(occupancy == 1) and np.all(remaining > 0))}


def summarize(rows):
    groups = {name: [r for r in rows if r['policy'] == name] for name in dict.fromkeys(r['policy'] for r in rows)}
    methods, paired = [], []
    for name, group in groups.items():
        row = {'policy': name, 'runs': len(group), 'all_windows_fully_saturated': all(r['window_fully_saturated'] for r in group)}
        for field in ['throughput_tokens_per_sec', 'window_throughput_tokens_per_sec', 'routing_p50_ms', 'routing_p99_ms']:
            values = [r[field] for r in group]
            row[field] = float(np.mean(values))
            row[field + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row['tpot_p95_ms'] = float(np.mean([r['tpot_p95'] for r in group]) * 1000)
        methods.append(row)
        for baseline in ['BR-0', 'JSQ', 'RR']:
            if baseline not in groups:
                continue
            control = {r['seed']: r for r in groups[baseline]}
            if set(control) != {r['seed'] for r in group}:
                raise ValueError('paired policies must use the same seeds')
            for field in ['throughput_tokens_per_sec', 'window_throughput_tokens_per_sec']:
                gains = []
                for r in group:
                    if r['input_sha256'] != control[r['seed']]['input_sha256']:
                        raise ValueError('paired inputs differ')
                    gains.append(100 * (r[field] / control[r['seed']][field] - 1))
                paired.append({'policy': name, 'baseline': baseline, 'metric': field,
                               'mean_gain_pct': float(np.mean(gains)), 'wins': sum(g > 1e-9 for g in gains),
                               'runs': len(gains), 'per_seed_gain_pct': dict(zip(map(str, [r['seed'] for r in group]), gains))})
    return {'methods': methods, 'paired_gains': paired,
            'scope': 'CPU simulation. Full trace includes drain. Throughput excludes measured router CPU execution time.'}


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def run(config, output, workers=1, mode='custom'):
    validate_config(config)
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    out = Path(output)
    # Never mix a new experiment into an existing result directory.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    (out / 'traces').mkdir()
    package = Path(__file__).parent
    source_hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(package.rglob('*.py'))}
    manifest = {'schema_version': 1, 'mode': mode, 'status': 'running', 'config': config,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'package_version': __version__,
                'python_version': platform.python_version(), 'numpy_version': np.__version__,
                'platform': platform.system(), 'workers': workers, 'source_sha256': source_hashes,
                'input_sha256': {}, 'notes': ['Oracle remaining lengths for active requests.',
                    'New requests use the constant-prompt F-score approximation.',
                    'Demand-revealed ready pool; no time-stamped production arrivals.',
                    'Throughput excludes routing CPU time.']}
    write_json(out / 'manifest.json', manifest)
    try:
        expected_hashes = {r['seed']: r['input_sha256'] for r in reference()['records']}
        for seed in config['seeds']:
            s, o = frozen_inputs(seed) if config['input_source'] == 'frozen' else generate_lengths(config['workload'], seed)
            digest = input_digest(s, o)
            if config['input_source'] == 'frozen' and digest != expected_hashes[seed]:
                raise ValueError(f'frozen input checksum mismatch for seed {seed}')
            manifest['input_sha256'][str(seed)] = digest
            np.savez_compressed(out / 'inputs' / f'{seed}.npz', s=s, o=o)
        write_json(out / 'manifest.json', manifest)
        tasks = [(config, seed, policy, str(out)) for seed in config['seeds'] for policy in config['policies']]
        rows = []
        with (out / 'metrics.jsonl').open('w', encoding='utf-8') as log:
            def record(row):
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                print(f"{len(rows)}/{len(tasks)} seed={row['seed']} {row['policy']}: "
                      f"window={row['window_throughput_tokens_per_sec']:.1f}, full={row['throughput_tokens_per_sec']:.1f} tok/s", flush=True)
            if workers == 1:
                for task in tasks:
                    record(run_one(task))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    for future in as_completed([pool.submit(run_one, task) for task in tasks]):
                        record(future.result())
        names = [p['name'] for p in config['policies']]
        rows.sort(key=lambda r: (config['seeds'].index(r['seed']), names.index(r['policy'])))
        report = summarize(rows)
        write_json(out / 'summary.json', report)
        write_csv(out / 'metrics.csv', rows)
        write_csv(out / 'summary.csv', report['methods'])
        write_csv(out / 'paired_gains.csv', report['paired_gains'])
        manifest['status'], manifest['completed_runs'] = 'complete', len(rows)
        write_json(out / 'manifest.json', manifest)
        return report
    except BaseException as error:
        manifest['status'], manifest['error'] = 'failed', str(error)
        write_json(out / 'manifest.json', manifest)
        raise


def _verify(output, require_complete=True):
    """Check inputs, configuration, coverage, saved traces and frozen metrics.

    Wall-clock routing timings are intentionally excluded from golden checks.
    """
    out = Path(output)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    validate_config(config)
    expected = reference()
    baseline = expected['config']
    if manifest['status'] != 'complete':
        raise ValueError('run is incomplete')
    if config['workload'] != baseline['workload'] or config['simulation'] != baseline['simulation']:
        raise ValueError('workload or simulation settings differ from the R=16 reference')
    specs = {p['name']: p for p in baseline['policies']}
    if any(p != specs.get(p['name']) for p in config['policies']):
        raise ValueError('policy configuration differs from the R=16 reference')
    expected_rows = {(r['seed'], r['policy']): r for r in expected['records']}
    required = {(s, p['name']) for s in config['seeds'] for p in config['policies']}
    if require_complete and required != set(expected_rows):
        raise ValueError(f"full reference verification requires all {len(baseline['seeds'])} seeds and all {len(specs)} policies")
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    keys = [(r['seed'], r['policy']) for r in rows]
    if len(set(keys)) != len(rows) or set(keys) != required or manifest['completed_runs'] != len(rows):
        raise ValueError('missing, duplicate or unexpected result rows')
    input_hashes = {}
    for seed in config['seeds']:
        with np.load(out / 'inputs' / f'{seed}.npz', allow_pickle=False) as data:
            digest = input_digest(data['s'], data['o'])
        if digest != manifest['input_sha256'][str(seed)]:
            raise ValueError(f'saved input checksum mismatch for seed {seed}')
        input_hashes[seed] = digest
    for row in rows:
        key = (row['seed'], row['policy'])
        ref = expected_rows.get(key)
        if ref is None or row['input_sha256'] != ref['input_sha256'] or input_hashes[row['seed']] != ref['input_sha256']:
            raise ValueError(f'reference input mismatch: {key}')
        if row['policy_config'] != ref['policy_config']:
            raise ValueError(f'recorded policy configuration mismatch: {key}')
        for field in expected['metric_fields']:
            value, target = row[field], ref[field]
            valid = type(value) is type(target) and value == target if type(target) in [int, bool] else np.isfinite(value) and np.isclose(value, target, rtol=1e-9, atol=1e-9)
            if not valid:
                raise ValueError(f'reference metric mismatch: {key}, {field}: {value} vs {target}')
        trace_path = (out / row['trace']).resolve()
        if not trace_path.is_relative_to(out.resolve()):
            raise ValueError('trace path escapes the result directory')
        with np.load(trace_path, allow_pickle=False) as trace:
            duration = config['simulation']['fixed_step_seconds'] + config['simulation']['seconds_per_context_token'] * trace['maximum']
            if len(duration) != row['window_end_step_exclusive'] - row['window_start_step']:
                raise ValueError('saved window length mismatch')
            slots = config['simulation']['n_workers'] * config['simulation']['max_requests_per_worker']
            if not np.all(trace['active'] == slots):
                raise ValueError('reference window is not fully saturated')
            checks = {'window_throughput_tokens_per_sec': trace['active'].sum() / duration.sum(),
                      'window_step_latency_p95_ms': np.percentile(duration, 95) * 1000,
                      'window_avg_max_minus_min': trace['spread'].mean()}
            for field, value in checks.items():
                if not np.isclose(value, row[field], rtol=1e-12, atol=1e-12):
                    raise ValueError(f'saved window trace mismatch: {key}, {field}')
    actual_summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    names = [p['name'] for p in config['policies']]
    ordered = sorted(rows, key=lambda r: (config['seeds'].index(r['seed']), names.index(r['policy'])))
    if actual_summary != summarize(ordered):
        raise ValueError('saved summary differs from per-seed metrics')
    report = {'passed': True, 'reference_id': expected['reference_id'], 'verified_runs': len(rows),
              'scope': 'full_reference' if required == set(expected_rows) else 'reference_subset',
              'input_checksums_match': True, 'saved_window_metrics_recomputed': True,
              'all_windows_fully_saturated_with_backlog': True,
              'routing_timings_excluded': True, 'relative_tolerance': 1e-9}
    write_json(out / 'verification.json', report)
    return report


def verify(output, require_complete=True):
    """Verify artifacts and replace a stale success report on any failure."""
    try:
        return _verify(output, require_complete=require_complete)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        out = Path(output)
        if out.is_dir() and (out / 'manifest.json').is_file():
            write_json(out / 'verification.json', {'passed': False, 'error': str(error)})
        raise
