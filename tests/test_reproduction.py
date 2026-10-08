import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from balanceroute.benchmark import preset, reference, run, validate_config, verify
from balanceroute.workload import frozen_inputs, generate_lengths, input_digest


def test_generated_inputs_match_every_frozen_trace():
    config = preset()
    hashes = {r['seed']: r['input_sha256'] for r in reference()['records']}
    for seed in config['seeds']:
        generated = generate_lengths(config['workload'], seed)
        frozen = frozen_inputs(seed)
        np.testing.assert_array_equal(generated, frozen)
        assert input_digest(*generated) == hashes[seed]


@pytest.mark.parametrize('change', ['duplicate_seed', 'unknown_field', 'wrong_frozen_workload', 'wrong_br0_beta', 'zero_workers'])
def test_invalid_configuration_cannot_silently_change_experiment(change):
    config = preset()
    if change == 'duplicate_seed':
        config['seeds'] = [100, 100]
    elif change == 'unknown_field':
        config['simulation']['candidate_machines'] = 16
    elif change == 'wrong_frozen_workload':
        config['workload']['mean_prompt'] = 4000
    elif change == 'wrong_br0_beta':
        next(p for p in config['policies'] if p['name'] == 'BR-0')['parameters']['beta'] = 43
    else:
        config['simulation']['n_workers'] = 0
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.fixture(scope='module')
def reference_run(tmp_path_factory):
    output = tmp_path_factory.mktemp('reference') / 'run'
    config = preset()
    config['seeds'] = [100]
    config['policies'] = [p for p in config['policies'] if p['name'] in ['JSQ', 'BR-0', 'BR-H']]
    run(config, output, mode='test')
    return output


def test_packaged_reference_replay(reference_run):
    report = verify(reference_run, require_complete=False)
    assert report['scope'] == 'reference_subset'
    assert report['verified_runs'] == 3
    with pytest.raises(ValueError, match='all 10 seeds'):
        verify(reference_run)


@pytest.mark.parametrize('artifact', ['inputs', 'metric', 'missing_row', 'trace', 'summary'])
def test_verifier_rejects_corrupted_artifacts(reference_run, tmp_path, artifact):
    output = tmp_path / 'modified'
    shutil.copytree(reference_run, output)
    records = output / 'metrics.jsonl'
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    if artifact == 'inputs':
        path = output / 'inputs/100.npz'
        with np.load(path) as data:
            s, o = data['s'], data['o']
        s[0] += 1
        np.savez_compressed(path, s=s, o=o)
    elif artifact == 'metric':
        rows[0]['throughput_tokens_per_sec'] *= 1.01
        records.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    elif artifact == 'missing_row':
        records.write_text(''.join(json.dumps(r) + '\n' for r in rows[1:]))
    elif artifact == 'trace':
        path = output / rows[0]['trace']
        with np.load(path) as data:
            arrays = dict(data)
        arrays['maximum'][0] += 100
        np.savez_compressed(path, **arrays)
    else:
        path = output / 'summary.json'
        summary = json.loads(path.read_text())
        summary['methods'][0]['throughput_tokens_per_sec'] *= 1.01
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify(output, require_complete=False)
    assert json.loads((output / 'verification.json').read_text())['passed'] is False


def test_run_never_overwrites_previous_results(reference_run):
    before = (reference_run / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(preset(), reference_run)
    assert (reference_run / 'manifest.json').read_bytes() == before


def test_cli_works_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'balanceroute', 'show-config'], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == preset()


def test_cli_verification_failure_has_nonzero_exit(reference_run, tmp_path):
    result = subprocess.run([sys.executable, '-m', 'balanceroute', 'verify', str(reference_run)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'all 10 seeds' in result.stderr
