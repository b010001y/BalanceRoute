"""Self-contained generation and loading of synthetic integer length traces."""
from importlib.resources import files
import hashlib

import numpy as np


def input_digest(prompt, output):
    # Specify byte order so the digest is portable across machine architectures.
    return hashlib.sha256(np.stack([prompt, output]).astype('<i8').tobytes()).hexdigest()


def frozen_inputs(seed):
    path = files('balanceroute').joinpath('reference', 'r16', 'inputs', f'{seed}.npz')
    with path.open('rb') as stream, np.load(stream, allow_pickle=False) as data:
        return data['s'].copy(), data['o'].copy()


def _fit_mean(raw, target, low, high):
    lo, hi = 0., target / float(raw.mean())
    while np.clip(hi * raw, low, high).mean() < target:
        hi *= 2
    for _ in range(55):
        mid = (lo + hi) / 2
        if np.clip(mid * raw, low, high).mean() < target:
            lo = mid
        else:
            hi = mid
    return np.rint(np.clip((lo + hi) / 2 * raw, low, high)).astype(np.int64)


def generate_lengths(config, seed):
    """Truncated lognormals; correlation refers to the latent normal variables."""
    rng = np.random.default_rng(seed)
    z_s, independent = rng.standard_normal((2, config['n_requests']))
    rho = config['correlation']
    z_o = rho * z_s + np.sqrt(1 - rho ** 2) * independent
    prompt = _fit_mean(np.exp(config['sigma_prompt'] * z_s), config['mean_prompt'],
                       config['min_prompt'], config['max_prompt'])
    output = _fit_mean(np.exp(config['sigma_output'] * z_o), config['mean_output'],
                       config['min_output'], config['max_output'])
    return prompt, output
