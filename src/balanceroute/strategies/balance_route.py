"""BalanceRoute scoring and two-stage routing (paper Algorithms 2/3).

h=0 is the next decode iteration; a request with r remaining tokens contributes
at offsets 0 <= h < r. New admissions consume a constant prompt-length increment
within a round, as in the paper's F-score approximation. At the next round the
projections are rebuilt from observed ages. Unseen lengths are never inspected.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, islice
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BalanceRouteConfig:
    horizon: int = 80
    candidate_window: int = 16
    greedy_slot_threshold: int = 7
    alpha: float = 1.0
    beta: float = 43.0
    decay: float = 0.86
    candidate_order: str = "prompt_desc"

    def __post_init__(self):
        for name in ("horizon", "candidate_window", "greedy_slot_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 1 <= self.candidate_window <= 16:
            raise ValueError("exact subset enumeration requires 1 <= candidate_window <= 16")
        if not np.isfinite([self.alpha, self.beta, self.decay]).all():
            raise ValueError("score parameters must be finite")
        if self.alpha <= 0 or self.beta <= 0 or not 0 < self.decay <= 1:
            raise ValueError("require alpha,beta > 0 and 0 < decay <= 1")
        if self.candidate_order not in {"fifo", "prompt_desc"}:
            raise ValueError("candidate_order must be fifo or prompt_desc")


class FScore:
    """Scalar/vector evaluation of the discounted piecewise-linear F-score."""

    def __init__(self, load_row, envelope, alpha=1.0, beta=43.0, decay=0.86):
        margin = np.maximum(np.asarray(envelope, dtype=float) - np.asarray(load_row, dtype=float), 0)
        if margin.ndim != 1 or not margin.size or not np.isfinite(margin).all():
            raise ValueError("load projections must be finite, nonempty one-dimensional arrays")
        weights = decay ** np.arange(margin.size)
        order = np.argsort(margin, kind="stable")
        self.margins = margin[order]
        self.prefix_d = np.r_[0., np.cumsum(weights[order])]
        self.prefix_dm = np.r_[0., np.cumsum(weights[order] * self.margins)]
        self.discount_sum = float(weights.sum())
        self.alpha = float(alpha)
        self.reward = float(alpha * self.discount_sum)
        self.beta = float(beta)
        self.safe_margin = int(np.floor(self.margins[0]))
        self.constant_margin = float(self.margins[0]) if self.margins[0] == self.margins[-1] else None

    def __call__(self, sums):
        values = np.asarray(sums, dtype=float)
        if self.constant_margin is not None:
            # F_H = sum(discounts) * F_0 for a constant margin. Factoring
            # before multiplying preserves exact score ties across horizons;
            # prefix subtraction could otherwise break ties at ~1e-12.
            scores = self.discount_sum * (
                self.alpha * values - self.beta * np.maximum(values - self.constant_margin, 0)
            )
            return float(scores) if values.ndim == 0 else scores
        indices = np.searchsorted(self.margins, values, side="left")
        scores = self.reward * values - self.beta * (
            values * self.prefix_d[indices] - self.prefix_dm[indices]
        )
        return float(scores) if values.ndim == 0 else scores


def project_future(sim, horizon: int):
    """Project active growth/departures without new arrivals.

    An optional remaining_predictor(sim, ids, horizon) returns positive remaining
    steps for active ids. Otherwise use supplied estimates, retaining exhausted
    estimates through the window. Exact output estimates implement an oracle.
    """
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    loads = np.zeros((horizon + 1, sim.n_worker), dtype=float)
    loads[0] = sim.sum_lengths
    if horizon == 0:
        return loads, loads.max(axis=1)
    offsets = np.arange(horizon + 1)
    predictor = getattr(sim, "remaining_predictor", None)
    for gid, ids in enumerate(sim.gpu_active):
        if not len(ids):
            continue
        if predictor is None:
            remaining = sim.o_est_arr[ids] - sim.gen_arr[ids]
            remaining = np.where(remaining > 0, remaining, horizon + 1)
        else:
            remaining = np.asarray(predictor(sim, ids, horizon), dtype=float)
            if remaining.shape != ids.shape or not np.isfinite(remaining).all() or (remaining <= 0).any():
                raise ValueError("remaining_predictor must return one finite positive value per active request")
        remaining = np.ceil(np.minimum(remaining, horizon + 1)).astype(np.int64)
        base = sim.s_arr[ids] + sim.gen_arr[ids]
        departed_base = np.bincount(remaining, weights=base, minlength=horizon + 2).cumsum()[:horizon + 1]
        departed_count = np.bincount(remaining, minlength=horizon + 2).cumsum()[:horizon + 1]
        loads[:, gid] = base.sum() - departed_base + offsets * (len(ids) - departed_count)
    return loads, loads.max(axis=1)


def select_subset(sizes: Sequence[int], slots: int, score: Callable):
    """Exact optimum over nonempty subsets of a bounded candidate window.

    Returns (indices, total_prompt_length, score). Every cardinality is checked;
    cost depends on candidate count rather than token-length magnitude.
    """
    if slots <= 0 or not len(sizes):
        return [], 0, float("-inf")
    if len(sizes) > 16:
        raise ValueError("subset selection accepts at most 16 candidates; use a bounded window")
    subsets = [c for k in range(1, min(slots, len(sizes)) + 1) for c in combinations(range(len(sizes)), k)]
    sums = [sum(int(sizes[i]) for i in c) for c in subsets]
    values = score(sums) if isinstance(score, FScore) else np.array([score(s) for s in sums])
    best = int(np.argmax(values))
    return list(subsets[best]), sums[best], float(values[best])


def route(sim, config: BalanceRouteConfig):
    if not sim.wait_q:
        return
    capacity = sim._gpu_capacity_left()
    total_slots = int(capacity.sum())
    if total_slots <= 0:
        return
    loads, envelope = project_future(sim, config.horizon)
    # BR-0 parameters are independent of any BR-H tuning.
    alpha = 1.0 if config.horizon == 0 else config.alpha
    beta = float(sim.n_worker) if config.horizon == 0 else config.beta

    def evaluator(gid):
        return FScore(loads[:, gid], envelope, alpha, beta, config.decay)

    def admit(ids, gid):
        nonlocal total_slots
        increment = int(sim.s_arr[ids].sum())
        for rid in ids:
            sim._assign_to_gpu(int(rid), int(gid))
        sim._remove_from_wait_q(ids)
        capacity[gid] -= len(ids)
        total_slots -= len(ids)
        loads[:, gid] += increment
        np.maximum(envelope, loads[:, gid], out=envelope)

    while sim.wait_q and total_slots > config.greedy_slot_threshold:
        available = np.flatnonzero(capacity == capacity.max())
        gid = int(available[np.argmin(loads[0, available])])
        ids = np.fromiter(sim.wait_q, dtype=np.int64)
        best = int(np.argmax(evaluator(gid)(sim.s_arr[ids])))
        admit([int(ids[best])], gid)

    while sim.wait_q and total_slots > 0:
        # Rebuild priority after each admission: the envelope changes margins
        # on other workers too, so retaining their heap keys would be stale.
        minimum_margin = (envelope[:, None] - loads).min(axis=0)
        available = np.flatnonzero(capacity > 0)
        gid = max(available, key=lambda g: (capacity[g], minimum_margin[g], -g))
        if config.candidate_order == "prompt_desc":
            ids = sorted(sim.wait_q, key=lambda r: -int(sim.s_arr[r]))[:config.candidate_window]
        else:
            ids = list(islice(sim.wait_q, config.candidate_window))
        score = evaluator(gid)
        indices, _, best_score = select_subset(sim.s_arr[ids], int(capacity[gid]), score)
        if best_score <= 0:
            indices = [int(np.argmax(score(sim.s_arr[ids])))]
        admit([ids[i] for i in indices], gid)


@dataclass(frozen=True)
class BalanceRoutePolicy:
    """Picklable configuration, isolated between independent worker processes."""
    config: BalanceRouteConfig = BalanceRouteConfig()

    def __call__(self, sim):
        route(sim, self.config)
