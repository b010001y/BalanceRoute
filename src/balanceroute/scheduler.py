"""Discrete decode simulation with explicit, consistent metric windows."""
from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Callable, Iterable, Optional

import numpy as np


def _positive_lengths(values, name):
    arr = np.asarray(values)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    if not np.isfinite(arr).all() or (arr <= 0).any() or (arr != np.floor(arr)).any():
        raise ValueError(f"{name} must contain finite positive integer lengths")
    if (arr >= 2**63).any():
        raise ValueError(f"{name} exceeds int64 range")
    return arr.astype(np.int64, copy=True)


class SchedulerSim:
    """Sticky assignments; one output token per active request per decode step.

    Default arrivals are a demand-driven pool of reveal_target candidates, not a
    time-stamped production replay. Arrival latency starts at reveal time.
    All primary metrics cover the complete replay including drain. Explicit
    admission_prefix_* metrics exclude steps from the last admission onward.
    """

    def __init__(
        self, n_worker: int, batch_size: int, C: float, t_length: float,
        s_list: np.ndarray, o_list: np.ndarray, o_est_list: np.ndarray,
        policy_fn: Callable, record_history: bool = True,
        reveal_target: Optional[int] = None, discover_fn: Optional[Callable] = None,
        remaining_predictor: Optional[Callable] = None, random_seed: int = 0,
    ):
        if int(n_worker) != n_worker or n_worker <= 0 or int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("n_worker and batch_size must be positive integers")
        if not np.isfinite([C, t_length]).all() or C < 0 or t_length < 0 or C + t_length <= 0:
            raise ValueError("C and t_length must be nonnegative, finite, and not both zero")
        self.n_worker, self.batch_size = int(n_worker), int(batch_size)
        self.C, self.t_length = float(C), float(t_length)
        self.policy_fn, self.record_hist = policy_fn, record_history
        self.remaining_predictor = remaining_predictor
        self.rng = np.random.default_rng(random_seed)
        self.s_arr = _positive_lengths(s_list, "s_list")
        self.o_arr = _positive_lengths(o_list, "o_list")
        self.o_est_arr = _positive_lengths(o_est_list, "o_est_list")
        self.N = len(self.s_arr)
        if self.N == 0 or len(self.o_arr) != self.N or len(self.o_est_arr) != self.N:
            raise ValueError("request arrays must be nonempty and have equal lengths")
        self.gen_arr = np.zeros(self.N, dtype=np.int64)
        self.gpu_of_req = np.full(self.N, -1, dtype=np.int64)
        self.start_time = np.full(self.N, np.nan)
        self.arrival_time = np.full(self.N, np.nan)
        self.first_token_time = np.full(self.N, np.nan)
        self.finish_time = np.full(self.N, np.nan)
        self.start_step = np.full(self.N, -1, dtype=np.int64)
        self._unfinished_mask = np.ones(self.N, dtype=bool)
        self.undiscovered_q = deque(range(self.N))
        self.wait_q = deque()
        self.reveal_target = self.n_worker * 4 if reveal_target is None else reveal_target
        if int(self.reveal_target) != self.reveal_target or self.reveal_target <= 0:
            raise ValueError("reveal_target must be a positive integer")
        self.reveal_target = int(self.reveal_target)
        self.discover_fn = discover_fn or self.discover_fn_default
        self.gpu_active = [np.empty(0, dtype=np.int64) for _ in range(self.n_worker)]
        self.sum_lengths = np.zeros(self.n_worker, dtype=np.int64)
        self.active_total = 0
        self.clock = 0.0
        self.step_idx = 0
        self.last_assign_step = -1
        self.hist_sum_lengths, self.hist_steps = [], []
        self.routing_durations = []
        self._has_run = False
        self._reveal_next()

    @staticmethod
    def discover_fn_default(sim, needed):
        return [sim.undiscovered_q.popleft() for _ in range(min(needed, len(sim.undiscovered_q)))]

    def _reveal_next(self):
        need = self.reveal_target - len(self.wait_q)
        if need <= 0 or not self.undiscovered_q:
            return
        new_ids = list(self.discover_fn(self, need))
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("discover_fn returned duplicate request ids")
        for rid in new_ids:
            if rid < 0 or rid >= self.N or not np.isnan(self.arrival_time[rid]):
                raise ValueError("discover_fn returned an invalid or previously revealed request")
            self.arrival_time[rid] = self.clock
        self.wait_q.extend(new_ids)

    def _gpu_capacity_left(self):
        return self.batch_size - np.array([len(ids) for ids in self.gpu_active], dtype=np.int64)

    def _remove_from_wait_q(self, ids: Iterable[int]):
        assigned = set(map(int, ids))
        self.wait_q = deque(rid for rid in self.wait_q if rid not in assigned)

    def _assign_to_gpu(self, req_id, gpu_id):
        if not 0 <= req_id < self.N or not 0 <= gpu_id < self.n_worker:
            raise ValueError("invalid request or worker id")
        if self.gpu_of_req[req_id] != -1 or not self._unfinished_mask[req_id]:
            raise ValueError("a request may be admitted exactly once")
        if np.isnan(self.arrival_time[req_id]):
            raise ValueError("cannot admit an unrevealed request")
        if len(self.gpu_active[gpu_id]) >= self.batch_size:
            raise AssertionError(f"worker {gpu_id} exceeds batch_size={self.batch_size}")
        self.gpu_active[gpu_id] = np.append(self.gpu_active[gpu_id], req_id)
        self.sum_lengths[gpu_id] += self.s_arr[req_id]
        self.gpu_of_req[req_id] = gpu_id
        self.active_total += 1
        self.start_time[req_id] = self.clock
        self.start_step[req_id] = self.step_idx
        self.last_assign_step = self.step_idx

    def _remove_finished_from_gpu(self, gpu_id, finished_ids):
        keep = self.gpu_active[gpu_id][~np.isin(self.gpu_active[gpu_id], finished_ids)]
        self.active_total -= len(self.gpu_active[gpu_id]) - len(keep)
        self.gpu_active[gpu_id] = keep
        self.sum_lengths[gpu_id] = (self.s_arr[keep] + self.gen_arr[keep]).sum()

    def _dispatch(self):
        if self.wait_q and self._gpu_capacity_left().sum() > 0:
            start = perf_counter()
            self.policy_fn(self)
            self.routing_durations.append(perf_counter() - start)

    def run(self, verbose=False):
        if self._has_run:
            raise RuntimeError("SchedulerSim.run() can only be called once")
        self._has_run = True
        self._dispatch()
        durations, tokens, deficits, spreads = [], [], [], []
        utilizations, idle_rates, clocks = [], [], []
        finished_count = 0
        while finished_count < self.N:
            if not self.active_total:
                self._reveal_next()
                self._dispatch()
                if not self.active_total:
                    raise RuntimeError("unfinished requests remain but policy/discovery made no progress")
            active_now = self.active_total
            # Every step statistic uses this same state, before token generation.
            maximum = int(self.sum_lengths.max())
            total = int(self.sum_lengths.sum())
            duration = self.C + self.t_length * maximum
            deficit = self.n_worker * maximum - total
            spread = maximum - int(self.sum_lengths.min())
            durations.append(duration)
            tokens.append(active_now)
            deficits.append(deficit)
            spreads.append(spread)
            utilizations.append(active_now / (self.n_worker * self.batch_size))
            idle_rates.append(1 - total / (self.n_worker * maximum))
            if self.record_hist:
                self.hist_sum_lengths.append(self.sum_lengths.copy())
                self.hist_steps.append(self.step_idx)
            self.clock += duration
            self.step_idx += 1
            clocks.append(self.clock)
            completed = 0
            for gid in range(self.n_worker):
                ids = self.gpu_active[gid]
                if not len(ids):
                    continue
                first = ids[self.gen_arr[ids] == 0]
                self.first_token_time[first] = self.clock
                self.gen_arr[ids] += 1
                self.sum_lengths[gid] += len(ids)
                finished = ids[self.gen_arr[ids] == self.o_arr[ids]]
                if len(finished):
                    self.finish_time[finished] = self.clock
                    self._unfinished_mask[finished] = False
                    completed += len(finished)
                    self._remove_finished_from_gpu(gid, finished)
            finished_count += completed
            if self.wait_q or self.undiscovered_q:
                self._reveal_next()
                self._dispatch()
            if verbose and self.step_idx % 100 == 0:
                print(f"step {self.step_idx}, t={self.clock:.6f}, finished={finished_count}/{self.N}")

        if self.active_total or not np.array_equal(self.gen_arr, self.o_arr):
            raise RuntimeError("token conservation or completion invariant failed")
        durations = np.asarray(durations)
        tokens = np.asarray(tokens)
        if int(tokens.sum()) != int(self.o_arr.sum()):
            raise RuntimeError("emitted token count does not match requested outputs")
        cut = max(0, self.last_assign_step)
        prefix_time = float(durations[:cut].sum())
        service_per_token = (self.finish_time - self.start_time) / self.o_arr
        multi_token = self.o_arr > 1
        # Standard per-request TPOT excludes time to the first output token.
        tpot = (self.finish_time[multi_token] - self.first_token_time[multi_token]) / (self.o_arr[multi_token] - 1)
        metrics = dict(
            makespan=float(self.clock),
            avg_latency=float(np.mean(self.finish_time - self.arrival_time)),
            avg_waiting_time=float(np.mean(self.start_time - self.arrival_time)),
            avg_service_time=float(np.mean(self.finish_time - self.start_time)),
            avg_completion_time=float(np.mean(self.finish_time)),
            avg_latency_per_token=float(service_per_token.mean()),
            tpot_mean=float(tpot.mean()) if tpot.size else 0.0,
            tpot_p95=float(np.percentile(tpot, 95)) if tpot.size else 0.0,
            tpot_request_count=int(tpot.size),
            throughput_tokens_per_sec=float(self.o_arr.sum() / self.clock),
            QPM=float(60 * self.N / self.clock),
            # Deficit objective and max-minus-min load spread.
            avg_gpu_imbalance=float(np.mean(deficits)),
            avg_load_spread=float(np.mean(spreads)),
            avg_gpu_utilization=float(np.mean(utilizations)),
            avg_gpu_idle_rate=float(np.mean(idle_rates)),
            avg_barrier_idle_rate=float(self.t_length * np.sum(deficits) / (self.n_worker * self.clock)),
            emitted_tokens=int(tokens.sum()),
            decode_steps=self.step_idx,
            admission_prefix_steps=cut,
            admission_prefix_throughput=float(tokens[:cut].sum() / prefix_time) if prefix_time else 0.0,
            admission_prefix_avg_imbalance=float(np.mean(deficits[:cut])) if cut else 0.0,
            admission_prefix_avg_load_spread=float(np.mean(spreads[:cut])) if cut else 0.0,
            routing_calls=len(self.routing_durations),
            routing_p50_ms=float(np.percentile(self.routing_durations, 50) * 1000),
            routing_p99_ms=float(np.percentile(self.routing_durations, 99) * 1000),
            routing_max_ms=float(max(self.routing_durations) * 1000),
            step_clock_times=clocks,
        )
        self.metrics = metrics
        return metrics
