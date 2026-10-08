import numpy as np
import pytest

from balanceroute import SchedulerSim
from balanceroute.strategies import (
    policy_join_shortest_queue, policy_least_token_load,
    policy_round_robin, policy_random, policy_power_of_two,
    BalanceRouteConfig, BalanceRoutePolicy,
)


def make_sim(s, o, policy=policy_round_robin, g=2, b=2, **kw):
    return SchedulerSim(g, b, 1., .01, np.asarray(s), np.asarray(o), np.asarray(o), policy, **kw)


@pytest.mark.parametrize("s,o", [([10], [0]), ([0], [1]), ([10], [-1]), ([10], [1.2]), ([10], [np.nan]), ([10, 20], [1])])
def test_invalid_requests_rejected(s, o):
    with pytest.raises(ValueError):
        make_sim(s, o)


def test_all_initial_admissions_produce_nonzero_full_metrics():
    sim = make_sim([100], [3], g=1, b=1)
    metrics = sim.run()
    assert metrics["makespan"] == pytest.approx(2. + 2.01 + 2.02)
    assert metrics["throughput_tokens_per_sec"] == pytest.approx(3 / 6.03)
    assert metrics["avg_gpu_utilization"] == 1
    assert metrics["admission_prefix_steps"] == 0
    assert metrics["tpot_p95"] == pytest.approx((2.01 + 2.02) / 2)


def test_completed_requests_do_not_erase_step_imbalance():
    sim = make_sim([100, 1], [1, 1], policy_round_robin, g=2, b=1)
    metrics = sim.run()
    np.testing.assert_array_equal(sim.hist_sum_lengths, [[100, 1]])
    assert metrics["avg_gpu_imbalance"] == metrics["avg_load_spread"] == 99
    assert metrics["emitted_tokens"] == 2
    assert metrics["tpot_request_count"] == 0


@pytest.mark.parametrize("policy", [policy_join_shortest_queue, policy_least_token_load,
    policy_round_robin, policy_random, policy_power_of_two,
    BalanceRoutePolicy(), BalanceRoutePolicy(BalanceRouteConfig(horizon=0))])
def test_full_replay_matches_independent_state_reconstruction(policy):
    rng = np.random.default_rng(13)
    s, o = rng.integers(1, 500, 80), rng.integers(1, 30, 80)
    sim = make_sim(s, o, policy, g=4, b=3)
    m = sim.run()
    loads = []
    for step in range(sim.step_idx):
        active = (sim.start_step <= step) & (step < sim.start_step + o)
        loads.append(np.bincount(sim.gpu_of_req[active], weights=s[active] + step - sim.start_step[active], minlength=4))
    loads = np.asarray(loads)
    np.testing.assert_array_equal(sim.hist_sum_lengths, loads)
    assert m["avg_gpu_imbalance"] == pytest.approx(np.mean(4 * loads.max(axis=1) - loads.sum(axis=1)))
    assert m["avg_load_spread"] == pytest.approx(np.mean(np.ptp(loads, axis=1)))
    assert m["makespan"] == pytest.approx(np.sum(1 + .01 * loads.max(axis=1)))
    assert m["throughput_tokens_per_sec"] == pytest.approx(o.sum() / m["makespan"])
    assert sim.active_total == 0 and not sim._unfinished_mask.any()
    np.testing.assert_array_equal(sim.gen_arr, o)


def test_round_robin_cursor_survives_dispatch_calls():
    sim = make_sim([10] * 3, [3] * 3, g=3, b=2, reveal_target=1)
    for _ in range(3):
        policy_round_robin(sim)
        sim._reveal_next()
    np.testing.assert_array_equal(sim.gpu_of_req, [0, 1, 2])


def test_jsq_and_token_load_have_distinct_semantics():
    for policy, expected in [(policy_join_shortest_queue, 0), (policy_least_token_load, 1)]:
        sim = make_sim([1000, 1, 1, 2], [10] * 4, g=2, b=3)
        for rid, gid in enumerate([0, 1, 1]):
            sim.wait_q.popleft()
            sim._assign_to_gpu(rid, gid)
        policy(sim)
        assert sim.gpu_of_req[3] == expected
