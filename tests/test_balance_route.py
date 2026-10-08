from itertools import combinations

import numpy as np
import pytest

from balanceroute import SchedulerSim
from balanceroute.strategies import policy_round_robin
from balanceroute.strategies.balance_route import (
    BalanceRouteConfig, FScore, project_future, route, select_subset,
)


def make_sim(s, o, g=2, b=2):
    return SchedulerSim(g, b, 1., .01, np.array(s), np.array(o), np.array(o), policy_round_robin, reveal_target=len(s))


@pytest.mark.parametrize("sizes,margin,slots,expected", [
    ([1, 101], 100, 1, 101), ([4, 4, 9], 10, 2, 9),
])
def test_subset_optimum_across_score_breakpoint(sizes, margin, slots, expected):
    score = FScore([0], [margin], beta=43)
    ids, total, value = select_subset(sizes, slots, score)
    assert total == expected == sum(sizes[i] for i in ids)
    assert value == score(expected)


def test_multiple_horizon_kinks():
    score = FScore(np.zeros(31), [1000.] * 30 + [0.])
    assert select_subset([1, 100], 1, score)[1] == 100


def test_selector_against_independent_exhaustive_formula():
    rng = np.random.default_rng(20261007)
    for _ in range(300):
        horizon = int(rng.integers(0, 25))
        margins = rng.uniform(0, 100, horizon + 1)
        alpha, beta, decay = 1.2, float(rng.uniform(1, 50)), float(rng.uniform(.5, 1))
        sizes = rng.integers(1, 70, 4)
        slots = int(rng.integers(1, 5))
        discounts = decay ** np.arange(horizon + 1)

        def reference(subset):
            total = sum(sizes[i] for i in subset)
            return alpha * discounts.sum() * total - beta * np.maximum(total - margins, 0) @ discounts

        exact = max(reference(c) for k in range(1, slots + 1) for c in combinations(range(4), k))
        ids, total, value = select_subset(sizes, slots, FScore(np.zeros(horizon + 1), margins, alpha, beta, decay))
        assert len(ids) == len(set(ids)) <= slots
        assert total == sizes[ids].sum()
        assert value == pytest.approx(exact, abs=1e-8)


def test_projection_matches_independent_decode_steps():
    sim = make_sim([100, 200, 50], [3, 5, 7])
    sim.wait_q.clear()
    for rid, gid in enumerate([0, 0, 1]):
        sim._assign_to_gpu(rid, gid)
    sim.gen_arr[:] = [1, 2, 0]
    sim.sum_lengths[:] = [303, 50]
    actual, envelope = project_future(sim, 8)
    expected = []
    for offset in range(9):
        expected.append([sum(int(sim.s_arr[i] + sim.gen_arr[i] + offset)
                             for i in ids if sim.gen_arr[i] + offset < sim.o_arr[i])
                         for ids in sim.gpu_active])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(envelope, np.max(expected, axis=1))


def test_br0_beta_is_worker_count_and_never_reads_prediction():
    sim = make_sim([100, 105, 3, 6], [10] * 4)
    for gid in range(2):
        rid = sim.wait_q.popleft()
        sim._assign_to_gpu(rid, gid)

    class Forbidden:
        def __getitem__(self, _):
            raise AssertionError("BR-0 accessed predictions")

    sim.o_est_arr = Forbidden()
    route(sim, BalanceRouteConfig(horizon=0, beta=999))
    assert sim.gpu_of_req[3] == 0  # F(6)=4 > F(3)=3 when G=2.
    assert not sim.wait_q


def test_no_access_to_unseen_predictions():
    sim = make_sim([100, 100, 100], [100, 100, 100], g=1, b=3)
    sim.wait_q.clear()
    sim.wait_q.append(1)
    sim._assign_to_gpu(0, 0)

    class ActiveOnly:
        def __getitem__(self, ids):
            assert list(ids) == [0]
            return np.array([100])

    sim.o_est_arr = ActiveOnly()
    route(sim, BalanceRouteConfig(horizon=4))
    assert sim.gpu_of_req[1] == 0


def test_nonpositive_score_guard_fills_capacity():
    sim = make_sim([100, 100, 100, 100], [3] * 4)
    route(sim, BalanceRouteConfig(horizon=80))
    assert sim.active_total == 4
    assert not sim.wait_q


def test_capacity_priority_precedes_margin():
    sim = make_sim([1000, 1, 1, 5], [10] * 4, g=2, b=3)
    for rid, gid in enumerate([0, 1, 1]):
        sim.wait_q.popleft()
        sim._assign_to_gpu(rid, gid)
    route(sim, BalanceRouteConfig(horizon=0, greedy_slot_threshold=7))
    assert sim.gpu_of_req[3] == 0


def test_horizon_zero_matches_br0_entry_point():
    from balanceroute.strategies import BalanceRoutePolicy
    a, b = make_sim([3, 6, 14, 8, 4], [2, 3, 4, 2, 1]), make_sim([3, 6, 14, 8, 4], [2, 3, 4, 2, 1])
    route(a, BalanceRouteConfig(horizon=0))
    BalanceRoutePolicy(BalanceRouteConfig(horizon=0))(b)
    np.testing.assert_array_equal(a.gpu_of_req, b.gpu_of_req)


@pytest.mark.parametrize("order", ["fifo", "prompt_desc"])
@pytest.mark.parametrize("threshold", [0, 7, 100])
def test_constant_margins_produce_identical_admissions_at_h0_and_h80(order, threshold):
    rng = np.random.default_rng(20261007)
    for _ in range(80):
        workers, batch = 4, 4
        s = rng.integers(1, 200, workers + 14)
        sims = []
        for horizon in [0, 80]:
            sim = make_sim(s, [1000] * len(s), g=workers, b=batch)
            for rid in range(workers):
                sim._assign_to_gpu(rid, rid)
                sim._remove_from_wait_q([rid])
            loads, envelope = project_future(sim, horizon)
            margins = envelope[:, None] - loads
            np.testing.assert_array_equal(margins, np.broadcast_to(margins[0], margins.shape))
            sim.admission_log = []
            assign = sim._assign_to_gpu

            def record(rid, gid, sim=sim, assign=assign):
                sim.admission_log.append((rid, gid))
                assign(rid, gid)

            sim._assign_to_gpu = record
            route(sim, BalanceRouteConfig(horizon=horizon, beta=workers,
                                          candidate_order=order, greedy_slot_threshold=threshold))
            sims.append(sim)
        assert sims[0].admission_log == sims[1].admission_log
        np.testing.assert_array_equal(sims[0].gpu_of_req, sims[1].gpu_of_req)
        assert list(sims[0].wait_q) == list(sims[1].wait_q)


def test_candidate_order_can_be_shared_across_horizons():
    for order in ["fifo", "prompt_desc"]:
        assert BalanceRouteConfig(horizon=0, candidate_order=order).candidate_order == order
        assert BalanceRouteConfig(horizon=80, candidate_order=order).candidate_order == order
    with pytest.raises(ValueError):
        BalanceRouteConfig(candidate_order="unknown")
