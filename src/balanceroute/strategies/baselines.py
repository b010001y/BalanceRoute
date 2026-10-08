"""Random, round-robin, power-of-two, JSQ, and least-token-load routing."""
import numpy as np


def _route_random(sim, two_choices):
    capacity = sim._gpu_capacity_left()
    assigned = []
    for rid in sim.wait_q:
        available = np.flatnonzero(capacity > 0)
        if not available.size:
            break
        if two_choices:
            choices = sim.rng.choice(available, size=min(2, len(available)), replace=False)
            gid = int(choices[np.argmax(capacity[choices])])
        else:
            gid = int(sim.rng.choice(available))
        sim._assign_to_gpu(int(rid), gid)
        capacity[gid] -= 1
        assigned.append(rid)
    sim._remove_from_wait_q(assigned)


def policy_random(sim):
    """Choose uniformly among workers with an available request slot."""
    _route_random(sim, False)


def policy_power_of_two(sim):
    """Sample two available workers and choose the smaller active count."""
    _route_random(sim, True)


def policy_round_robin(sim):
    """Cycle through available workers, retaining the cursor across rounds."""
    capacity = sim._gpu_capacity_left()
    if not sim.wait_q or capacity.sum() == 0:
        return
    count = min(len(sim.wait_q), int(capacity.sum()))
    assigned = []
    gid = getattr(sim, "_round_robin_next", 0)
    for rid in list(sim.wait_q)[:count]:
        while capacity[gid] <= 0:
            gid = (gid + 1) % sim.n_worker
        sim._assign_to_gpu(int(rid), gid)
        assigned.append(rid)
        capacity[gid] -= 1
        gid = (gid + 1) % sim.n_worker
    sim._remove_from_wait_q(assigned)
    sim._round_robin_next = gid


def policy_join_shortest_queue(sim):
    """Minimize active request count, breaking ties by worker id."""
    capacity = sim._gpu_capacity_left()
    assigned = []
    for rid in sim.wait_q:
        available = np.flatnonzero(capacity > 0)
        if not available.size:
            break
        gid = int(available[np.argmax(capacity[available])])
        sim._assign_to_gpu(int(rid), gid)
        capacity[gid] -= 1
        assigned.append(rid)
    sim._remove_from_wait_q(assigned)


def policy_least_token_load(sim):
    """Minimize current context-token load, breaking ties by worker id."""
    capacity = sim._gpu_capacity_left()
    assigned = []
    for rid in sim.wait_q:
        available = np.flatnonzero(capacity > 0)
        if not available.size:
            break
        gid = int(available[np.argmin(sim.sum_lengths[available])])
        sim._assign_to_gpu(int(rid), gid)
        capacity[gid] -= 1
        assigned.append(rid)
    sim._remove_from_wait_q(assigned)
