# Reproduction protocol

## Configuration and inputs

The [R=16 preset](../src/balanceroute/presets/r16.json) specifies every workload, simulation, and policy parameter. Ten frozen inputs (seeds 100–109) and 70 expected records are bundled in the package. They contain synthetic integer lengths only.

Prompt and output lengths follow truncated lognormal samples with independent latent normal variables and log-standard-deviations 1.0 and 0.9. Bounds are 128–65,536 prompt tokens and 200–8,192 output tokens. A fixed 55-step scaling bisection fits each sample's pre-rounding means to 3,000 and 1,000; lengths are then rounded to integers. Input hashes cover the stacked arrays as little-endian signed 64-bit integers. Tests regenerate and compare all ten inputs.

All seven policies share the same inputs, eight workers, 72 request slots per worker, and ready-pool limit of 608. Both BR variants use prompt-descending candidate order, R=16, and a greedy-stage threshold of seven free slots. BR-0 uses H=0, α=1, and β=G; BR-H uses H=80, α=1, β=43, and γ=0.86. R bounds candidate requests, while G denotes workers. Exact subset enumeration supports R from 1 to 16.

## Simulation

Requests remain on their assigned worker. Each active request emits one token per synchronized decode step. With context load defined as prompt tokens plus generated tokens before the step:

```text
step_seconds = 0.0048875 + 1.0052490234375e-7 * max_worker_context_load
```

The ready pool replenishes on demand. BR-H projects growth and departure using the exact remaining lengths of active requests. Within a routing round, newly admitted requests add a constant prompt-length increment to the projected loads; projections are rebuilt at the next round. The simulator does not model prefill execution or time-stamped production arrivals. Router wall time is measured separately and excluded from the simulated clock.

## Metrics

- **Full throughput:** all emitted output tokens divided by final completion time, including the drain phase.
- **Measurement window:** with `ideal_steps = sum(output)/(G*B)`, use decode steps `[floor(0.1*ideal_steps), floor(0.6*ideal_steps))` for every policy. The reference verifies full slot occupancy and positive unadmitted backlog throughout this window.
- **Window throughput:** emitted tokens divided by elapsed simulated time within the window.
- **Request TPOT:** `(finish_time - first_token_time)/(output_length - 1)`. P95 is computed over requests with more than one output token, then averaged over seeds.
- **Load spread:** `max(load)-min(load)`. The deficit objective is `G*max(load)-sum(load)`.
- **Paired gains:** `mean_seed(100*(method/reference-1))` for the same metric and input. Summaries include both full and window throughput.

## Outputs and verification

```bash
balanceroute reproduce --workers 3 --output results/r16
balanceroute verify results/r16
```

`manifest.json` records the configuration, software versions, source and input hashes, and completion status. `metrics.jsonl` and `metrics.csv` contain per-seed records; `summary.json`, `summary.csv`, and `paired_gains.csv` contain aggregate results. Summary standard deviations use `ddof=1` and are null for a single seed. Saved inputs and window traces are in `inputs/` and `traces/`.

Verification checks input hashes, configuration, complete seed/policy coverage, 23 deterministic metrics, saved window traces, and the JSON summary. Metric comparisons use relative and absolute tolerances of 1e-9. Hardware-dependent routing timings are excluded. A mismatch writes a failed `verification.json` and exits with a nonzero status.

Full verification requires all ten seeds and seven policies. `smoke` runs all seven policies on seed 100 and verifies this subset automatically. To recheck its saved directory, use `balanceroute verify <directory> --allow-subset`. Each run requires a new output directory.

## Custom runs and plotting

Export with `balanceroute show-config --output experiment.json`, then use `balanceroute run --config experiment.json --output results/custom`. Set `input_source` to `generated` for new seeds or workload distributions. When changing G, also set BR-0's β to G. Custom runs save the same metrics without checking against the fixed reference; inspect `window_fully_saturated` when interpreting their window results. Small workloads may require adjusted window fractions.

To plot a completed run:

```bash
python -m pip install -e ".[plot]"
python examples/plot_results.py results/r16
```

GitHub Actions runs the test suite on Python 3.11–3.13. Manually running the workflow also executes the full reference benchmark and uploads its results.
